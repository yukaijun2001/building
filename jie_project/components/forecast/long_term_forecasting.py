import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from utils.early_stopping import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.npy_result import save_npy_results
from utils.save_result import record_metrics, save_result_data
from .forecast_basic import ForecastBasic
from ..data_provider.data_factory import data_provider

warnings.filterwarnings('ignore')


class LongTermForecast(ForecastBasic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args)
        self.device = self.args.gpu
        model = model.to(self.device)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        p_list = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            else:
                p_list.append(p)
                print(n, p.dtype, p.shape)
        model_optim = optim.Adam([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        print('trainable parameters is {}'.format(sum(p.numel() for p in p_list)))
        print('next learning rate is {}'.format(self.args.learning_rate))
        return model_optim

    @classmethod
    def _select_criterion(cls):
        criterion = nn.MSELoss()
        return criterion

    def _model_profile(self):
        total_parameters = sum(p.numel() for p in self.model.parameters())
        trainable_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        parameter_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        return {
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "parameter_size_mb": parameter_size / (1024 ** 2),
        }

    def _cuda_synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _prediction_window(self, outputs, batch_y, pred_len=None):
        pred_len = pred_len or self.args.pred_len
        return outputs[:, -pred_len:, :], batch_y[:, -pred_len:, :]

    @staticmethod
    def _inverse_target(data, data_set):
        scaler = getattr(data_set, "scaler", None)
        if scaler is None or not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
            return data
        target_mean = float(scaler.mean_[-1])
        target_scale = float(scaler.scale_[-1])
        return data * target_scale + target_mean

    def vali(self, vali_data, vali_loader, criterion, is_test=False):
        total_loss = []
        total_count = []
        time_now = time.time()
        test_steps = len(vali_loader)
        iter_count = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                outputs, batch_y = self._prediction_window(outputs, batch_y.to(self.device))

                loss = criterion(outputs, batch_y)

                loss = loss.detach().cpu()
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])
                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (test_steps - i)
                    print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                    iter_count = 0
                    time_now = time.time()
        total_loss = np.average(total_loss, weights=total_count)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        self._cuda_synchronize()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        train_start_time = time.time()
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)

        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        criterion = self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0

            loss_val = torch.tensor(0., device="cuda")
            count = torch.tensor(0., device="cuda")

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                        outputs, batch_y = self._prediction_window(outputs, batch_y)
                        loss = criterion(outputs, batch_y)
                        loss_val += loss
                        count += 1
                else:
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                    outputs, batch_y = self._prediction_window(outputs, batch_y)
                    loss = criterion(outputs, batch_y)
                    loss_val += loss
                    count += 1

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = loss_val.item() / count.item()

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion, is_test=True)
            print("Epoch: {}, Steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f} Test Loss: {:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.cosine:
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path), strict=False)
        self._cuda_synchronize()
        training_profile = self._model_profile()
        training_profile["training_time_seconds"] = time.time() - train_start_time
        if torch.cuda.is_available():
            training_profile["training_gpu_memory_mb"] = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            training_profile["training_gpu_reserved_mb"] = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
        self.training_profile = training_profile
        print("training time: {:.6f}s".format(training_profile["training_time_seconds"]))
        print("parameter size: {:.6f} MB".format(training_profile["parameter_size_mb"]))
        if torch.cuda.is_available():
            print("training gpu memory: {:.6f} MB".format(training_profile["training_gpu_memory_mb"]))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        token_len = self.args.seq_len - self.args.label_len
        print("info:", self.args.test_seq_len, self.args.test_label_len, token_len, self.args.test_pred_len)
        if test:
            print('loading model')
            setting = self.args.test_dir
            best_model_path = self.args.test_file_name

            print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
            load_item = torch.load(os.path.join(self.args.checkpoints, setting, best_model_path))
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in load_item.items()}, strict=False)

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        time_now = time.time()
        test_steps = len(test_loader)
        iter_count = 0
        self.model.eval()
        self._cuda_synchronize()
        epoch_time = time.time()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                inference_steps = self.args.test_pred_len // self.args.pred_len
                dis = self.args.test_pred_len - inference_steps * self.args.pred_len
                if dis != 0:
                    inference_steps += 1
                pred_y = []
                for j in range(inference_steps):
                    if len(pred_y) != 0:
                        batch_x = torch.cat([batch_x[:, self.args.pred_len:, :], pred_y[-1]], dim=1)
                        tmp = batch_y_mark[:, j - 1:j, :]
                        batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)

                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                    pred_y.append(outputs[:, -self.args.pred_len:, :])
                pred_y = torch.cat(pred_y, dim=1)
                if dis != 0:
                    pred_y = pred_y[:, :-dis, :]
                batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)
                outputs = pred_y.detach().cpu()
                batch_y = batch_y.detach().cpu()

                pred = outputs
                true = batch_y
                if self.args.inverse:
                    pred = self._inverse_target(pred, test_data)
                    true = self._inverse_target(true, test_data)

                preds.append(pred)
                trues.append(true)
                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (test_steps - i)
                    print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.visualize and i % 4 == 0:
                    gt = np.array(true[0, :, -1])
                    pd = np.array(pred[0, :, -1])
                    lookback = batch_x[0, :, -1].detach().cpu().numpy()
                    if self.args.inverse:
                        lookback = np.asarray(self._inverse_target(lookback, test_data))
                    gt = np.concatenate([lookback, gt], axis=0)
                    pd = np.concatenate([lookback, pd], axis=0)
                    dir_path = folder_path + f'{self.args.test_pred_len}/'
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    visual(gt, pd, os.path.join(dir_path, f'{i}.png'))
        self._cuda_synchronize()
        inference_time = time.time() - epoch_time
        print("cost time: {}".format(inference_time))
        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()

        mae, mse, rmse, mape, mspe, r2 = metric(preds, trues)
        print('mse:{}, mae:{}, mape:{}, r2:{}'.format(mse, mae, mape, r2))
        save_npy_results(setting, preds, trues, [mae, mse, rmse, mape, mspe, r2])
        record_metrics(self.args.data_path, self.args.model_id,
                       [self.args.test_seq_len, self.args.test_label_len, self.args.test_pred_len],
                       mae, mse, mape, r2)
        # save_result_data(preds, trues, self.args.data_path)
        inference_profile = self._model_profile()
        inference_profile["inference_time_seconds"] = inference_time
        return inference_profile
