from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_provider.building_energy import SplitData
from exp.exp_basic import Exp_Basic


class Exp_Building_Energy(Exp_Basic):
    def __init__(self, args, split: SplitData) -> None:
        self.split = split
        super().__init__(args)
        self.criterion = self._select_criterion()
        self.optimizer = self._select_optimizer()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
        )
        self.best_epoch = 0
        self.stopped_epoch = 0
        self.best_val_loss = float("inf")

    def _build_model(self):
        self.args.enc_in = len(self.split.feature_cols)
        self.args.dec_in = len(self.split.feature_cols)
        self.args.c_out = 1
        self.args.target_index = self.split.target_index
        model = self.model_dict[self.args.model](self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        if flag == "train":
            data_set = self.split.train
            shuffle_flag = True
        elif flag == "val":
            data_set = self.split.val
            shuffle_flag = False
        elif flag == "test":
            data_set = self.split.test
            shuffle_flag = False
        else:
            raise ValueError(f"Unknown data flag: {flag}")

        data_loader = DataLoader(
            data_set,
            batch_size=self.args.batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args.num_workers,
            drop_last=False,
        )
        print(flag, len(data_set))
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

    def _select_criterion(self):
        return nn.MSELoss()

    def train_one_epoch(self, loader) -> float:
        self.model.train()
        total = 0.0
        count = 0
        for x, y in tqdm(loader, desc="train", leave=False):
            x = x.float().to(self.device)
            y = y.float().to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            out = self._select_target_output(self.model(x, None, None, None))
            loss = self.criterion(out, y)
            loss.backward()
            if self.args.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()
            total += loss.item() * x.size(0)
            count += x.size(0)
        return total / max(1, count)

    @torch.no_grad()
    def vali(self, loader=None) -> float:
        if loader is None:
            _data, loader = self._get_data("val")
        self.model.eval()
        total = 0.0
        count = 0
        for x, y in loader:
            x = x.float().to(self.device)
            y = y.float().to(self.device)
            out = self._select_target_output(self.model(x, None, None, None))
            loss = self.criterion(out, y)
            total += loss.item() * x.size(0)
            count += x.size(0)
        return total / max(1, count)

    def train(self, setting):
        _train_data, train_loader = self._get_data("train")
        _val_data, val_loader = self._get_data("val")
        checkpoint_dir = Path(self.args.checkpoints) / setting
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = checkpoint_dir / "checkpoint.pth"

        best_state = None
        best_val = float("inf")
        bad_epochs = 0
        history = []

        for epoch in range(1, self.args.train_epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.vali(val_loader)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            print(f"{setting} epoch {epoch:03d}: train={train_loss:.5f} val={val_loss:.5f}")

            self.scheduler.step(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                self.best_epoch = epoch
                model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, best_model_path)
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.args.patience:
                    print("Early stopping")
                    break

        self.stopped_epoch = history[-1]["epoch"] if history else 0
        self.best_val_loss = float(best_val)
        if best_state is not None:
            model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            model.load_state_dict(best_state)
        return self.model

    @torch.no_grad()
    def predict(self, loader=None):
        if loader is None:
            _data, loader = self._get_data("test")
        self.model.eval()
        preds = []
        trues = []
        histories = []
        for x, y in loader:
            histories.append(x.numpy())
            x = x.float().to(self.device)
            out = self._select_target_output(self.model(x, None, None, None)).detach().cpu().numpy()
            preds.append(out)
            trues.append(y.numpy())
        return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0), np.concatenate(histories, axis=0)

    def _select_target_output(self, outputs: torch.Tensor) -> torch.Tensor:
        if outputs.shape[-1] == 1:
            return outputs
        return outputs[:, :, self.split.target_index : self.split.target_index + 1]

    def test(self, setting, test=0):
        _test_data, test_loader = self._get_data("test")
        if test:
            print("loading model")
            model_path = Path(self.args.checkpoints) / setting / "checkpoint.pth"
            model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            model.load_state_dict(torch.load(model_path, map_location=self.device))

        pred_scaled, true_scaled, history = self.predict(test_loader)
        if getattr(self.args, "inverse", False):
            pred = inverse_target(pred_scaled, self.split)
            true = inverse_target(true_scaled, self.split)
        else:
            pred = pred_scaled
            true = true_scaled

        metrics = source_metrics(pred, true)
        mae, mse = metrics["mae"], metrics["mse"]
        dtw = "Not calculated"
        print(f"mse:{mse}, mae:{mae}, dtw:{dtw}")

        result_dir = Path("./results") / setting
        result_dir.mkdir(parents=True, exist_ok=True)
        np.save(result_dir / "metrics.npy", np.array([mae, mse, metrics["rmse"], metrics["mape"] / 100.0, metrics["mspe"]]))
        np.save(result_dir / "pred.npy", pred)
        np.save(result_dir / "true.npy", true)

        test_result_dir = Path("./test_results") / setting
        test_result_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(test_result_dir / "0.npz", pred=pred[0], true=true[0])
        save_visual(test_result_dir / "0.pdf", pred[0], true[0])

        with open("result_long_term_forecast.txt", "a") as f:
            f.write(setting + "  \n")
            f.write(f"mse:{mse}, mae:{mae}, dtw:{dtw}")
            f.write("\n\n")

        self.last_pred_scaled = pred_scaled
        self.last_true_scaled = true_scaled
        self.last_history_scaled = history
        self.last_metrics = metrics
        return metrics

    def save_source_outputs(self, building_name: str, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_raw = inverse_target(self.last_pred_scaled, self.split)
        true_raw = inverse_target(self.last_true_scaled, self.split)
        np.savez_compressed(
            output_dir / f"{building_name}_predictions.npz",
            pred=pred_raw,
            true=true_raw,
            pred_scaled=self.last_pred_scaled,
            true_scaled=self.last_true_scaled,
        )
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        torch.save(model.state_dict(), output_dir / f"{building_name}_model.pt")


def source_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    p = pred.reshape(-1)
    y = true.reshape(-1)
    err = p - y
    abs_err = np.abs(err)
    mse = float(np.mean(err**2))
    mae = float(np.mean(abs_err))
    denom = np.maximum(np.abs(y), 1e-6)
    mape = float(np.mean(abs_err / denom) * 100)
    mspe = float(np.mean(np.square(err / denom)))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"mse": mse, "mae": mae, "mape": mape, "r2": r2, "rmse": rmse, "mspe": mspe}


def inverse_target(values: np.ndarray, split: SplitData) -> np.ndarray:
    target = split.target_index
    return values * split.scaler["std"][target] + split.scaler["mean"][target]


def save_visual(path: Path, pred: np.ndarray, true: np.ndarray) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    plt.figure()
    plt.plot(true.reshape(-1), label="GroundTruth", linewidth=2)
    plt.plot(pred.reshape(-1), label="Prediction", linewidth=2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
