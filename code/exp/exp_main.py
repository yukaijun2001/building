from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from data_provider.data_loader import SplitData
from exp.exp_basic import ExpBasic
from utils.config import ExperimentConfig


class ExpMain(ExpBasic):
    def __init__(self, cfg: ExperimentConfig, split: SplitData) -> None:
        super().__init__(cfg, split)
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
        self.final_lr = cfg.lr

    def _select_optimizer(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

    def _select_criterion(self):
        return nn.MSELoss()

    def train_one_epoch(self, loader) -> float:
        self.model.train()
        total = 0.0
        count = 0
        for x, y in tqdm(loader, desc="train", leave=False):
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(x).prediction
            loss = self.criterion(out, y)
            loss.backward()
            if self.cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
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
            x = x.to(self.device)
            y = y.to(self.device)
            out = self.model(x).prediction
            loss = self.criterion(out, y)
            total += loss.item() * x.size(0)
            count += x.size(0)
        return total / max(1, count)

    def train(self, building_name: str, checkpoint_path: Path | None = None) -> list[dict[str, float]]:
        _train_data, train_loader = self._get_data("train")
        _val_data, val_loader = self._get_data("val")
        best_state = None
        best_val = float("inf")
        bad_epochs = 0
        history = []

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.vali(val_loader)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            print(f"{building_name} epoch {epoch:03d}: train={train_loss:.5f} val={val_loss:.5f}")

            self.scheduler.step(val_loss)
            self.final_lr = float(self.optimizer.param_groups[0]["lr"])

            if val_loss < best_val:
                best_val = val_loss
                self.best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                if checkpoint_path is not None:
                    torch.save(best_state, checkpoint_path)
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.cfg.patience:
                    break

        self.stopped_epoch = history[-1]["epoch"] if history else 0
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_val_loss = float(best_val)
        return history

    @torch.no_grad()
    def predict(self, loader=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if loader is None:
            _data, loader = self._get_data("test")
        self.model.eval()
        preds = []
        trues = []
        histories = []
        for x, y in loader:
            histories.append(x.numpy())
            x = x.to(self.device)
            out = self.model(x).prediction.cpu().numpy()
            preds.append(out)
            trues.append(y.numpy())
        return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0), np.concatenate(histories, axis=0)

    def test(self) -> dict[str, float]:
        _test_data, test_loader = self._get_data("test")
        pred, true, _history = self.predict(test_loader)
        pred_raw = inverse_target(pred, self.split)
        true_raw = inverse_target(true, self.split)
        result = metrics(pred, true)
        self.last_pred = pred_raw
        self.last_true = true_raw
        self.last_pred_scaled = pred
        self.last_true_scaled = true
        return result

    def save_results(self, building_name: str, output_dir: Path) -> None:
        np.savez_compressed(
            output_dir / f"{building_name}_predictions.npz",
            pred=self.last_pred,
            true=self.last_true,
            pred_scaled=self.last_pred_scaled,
            true_scaled=self.last_true_scaled,
        )
        torch.save(self.model.state_dict(), output_dir / f"{building_name}_model.pt")


def metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float | list[float]]:
    p = pred.reshape(-1)
    y = true.reshape(-1)
    err = p - y
    abs_err = np.abs(err)
    mse = float(np.mean(err**2))
    mae = float(np.mean(abs_err))
    denom = np.maximum(np.abs(y), 1e-6)
    mape = float(np.mean(abs_err / denom) * 100)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "mse": mse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
    }


def naive_metrics(pred: np.ndarray, history: np.ndarray, true: np.ndarray, target_index: int) -> dict[str, float]:
    pred_len = true.shape[1]
    target_history = history[:, :, target_index : target_index + 1]
    daily_pred = target_history[:, -pred_len:, :]
    weekly_pred = target_history[:, -168 : -168 + pred_len, :] if target_history.shape[1] >= 168 else daily_pred

    model_mae = float(np.mean(np.abs(pred - true)))
    daily_mae = float(np.mean(np.abs(daily_pred - true)))
    weekly_mae = float(np.mean(np.abs(weekly_pred - true)))
    return {
        "daily_naive_mae": daily_mae,
        "weekly_naive_mae": weekly_mae,
        "mase_daily": float(model_mae / max(daily_mae, 1e-6)),
        "mase_weekly": float(model_mae / max(weekly_mae, 1e-6)),
    }


def inverse_target(values: np.ndarray, split: SplitData) -> np.ndarray:
    target = split.target_index
    return values * split.scaler["std"][target] + split.scaler["mean"][target]


def inverse_all(values: np.ndarray, split: SplitData) -> np.ndarray:
    return values * split.scaler["std"].reshape(1, 1, -1) + split.scaler["mean"].reshape(1, 1, -1)


def fit_building(
    cfg: ExperimentConfig,
    building_name: str,
    split: SplitData,
    output_dir: Path,
) -> dict[str, float]:
    exp = ExpMain(cfg, split)
    exp.train(building_name, output_dir / f"{building_name}_model.pt")
    result = exp.test()
    exp.save_results(building_name, output_dir)
    return result
