from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SplitData:
    train: "WindowDataset"
    val: "WindowDataset"
    test: "WindowDataset"
    feature_cols: list[str]
    target_index: int
    scaler: dict[str, np.ndarray]


class WindowDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        seq_len: int,
        pred_len: int,
        target_index: int,
        stride: int = 1,
        limit_windows: int | None = None,
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if pred_len <= 0:
            raise ValueError("pred_len must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        self.values = values.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_index = target_index
        total = max(0, len(values) - seq_len - pred_len + 1)
        starts = np.arange(0, total, stride, dtype=np.int64)
        if limit_windows is not None:
            starts = starts[:limit_windows]
        self.starts = starts

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[idx])
        mid = start + self.seq_len
        end = mid + self.pred_len
        x = self.values[start:mid]
        y = self.values[mid:end, self.target_index : self.target_index + 1]
        return torch.from_numpy(x), torch.from_numpy(y)


def _is_subhour_frequency(freq: str) -> bool:
    try:
        return pd.tseries.frequencies.to_offset(freq).nanos < pd.Timedelta(hours=1).value
    except ValueError:
        return any(token in str(freq).lower() for token in ("min", "t", "s"))


def load_building_csv(path: Path, target_col: str, freq: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "timestamp"})
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} does not contain timestamp column")
    if target_col not in df.columns:
        raise ValueError(f"{path} does not contain target column {target_col!r}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp")

    numeric_cols = [c for c in df.columns if c != target_col] + [target_col]
    df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df = df.reindex(full_index)
    df.index.name = "timestamp"
    return add_time_features(df, freq)


def add_time_features(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    hour = idx.hour.to_numpy()
    minute = idx.minute.to_numpy()
    minute_of_day = hour * 60 + minute
    dayofweek = idx.dayofweek.to_numpy()
    month = idx.month.to_numpy()
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    if _is_subhour_frequency(freq):
        out["minute_sin"] = np.sin(2 * np.pi * minute_of_day / (24 * 60))
        out["minute_cos"] = np.cos(2 * np.pi * minute_of_day / (24 * 60))
    out["dow_sin"] = np.sin(2 * np.pi * dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dayofweek / 7)
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    return out


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    return df.interpolate(method="time", limit_direction="both").ffill().bfill().fillna(0.0)


def fit_preprocessor(train_df: pd.DataFrame) -> dict[str, np.ndarray]:
    values = train_df.to_numpy(dtype=np.float32)
    q01 = np.nanquantile(values, 0.01, axis=0)
    q99 = np.nanquantile(values, 0.99, axis=0)
    clipped = np.clip(values, q01, q99)
    mean = clipped.mean(axis=0)
    std = clipped.std(axis=0)
    std[std < 1e-6] = 1.0
    return {"q01": q01, "q99": q99, "mean": mean, "std": std}


def transform(df: pd.DataFrame, scaler: dict[str, np.ndarray]) -> np.ndarray:
    values = df.to_numpy(dtype=np.float32)
    values = np.clip(values, scaler["q01"], scaler["q99"])
    return (values - scaler["mean"]) / scaler["std"]


def prepare_building_split(
    path: Path,
    target_col: str,
    seq_len: int,
    pred_len: int,
    train_ratio: float,
    val_ratio: float,
    stride: int,
    freq: str,
    limit_windows: int | None = None,
) -> SplitData:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    df = fill_missing(load_building_csv(path, target_col, freq))
    feature_cols = [c for c in df.columns if c != target_col] + [target_col]
    df = df[feature_cols]
    target_index = feature_cols.index(target_col)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_df = df.iloc[:train_end]
    val_df = df.iloc[max(0, train_end - seq_len) : val_end]
    test_df = df.iloc[max(0, val_end - seq_len) :]

    scaler = fit_preprocessor(train_df)
    train_values = transform(train_df, scaler)
    val_values = transform(val_df, scaler)
    test_values = transform(test_df, scaler)

    train = WindowDataset(train_values, seq_len, pred_len, target_index, stride, limit_windows)
    val = WindowDataset(val_values, seq_len, pred_len, target_index, stride, limit_windows)
    test = WindowDataset(test_values, seq_len, pred_len, target_index, stride, limit_windows)
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError(
            f"{path} does not have enough rows for seq_len={seq_len}, pred_len={pred_len}, "
            f"and split ratios train={train_ratio}, val={val_ratio}"
        )
    return SplitData(train, val, test, feature_cols, target_index, scaler)
