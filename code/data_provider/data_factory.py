from __future__ import annotations

from torch.utils.data import DataLoader

from utils.config import ExperimentConfig

from .data_loader import SplitData, WindowDataset, prepare_split


def data_provider(split: SplitData, flag: str, cfg: ExperimentConfig):
    if flag == "train":
        data_set = split.train
        shuffle_flag = True
    elif flag == "val":
        data_set = split.val
        shuffle_flag = False
    elif flag == "test":
        data_set = split.test
        shuffle_flag = False
    else:
        raise ValueError(f"Unknown data flag: {flag}")

    data_loader = DataLoader(
        data_set,
        batch_size=cfg.batch_size,
        shuffle=shuffle_flag,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    print(flag, len(data_set))
    return data_set, data_loader


__all__ = ["SplitData", "WindowDataset", "data_provider", "prepare_split"]
