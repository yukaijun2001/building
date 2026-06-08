from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

TSL_ROOT = Path(__file__).resolve().parents[1]
if str(TSL_ROOT) not in sys.path:
    sys.path.insert(0, str(TSL_ROOT))

from data_provider.building_energy import SplitData, prepare_building_split
from exp.exp_building_energy import inverse_target, save_visual, source_metrics
from 热插拔实验.hotplug_model import HotPlugPatchGatedResidual, normalize_baseline_name


METRIC_NAMES = ["mse", "mae", "mape", "r2"]


def parse_args():
    parser = argparse.ArgumentParser(description="Hot-plug baseline + patch-gated residual for BuildingEnergy")
    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--is_training", type=int, default=1)
    parser.add_argument("--model_id", type=str, default="HotPlug_672_24")
    parser.add_argument("--root_path", type=str, default=str(TSL_ROOT.parent / "data"))
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--target", type=str, default="electricity")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--checkpoints", type=str, default=str(TSL_ROOT / "热插拔实验" / "checkpoints"))
    parser.add_argument("--source_output_dir", type=str, default=str(TSL_ROOT / "热插拔实验" / "outputs"))

    parser.add_argument("--seq_len", type=int, default=672)
    parser.add_argument("--label_len", type=int, default=24)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--limit_windows", type=int, default=None)
    parser.add_argument("--max_files", type=int, default=None)

    parser.add_argument("--baseline", type=str, default="nlinear")
    parser.add_argument("--disable_residual", action="store_true", default=False)
    parser.add_argument("--baseline_patch_len", type=int, default=16)
    parser.add_argument("--baseline_patch_stride", type=int, default=8)
    parser.add_argument("--baseline_d_model", type=int, default=128)
    parser.add_argument("--baseline_n_heads", type=int, default=4)
    parser.add_argument("--baseline_e_layers", type=int, default=2)
    parser.add_argument("--baseline_d_ff", type=int, default=256)
    parser.add_argument("--baseline_use_norm", type=int, default=1)
    parser.add_argument("--mapl_project", type=str, default="/home/ykj/build/jie_project")
    parser.add_argument("--mapl_mark_root", type=str, default="/home/ykj/build/jie_project/dataset/building")
    parser.add_argument("--llm_ckp_dir", type=str, default="/home/ykj/build/llama_model")
    parser.add_argument("--mlp_hidden_dim", type=int, default=1024)
    parser.add_argument("--mlp_hidden_layers", type=int, default=2)
    parser.add_argument("--mlp_activation", type=str, default="relu")
    parser.add_argument("--num_experts", type=int, default=1)
    parser.add_argument("--lambda_reg", type=float, default=0.01)
    parser.add_argument("--mapl_use_amp", action="store_true", default=False)

    parser.add_argument("--patch_len", type=int, default=24)
    parser.add_argument("--patch_stride", type=int, default=12)
    parser.add_argument("--top_k_patches", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--model_dim", type=int, default=64)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fft_keep_ratio", type=float, default=0.35)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--decomp_method", type=str, default="moving_avg")
    parser.add_argument("--channel_independence", type=int, default=1)
    parser.add_argument("--use_norm", type=int, default=1)
    parser.add_argument("--down_sampling_layers", type=int, default=0)
    parser.add_argument("--down_sampling_window", type=int, default=1)
    parser.add_argument("--down_sampling_method", type=str, default=None)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--des", type=str, default="HotPlug")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--no_use_gpu", action="store_false", dest="use_gpu")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    args.baseline = normalize_baseline_name(args.baseline)
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_device(args) -> torch.device:
    if torch.cuda.is_available() and args.use_gpu:
        print("Using GPU")
        return torch.device(f"cuda:{args.gpu}")
    args.use_gpu = False
    print("Using CPU")
    return torch.device("cpu")


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def summarize_results(results: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        metric: float(np.mean([r[metric] for r in results.values()]))
        for metric in METRIC_NAMES
        if results and all(metric in r for r in results.values())
    }


def setting_name(args, building_name: str, ii: int) -> str:
    residual_suffix = "_baseline_only" if args.disable_residual else "_residual"
    return (
        f"{args.task_name}_{args.model_id}_{building_name}_HotPlug_{args.baseline}{residual_suffix}_BuildingEnergy"
        f"_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}"
        f"_bpl{args.baseline_patch_len}_bdm{args.baseline_d_model}"
        f"_plen{args.patch_len}_pstride{args.patch_stride}_topkp{args.top_k_patches}"
        f"_hs{args.hidden_size}_mdim{args.model_dim}_lstm{args.lstm_layers}"
        f"_fft{args.fft_keep_ratio}_seed{args.seed}_{args.des}_{ii}"
    )


class MarkedWindowDataset(Dataset):
    def __init__(self, base: Dataset, marks: torch.Tensor, global_offset: int, label_len: int) -> None:
        self.base = base
        self.marks = marks.float().cpu()
        self.global_offset = int(global_offset)
        self.token_len = int(base.seq_len - label_len)
        if self.token_len <= 0:
            raise ValueError("MaPL-LLM requires label_len < seq_len so token_len is positive.")
        self.expected_steps = (base.seq_len + self.token_len - 1) // self.token_len

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        start = self.global_offset + int(self.base.starts[idx])
        end = start + self.base.seq_len
        x_mark = self.marks[start:end:self.token_len]
        if x_mark.shape[0] < self.expected_steps:
            if x_mark.shape[0] == 0:
                pad = torch.zeros(self.expected_steps, self.marks.shape[-1], dtype=self.marks.dtype)
                x_mark = pad
            else:
                pad = x_mark[-1:].repeat(self.expected_steps - x_mark.shape[0], 1)
                x_mark = torch.cat([x_mark, pad], dim=0)
        elif x_mark.shape[0] > self.expected_steps:
            x_mark = x_mark[: self.expected_steps]
        return x, y, x_mark


def attach_mapl_marks(split: SplitData, csv_path: Path, args) -> SplitData:
    mark_path = Path(args.mapl_mark_root) / f"{csv_path.stem}.pt"
    if not mark_path.exists():
        raise FileNotFoundError(
            f"MaPL-LLM baseline needs {mark_path}. Set --mapl_mark_root to the directory containing the .pt files."
        )
    marks = torch.load(mark_path, map_location="cpu")
    if not isinstance(marks, torch.Tensor):
        marks = torch.as_tensor(marks)
    if marks.ndim != 2:
        raise ValueError(f"{mark_path} must contain a 2D tensor [time, hidden_dim], got shape {tuple(marks.shape)}")

    train_offset = 0
    val_offset = max(0, len(split.train.values) - args.seq_len)
    val_end = val_offset + len(split.val.values)
    test_offset = max(0, val_end - args.seq_len)

    required = test_offset + len(split.test.values)
    if marks.shape[0] < required:
        raise ValueError(f"{mark_path} has {marks.shape[0]} rows, but this split needs at least {required}.")

    return SplitData(
        train=MarkedWindowDataset(split.train, marks, train_offset, args.label_len),
        val=MarkedWindowDataset(split.val, marks, val_offset, args.label_len),
        test=MarkedWindowDataset(split.test, marks, test_offset, args.label_len),
        feature_cols=split.feature_cols,
        target_index=split.target_index,
        scaler=split.scaler,
    )


def unpack_batch(batch, device: torch.device):
    if len(batch) == 2:
        x, y = batch
        x_mark = None
    elif len(batch) == 3:
        x, y, x_mark = batch
        x_mark = x_mark.float().to(device)
    else:
        raise ValueError(f"Expected batch of length 2 or 3, got {len(batch)}")
    return x.float().to(device), y.float().to(device), x_mark


class HotPlugExperiment:
    def __init__(self, args, split: SplitData, device: torch.device) -> None:
        self.args = args
        self.split = split
        self.device = device
        args.enc_in = len(split.feature_cols)
        args.dec_in = len(split.feature_cols)
        args.c_out = 1
        args.target_index = split.target_index
        self.model = HotPlugPatchGatedResidual(args).float().to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
        )

    def loader(self, flag: str) -> DataLoader:
        data_set = getattr(self.split, flag)
        print(flag, len(data_set))
        return DataLoader(
            data_set,
            batch_size=self.args.batch_size,
            shuffle=(flag == "train"),
            num_workers=self.args.num_workers,
            drop_last=False,
        )

    def train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total = 0.0
        count = 0
        for batch in tqdm(loader, desc="train", leave=False):
            x, y, x_mark = unpack_batch(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(x, x_mark)
            loss = self.criterion(out, y)
            loss.backward()
            if self.args.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()
            total += loss.item() * x.size(0)
            count += x.size(0)
        return total / max(1, count)

    @torch.no_grad()
    def evaluate_loss(self, loader: DataLoader) -> float:
        self.model.eval()
        total = 0.0
        count = 0
        for batch in loader:
            x, y, x_mark = unpack_batch(batch, self.device)
            out = self.model(x, x_mark)
            loss = self.criterion(out, y)
            total += loss.item() * x.size(0)
            count += x.size(0)
        return total / max(1, count)

    def train(self, setting: str) -> None:
        train_loader = self.loader("train")
        val_loader = self.loader("val")
        checkpoint_dir = Path(self.args.checkpoints) / setting
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = checkpoint_dir / "checkpoint.pth"

        best_state = None
        best_val = float("inf")
        bad_epochs = 0
        for epoch in range(1, self.args.train_epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.evaluate_loss(val_loader)
            print(f"{setting} epoch {epoch:03d}: train={train_loss:.5f} val={val_loss:.5f}")
            self.scheduler.step(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                torch.save(best_state, best_model_path)
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.args.patience:
                    print("Early stopping")
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)

    @torch.no_grad()
    def predict(self, loader: DataLoader):
        self.model.eval()
        preds = []
        trues = []
        for batch in loader:
            x, y, x_mark = unpack_batch(batch, self.device)
            out = self.model(x, x_mark).detach().cpu().numpy()
            preds.append(out)
            trues.append(y.detach().cpu().numpy())
        return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)

    def test(self, setting: str, building_name: str, output_dir: Path, load_checkpoint: bool = False):
        if load_checkpoint:
            model_path = Path(self.args.checkpoints) / setting / "checkpoint.pth"
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        test_loader = self.loader("test")
        pred, true = self.predict(test_loader)
        metrics = source_metrics(pred, true)
        print(f"mse:{metrics['mse']}, mae:{metrics['mae']}, dtw:Not calculated")

        result_dir = TSL_ROOT / "热插拔实验" / "results" / setting
        result_dir.mkdir(parents=True, exist_ok=True)
        np.save(result_dir / "metrics.npy", np.array([metrics["mae"], metrics["mse"], metrics["rmse"], metrics["mape"] / 100.0, metrics["mspe"]]))
        np.save(result_dir / "pred.npy", pred)
        np.save(result_dir / "true.npy", true)

        test_result_dir = TSL_ROOT / "热插拔实验" / "test_results" / setting
        test_result_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(test_result_dir / "0.npz", pred=pred[0], true=true[0])
        save_visual(test_result_dir / "0.pdf", pred[0], true[0])

        pred_raw = inverse_target(pred, self.split)
        true_raw = inverse_target(true, self.split)
        np.savez_compressed(
            output_dir / f"{building_name}_predictions.npz",
            pred=pred_raw,
            true=true_raw,
            pred_scaled=pred,
            true_scaled=true,
        )
        torch.save(self.model.state_dict(), output_dir / f"{building_name}_model.pt")
        return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = configure_device(args)

    root_path = Path(args.root_path)
    if args.data_path:
        csv_files = [root_path / args.data_path]
    else:
        csv_files = sorted(root_path.glob("*.csv"))
    if args.max_files is not None:
        csv_files = csv_files[: args.max_files]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {root_path}")

    source_output_dir = Path(args.source_output_dir) / f"source_{args.baseline}_{args.seq_len}_{args.pred_len}"
    source_output_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), source_output_dir / "config.json")

    results = {}
    for ii in range(args.itr):
        for path in csv_files:
            building_name = path.stem
            print(f"\n=== {building_name} | baseline={args.baseline} ===")
            split = prepare_building_split(
                path=path,
                target_col=args.target,
                seq_len=args.seq_len,
                pred_len=args.pred_len,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                stride=args.stride,
                limit_windows=args.limit_windows,
            )
            if args.baseline == "mapl_llm":
                split = attach_mapl_marks(split, path, args)
            exp = HotPlugExperiment(args, split, device)
            setting = setting_name(args, building_name, ii)
            print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            if args.is_training:
                exp.train(setting)
            print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            result = exp.test(setting, building_name, source_output_dir, load_checkpoint=not args.is_training)
            public_result = {k: result[k] for k in METRIC_NAMES}
            results[building_name] = public_result
            save_json({**results, "_mean": summarize_results(results)}, source_output_dir / "metrics.json")
            print(f"{building_name} test: {public_result}")

    results["_mean"] = summarize_results(results)
    save_json(results, source_output_dir / "metrics.json")
    print("\n=== Mean metrics ===")
    print(results["_mean"])


if __name__ == "__main__":
    main()
