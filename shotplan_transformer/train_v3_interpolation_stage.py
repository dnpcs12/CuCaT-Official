from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from shotplan_transformer.dataset import load_manifest
from shotplan_transformer.v3_interpolation_stage import (
    V3InterpolationStageConfig,
    V3InterpolationStageModel,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (
    WORKSPACE_ROOT
    / "shotplan_transformer"
    / "results"
    / "discrete_v3_1_30fps_close_dense_temporal_delta_interp_dcm_split_dataset"
)
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "shotplan_transformer" / "results" / "v3_interpolation_stage_runs"
TARGET_FIELD = "interpolation"
IGNORED_INPUT_FIELDS = {TARGET_FIELD, "transition", "vertical_composition"}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a post-stage ReverseV3 interpolation classifier.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--class-weight-power", type=float, default=0.35)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="base")
    return parser


def _device_from_arg(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


class V3InterpolationDataset(Dataset):
    """Build prev/current/next shot-plan context samples from a built V3 dataset."""

    def __init__(self, dataset_root: Path, split_name: str) -> None:
        self.dataset_root = Path(dataset_root)
        self.manifest = load_manifest(self.dataset_root)
        label_fields = [str(field) for field in self.manifest["label_fields"]]
        if TARGET_FIELD not in label_fields:
            raise ValueError(f"Dataset must include {TARGET_FIELD!r} in label_fields: {self.dataset_root}")
        self.input_fields = tuple(field for field in label_fields if field not in IGNORED_INPUT_FIELDS)
        self.vocabularies = {
            str(field): [str(value) for value in values]
            for field, values in self.manifest["vocabularies"].items()
        }
        self.pad_ids = {field: len(self.vocabularies[field]) for field in self.input_fields}
        song_lookup = {entry["song_id"]: entry for entry in self.manifest["songs"]}

        chunks: dict[str, list[np.ndarray]] = {}
        for prefix in ("prev", "current", "next"):
            for field in self.input_fields:
                chunks[f"{prefix}_{field}"] = []
        prev_delta_chunks: list[np.ndarray] = []
        next_delta_chunks: list[np.ndarray] = []
        target_chunks: list[np.ndarray] = []

        for song_id in self.manifest["splits"][split_name]:
            entry = song_lookup.get(song_id)
            if entry is None:
                continue
            arrays = np.load(entry["path"], allow_pickle=False)
            frames = np.asarray(arrays["sampled_frames"], dtype=np.int32)
            if frames.size == 0:
                continue
            targets = np.asarray(arrays[f"label_{TARGET_FIELD}"], dtype=np.int64)[:, 0]
            valid = targets >= 0

            for field in self.input_fields:
                labels = np.asarray(arrays[f"label_{field}"], dtype=np.int64)[:, 0]
                pad_id = self.pad_ids[field]
                prev_labels = np.full_like(labels, pad_id)
                next_labels = np.full_like(labels, pad_id)
                if labels.size > 1:
                    prev_labels[1:] = labels[:-1]
                    next_labels[:-1] = labels[1:]
                chunks[f"prev_{field}"].append(prev_labels[valid])
                chunks[f"current_{field}"].append(labels[valid])
                chunks[f"next_{field}"].append(next_labels[valid])

            prev_delta = np.full((frames.shape[0],), 0, dtype=np.float32)
            next_delta = np.full((frames.shape[0],), 0, dtype=np.float32)
            if frames.size > 1:
                prev_delta[1:] = frames[1:] - frames[:-1]
                next_delta[:-1] = frames[1:] - frames[:-1]
            prev_delta_chunks.append(prev_delta[valid])
            next_delta_chunks.append(next_delta[valid])
            target_chunks.append(targets[valid])

        if not target_chunks:
            raise ValueError(f"No interpolation samples found for split {split_name}")
        self.arrays = {key: np.concatenate(values, axis=0).astype(np.int64) for key, values in chunks.items()}
        self.prev_delta = np.concatenate(prev_delta_chunks, axis=0).astype(np.float32)
        self.next_delta = np.concatenate(next_delta_chunks, axis=0).astype(np.float32)
        self.targets = np.concatenate(target_chunks, axis=0).astype(np.int64)

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(int(values[index]), dtype=torch.long)
            for key, values in self.arrays.items()
        }
        item["prev_delta"] = torch.tensor(float(self.prev_delta[index]), dtype=torch.float32)
        item["next_delta"] = torch.tensor(float(self.next_delta[index]), dtype=torch.float32)
        item["target"] = torch.tensor(int(self.targets[index]), dtype=torch.long)
        return item


def _class_weights(targets: np.ndarray, class_count: int, power: float, device: torch.device) -> torch.Tensor | None:
    if power <= 0.0:
        return None
    counts = np.bincount(targets.astype(np.int64), minlength=class_count).astype(np.float64)
    weights = np.power(np.maximum(counts, 1.0), -float(power))
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _run_epoch(
    model: V3InterpolationStageModel,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    class_weights: torch.Tensor | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total = 0
    confusion: dict[int, list[int]] = {}
    for batch in loader:
        labels = {
            key: value.to(device)
            for key, value in batch.items()
            if key not in {"prev_delta", "next_delta", "target"}
        }
        target = batch["target"].to(device)
        logits = model(
            labels,
            prev_delta=batch["prev_delta"].to(device),
            next_delta=batch["next_delta"].to(device),
        )
        loss = F.cross_entropy(logits, target, weight=class_weights)
        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        pred = logits.argmax(dim=-1)
        total_loss += float(loss.detach().cpu().item()) * int(target.numel())
        total_correct += int(pred.eq(target).sum().detach().cpu().item())
        total += int(target.numel())
        for cls in torch.unique(target).detach().cpu().tolist():
            mask = target == int(cls)
            correct = int(pred[mask].eq(target[mask]).sum().detach().cpu().item())
            count = int(mask.sum().detach().cpu().item())
            current = confusion.setdefault(int(cls), [0, 0])
            current[0] += correct
            current[1] += count
    class_accs = [correct / count for correct, count in confusion.values() if count > 0]
    return {
        "loss": total_loss / max(1, total),
        "acc": total_correct / max(1, total),
        "macro_acc": float(sum(class_accs) / max(1, len(class_accs))),
    }


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = _device_from_arg(str(args.device))
    dataset_root = args.dataset_root.resolve()
    train_ds = V3InterpolationDataset(dataset_root, "train")
    val_ds = V3InterpolationDataset(dataset_root, "val")
    test_ds = V3InterpolationDataset(dataset_root, "test")
    interp_vocab = train_ds.vocabularies[TARGET_FIELD]

    config = V3InterpolationStageConfig(
        field_vocab_sizes={field: len(train_ds.vocabularies[field]) for field in train_ds.input_fields},
        interpolation_vocab_size=len(interp_vocab),
        embedding_dim=int(args.embedding_dim),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
    )
    model = V3InterpolationStageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    class_weights = _class_weights(train_ds.targets, len(interp_vocab), float(args.class_weight_power), device)

    run_dir = args.output_root.resolve() / f"v3_interpolation_stage_{time.strftime('%Y%m%d_%H%M%S')}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "config.json",
        {
            "args": vars(args) | {"dataset_root": str(dataset_root), "output_root": str(args.output_root)},
            "run_dir": str(run_dir),
            "input_fields": list(train_ds.input_fields),
            "interpolation_vocab": interp_vocab,
            "vocabularies": {field: train_ds.vocabularies[field] for field in train_ds.input_fields},
            "model_config": config.__dict__,
            "num_train": len(train_ds),
            "num_val": len(val_ds),
            "num_test": len(test_ds),
            "note": "Post-stage interpolation classifier. Main shot-plan prediction remains interpolation-free.",
        },
    )

    def save_checkpoint(path: Path, epoch: int, val_metrics: dict[str, float]) -> None:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
                "fields": list(train_ds.input_fields),
                "vocabularies": {field: train_ds.vocabularies[field] for field in train_ds.input_fields},
                "interpolation_vocab": interp_vocab,
                "epoch": int(epoch),
                "val_metrics": val_metrics,
                "args": vars(args) | {"dataset_root": str(dataset_root), "output_root": str(args.output_root)},
            },
            path,
        )

    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0)

    best_loss = math.inf
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer=optimizer, device=device, class_weights=class_weights)
        val_metrics = _run_epoch(model, val_loader, optimizer=None, device=device, class_weights=class_weights)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch {epoch:03d} | train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.3f} "
            f"| val loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.3f} macro={val_metrics['macro_acc']:.3f}",
            flush=True,
        )
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            save_checkpoint(run_dir / "best.pt", epoch, val_metrics)
        if epoch == int(args.epochs):
            save_checkpoint(run_dir / "last.pt", epoch, val_metrics)
        _write_json(run_dir / "history.json", history)

    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = _run_epoch(model, test_loader, optimizer=None, device=device, class_weights=class_weights)
    _write_json(run_dir / "metrics.json", {"test": test_metrics, "best_epoch": int(checkpoint["epoch"])})
    print(f"run_dir: {run_dir}", flush=True)
    print(
        f"test loss={test_metrics['loss']:.4f} acc={test_metrics['acc']:.3f} macro={test_metrics['macro_acc']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
