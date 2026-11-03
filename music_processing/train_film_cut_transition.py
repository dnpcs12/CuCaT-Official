from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from music_processing.train_cut_transition_baseline import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_WANDB_KEY_PATH,
    DilatedResidualBlock,
    WindowedSongDataset,
    _compute_feature_stats,
    _device_from_arg,
    _evaluate_entries,
    _init_wandb,
    _load_json,
    _loss_from_logits,
    _resolve_split,
    _save_checkpoint,
    _search_best_threshold,
    _set_seed,
)


DEFAULT_DATASET_ROOT = (
    Path(__file__).resolve().parent / "results" / "cut_transition_dataset_audio_compact_dance_from_multimodal_dcm"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a frame-level cut-transition model with FiLM fusion between audio and dance features."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window-frames", type=int, default=240)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--positive-repeat", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss-positive-weight", type=float, default=6.0)
    parser.add_argument("--sequence-loss-weight", type=float, default=0.75)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--tolerance-frames", type=int, default=3)
    parser.add_argument("--min-peak-gap-frames", type=int, default=6)
    parser.add_argument(
        "--threshold-grid",
        type=float,
        nargs="*",
        default=[0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
    )
    parser.add_argument(
        "--selection-metric",
        choices=["window_sequence_f1_active", "window_sequence_f1_all", "global_event_f1"],
        default="window_sequence_f1_active",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tag", default="")
    parser.add_argument("--audio-dim", type=int, default=None)
    parser.add_argument(
        "--film-scale",
        type=float,
        default=0.5,
        help="Scale for dance-conditioned FiLM modulation. Lower values keep the model more audio-primary.",
    )
    parser.add_argument(
        "--save-epoch-checkpoints",
        action="store_true",
        help="Save one checkpoint per epoch under checkpoints/ for later visual comparison.",
    )
    parser.add_argument(
        "--min-selection-recall",
        type=float,
        default=None,
        help=(
            "When set, also save best_recall_floor_model.pt using the selection metric "
            "among epochs whose validation recall is at least this value."
        ),
    )
    parser.add_argument("--wandb-project", default="music-cut-transition")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-key-file", type=Path, default=DEFAULT_WANDB_KEY_PATH)
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def _run_dir(output_root: Path, *, seed: int, tag: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    return output_root / f"film_cnn_seed{seed}_{stamp}{suffix}"


def _infer_audio_dim(feature_names: list[str], explicit_audio_dim: int | None) -> int:
    if explicit_audio_dim is not None:
        return int(explicit_audio_dim)
    audio_dim = 0
    for name in feature_names:
        if str(name).startswith("dance_"):
            break
        audio_dim += 1
    if audio_dim <= 0 or audio_dim >= len(feature_names):
        raise ValueError(
            "Could not infer audio/dance split from feature names. "
            "Pass --audio-dim explicitly for this dataset."
        )
    return audio_dim


def _snapshot_model_state(model: nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


class AudioDanceFiLMCutPredictor(nn.Module):
    """Temporal CNN where dance features FiLM-condition the audio stream.

    FiLM and dance residual layers are zero-initialized, so the model starts
    close to an audio-only CNN and learns to use dance only when helpful.
    """

    def __init__(self, *, audio_dim: int, dance_dim: int, hidden_dim: int, dropout: float, film_scale: float = 0.5) -> None:
        super().__init__()
        self.audio_dim = int(audio_dim)
        self.dance_dim = int(dance_dim)
        self.film_scale = float(film_scale)

        self.audio_stem = nn.Sequential(
            nn.Conv1d(self.audio_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.dance_stem = nn.Sequential(
            nn.Conv1d(self.dance_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.audio_blocks = nn.Sequential(
            DilatedResidualBlock(hidden_dim, dilation=1, dropout=dropout),
            DilatedResidualBlock(hidden_dim, dilation=2, dropout=dropout),
        )
        self.dance_blocks = nn.Sequential(
            DilatedResidualBlock(hidden_dim, dilation=1, dropout=dropout),
            DilatedResidualBlock(hidden_dim, dilation=2, dropout=dropout),
        )
        self.film = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1)
        self.dance_residual = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.fusion_blocks = nn.Sequential(
            DilatedResidualBlock(hidden_dim, dilation=4, dropout=dropout),
            DilatedResidualBlock(hidden_dim, dilation=8, dropout=dropout),
            DilatedResidualBlock(hidden_dim, dilation=16, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )
        self._reset_film_to_identity()

    def _reset_film_to_identity(self) -> None:
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.dance_residual.weight)
        nn.init.zeros_(self.dance_residual.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        audio = x[:, :, : self.audio_dim].transpose(1, 2)
        dance = x[:, :, self.audio_dim :].transpose(1, 2)

        audio = self.audio_blocks(self.audio_stem(audio))
        dance = self.dance_blocks(self.dance_stem(dance))
        gamma, beta = self.film(dance).chunk(2, dim=1)
        audio = audio * (1.0 + self.film_scale * torch.tanh(gamma)) + self.film_scale * beta
        fused = audio + self.dance_residual(dance)
        fused = self.fusion_blocks(fused)
        return self.head(fused).squeeze(1)


def main() -> None:
    args = parse_args()
    _set_seed(int(args.seed))
    random.seed(int(args.seed))
    device = _device_from_arg(str(args.device))

    manifest_path = args.dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Dataset manifest not found: {manifest_path}")
    manifest = _load_json(manifest_path)
    songs = manifest.get("songs", [])
    if not songs:
        raise SystemExit(f"No songs found in dataset manifest: {manifest_path}")

    feature_names = list(manifest.get("feature_names", []))
    input_dim = len(feature_names)
    audio_dim = _infer_audio_dim(feature_names, args.audio_dim)
    dance_dim = input_dim - audio_dim
    if dance_dim <= 0:
        raise SystemExit(f"Invalid audio/dance dimensions: audio={audio_dim}, total={input_dim}")

    split, split_source = _resolve_split(
        songs=songs,
        manifest=manifest,
        split_manifest_path=args.split_manifest,
        seed=int(args.seed),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
    )
    entries_by_id = {song["id"]: song for song in songs}
    train_entries = [entries_by_id[song_id] for song_id in split["train"]]
    val_entries = [entries_by_id[song_id] for song_id in split["val"]]
    test_entries = [entries_by_id[song_id] for song_id in split["test"]]

    feature_mean, feature_std = _compute_feature_stats(train_entries)
    train_dataset = WindowedSongDataset(
        entries=train_entries,
        feature_mean=feature_mean,
        feature_std=feature_std,
        window_frames=int(args.window_frames),
        stride=int(args.stride),
        positive_repeat=int(args.positive_repeat),
        training=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
    )

    model = AudioDanceFiLMCutPredictor(
        audio_dim=audio_dim,
        dance_dim=dance_dim,
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        film_scale=float(args.film_scale),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    run_dir = _run_dir(args.output_root, seed=int(args.seed), tag=str(args.tag))
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = _init_wandb(
        args=args,
        run_dir=run_dir,
        split=split,
        manifest=manifest,
        device=device,
        resume_run_id=None,
    )

    print(f"Run dir: {run_dir}")
    print(f"Device: {device}")
    print(f"Split source: {split_source}")
    print(f"Input dims: audio={audio_dim}, dance={dance_dim}, total={input_dim}")
    print(f"FiLM scale: {float(args.film_scale):.3f}")
    print(f"Train/Val/Test songs: {len(train_entries)}/{len(val_entries)}/{len(test_entries)}")
    print(f"Train windows: {len(train_dataset)}")

    history: list[dict] = []
    best_val_metrics: dict | None = None
    best_state: dict | None = None
    best_recall_floor_metrics: dict | None = None
    best_recall_floor_state: dict | None = None
    checkpoints_dir = run_dir / "checkpoints"
    if args.save_epoch_checkpoints:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            x = batch["x"].to(device)
            y_soft = batch["y_soft"].to(device)
            valid = batch["valid"].to(device)
            logits = model(x)
            loss = _loss_from_logits(
                logits,
                y_soft,
                valid,
                loss_positive_weight=float(args.loss_positive_weight),
                sequence_loss_weight=float(args.sequence_loss_weight),
            )
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            batch_count += 1

        train_loss = running_loss / float(max(1, batch_count))
        val_metrics = _search_best_threshold(
            model,
            entries=val_entries,
            feature_mean=feature_mean,
            feature_std=feature_std,
            window_frames=int(args.window_frames),
            stride=int(args.stride),
            threshold_grid=[float(x) for x in args.threshold_grid],
            tolerance_frames=int(args.tolerance_frames),
            min_peak_gap_frames=int(args.min_peak_gap_frames),
            selection_metric=str(args.selection_metric),
            device=device,
        )
        epoch_row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_global_event_f1": val_metrics["global_event_f1"],
            "val_window_sequence_f1_active": val_metrics["window_sequence_f1_active"],
            "val_window_sequence_f1_all": val_metrics["window_sequence_f1_all"],
            "val_threshold": val_metrics["threshold"],
        }
        history.append(epoch_row)
        print(
            f"Epoch {epoch:02d}: loss={train_loss:.5f} "
            f"val_window_f1={val_metrics['window_sequence_f1_active']:.4f} "
            f"val_global_f1={val_metrics['global_event_f1']:.4f} "
            f"(p={val_metrics['precision']:.4f}, r={val_metrics['recall']:.4f}, th={val_metrics['threshold']:.2f})"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/f1": val_metrics["f1"],
                    "val/global_event_f1": val_metrics["global_event_f1"],
                    "val/window_sequence_f1_active": val_metrics["window_sequence_f1_active"],
                    "val/window_sequence_f1_all": val_metrics["window_sequence_f1_all"],
                    "val/precision": val_metrics["precision"],
                    "val/recall": val_metrics["recall"],
                    "val/threshold": val_metrics["threshold"],
                    "val/tp": val_metrics["tp"],
                    "val/fp": val_metrics["fp"],
                    "val/fn": val_metrics["fn"],
                },
                step=epoch,
            )

        current_state = {
            "epoch": epoch,
            "model_type": "audio_dance_film_cnn",
            "model_state": _snapshot_model_state(model),
            "optimizer_state": copy.deepcopy(optimizer.state_dict()),
            "audio_dim": audio_dim,
            "dance_dim": dance_dim,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "feature_names": feature_names,
            "args": vars(args),
            "split": split,
            "best_val_metrics": best_val_metrics,
            "current_val_metrics": val_metrics,
            "history": history,
            "run_dir": str(run_dir),
            "wandb_run_id": getattr(wandb_run, "id", None),
        }
        if args.save_epoch_checkpoints:
            _save_checkpoint(checkpoints_dir / f"epoch_{epoch:03d}.pt", current_state)

        recall_floor = args.min_selection_recall
        if recall_floor is not None and float(val_metrics["recall"]) >= float(recall_floor):
            if best_recall_floor_metrics is None or float(val_metrics[str(args.selection_metric)]) > float(
                best_recall_floor_metrics[str(args.selection_metric)]
            ):
                best_recall_floor_metrics = val_metrics
                best_recall_floor_state = {
                    **current_state,
                    "recall_floor": float(recall_floor),
                    "best_recall_floor_metrics": best_recall_floor_metrics,
                }
                _save_checkpoint(run_dir / "best_recall_floor_model.pt", best_recall_floor_state)
                if wandb_run is not None:
                    wandb_run.summary["best_recall_floor_epoch"] = epoch
                    wandb_run.summary["best_recall_floor_recall"] = val_metrics["recall"]
                    wandb_run.summary["best_recall_floor_selection_score"] = val_metrics[str(args.selection_metric)]

        if best_val_metrics is None or float(val_metrics[str(args.selection_metric)]) > float(
            best_val_metrics[str(args.selection_metric)]
        ):
            best_val_metrics = val_metrics
            best_state = {
                "epoch": epoch,
                "model_type": "audio_dance_film_cnn",
                "model_state": _snapshot_model_state(model),
                "optimizer_state": copy.deepcopy(optimizer.state_dict()),
                "audio_dim": audio_dim,
                "dance_dim": dance_dim,
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "feature_names": feature_names,
                "args": vars(args),
                "split": split,
                "best_val_metrics": best_val_metrics,
                "history": history,
                "run_dir": str(run_dir),
                "wandb_run_id": getattr(wandb_run, "id", None),
            }
            _save_checkpoint(run_dir / "best_model.pt", best_state)
            if wandb_run is not None:
                wandb_run.summary["best_epoch"] = epoch
                wandb_run.summary["best_val_selection_score"] = val_metrics[str(args.selection_metric)]
                wandb_run.summary["best_val_global_event_f1"] = val_metrics["global_event_f1"]
                wandb_run.summary["best_val_window_sequence_f1_active"] = val_metrics["window_sequence_f1_active"]
                wandb_run.summary["best_val_threshold"] = val_metrics["threshold"]

        last_state = {
            "epoch": epoch,
            "model_type": "audio_dance_film_cnn",
            "model_state": _snapshot_model_state(model),
            "optimizer_state": copy.deepcopy(optimizer.state_dict()),
            "audio_dim": audio_dim,
            "dance_dim": dance_dim,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "feature_names": feature_names,
            "args": vars(args),
            "split": split,
            "best_val_metrics": best_val_metrics,
            "history": history,
            "run_dir": str(run_dir),
            "wandb_run_id": getattr(wandb_run, "id", None),
        }
        _save_checkpoint(run_dir / "checkpoint_last.pt", last_state)
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "dataset_manifest": str(manifest_path),
                    "device": str(device),
                    "model_type": "audio_dance_film_cnn",
                    "audio_dim": audio_dim,
                    "dance_dim": dance_dim,
                    "feature_names": feature_names,
                    "split": split,
                    "history": history,
                    "best_val_metrics": best_val_metrics,
                    "best_recall_floor_metrics": best_recall_floor_metrics,
                    "test_metrics": None,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    if best_state is None or best_val_metrics is None:
        raise SystemExit("Training did not produce a best checkpoint.")

    model.load_state_dict(best_state["model_state"])
    test_metrics = _evaluate_entries(
        model,
        entries=test_entries,
        feature_mean=feature_mean,
        feature_std=feature_std,
        window_frames=int(args.window_frames),
        stride=int(args.stride),
        threshold=float(best_val_metrics["threshold"]),
        tolerance_frames=int(args.tolerance_frames),
        min_peak_gap_frames=int(args.min_peak_gap_frames),
        device=device,
    )
    summary = {
        "run_dir": str(run_dir),
        "dataset_manifest": str(manifest_path),
        "device": str(device),
        "model_type": "audio_dance_film_cnn",
        "audio_dim": audio_dim,
        "dance_dim": dance_dim,
        "feature_names": feature_names,
        "split": split,
        "history": history,
        "best_val_metrics": best_val_metrics,
        "best_recall_floor_metrics": best_recall_floor_metrics,
        "test_metrics": test_metrics,
    }
    if best_recall_floor_state is not None and best_recall_floor_metrics is not None:
        model.load_state_dict(best_recall_floor_state["model_state"])
        summary["test_metrics_recall_floor"] = _evaluate_entries(
            model,
            entries=test_entries,
            feature_mean=feature_mean,
            feature_std=feature_std,
            window_frames=int(args.window_frames),
            stride=int(args.stride),
            threshold=float(best_recall_floor_metrics["threshold"]),
            tolerance_frames=int(args.tolerance_frames),
            min_peak_gap_frames=int(args.min_peak_gap_frames),
            device=device,
        )
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "split.json").write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")

    if wandb_run is not None:
        wandb_run.summary["test_f1"] = test_metrics["f1"]
        wandb_run.summary["test_global_event_f1"] = test_metrics["global_event_f1"]
        wandb_run.summary["test_window_sequence_f1_active"] = test_metrics["window_sequence_f1_active"]
        wandb_run.summary["test_precision"] = test_metrics["precision"]
        wandb_run.summary["test_recall"] = test_metrics["recall"]
        wandb_run.summary["run_dir"] = str(run_dir)
        wandb_run.log(
            {
                "test/f1": test_metrics["f1"],
                "test/global_event_f1": test_metrics["global_event_f1"],
                "test/window_sequence_f1_active": test_metrics["window_sequence_f1_active"],
                "test/window_sequence_f1_all": test_metrics["window_sequence_f1_all"],
                "test/precision": test_metrics["precision"],
                "test/recall": test_metrics["recall"],
                "test/threshold": best_val_metrics["threshold"],
            },
            step=int(args.epochs) + 1,
        )
        wandb_run.finish()

    print(f"Saved checkpoint: {run_dir / 'best_model.pt'}")
    print(f"Saved metrics: {run_dir / 'metrics.json'}")
    print(
        f"Test window_f1={test_metrics['window_sequence_f1_active']:.4f} "
        f"global_f1={test_metrics['global_event_f1']:.4f} "
        f"(p={test_metrics['precision']:.4f}, r={test_metrics['recall']:.4f}, th={best_val_metrics['threshold']:.2f})"
    )


if __name__ == "__main__":
    main()
