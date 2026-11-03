from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks
from torch import nn
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import wandb
except ImportError:
    wandb = None


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent / "results" / "cut_transition_dataset"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "cut_transition_models"
DEFAULT_WANDB_KEY_PATH = Path(__file__).resolve().parent / ".wandb_api_key"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a frame-level 1D CNN baseline for predicting cut-transition moments from audio-derived features."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root folder containing manifest.json and per-song .npz dataset files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root folder where model checkpoints and metrics will be written.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Optional JSON file containing explicit train/val/test ids for this dataset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split and training.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--window-frames", type=int, default=240, help="Training window length in frames.")
    parser.add_argument("--stride", type=int, default=60, help="Window stride in frames.")
    parser.add_argument(
        "--positive-repeat",
        type=int,
        default=4,
        help="Repeat windows containing at least one cut this many times in the training set.",
    )
    parser.add_argument(
        "--model-type",
        choices=["temporal_cnn", "multi_branch_cnn"],
        default="temporal_cnn",
        help="Temporal CNN architecture variant.",
    )
    parser.add_argument("--hidden-dim", type=int, default=96, help="Channel width for the temporal CNN.")
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout rate inside the model.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument(
        "--loss-positive-weight",
        type=float,
        default=6.0,
        help="Additional multiplier applied to frames near soft labels during training.",
    )
    parser.add_argument(
        "--sequence-loss-weight",
        type=float,
        default=0.75,
        help="Weight for the window-level soft Dice sequence loss added on top of BCE.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Validation-song fraction. Remaining songs after test split go to train.",
    )
    parser.add_argument("--test-frac", type=float, default=0.15, help="Test-song fraction.")
    parser.add_argument(
        "--tolerance-frames",
        type=int,
        default=3,
        help="Matching tolerance when scoring predicted cut peaks.",
    )
    parser.add_argument(
        "--min-peak-gap-frames",
        type=int,
        default=6,
        help="Minimum gap between predicted cut peaks passed to find_peaks.",
    )
    parser.add_argument(
        "--threshold-grid",
        type=float,
        nargs="*",
        default=[0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
        help="Candidate peak thresholds searched on the validation split.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["window_sequence_f1_active", "window_sequence_f1_all", "global_event_f1"],
        default="window_sequence_f1_active",
        help="Validation metric used to pick the best threshold and checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device. Use `auto`, `cpu`, or a torch device string like `cuda`.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix for the run directory name.",
    )
    parser.add_argument(
        "--wandb-project",
        default="music-cut-transition",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default="",
        help="Optional W&B run name. Defaults to the output run directory name.",
    )
    parser.add_argument(
        "--wandb-key-file",
        type=Path,
        default=DEFAULT_WANDB_KEY_PATH,
        help="Local file containing the W&B API key.",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable W&B logging even if the package and API key are available.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Checkpoint file or run directory to resume from.",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume from the most recently modified run directory under --output-root.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_local_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (WORKSPACE_ROOT / path).resolve()


def _checkpoint_candidates(run_dir: Path) -> list[Path]:
    return [run_dir / "checkpoint_last.pt", run_dir / "best_model.pt"]


def _find_latest_run_dir(output_root: Path, *, tag: str) -> Path | None:
    if not output_root.exists():
        return None
    run_dirs = [path for path in output_root.iterdir() if path.is_dir()]
    if tag:
        suffix = f"_{tag}"
        tagged_dirs = [path for path in run_dirs if path.name.endswith(suffix)]
        if tagged_dirs:
            run_dirs = tagged_dirs
    valid_dirs = [path for path in run_dirs if any(candidate.exists() for candidate in _checkpoint_candidates(path))]
    if not valid_dirs:
        return None
    return max(valid_dirs, key=lambda path: path.stat().st_mtime)


def _resolve_resume_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.resume_from is not None:
        target = args.resume_from
        if target.is_dir():
            for candidate in _checkpoint_candidates(target):
                if candidate.exists():
                    return candidate
            raise SystemExit(f"No checkpoint file found in run directory: {target}")
        if not target.exists():
            raise SystemExit(f"Resume checkpoint not found: {target}")
        return target

    if args.resume_latest:
        latest_run_dir = _find_latest_run_dir(args.output_root, tag=args.tag)
        if latest_run_dir is None:
            raise SystemExit(f"No resumable run directory found under: {args.output_root}")
        for candidate in _checkpoint_candidates(latest_run_dir):
            if candidate.exists():
                return candidate
        raise SystemExit(f"No checkpoint file found in latest run directory: {latest_run_dir}")
    return None


def _load_wandb_key(path: Path) -> str:
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"W&B key file is empty: {path}")
    return key


def _wandb_enabled(args: argparse.Namespace) -> bool:
    return not args.disable_wandb


def _infer_wandb_run_id(run_dir: Path) -> str | None:
    wandb_root = run_dir / "wandb"
    if not wandb_root.exists():
        return None
    run_files = sorted(wandb_root.glob("run-*"))
    if not run_files:
        return None
    latest = run_files[-1]
    name = latest.name
    if "-" not in name:
        return None
    return name.rsplit("-", 1)[-1]


def _init_wandb(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    split: dict,
    manifest: dict,
    device: torch.device,
    resume_run_id: str | None,
):
    if not _wandb_enabled(args):
        return None
    if wandb is None:
        raise SystemExit("wandb is not installed in the current environment. Install it or pass --disable-wandb.")

    key_path = args.wandb_key_file
    if not key_path.exists():
        raise SystemExit(f"W&B key file not found: {key_path}")
    key = _load_wandb_key(key_path)
    os.environ["WANDB_API_KEY"] = key
    wandb.login(key=key, relogin=True)
    run_name = args.wandb_run_name or run_dir.name
    return wandb.init(
        project=args.wandb_project,
        id=resume_run_id,
        resume="allow" if resume_run_id else None,
        name=run_name,
        dir=str(run_dir),
        config={
            **vars(args),
            "device": str(device),
            "dataset_manifest": str(args.dataset_root / "manifest.json"),
            "feature_dim": len(manifest.get("feature_names", [])),
            "song_count": manifest.get("song_count"),
            "split": split,
        },
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device_from_arg(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _make_song_split(song_ids: list[str], *, seed: int, val_frac: float, test_frac: float) -> dict[str, list[str]]:
    ids = list(song_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    song_count = len(ids)
    test_count = max(1, int(round(song_count * test_frac)))
    val_count = max(1, int(round(song_count * val_frac)))
    if test_count + val_count >= song_count:
        test_count = max(1, int(math.floor(song_count * test_frac)))
        val_count = max(1, min(song_count - test_count - 1, int(math.floor(song_count * val_frac))))
    train_count = max(1, song_count - val_count - test_count)
    train_ids = ids[:train_count]
    val_ids = ids[train_count : train_count + val_count]
    test_ids = ids[train_count + val_count :]
    return {"train": sorted(train_ids), "val": sorted(val_ids), "test": sorted(test_ids)}


def _build_window_starts(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, max(1, stride)))
    last_start = length - window
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _triage_predictions(probabilities: np.ndarray, *, threshold: float, min_gap_frames: int) -> np.ndarray:
    if len(probabilities) == 0:
        return np.zeros(0, dtype=np.int32)
    peaks, properties = find_peaks(probabilities, height=float(threshold), distance=max(1, int(min_gap_frames)))
    if len(peaks) == 0:
        return np.zeros(0, dtype=np.int32)
    order = np.argsort(properties["peak_heights"])[::-1]
    return np.sort(peaks[order].astype(np.int32))


def _match_cut_events(pred_frames: np.ndarray, gt_frames: np.ndarray, *, tolerance: int) -> tuple[int, int, int]:
    pred = np.sort(pred_frames.astype(np.int32))
    gt = np.sort(gt_frames.astype(np.int32))
    matched_gt = np.zeros(len(gt), dtype=bool)

    true_positive = 0
    false_positive = 0
    for frame in pred.tolist():
        left = np.searchsorted(gt, frame - tolerance, side="left")
        right = np.searchsorted(gt, frame + tolerance, side="right")
        matched = False
        for idx in range(left, right):
            if not matched_gt[idx]:
                matched_gt[idx] = True
                true_positive += 1
                matched = True
                break
        if not matched:
            false_positive += 1

    false_negative = int(np.sum(~matched_gt))
    return true_positive, false_positive, false_negative


def _precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = float(tp / (tp + fp)) if tp + fp > 0 else 0.0
    recall = float(tp / (tp + fn)) if tp + fn > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    return {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}


def _safe_f1(tp: int, fp: int, fn: int, *, empty_value: float) -> float:
    if tp == 0 and fp == 0 and fn == 0:
        return float(empty_value)
    precision = float(tp / (tp + fp)) if tp + fp > 0 else 0.0
    recall = float(tp / (tp + fn)) if tp + fn > 0 else 0.0
    if precision + recall <= 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _compute_feature_stats(entries: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    feature_sum: np.ndarray | None = None
    feature_sq_sum: np.ndarray | None = None
    frame_count = 0

    for entry in entries:
        data = np.load(_resolve_local_path(entry["feature_path"]))
        features = np.asarray(data["features"], dtype=np.float32)
        if feature_sum is None:
            feature_sum = np.zeros(features.shape[1], dtype=np.float64)
            feature_sq_sum = np.zeros(features.shape[1], dtype=np.float64)
        feature_sum += np.sum(features, axis=0, dtype=np.float64)
        feature_sq_sum += np.sum(np.square(features, dtype=np.float32), axis=0, dtype=np.float64)
        frame_count += int(features.shape[0])

    if feature_sum is None or feature_sq_sum is None or frame_count <= 0:
        raise ValueError("No training features available.")

    mean = feature_sum / float(frame_count)
    variance = np.maximum(1e-8, feature_sq_sum / float(frame_count) - np.square(mean))
    std = np.sqrt(variance)
    return mean.astype(np.float32), std.astype(np.float32)


@dataclass
class SongArrays:
    song_id: str
    features: np.ndarray
    labels: np.ndarray
    soft_labels: np.ndarray
    cut_frames: np.ndarray


class WindowedSongDataset(Dataset):
    def __init__(
        self,
        *,
        entries: list[dict],
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        window_frames: int,
        stride: int,
        positive_repeat: int,
        training: bool,
    ) -> None:
        self.training = training
        self.feature_mean = feature_mean.astype(np.float32)
        self.feature_std = feature_std.astype(np.float32)
        self.window_frames = int(window_frames)
        self.samples: list[tuple[int, int, int]] = []
        self.songs: list[SongArrays] = []

        for song_idx, entry in enumerate(entries):
            data = np.load(_resolve_local_path(entry["feature_path"]))
            features = np.asarray(data["features"], dtype=np.float32)
            labels = np.asarray(data["labels"], dtype=np.float32)
            soft_labels = np.asarray(data["soft_labels"], dtype=np.float32)
            cut_frames = np.asarray(data["cut_frames"], dtype=np.int32)
            normalized = (features - self.feature_mean[None, :]) / self.feature_std[None, :]
            song = SongArrays(song_id=entry["id"], features=normalized, labels=labels, soft_labels=soft_labels, cut_frames=cut_frames)
            self.songs.append(song)

            starts = _build_window_starts(len(features), self.window_frames, stride)
            for start in starts:
                end = min(len(features), start + self.window_frames)
                repeat = int(max(1, positive_repeat)) if labels[start:end].max() > 0 else 1
                for _ in range(repeat if training else 1):
                    self.samples.append((song_idx, start, end))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        song_idx, start, end = self.samples[index]
        song = self.songs[song_idx]
        valid_length = end - start

        x = np.zeros((self.window_frames, song.features.shape[1]), dtype=np.float32)
        y = np.zeros(self.window_frames, dtype=np.float32)
        y_soft = np.zeros(self.window_frames, dtype=np.float32)
        x[:valid_length] = song.features[start:end]
        y[:valid_length] = song.labels[start:end]
        y_soft[:valid_length] = song.soft_labels[start:end]
        valid = np.zeros(self.window_frames, dtype=np.float32)
        valid[:valid_length] = 1.0

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "y_soft": torch.from_numpy(y_soft),
            "valid": torch.from_numpy(valid),
        }


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.dropout(x)
        return F.gelu(x + residual)


class TemporalCutPredictor(nn.Module):
    def __init__(self, input_dim: int, *, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            DilatedResidualBlock(hidden_dim, dilation=1, dropout=dropout),
            DilatedResidualBlock(hidden_dim, dilation=2, dropout=dropout),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.blocks(x)
        logits = self.head(x).squeeze(1)
        return logits


class MultiBranchTemporalCutPredictor(nn.Module):
    """Temporal CNN with separate local, mid, and long context branches."""

    def __init__(self, input_dim: int, *, hidden_dim: int, dropout: float) -> None:
        super().__init__()

        def branch(kernel_size: int, dilations: tuple[int, ...]) -> nn.Sequential:
            padding = kernel_size // 2
            return nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                *[DilatedResidualBlock(hidden_dim, dilation=dilation, dropout=dropout) for dilation in dilations],
            )

        self.local_branch = branch(3, (1, 2))
        self.mid_branch = branch(7, (2, 4, 8))
        self.long_branch = branch(5, (4, 8, 16))
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim * 3, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            DilatedResidualBlock(hidden_dim, dilation=1, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        local = self.local_branch(x)
        mid = self.mid_branch(x)
        long = self.long_branch(x)
        fused = self.fusion(torch.cat([local, mid, long], dim=1))
        return self.head(fused).squeeze(1)


def _loss_from_logits(
    logits: torch.Tensor,
    targets_soft: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    loss_positive_weight: float,
    sequence_loss_weight: float,
) -> torch.Tensor:
    weights = 1.0 + float(loss_positive_weight) * targets_soft
    bce = F.binary_cross_entropy_with_logits(logits, targets_soft, reduction="none")
    weighted = bce * weights * valid_mask
    denom = torch.clamp(valid_mask.sum(), min=1.0)
    bce_loss = weighted.sum() / denom

    probs = torch.sigmoid(logits) * valid_mask
    targets = targets_soft * valid_mask
    intersection = torch.sum(probs * targets, dim=1)
    cardinality = torch.sum(probs, dim=1) + torch.sum(targets, dim=1)
    dice = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
    sequence_loss = 1.0 - dice.mean()
    return bce_loss + float(sequence_loss_weight) * sequence_loss


def _predict_song(
    model: nn.Module,
    *,
    features: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    window_frames: int,
    stride: int,
    device: torch.device,
) -> np.ndarray:
    normalized = (features.astype(np.float32) - feature_mean[None, :]) / feature_std[None, :]
    length = len(normalized)
    starts = _build_window_starts(length, window_frames, stride)
    total = np.zeros(length, dtype=np.float32)
    counts = np.zeros(length, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for start in starts:
            end = min(length, start + window_frames)
            valid_length = end - start
            batch = np.zeros((1, window_frames, normalized.shape[1]), dtype=np.float32)
            batch[0, :valid_length] = normalized[start:end]
            logits = model(torch.from_numpy(batch).to(device))
            probs = torch.sigmoid(logits)[0, :valid_length].detach().cpu().numpy().astype(np.float32)
            total[start:end] += probs
            counts[start:end] += 1.0

    counts = np.maximum(counts, 1.0)
    return total / counts


def _window_sequence_scores(
    pred_frames: np.ndarray,
    gt_frames: np.ndarray,
    *,
    num_frames: int,
    window_frames: int,
    stride: int,
    tolerance_frames: int,
) -> dict[str, float]:
    starts = _build_window_starts(num_frames, window_frames, stride)
    active_scores: list[float] = []
    all_scores: list[float] = []

    for start in starts:
        end = min(num_frames, start + window_frames)
        pred_local = pred_frames[(pred_frames >= start) & (pred_frames < end)] - start
        gt_local = gt_frames[(gt_frames >= start) & (gt_frames < end)] - start
        tp, fp, fn = _match_cut_events(pred_local, gt_local, tolerance=tolerance_frames)
        active = len(pred_local) > 0 or len(gt_local) > 0
        score_all = _safe_f1(tp, fp, fn, empty_value=1.0)
        all_scores.append(score_all)
        if active:
            active_scores.append(_safe_f1(tp, fp, fn, empty_value=1.0))

    return {
        "window_sequence_f1_active": round(float(np.mean(active_scores)) if active_scores else 0.0, 6),
        "window_sequence_f1_all": round(float(np.mean(all_scores)) if all_scores else 0.0, 6),
        "window_count": int(len(starts)),
        "active_window_count": int(len(active_scores)),
    }


def _evaluate_entries(
    model: nn.Module,
    *,
    entries: list[dict],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    window_frames: int,
    stride: int,
    threshold: float,
    tolerance_frames: int,
    min_peak_gap_frames: int,
    device: torch.device,
) -> dict:
    tp = fp = fn = 0
    per_song: list[dict] = []
    window_score_sum_active = 0.0
    window_score_sum_all = 0.0
    window_song_count = 0
    window_count_total = 0
    active_window_count_total = 0

    for entry in entries:
        data = np.load(_resolve_local_path(entry["feature_path"]))
        features = np.asarray(data["features"], dtype=np.float32)
        gt_frames = np.asarray(data["cut_frames"], dtype=np.int32)
        probabilities = _predict_song(
            model,
            features=features,
            feature_mean=feature_mean,
            feature_std=feature_std,
            window_frames=window_frames,
            stride=stride,
            device=device,
        )
        pred_frames = _triage_predictions(
            probabilities,
            threshold=threshold,
            min_gap_frames=min_peak_gap_frames,
        )
        window_scores = _window_sequence_scores(
            pred_frames,
            gt_frames,
            num_frames=len(features),
            window_frames=window_frames,
            stride=stride,
            tolerance_frames=tolerance_frames,
        )
        song_tp, song_fp, song_fn = _match_cut_events(pred_frames, gt_frames, tolerance=tolerance_frames)
        tp += song_tp
        fp += song_fp
        fn += song_fn
        window_score_sum_active += window_scores["window_sequence_f1_active"]
        window_score_sum_all += window_scores["window_sequence_f1_all"]
        window_song_count += 1
        window_count_total += window_scores["window_count"]
        active_window_count_total += window_scores["active_window_count"]
        per_song.append(
            {
                "id": entry["id"],
                "gt_cut_count": int(len(gt_frames)),
                "pred_cut_count": int(len(pred_frames)),
                "tp": int(song_tp),
                "fp": int(song_fp),
                "fn": int(song_fn),
                "window_sequence_f1_active": window_scores["window_sequence_f1_active"],
                "window_sequence_f1_all": window_scores["window_sequence_f1_all"],
                "window_count": window_scores["window_count"],
                "active_window_count": window_scores["active_window_count"],
            }
        )

    metrics = _precision_recall_f1(tp, fp, fn)
    metrics.update(
        {
            "global_event_f1": metrics["f1"],
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "threshold": float(threshold),
            "tolerance_frames": int(tolerance_frames),
            "min_peak_gap_frames": int(min_peak_gap_frames),
            "window_sequence_f1_active": round(float(window_score_sum_active / window_song_count) if window_song_count else 0.0, 6),
            "window_sequence_f1_all": round(float(window_score_sum_all / window_song_count) if window_song_count else 0.0, 6),
            "window_count": int(window_count_total),
            "active_window_count": int(active_window_count_total),
            "per_song": per_song,
        }
    )
    return metrics


def _search_best_threshold(
    model: nn.Module,
    *,
    entries: list[dict],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    window_frames: int,
    stride: int,
    threshold_grid: list[float],
    tolerance_frames: int,
    min_peak_gap_frames: int,
    selection_metric: str,
    device: torch.device,
) -> dict:
    best_metrics: dict | None = None
    for threshold in threshold_grid:
        metrics = _evaluate_entries(
            model,
            entries=entries,
            feature_mean=feature_mean,
            feature_std=feature_std,
            window_frames=window_frames,
            stride=stride,
            threshold=float(threshold),
            tolerance_frames=tolerance_frames,
            min_peak_gap_frames=min_peak_gap_frames,
            device=device,
        )
        if best_metrics is None or float(metrics[selection_metric]) > float(best_metrics[selection_metric]):
            best_metrics = metrics
    if best_metrics is None:
        raise ValueError("Threshold search failed.")
    return best_metrics


def _run_dir(output_root: Path, *, seed: int, tag: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    return output_root / f"frame_cnn_seed{seed}_{stamp}{suffix}"


def _load_resume_payload(resume_checkpoint: Path) -> dict:
    payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise SystemExit(f"Unexpected checkpoint format: {resume_checkpoint}")
    return payload


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _resolve_split(
    *,
    songs: list[dict],
    manifest: dict,
    split_manifest_path: Path | None,
    seed: int,
    val_frac: float,
    test_frac: float,
) -> tuple[dict[str, list[str]], str]:
    entries_by_id = {str(song["id"]): song for song in songs}

    def extract_split(payload: dict) -> dict[str, list[str]]:
        source = payload["split"] if isinstance(payload.get("split"), dict) else payload
        return {
            "train": [str(song_id) for song_id in source.get("train", [])],
            "val": [str(song_id) for song_id in source.get("val", [])],
            "test": [str(song_id) for song_id in source.get("test", [])],
        }

    def validate_split(split: dict[str, list[str]], *, source_name: str) -> dict[str, list[str]]:
        overlap = (set(split["train"]) & set(split["val"])) | (set(split["train"]) & set(split["test"])) | (
            set(split["val"]) & set(split["test"])
        )
        if overlap:
            raise ValueError(f"split overlap detected in {source_name}: {sorted(overlap)}")
        missing = {
            split_name: [song_id for song_id in split[split_name] if song_id not in entries_by_id]
            for split_name in ("train", "val", "test")
        }
        if any(missing.values()):
            raise ValueError(f"split ids missing from dataset manifest in {source_name}: {missing}")
        return split

    if split_manifest_path is not None:
        split = validate_split(extract_split(_load_json(split_manifest_path)), source_name=str(split_manifest_path))
        return split, str(split_manifest_path)

    if isinstance(manifest.get("split"), dict):
        split = validate_split(extract_split(manifest), source_name="dataset manifest split")
        return split, "dataset_manifest"

    split = _make_song_split(
        [song["id"] for song in songs],
        seed=int(seed),
        val_frac=float(val_frac),
        test_frac=float(test_frac),
    )
    return split, "random_song_split"


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    device = _device_from_arg(args.device)
    resume_checkpoint = _resolve_resume_checkpoint(args)
    resume_payload = _load_resume_payload(resume_checkpoint) if resume_checkpoint is not None else None
    resume_run_dir = resume_checkpoint.parent if resume_checkpoint is not None else None

    manifest_path = args.dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Dataset manifest not found: {manifest_path}")

    manifest = _load_json(manifest_path)
    songs = manifest.get("songs", [])
    if not songs:
        raise SystemExit(f"No songs found in dataset manifest: {manifest_path}")

    if resume_payload is not None and "split" in resume_payload:
        split = resume_payload["split"]
        split_source = "resume_payload"
    else:
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

    if resume_payload is not None and "feature_mean" in resume_payload and "feature_std" in resume_payload:
        feature_mean = np.asarray(resume_payload["feature_mean"], dtype=np.float32)
        feature_std = np.asarray(resume_payload["feature_std"], dtype=np.float32)
    else:
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

    input_dim = len(manifest.get("feature_names", []))
    if input_dim <= 0:
        input_dim = int(train_dataset.songs[0].features.shape[1])
    model_cls = MultiBranchTemporalCutPredictor if str(args.model_type) == "multi_branch_cnn" else TemporalCutPredictor
    model = model_cls(
        input_dim=input_dim,
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    run_dir = resume_run_dir if resume_run_dir is not None else _run_dir(args.output_root, seed=int(args.seed), tag=args.tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    resume_mode = False
    resume_note = ""
    start_epoch = 0
    history: list[dict] = []
    best_val_metrics: dict | None = None
    best_state: dict | None = None
    resume_run_id = None

    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state"])
        resume_run_id = resume_payload.get("wandb_run_id") or _infer_wandb_run_id(run_dir)
        history = list(resume_payload.get("history", []))
        best_val_metrics = resume_payload.get("best_val_metrics")
        start_epoch = int(resume_payload.get("epoch", 0))
        optimizer_state = resume_payload.get("optimizer_state")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            resume_mode = True
            resume_note = f"Resuming exact training state from {resume_checkpoint}"
        else:
            resume_mode = True
            start_epoch = 0
            resume_note = (
                f"Warm-starting from model weights in {resume_checkpoint} "
                f"(optimizer/last-epoch state was not available)."
            )
        best_state = dict(resume_payload)

    wandb_run = _init_wandb(
        args=args,
        run_dir=run_dir,
        split=split,
        manifest=manifest,
        device=device,
        resume_run_id=resume_run_id,
    )

    print(f"Run dir: {run_dir}")
    print(f"Device: {device}")
    print(f"Model type: {args.model_type}")
    print(f"Split source: {split_source}")
    print(f"Train/Val/Test songs: {len(train_entries)}/{len(val_entries)}/{len(test_entries)}")
    print(f"Train windows: {len(train_dataset)}")
    if resume_mode:
        print(resume_note)
        print(f"Starting from epoch {start_epoch + 1} up to epoch {int(args.epochs)}")

    for epoch in range(start_epoch + 1, int(args.epochs) + 1):
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
            wandb.log(
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

        if best_val_metrics is None or float(val_metrics[str(args.selection_metric)]) > float(best_val_metrics[str(args.selection_metric)]):
            best_val_metrics = val_metrics
            best_state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "feature_names": manifest.get("feature_names", []),
                "args": vars(args),
                "split": split,
                "best_val_metrics": best_val_metrics,
                "history": history,
                "run_dir": str(run_dir),
                "wandb_run_id": getattr(wandb_run, "id", None),
                "model_type": str(args.model_type),
            }
            _save_checkpoint(run_dir / "best_model.pt", best_state)
            if wandb_run is not None:
                wandb.summary["best_val_f1"] = val_metrics["f1"]
                wandb.summary["best_val_global_event_f1"] = val_metrics["global_event_f1"]
                wandb.summary["best_val_window_sequence_f1_active"] = val_metrics["window_sequence_f1_active"]
                wandb.summary["best_val_window_sequence_f1_all"] = val_metrics["window_sequence_f1_all"]
                wandb.summary["selection_metric"] = str(args.selection_metric)
                wandb.summary["best_val_selection_score"] = val_metrics[str(args.selection_metric)]
                wandb.summary["best_val_precision"] = val_metrics["precision"]
                wandb.summary["best_val_recall"] = val_metrics["recall"]
                wandb.summary["best_val_threshold"] = val_metrics["threshold"]
                wandb.summary["best_epoch"] = epoch

        last_state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "feature_names": manifest.get("feature_names", []),
            "args": vars(args),
            "split": split,
            "best_val_metrics": best_val_metrics,
            "history": history,
            "run_dir": str(run_dir),
            "wandb_run_id": getattr(wandb_run, "id", None),
            "model_type": str(args.model_type),
        }
        _save_checkpoint(run_dir / "checkpoint_last.pt", last_state)
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "dataset_manifest": str(manifest_path),
                    "device": str(device),
                    "model_type": str(args.model_type),
                    "feature_names": manifest.get("feature_names", []),
                    "split": split,
                    "history": history,
                    "best_val_metrics": best_val_metrics,
                    "test_metrics": None,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (run_dir / "split.json").write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")

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
        "feature_names": manifest.get("feature_names", []),
        "split": split,
        "history": history,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "split.json").write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
    if wandb_run is not None:
        wandb.summary["test_f1"] = test_metrics["f1"]
        wandb.summary["test_global_event_f1"] = test_metrics["global_event_f1"]
        wandb.summary["test_window_sequence_f1_active"] = test_metrics["window_sequence_f1_active"]
        wandb.summary["test_window_sequence_f1_all"] = test_metrics["window_sequence_f1_all"]
        wandb.summary["test_precision"] = test_metrics["precision"]
        wandb.summary["test_recall"] = test_metrics["recall"]
        wandb.summary["test_threshold"] = best_val_metrics["threshold"]
        wandb.summary["run_dir"] = str(run_dir)
        wandb.summary["metrics_json"] = str(run_dir / "metrics.json")
        wandb.summary["checkpoint_path"] = str(run_dir / "best_model.pt")
        wandb.log(
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
        wandb.finish()
    print(f"Saved checkpoint: {run_dir / 'best_model.pt'}")
    print(f"Saved metrics: {run_dir / 'metrics.json'}")
    print(
        f"Test window_f1={test_metrics['window_sequence_f1_active']:.4f} "
        f"global_f1={test_metrics['global_event_f1']:.4f} "
        f"(p={test_metrics['precision']:.4f}, r={test_metrics['recall']:.4f}, th={best_val_metrics['threshold']:.2f})"
    )


if __name__ == "__main__":
    main()
