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
from torch import nn
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import wandb
except ImportError:
    wandb = None

from music_processing.train_cut_transition_baseline import _device_from_arg


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent / "results" / "keyframe_density_stagea_dataset_dcm"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "keyframe_density_stagea_models"
DEFAULT_WANDB_KEY_PATH = Path(__file__).resolve().parent / ".wandb_api_key"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Stage-A 3-second-window keyframe-density regressor from music, dance, and previous-cut context."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--positive-repeat", type=int, default=3)
    parser.add_argument("--model-type", choices=["transformer", "cnn", "film_cnn"], default="transformer")
    parser.add_argument("--model-dim", type=int, default=192)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--cnn-kernel-size", type=int, default=5)
    parser.add_argument("--cnn-dilations", default="1,2,4,8")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--smooth-l1-weight", type=float, default=0.2)
    parser.add_argument("--max-prev-cut-age-sec", type=float, default=10.0)
    parser.add_argument("--max-prev-cut-gap-sec", type=float, default=10.0)
    parser.add_argument(
        "--selection-metric",
        choices=["mae", "pearson", "topk_density_capture", "positive_f1"],
        default="topk_density_capture",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tag", default="")
    parser.add_argument("--wandb-project", default="music-keyframe-density-stagea")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-key-file", type=Path, default=DEFAULT_WANDB_KEY_PATH)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--no-save-last-checkpoint", action="store_true")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_repo_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (WORKSPACE_ROOT / path).resolve()


def _run_dir(output_root: Path, *, seed: int, tag: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    return output_root / f"stagea_density_seed{seed}_{stamp}{suffix}"


def _load_wandb_key(path: Path) -> str:
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"W&B key file is empty: {path}")
    return key


def _init_wandb(*, args: argparse.Namespace, run_dir: Path, manifest: dict, device: torch.device):
    if args.disable_wandb:
        return None
    if wandb is None:
        raise SystemExit("wandb is not installed in the current environment. Install it or pass --disable-wandb.")
    if not args.wandb_key_file.exists():
        raise SystemExit(f"W&B key file not found: {args.wandb_key_file}")
    os.environ["WANDB_API_KEY"] = _load_wandb_key(args.wandb_key_file)
    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
    run_name = args.wandb_run_name or run_dir.name
    return wandb.init(
        project=args.wandb_project,
        name=run_name,
        dir=str(run_dir),
        config={
            **vars(args),
            "device": str(device),
            "dataset_manifest": str(args.dataset_root / "manifest.json"),
            "audio_feature_dim": manifest.get("audio_feature_dim"),
            "dance_feature_dim": manifest.get("dance_feature_dim"),
            "window_frames": manifest.get("window_frames"),
            "stride_frames": manifest.get("stride_frames"),
            "window_count": manifest.get("window_count"),
        },
    )


@dataclass
class FeatureArrays:
    audio: np.ndarray
    dance: np.ndarray
    cut_frames: np.ndarray


def _load_feature_arrays(path: Path) -> FeatureArrays:
    data = np.load(path)
    return FeatureArrays(
        audio=np.asarray(data["audio_features"], dtype=np.float32),
        dance=np.asarray(data["dance_features"], dtype=np.float32),
        cut_frames=np.asarray(data["cut_frames"], dtype=np.int32),
    )


def _compute_modality_stats(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_paths = sorted({_resolve_repo_path(row["feature_path"]) for row in rows})
    audio_sum = dance_sum = audio_sq = dance_sq = None
    audio_count = dance_count = 0
    for path in unique_paths:
        arrays = _load_feature_arrays(path)
        if audio_sum is None:
            audio_sum = np.zeros(arrays.audio.shape[1], dtype=np.float64)
            audio_sq = np.zeros(arrays.audio.shape[1], dtype=np.float64)
            dance_sum = np.zeros(arrays.dance.shape[1], dtype=np.float64)
            dance_sq = np.zeros(arrays.dance.shape[1], dtype=np.float64)
        audio_sum += arrays.audio.sum(axis=0, dtype=np.float64)
        audio_sq += np.square(arrays.audio, dtype=np.float64).sum(axis=0)
        dance_sum += arrays.dance.sum(axis=0, dtype=np.float64)
        dance_sq += np.square(arrays.dance, dtype=np.float64).sum(axis=0)
        audio_count += arrays.audio.shape[0]
        dance_count += arrays.dance.shape[0]

    assert audio_sum is not None and audio_sq is not None and dance_sum is not None and dance_sq is not None
    audio_mean = audio_sum / max(1, audio_count)
    dance_mean = dance_sum / max(1, dance_count)
    audio_var = np.maximum(audio_sq / max(1, audio_count) - np.square(audio_mean), 1e-6)
    dance_var = np.maximum(dance_sq / max(1, dance_count) - np.square(dance_mean), 1e-6)
    return (
        audio_mean.astype(np.float32),
        np.sqrt(audio_var).astype(np.float32),
        dance_mean.astype(np.float32),
        np.sqrt(dance_var).astype(np.float32),
    )


def _prev_cut_context(
    *,
    start_frame: int,
    window_frames: int,
    fps: float,
    last_cut_frame: int | None,
    previous_cut_frame: int | None,
    max_age_sec: float,
    max_gap_sec: float,
) -> np.ndarray:
    frame_offsets = np.arange(int(window_frames), dtype=np.float32)
    context = np.zeros((int(window_frames), 4), dtype=np.float32)
    if last_cut_frame is None:
        context[:, 1] = 1.0
        context[:, 2] = 1.0
        return context

    frame_indices = float(start_frame) + frame_offsets
    since_sec = (frame_indices - float(last_cut_frame)) / float(max(fps, 1e-6))
    prev_gap_sec = (
        (float(last_cut_frame) - float(previous_cut_frame)) / float(max(fps, 1e-6))
        if previous_cut_frame is not None
        else float(max_gap_sec)
    )
    context[:, 0] = 1.0
    context[:, 1] = np.clip(since_sec / max(float(max_age_sec), 1e-6), 0.0, 1.0)
    context[:, 2] = np.clip(prev_gap_sec / max(float(max_gap_sec), 1e-6), 0.0, 1.0)
    context[:, 3] = np.clip(since_sec / max(prev_gap_sec, 1e-6), 0.0, 2.0) / 2.0
    return context


class StageAWindowDataset(Dataset):
    def __init__(
        self,
        *,
        rows: list[dict],
        audio_mean: np.ndarray,
        audio_std: np.ndarray,
        dance_mean: np.ndarray,
        dance_std: np.ndarray,
        positive_repeat: int,
        training: bool,
        max_prev_cut_age_sec: float,
        max_prev_cut_gap_sec: float,
    ) -> None:
        self.rows: list[dict] = []
        for row in rows:
            repeat = int(max(1, positive_repeat)) if training and float(row["target_density"]) > 0 else 1
            for _ in range(repeat):
                self.rows.append(row)
        self.audio_mean = audio_mean.astype(np.float32)
        self.audio_std = np.maximum(audio_std.astype(np.float32), 1e-6)
        self.dance_mean = dance_mean.astype(np.float32)
        self.dance_std = np.maximum(dance_std.astype(np.float32), 1e-6)
        self.max_prev_cut_age_sec = float(max_prev_cut_age_sec)
        self.max_prev_cut_gap_sec = float(max_prev_cut_gap_sec)
        self.cache: dict[str, FeatureArrays] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _arrays(self, feature_path: str) -> FeatureArrays:
        if feature_path not in self.cache:
            self.cache[feature_path] = _load_feature_arrays(_resolve_repo_path(feature_path))
        return self.cache[feature_path]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        arrays = self._arrays(row["feature_path"])
        start = int(row["start_frame"])
        end = int(row["end_frame_exclusive"])
        window_frames = int(row["window_frames"])
        fps = float(row.get("fps", 30.0))

        audio = (arrays.audio[start:end] - self.audio_mean[None, :]) / self.audio_std[None, :]
        dance = (arrays.dance[start:end] - self.dance_mean[None, :]) / self.dance_std[None, :]
        cut_context = _prev_cut_context(
            start_frame=start,
            window_frames=window_frames,
            fps=fps,
            last_cut_frame=row.get("last_cut_frame"),
            previous_cut_frame=row.get("previous_cut_frame"),
            max_age_sec=self.max_prev_cut_age_sec,
            max_gap_sec=self.max_prev_cut_gap_sec,
        )
        valid = np.ones(window_frames, dtype=np.float32)
        return {
            "audio": torch.from_numpy(audio.astype(np.float32)),
            "dance": torch.from_numpy(dance.astype(np.float32)),
            "cut_context": torch.from_numpy(cut_context.astype(np.float32)),
            "valid": torch.from_numpy(valid),
            "target": torch.tensor(float(row["target_density"]), dtype=torch.float32),
        }


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class StageAKeyframeDensityTransformer(nn.Module):
    def __init__(
        self,
        *,
        audio_dim: int,
        dance_dim: int,
        cut_context_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.audio_proj = nn.Sequential(nn.Linear(audio_dim, model_dim), nn.LayerNorm(model_dim), nn.GELU())
        self.dance_proj = nn.Sequential(nn.Linear(dance_dim, model_dim), nn.LayerNorm(model_dim), nn.GELU())
        self.cut_proj = nn.Sequential(nn.Linear(cut_context_dim, model_dim), nn.LayerNorm(model_dim), nn.GELU())
        self.fusion = nn.Sequential(
            nn.LayerNorm(model_dim * 3),
            nn.Linear(model_dim * 3, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.position = SinusoidalPositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool_score = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )

    def forward(
        self,
        audio: torch.Tensor,
        dance: torch.Tensor,
        cut_context: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([self.audio_proj(audio), self.dance_proj(dance), self.cut_proj(cut_context)], dim=-1)
        x = self.position(self.fusion(x))
        key_padding_mask = valid <= 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        scores = self.pool_score(x).squeeze(-1).masked_fill(key_padding_mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return self.head(pooled).squeeze(-1)


class ResidualTemporalConvBlock(nn.Module):
    def __init__(self, *, dim: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (int(kernel_size) // 2) * int(dilation)
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=int(kernel_size), padding=padding, dilation=int(dilation)),
            nn.GroupNorm(1, dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.GroupNorm(1, dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class StageAKeyframeDensityCNN(nn.Module):
    def __init__(
        self,
        *,
        audio_dim: int,
        dance_dim: int,
        cut_context_dim: int,
        model_dim: int,
        kernel_size: int,
        dilations: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        input_dim = int(audio_dim) + int(dance_dim) + int(cut_context_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualTemporalConvBlock(
                    dim=int(model_dim),
                    kernel_size=int(kernel_size),
                    dilation=max(1, int(dilation)),
                    dropout=float(dropout),
                )
                for dilation in dilations
            ]
        )
        self.pool_score = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim * 3),
            nn.Linear(model_dim * 3, model_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(model_dim, 1),
        )

    def forward(
        self,
        audio: torch.Tensor,
        dance: torch.Tensor,
        cut_context: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        key_padding_mask = valid <= 0
        x = torch.cat([audio, dance, cut_context], dim=-1)
        x = self.input_proj(x).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)

        mask = valid.unsqueeze(-1).to(dtype=x.dtype)
        valid_count = torch.clamp(mask.sum(dim=1), min=1.0)
        mean_pool = torch.sum(x * mask, dim=1) / valid_count
        max_pool = x.masked_fill(key_padding_mask.unsqueeze(-1), -1e4).amax(dim=1)
        scores = self.pool_score(x).squeeze(-1).masked_fill(key_padding_mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        attention_pool = torch.sum(x * weights.unsqueeze(-1), dim=1)
        pooled = torch.cat([mean_pool, max_pool, attention_pool], dim=-1)
        return self.head(pooled).squeeze(-1)


class StageAKeyframeDensityFiLMCNN(nn.Module):
    def __init__(
        self,
        *,
        audio_dim: int,
        dance_dim: int,
        cut_context_dim: int,
        model_dim: int,
        kernel_size: int,
        dilations: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        condition_dim = int(audio_dim) + int(cut_context_dim)
        self.dance_proj = nn.Sequential(
            nn.Linear(dance_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.film = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
        )
        self.fusion_norm = nn.LayerNorm(model_dim)
        self.blocks = nn.ModuleList(
            [
                ResidualTemporalConvBlock(
                    dim=int(model_dim),
                    kernel_size=int(kernel_size),
                    dilation=max(1, int(dilation)),
                    dropout=float(dropout),
                )
                for dilation in dilations
            ]
        )
        self.pool_score = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim * 3),
            nn.Linear(model_dim * 3, model_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(model_dim, 1),
        )

    def forward(
        self,
        audio: torch.Tensor,
        dance: torch.Tensor,
        cut_context: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        key_padding_mask = valid <= 0
        dance_repr = self.dance_proj(dance)
        condition = self.condition_proj(torch.cat([audio, cut_context], dim=-1))
        gamma, beta = self.film(condition).chunk(2, dim=-1)
        # Audio and previous-cut context modulate the dance stream without discarding their own signal.
        x = self.fusion_norm(dance_repr * (1.0 + torch.tanh(gamma)) + beta + condition)
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)

        mask = valid.unsqueeze(-1).to(dtype=x.dtype)
        valid_count = torch.clamp(mask.sum(dim=1), min=1.0)
        mean_pool = torch.sum(x * mask, dim=1) / valid_count
        max_pool = x.masked_fill(key_padding_mask.unsqueeze(-1), -1e4).amax(dim=1)
        scores = self.pool_score(x).squeeze(-1).masked_fill(key_padding_mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        attention_pool = torch.sum(x * weights.unsqueeze(-1), dim=1)
        pooled = torch.cat([mean_pool, max_pool, attention_pool], dim=-1)
        return self.head(pooled).squeeze(-1)


def _parse_dilations(text: str) -> list[int]:
    dilations: list[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"CNN dilation must be positive: {item}")
        dilations.append(value)
    return dilations or [1, 2, 4, 8]


def build_stagea_model(
    *,
    model_type: str,
    audio_dim: int,
    dance_dim: int,
    cut_context_dim: int,
    model_dim: int,
    num_heads: int,
    num_layers: int,
    cnn_kernel_size: int,
    cnn_dilations: str | list[int],
    dropout: float,
) -> nn.Module:
    if str(model_type) in {"cnn", "film_cnn"}:
        dilations = cnn_dilations if isinstance(cnn_dilations, list) else _parse_dilations(str(cnn_dilations))
        if str(model_type) == "film_cnn":
            return StageAKeyframeDensityFiLMCNN(
                audio_dim=audio_dim,
                dance_dim=dance_dim,
                cut_context_dim=cut_context_dim,
                model_dim=model_dim,
                kernel_size=cnn_kernel_size,
                dilations=dilations,
                dropout=dropout,
            )
        return StageAKeyframeDensityCNN(
            audio_dim=audio_dim,
            dance_dim=dance_dim,
            cut_context_dim=cut_context_dim,
            model_dim=model_dim,
            kernel_size=cnn_kernel_size,
            dilations=dilations,
            dropout=dropout,
        )
    return StageAKeyframeDensityTransformer(
        audio_dim=audio_dim,
        dance_dim=dance_dim,
        cut_context_dim=cut_context_dim,
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )


def _loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, *, bce_weight: float, smooth_l1_weight: float) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    smooth = F.smooth_l1_loss(torch.sigmoid(logits), targets)
    return float(bce_weight) * bce + float(smooth_l1_weight) * smooth


def _pearson(preds: np.ndarray, targets: np.ndarray) -> float:
    if len(preds) < 2:
        return 0.0
    pred_std = float(np.std(preds))
    target_std = float(np.std(targets))
    if pred_std < 1e-8 or target_std < 1e-8:
        return 0.0
    return float(np.corrcoef(preds, targets)[0, 1])


def _binary_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    positive = targets > 0
    if not np.any(positive):
        return {"positive_threshold": 0.5, "positive_precision": 0.0, "positive_recall": 0.0, "positive_f1": 0.0}
    best = {"positive_threshold": 0.5, "positive_precision": 0.0, "positive_recall": 0.0, "positive_f1": 0.0}
    for threshold in np.linspace(0.01, 0.5, 50):
        predicted = preds >= threshold
        tp = int(np.sum(predicted & positive))
        fp = int(np.sum(predicted & ~positive))
        fn = int(np.sum(~predicted & positive))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best["positive_f1"]:
            best = {
                "positive_threshold": round(float(threshold), 4),
                "positive_precision": round(float(precision), 6),
                "positive_recall": round(float(recall), 6),
                "positive_f1": round(float(f1), 6),
            }
    return best


def _topk_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    positive_count = int(np.sum(targets > 0))
    total_density = float(np.sum(targets))
    if positive_count <= 0 or total_density <= 0:
        return {"positive_window_count": positive_count, "topk_recall": 0.0, "topk_density_capture": 0.0}
    k = min(positive_count, len(preds))
    top_indices = np.argsort(preds)[-k:]
    top_positive = int(np.sum(targets[top_indices] > 0))
    return {
        "positive_window_count": positive_count,
        "topk_recall": round(float(top_positive / positive_count), 6),
        "topk_density_capture": round(float(np.sum(targets[top_indices]) / total_density), 6),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, *, device: torch.device) -> dict[str, float]:
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        audio = batch["audio"].to(device)
        dance = batch["dance"].to(device)
        cut_context = batch["cut_context"].to(device)
        valid = batch["valid"].to(device)
        target = batch["target"].to(device)
        logits = model(audio, dance, cut_context, valid)
        loss = F.smooth_l1_loss(torch.sigmoid(logits), target, reduction="sum")
        total_loss += float(loss.item())
        total_count += int(target.numel())
        all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())
    preds = np.concatenate(all_preds).astype(np.float32) if all_preds else np.zeros(0, dtype=np.float32)
    targets = np.concatenate(all_targets).astype(np.float32) if all_targets else np.zeros(0, dtype=np.float32)
    mae = float(np.mean(np.abs(preds - targets))) if len(preds) else 0.0
    rmse = float(np.sqrt(np.mean(np.square(preds - targets)))) if len(preds) else 0.0
    metrics = {
        "smooth_l1": round(float(total_loss / max(1, total_count)), 6),
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "pearson": round(_pearson(preds, targets), 6),
        "pred_mean": round(float(np.mean(preds)) if len(preds) else 0.0, 6),
        "target_mean": round(float(np.mean(targets)) if len(targets) else 0.0, 6),
        "window_count": int(len(preds)),
    }
    metrics.update(_topk_metrics(preds, targets))
    metrics.update(_binary_metrics(preds, targets))
    return metrics


def _is_better(metric_name: str, current: dict[str, float], best: dict[str, float] | None) -> bool:
    if best is None:
        return True
    if metric_name == "mae":
        return float(current["mae"]) < float(best["mae"])
    return float(current[metric_name]) > float(best[metric_name])


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _snapshot_model_state(model: nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def main() -> None:
    args = parse_args()
    _set_seed(int(args.seed))
    device = _device_from_arg(str(args.device))

    manifest_path = args.dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Dataset manifest not found: {manifest_path}")
    manifest = _load_json(manifest_path)
    rows_path = _resolve_repo_path(manifest["window_index_path"])
    rows = _read_jsonl(rows_path)
    split_rows = {
        split_name: [row for row in rows if str(row.get("split")) == split_name]
        for split_name in ("train", "val", "test")
    }
    if not split_rows["train"] or not split_rows["val"] or not split_rows["test"]:
        raise SystemExit(f"Missing train/val/test rows in dataset: {manifest_path}")

    audio_mean, audio_std, dance_mean, dance_std = _compute_modality_stats(split_rows["train"])
    datasets = {
        "train": StageAWindowDataset(
            rows=split_rows["train"],
            audio_mean=audio_mean,
            audio_std=audio_std,
            dance_mean=dance_mean,
            dance_std=dance_std,
            positive_repeat=int(args.positive_repeat),
            training=True,
            max_prev_cut_age_sec=float(args.max_prev_cut_age_sec),
            max_prev_cut_gap_sec=float(args.max_prev_cut_gap_sec),
        ),
        "val": StageAWindowDataset(
            rows=split_rows["val"],
            audio_mean=audio_mean,
            audio_std=audio_std,
            dance_mean=dance_mean,
            dance_std=dance_std,
            positive_repeat=1,
            training=False,
            max_prev_cut_age_sec=float(args.max_prev_cut_age_sec),
            max_prev_cut_gap_sec=float(args.max_prev_cut_gap_sec),
        ),
        "test": StageAWindowDataset(
            rows=split_rows["test"],
            audio_mean=audio_mean,
            audio_std=audio_std,
            dance_mean=dance_mean,
            dance_std=dance_std,
            positive_repeat=1,
            training=False,
            max_prev_cut_age_sec=float(args.max_prev_cut_age_sec),
            max_prev_cut_gap_sec=float(args.max_prev_cut_gap_sec),
        ),
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=int(args.batch_size), shuffle=True, drop_last=False),
        "val": DataLoader(datasets["val"], batch_size=int(args.batch_size), shuffle=False, drop_last=False),
        "test": DataLoader(datasets["test"], batch_size=int(args.batch_size), shuffle=False, drop_last=False),
    }

    audio_dim = int(manifest["audio_feature_dim"])
    dance_dim = int(manifest["dance_feature_dim"])
    cut_context_dim = int(len(manifest.get("prev_cut_feature_names", [])) or 4)
    model = build_stagea_model(
        model_type=str(args.model_type),
        audio_dim=audio_dim,
        dance_dim=dance_dim,
        cut_context_dim=cut_context_dim,
        model_dim=int(args.model_dim),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        cnn_kernel_size=int(args.cnn_kernel_size),
        cnn_dilations=str(args.cnn_dilations),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    run_dir = _run_dir(args.output_root, seed=int(args.seed), tag=str(args.tag))
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = _init_wandb(args=args, run_dir=run_dir, manifest=manifest, device=device)

    print(f"Run dir: {run_dir}")
    print(f"Device: {device}")
    print(f"Train/Val/Test windows: {len(datasets['train'])}/{len(datasets['val'])}/{len(datasets['test'])}")
    print(f"Feature dims: audio={audio_dim}, dance={dance_dim}, prev_cut={cut_context_dim}")
    print(f"Model type: {args.model_type}")
    print(f"Selection metric: {args.selection_metric}")

    history: list[dict] = []
    best_metrics: dict[str, float] | None = None
    best_state: dict | None = None
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in loaders["train"]:
            audio = batch["audio"].to(device)
            dance = batch["dance"].to(device)
            cut_context = batch["cut_context"].to(device)
            valid = batch["valid"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(audio, dance, cut_context, valid)
            loss = _loss_from_logits(
                logits,
                target,
                bce_weight=float(args.bce_weight),
                smooth_l1_weight=float(args.smooth_l1_weight),
            )
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            total_loss += float(loss.item()) * int(target.numel())
            total_count += int(target.numel())

        train_loss = total_loss / max(1, total_count)
        val_metrics = evaluate(model, loaders["val"], device=device)
        row = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 6),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        if _is_better(str(args.selection_metric), val_metrics, best_metrics):
            best_metrics = dict(val_metrics)
            best_state = {
                "model_state": _snapshot_model_state(model),
                "args": vars(args),
                "manifest": manifest,
                "dataset_root": str(args.dataset_root),
                "audio_mean": audio_mean,
                "audio_std": audio_std,
                "dance_mean": dance_mean,
                "dance_std": dance_std,
                "best_epoch": epoch,
                "best_val_metrics": best_metrics,
                "feature_dims": {
                    "audio": audio_dim,
                    "dance": dance_dim,
                    "prev_cut": cut_context_dim,
                },
            }
            _save_checkpoint(run_dir / "best_model.pt", best_state)
        print(
            f"Epoch {epoch:02d}: loss={train_loss:.5f} "
            f"val_mae={val_metrics['mae']:.4f} "
            f"val_topk_capture={val_metrics['topk_density_capture']:.4f} "
            f"val_pos_f1={val_metrics['positive_f1']:.4f} "
            f"val_pearson={val_metrics['pearson']:.4f}"
        )
        if wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    **{f"val/{key}": value for key, value in val_metrics.items()},
                },
                step=epoch,
            )

    if best_state is None or best_metrics is None:
        raise SystemExit("Training did not produce a best checkpoint.")

    last_model_state = _snapshot_model_state(model)
    model.load_state_dict(best_state["model_state"])
    test_metrics = evaluate(model, loaders["test"], device=device)
    summary = {
        "run_dir": str(run_dir),
        "dataset_manifest": str(manifest_path),
        "device": str(device),
        "history": history,
        "best_epoch": int(best_state["best_epoch"]),
        "best_val_metrics": best_metrics,
        "test_metrics": test_metrics,
        "feature_dims": best_state["feature_dims"],
        "model_type": str(args.model_type),
    }
    _write_json(run_dir / "metrics.json", summary)
    if not args.no_save_last_checkpoint:
        _save_checkpoint(
            run_dir / "checkpoint_last.pt",
            {
                **best_state,
                "last_model_state": last_model_state,
                "history": history,
                "test_metrics": test_metrics,
            },
        )
    if wandb_run is not None:
        wandb.summary["best_epoch"] = int(best_state["best_epoch"])
        for key, value in best_metrics.items():
            wandb.summary[f"best_val_{key}"] = value
        for key, value in test_metrics.items():
            wandb.summary[f"test_{key}"] = value
        wandb.summary["metrics_json"] = str(run_dir / "metrics.json")
        wandb.finish()

    print(f"Saved checkpoint: {run_dir / 'best_model.pt'}")
    print(f"Saved metrics: {run_dir / 'metrics.json'}")
    print(
        f"Test mae={test_metrics['mae']:.4f} "
        f"topk_capture={test_metrics['topk_density_capture']:.4f} "
        f"positive_f1={test_metrics['positive_f1']:.4f} "
        f"pearson={test_metrics['pearson']:.4f}"
    )


if __name__ == "__main__":
    main()
