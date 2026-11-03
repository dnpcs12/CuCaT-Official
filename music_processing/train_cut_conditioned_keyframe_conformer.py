from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import wandb
except ImportError:
    wandb = None

from music_processing.cut_conditioned_keyframe_conformer import (
    CutConditionedKeyframeConformer,
    DancePrimaryFiLMConcatTransformer,
)
from music_processing.train_cut_transition_baseline import (
    _device_from_arg,
    _match_cut_events,
    _safe_f1,
    _triage_predictions,
)


DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent / "results" / "cut_conditioned_keyframe_dataset_dcm"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "cut_conditioned_keyframe_models"
DEFAULT_WANDB_KEY_PATH = Path(__file__).resolve().parent / ".wandb_api_key"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a keyframe-only multimodal conformer conditioned on GT cut boundaries."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--positive-repeat", type=int, default=4)
    parser.add_argument(
        "--architecture",
        choices=["conformer", "dance_primary_film_concat"],
        default="dance_primary_film_concat",
        help="Stage-2 keyframe model architecture.",
    )
    parser.add_argument("--model-dim", type=int, default=192)
    parser.add_argument("--audio-model-dim", type=int, default=96)
    parser.add_argument("--condition-model-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--audio-layers", type=int, default=2)
    parser.add_argument("--dance-layers", type=int, default=2)
    parser.add_argument("--cross-layers", type=int, default=2)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--film-scale", type=float, default=0.2)
    parser.add_argument("--fusion-scale", type=float, default=1.0)
    parser.add_argument("--audio-gate-init", type=float, default=0.35)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--positive-weight", type=float, default=4.0)
    parser.add_argument("--sequence-loss-weight", type=float, default=0.75)
    parser.add_argument("--tolerance-frames", type=int, default=3)
    parser.add_argument("--min-peak-gap-frames", type=int, default=6)
    parser.add_argument(
        "--threshold-grid",
        type=float,
        nargs="*",
        default=[0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    )
    parser.add_argument(
        "--selection-metric",
        choices=["keyframe_segment_f1_active", "keyframe_global_event_f1"],
        default="keyframe_segment_f1_active",
    )
    parser.add_argument("--wandb-project", default="music-cut-keyframe-stage2")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-key-file", type=Path, default=DEFAULT_WANDB_KEY_PATH)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_local_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parents[1] / path).resolve()


def _load_wandb_key(path: Path) -> str:
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"W&B key file is empty: {path}")
    return key


def _init_wandb(*, args: argparse.Namespace, run_dir: Path, split: dict, manifest: dict, device: torch.device):
    if args.disable_wandb:
        return None
    if wandb is None:
        raise SystemExit("wandb is not installed in the current environment. Install it or pass --disable-wandb.")
    if not args.wandb_key_file.exists():
        raise SystemExit(f"W&B key file not found: {args.wandb_key_file}")
    key = _load_wandb_key(args.wandb_key_file)
    os.environ["WANDB_API_KEY"] = key
    wandb.login(key=key, relogin=True)
    run_name = args.wandb_run_name or run_dir.name
    return wandb.init(
        project=args.wandb_project,
        name=run_name,
        dir=str(run_dir),
        config={
            **vars(args),
            "device": str(device),
            "dataset_manifest": str(args.dataset_root / "manifest.json"),
            "audio_feature_dim": len(manifest.get("audio_feature_names", [])),
            "dance_feature_dim": len(manifest.get("dance_feature_names", [])),
            "condition_feature_dim": len(manifest.get("condition_feature_names", [])),
            "segment_count": manifest.get("segment_count"),
            "split": split,
        },
    )


def _resolve_split(manifest: dict, split_manifest_path: Path | None) -> tuple[dict[str, list[str]], str]:
    segments = manifest.get("segments", [])
    entries_by_id = {str(entry["id"]): entry for entry in segments}

    def extract(payload: dict) -> dict[str, list[str]]:
        source = payload["split"] if isinstance(payload.get("split"), dict) else payload
        return {
            "train": [str(item) for item in source.get("train", [])],
            "val": [str(item) for item in source.get("val", [])],
            "test": [str(item) for item in source.get("test", [])],
        }

    if split_manifest_path is not None:
        split = extract(_load_json(split_manifest_path))
        source_name = str(split_manifest_path)
    elif isinstance(manifest.get("split"), dict):
        split = extract(manifest)
        source_name = "dataset_manifest"
    else:
        raise SystemExit("This stage-2 dataset requires an explicit or embedded split.")

    overlap = (set(split["train"]) & set(split["val"])) | (set(split["train"]) & set(split["test"])) | (set(split["val"]) & set(split["test"]))
    if overlap:
        raise SystemExit(f"Split overlap detected: {sorted(overlap)}")
    missing = {name: [item for item in split[name] if item not in entries_by_id] for name in ("train", "val", "test")}
    if any(missing.values()):
        raise SystemExit(f"Split ids missing from stage-2 dataset: {missing}")
    return split, source_name


def _compute_stats(entries: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    audio_sum = audio_sq = dance_sum = dance_sq = cond_sum = cond_sq = None
    frame_count = 0
    for entry in entries:
        payload = np.load(_resolve_local_path(entry["feature_path"]))
        audio = np.asarray(payload["audio_features"], dtype=np.float32)
        dance = np.asarray(payload["dance_features"], dtype=np.float32)
        condition = np.asarray(payload["condition_features"], dtype=np.float32)
        if audio_sum is None:
            audio_sum = np.zeros(audio.shape[1], dtype=np.float64)
            audio_sq = np.zeros(audio.shape[1], dtype=np.float64)
            dance_sum = np.zeros(dance.shape[1], dtype=np.float64)
            dance_sq = np.zeros(dance.shape[1], dtype=np.float64)
            cond_sum = np.zeros(condition.shape[1], dtype=np.float64)
            cond_sq = np.zeros(condition.shape[1], dtype=np.float64)
        audio_sum += np.sum(audio, axis=0, dtype=np.float64)
        audio_sq += np.sum(np.square(audio, dtype=np.float32), axis=0, dtype=np.float64)
        dance_sum += np.sum(dance, axis=0, dtype=np.float64)
        dance_sq += np.sum(np.square(dance, dtype=np.float32), axis=0, dtype=np.float64)
        cond_sum += np.sum(condition, axis=0, dtype=np.float64)
        cond_sq += np.sum(np.square(condition, dtype=np.float32), axis=0, dtype=np.float64)
        frame_count += int(audio.shape[0])
    if frame_count <= 0 or audio_sum is None or dance_sum is None or cond_sum is None:
        raise ValueError("No training features available.")

    def finalize(total: np.ndarray, sq_total: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = total / float(frame_count)
        var = np.maximum(1e-8, sq_total / float(frame_count) - np.square(mean))
        return mean.astype(np.float32), np.sqrt(var).astype(np.float32)

    audio_mean, audio_std = finalize(audio_sum, audio_sq)
    dance_mean, dance_std = finalize(dance_sum, dance_sq)
    cond_mean, cond_std = finalize(cond_sum, cond_sq)
    return audio_mean, audio_std, dance_mean, dance_std, cond_mean, cond_std


class CutConditionedSegmentDataset(Dataset):
    def __init__(
        self,
        *,
        entries: list[dict],
        audio_mean: np.ndarray,
        audio_std: np.ndarray,
        dance_mean: np.ndarray,
        dance_std: np.ndarray,
        cond_mean: np.ndarray,
        cond_std: np.ndarray,
        max_frames: int,
        positive_repeat: int,
        training: bool,
    ) -> None:
        self.max_frames = int(max_frames)
        self.samples: list[dict] = []
        for entry in entries:
            payload = np.load(_resolve_local_path(entry["feature_path"]))
            audio = (np.asarray(payload["audio_features"], dtype=np.float32) - audio_mean[None, :]) / audio_std[None, :]
            dance = (np.asarray(payload["dance_features"], dtype=np.float32) - dance_mean[None, :]) / dance_std[None, :]
            condition = (np.asarray(payload["condition_features"], dtype=np.float32) - cond_mean[None, :]) / cond_std[None, :]
            key_soft = np.asarray(payload["keyframe_soft_labels"], dtype=np.float32)
            key_mask = np.asarray(payload["keyframe_label_mask"], dtype=np.float32)
            key_frames = np.asarray(payload["keyframe_frames"], dtype=np.int32)
            repeat = int(max(1, positive_repeat)) if training and len(key_frames) > 0 else 1
            for _ in range(repeat):
                self.samples.append(
                    {
                        "id": str(entry["id"]),
                        "audio": audio,
                        "dance": dance,
                        "condition": condition,
                        "key_soft": key_soft,
                        "key_mask": key_mask,
                        "key_frames": key_frames,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.samples[index]
        length = int(len(item["audio"]))
        audio = np.zeros((self.max_frames, item["audio"].shape[1]), dtype=np.float32)
        dance = np.zeros((self.max_frames, item["dance"].shape[1]), dtype=np.float32)
        condition = np.zeros((self.max_frames, item["condition"].shape[1]), dtype=np.float32)
        key_soft = np.zeros(self.max_frames, dtype=np.float32)
        key_mask = np.zeros(self.max_frames, dtype=np.float32)
        valid = np.zeros(self.max_frames, dtype=np.float32)
        audio[:length] = item["audio"]
        dance[:length] = item["dance"]
        condition[:length] = item["condition"]
        key_soft[:length] = item["key_soft"]
        key_mask[:length] = item["key_mask"]
        valid[:length] = 1.0
        return {
            "audio": torch.from_numpy(audio),
            "dance": torch.from_numpy(dance),
            "condition": torch.from_numpy(condition),
            "key_soft": torch.from_numpy(key_soft),
            "key_mask": torch.from_numpy(key_mask),
            "valid": torch.from_numpy(valid),
        }


def _sequence_bce_dice_loss(
    logits: torch.Tensor,
    targets_soft: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    positive_weight: float,
    sequence_loss_weight: float,
) -> torch.Tensor:
    denom = torch.clamp(valid_mask.sum(), min=1.0)
    if float(denom.detach().cpu()) <= 0.0:
        return logits.new_tensor(0.0)
    weights = 1.0 + float(positive_weight) * targets_soft
    bce = F.binary_cross_entropy_with_logits(logits, targets_soft, reduction="none")
    bce_loss = torch.sum(bce * weights * valid_mask) / denom
    probs = torch.sigmoid(logits) * valid_mask
    targets = targets_soft * valid_mask
    intersection = torch.sum(probs * targets, dim=1)
    cardinality = torch.sum(probs, dim=1) + torch.sum(targets, dim=1)
    dice = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
    sequence_loss = 1.0 - dice.mean()
    return bce_loss + float(sequence_loss_weight) * sequence_loss


def _predict_segment(
    model: torch.nn.Module,
    *,
    entry: dict,
    audio_mean: np.ndarray,
    audio_std: np.ndarray,
    dance_mean: np.ndarray,
    dance_std: np.ndarray,
    cond_mean: np.ndarray,
    cond_std: np.ndarray,
    max_frames: int,
    device: torch.device,
) -> np.ndarray:
    payload = np.load(_resolve_local_path(entry["feature_path"]))
    audio = (np.asarray(payload["audio_features"], dtype=np.float32) - audio_mean[None, :]) / audio_std[None, :]
    dance = (np.asarray(payload["dance_features"], dtype=np.float32) - dance_mean[None, :]) / dance_std[None, :]
    condition = (np.asarray(payload["condition_features"], dtype=np.float32) - cond_mean[None, :]) / cond_std[None, :]
    length = int(len(audio))
    batch_audio = np.zeros((1, max_frames, audio.shape[1]), dtype=np.float32)
    batch_dance = np.zeros((1, max_frames, dance.shape[1]), dtype=np.float32)
    batch_condition = np.zeros((1, max_frames, condition.shape[1]), dtype=np.float32)
    valid = np.zeros((1, max_frames), dtype=np.float32)
    batch_audio[0, :length] = audio
    batch_dance[0, :length] = dance
    batch_condition[0, :length] = condition
    valid[0, :length] = 1.0
    model.eval()
    with torch.no_grad():
        outputs = model(
            torch.from_numpy(batch_audio).to(device),
            torch.from_numpy(batch_dance).to(device),
            torch.from_numpy(batch_condition).to(device),
            valid_mask=torch.from_numpy(valid).to(device),
        )
        probs = torch.sigmoid(outputs["keyframe_logits"])[0, :length].detach().cpu().numpy().astype(np.float32)
    return probs


def _precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = float(tp / (tp + fp)) if tp + fp > 0 else 0.0
    recall = float(tp / (tp + fn)) if tp + fn > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    return {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}


def _build_model(
    *,
    args: argparse.Namespace,
    audio_input_dim: int,
    dance_input_dim: int,
    condition_input_dim: int,
) -> torch.nn.Module:
    if str(args.architecture) == "conformer":
        return CutConditionedKeyframeConformer(
            audio_input_dim=int(audio_input_dim),
            dance_input_dim=int(dance_input_dim),
            condition_input_dim=int(condition_input_dim),
            model_dim=int(args.model_dim),
            num_heads=int(args.num_heads),
            audio_layers=int(args.audio_layers),
            dance_layers=int(args.dance_layers),
            cross_layers=int(args.cross_layers),
            fusion_layers=int(args.fusion_layers),
            dropout=float(args.dropout),
        )
    if str(args.architecture) == "dance_primary_film_concat":
        return DancePrimaryFiLMConcatTransformer(
            audio_input_dim=int(audio_input_dim),
            dance_input_dim=int(dance_input_dim),
            condition_input_dim=int(condition_input_dim),
            model_dim=int(args.model_dim),
            audio_model_dim=int(args.audio_model_dim),
            condition_model_dim=int(args.condition_model_dim),
            num_heads=int(args.num_heads),
            transformer_layers=int(args.transformer_layers),
            dropout=float(args.dropout),
            film_scale=float(args.film_scale),
            fusion_scale=float(args.fusion_scale),
            audio_gate_init=float(args.audio_gate_init),
        )
    raise ValueError(f"Unsupported architecture: {args.architecture}")


def _evaluate_entries(
    model: torch.nn.Module,
    *,
    entries: list[dict],
    audio_mean: np.ndarray,
    audio_std: np.ndarray,
    dance_mean: np.ndarray,
    dance_std: np.ndarray,
    cond_mean: np.ndarray,
    cond_std: np.ndarray,
    max_frames: int,
    threshold: float,
    tolerance_frames: int,
    min_peak_gap_frames: int,
    device: torch.device,
) -> dict:
    tp = fp = fn = 0
    segment_scores_active: list[float] = []
    segment_scores_all: list[float] = []
    for entry in entries:
        payload = np.load(_resolve_local_path(entry["feature_path"]))
        gt_frames = np.asarray(payload["keyframe_frames"], dtype=np.int32)
        probs = _predict_segment(
            model,
            entry=entry,
            audio_mean=audio_mean,
            audio_std=audio_std,
            dance_mean=dance_mean,
            dance_std=dance_std,
            cond_mean=cond_mean,
            cond_std=cond_std,
            max_frames=max_frames,
            device=device,
        )
        pred_frames = _triage_predictions(probs, threshold=float(threshold), min_gap_frames=int(min_peak_gap_frames))
        seg_tp, seg_fp, seg_fn = _match_cut_events(pred_frames, gt_frames, tolerance=int(tolerance_frames))
        tp += seg_tp
        fp += seg_fp
        fn += seg_fn
        score = _safe_f1(seg_tp, seg_fp, seg_fn, empty_value=1.0)
        segment_scores_all.append(score)
        if len(pred_frames) > 0 or len(gt_frames) > 0:
            segment_scores_active.append(score)
    metrics = _precision_recall_f1(tp, fp, fn)
    metrics.update(
        {
            "threshold": float(threshold),
            "keyframe_global_event_f1": metrics["f1"],
            "keyframe_segment_f1_active": round(float(np.mean(segment_scores_active)) if segment_scores_active else 0.0, 6),
            "keyframe_segment_f1_all": round(float(np.mean(segment_scores_all)) if segment_scores_all else 0.0, 6),
            "keyframe_tp": int(tp),
            "keyframe_fp": int(fp),
            "keyframe_fn": int(fn),
        }
    )
    return metrics


def _search_best_threshold(
    *,
    model: torch.nn.Module,
    entries: list[dict],
    audio_mean: np.ndarray,
    audio_std: np.ndarray,
    dance_mean: np.ndarray,
    dance_std: np.ndarray,
    cond_mean: np.ndarray,
    cond_std: np.ndarray,
    max_frames: int,
    threshold_grid: list[float],
    tolerance_frames: int,
    min_peak_gap_frames: int,
    selection_metric: str,
    device: torch.device,
) -> dict:
    best_metrics = None
    for threshold in threshold_grid:
        metrics = _evaluate_entries(
            model,
            entries=entries,
            audio_mean=audio_mean,
            audio_std=audio_std,
            dance_mean=dance_mean,
            dance_std=dance_std,
            cond_mean=cond_mean,
            cond_std=cond_std,
            max_frames=max_frames,
            threshold=float(threshold),
            tolerance_frames=int(tolerance_frames),
            min_peak_gap_frames=int(min_peak_gap_frames),
            device=device,
        )
        if best_metrics is None or float(metrics[selection_metric]) > float(best_metrics[selection_metric]):
            best_metrics = metrics
    if best_metrics is None:
        raise RuntimeError("Threshold search failed.")
    return best_metrics


def _run_dir(output_root: Path, *, seed: int, tag: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    return output_root / f"cut_conditioned_key_seed{seed}_{stamp}{suffix}"


def main() -> None:
    args = parse_args()
    _set_seed(int(args.seed))
    device = _device_from_arg(str(args.device))

    manifest_path = args.dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Dataset manifest not found: {manifest_path}")
    manifest = _load_json(manifest_path)
    segments = manifest.get("segments", [])
    if not isinstance(segments, list) or not segments:
        raise SystemExit(f"No segments found in dataset manifest: {manifest_path}")

    split, split_source = _resolve_split(manifest, args.split_manifest)
    entries_by_id = {str(entry["id"]): entry for entry in segments}
    train_entries = [entries_by_id[item] for item in split["train"]]
    val_entries = [entries_by_id[item] for item in split["val"]]
    test_entries = [entries_by_id[item] for item in split["test"]]

    audio_mean, audio_std, dance_mean, dance_std, cond_mean, cond_std = _compute_stats(train_entries)
    train_dataset = CutConditionedSegmentDataset(
        entries=train_entries,
        audio_mean=audio_mean,
        audio_std=audio_std,
        dance_mean=dance_mean,
        dance_std=dance_std,
        cond_mean=cond_mean,
        cond_std=cond_std,
        max_frames=int(manifest["max_segment_frames"]),
        positive_repeat=int(args.positive_repeat),
        training=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=True, drop_last=False)

    model = _build_model(
        args=args,
        audio_input_dim=int(len(manifest.get("audio_feature_names", [])) or train_dataset.samples[0]["audio"].shape[1]),
        dance_input_dim=int(len(manifest.get("dance_feature_names", [])) or train_dataset.samples[0]["dance"].shape[1]),
        condition_input_dim=int(len(manifest.get("condition_feature_names", [])) or train_dataset.samples[0]["condition"].shape[1]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    run_dir = _run_dir(args.output_root, seed=int(args.seed), tag=str(args.tag))
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = _init_wandb(args=args, run_dir=run_dir, split=split, manifest=manifest, device=device)
    print(f"Run dir: {run_dir}")
    print(f"Device: {device}")
    print(f"Split source: {split_source}")
    print(f"Train/Val/Test segments: {len(train_entries)}/{len(val_entries)}/{len(test_entries)}")
    print(f"Train samples: {len(train_dataset)}")

    history: list[dict] = []
    best_val_metrics: dict | None = None
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        loss_sum = 0.0
        step_count = 0
        for batch in train_loader:
            audio = batch["audio"].to(device)
            dance = batch["dance"].to(device)
            condition = batch["condition"].to(device)
            key_soft = batch["key_soft"].to(device)
            key_mask = batch["key_mask"].to(device)
            valid = batch["valid"].to(device)
            outputs = model(audio, dance, condition, valid_mask=valid)
            key_valid = valid * key_mask
            loss = _sequence_bce_dice_loss(
                outputs["keyframe_logits"],
                key_soft,
                key_valid,
                positive_weight=float(args.positive_weight),
                sequence_loss_weight=float(args.sequence_loss_weight),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            step_count += 1

        train_loss = loss_sum / max(1, step_count)
        val_metrics = _search_best_threshold(
            model=model,
            entries=val_entries,
            audio_mean=audio_mean,
            audio_std=audio_std,
            dance_mean=dance_mean,
            dance_std=dance_std,
            cond_mean=cond_mean,
            cond_std=cond_std,
            max_frames=int(manifest["max_segment_frames"]),
            threshold_grid=[float(x) for x in args.threshold_grid],
            tolerance_frames=int(args.tolerance_frames),
            min_peak_gap_frames=int(args.min_peak_gap_frames),
            selection_metric=str(args.selection_metric),
            device=device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "keyframe_global_event_f1": val_metrics["keyframe_global_event_f1"],
                "keyframe_segment_f1_active": val_metrics["keyframe_segment_f1_active"],
                "keyframe_segment_f1_all": val_metrics["keyframe_segment_f1_all"],
                "keyframe_precision": val_metrics["precision"],
                "keyframe_recall": val_metrics["recall"],
                "keyframe_threshold": val_metrics["threshold"],
            }
        )
        print(
            f"Epoch {epoch:02d}: loss={train_loss:.5f} "
            f"val_key_active_f1={val_metrics['keyframe_segment_f1_active']:.4f} "
            f"val_key_global_f1={val_metrics['keyframe_global_event_f1']:.4f} "
            f"(th={val_metrics['threshold']:.2f})"
        )
        if wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/keyframe_global_event_f1": val_metrics["keyframe_global_event_f1"],
                    "val/keyframe_segment_f1_active": val_metrics["keyframe_segment_f1_active"],
                    "val/keyframe_segment_f1_all": val_metrics["keyframe_segment_f1_all"],
                    "val/keyframe_precision": val_metrics["precision"],
                    "val/keyframe_recall": val_metrics["recall"],
                    "val/keyframe_threshold": val_metrics["threshold"],
                    "val/keyframe_tp": val_metrics["keyframe_tp"],
                    "val/keyframe_fp": val_metrics["keyframe_fp"],
                    "val/keyframe_fn": val_metrics["keyframe_fn"],
                },
                step=epoch,
            )

        if best_val_metrics is None or float(val_metrics[str(args.selection_metric)]) > float(best_val_metrics[str(args.selection_metric)]):
            best_val_metrics = val_metrics
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "audio_mean": audio_mean,
                    "audio_std": audio_std,
                    "dance_mean": dance_mean,
                    "dance_std": dance_std,
                    "cond_mean": cond_mean,
                    "cond_std": cond_std,
                    "split": split,
                    "args": vars(args),
                    "best_val_metrics": best_val_metrics,
                    "history": history,
                    "audio_feature_names": manifest.get("audio_feature_names", []),
                    "dance_feature_names": manifest.get("dance_feature_names", []),
                    "condition_feature_names": manifest.get("condition_feature_names", []),
                    "wandb_run_id": getattr(wandb_run, "id", None),
                },
                run_dir / "best_model.pt",
            )
            if wandb_run is not None:
                wandb.summary["best_epoch"] = epoch
                wandb.summary["selection_metric"] = str(args.selection_metric)
                wandb.summary["best_val_keyframe_global_event_f1"] = val_metrics["keyframe_global_event_f1"]
                wandb.summary["best_val_keyframe_segment_f1_active"] = val_metrics["keyframe_segment_f1_active"]
                wandb.summary["best_val_keyframe_segment_f1_all"] = val_metrics["keyframe_segment_f1_all"]
                wandb.summary["best_val_keyframe_precision"] = val_metrics["precision"]
                wandb.summary["best_val_keyframe_recall"] = val_metrics["recall"]
                wandb.summary["best_val_keyframe_threshold"] = val_metrics["threshold"]

    if best_val_metrics is None:
        raise RuntimeError("No validation metrics were produced.")

    best_payload = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_payload["model_state"])
    model.to(device)

    test_metrics = _evaluate_entries(
        model,
        entries=test_entries,
        audio_mean=audio_mean,
        audio_std=audio_std,
        dance_mean=dance_mean,
        dance_std=dance_std,
        cond_mean=cond_mean,
        cond_std=cond_std,
        max_frames=int(manifest["max_segment_frames"]),
        threshold=float(best_val_metrics["threshold"]),
        tolerance_frames=int(args.tolerance_frames),
        min_peak_gap_frames=int(args.min_peak_gap_frames),
        device=device,
    )

    metrics_payload = {
        "run_dir": str(run_dir),
        "dataset_manifest": str(manifest_path),
        "split_source": split_source,
        "device": str(device),
        "split": split,
        "history": history,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "split.json").write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")

    if wandb_run is not None:
        wandb.summary["run_dir"] = str(run_dir)
        wandb.summary["metrics_json"] = str(run_dir / "metrics.json")
        wandb.summary["checkpoint_path"] = str(run_dir / "best_model.pt")
        wandb.summary["test_keyframe_global_event_f1"] = test_metrics["keyframe_global_event_f1"]
        wandb.summary["test_keyframe_segment_f1_active"] = test_metrics["keyframe_segment_f1_active"]
        wandb.summary["test_keyframe_segment_f1_all"] = test_metrics["keyframe_segment_f1_all"]
        wandb.summary["test_keyframe_precision"] = test_metrics["precision"]
        wandb.summary["test_keyframe_recall"] = test_metrics["recall"]
        wandb.summary["test_keyframe_threshold"] = best_val_metrics["threshold"]
        wandb.log(
            {
                "test/keyframe_global_event_f1": test_metrics["keyframe_global_event_f1"],
                "test/keyframe_segment_f1_active": test_metrics["keyframe_segment_f1_active"],
                "test/keyframe_segment_f1_all": test_metrics["keyframe_segment_f1_all"],
                "test/keyframe_precision": test_metrics["precision"],
                "test/keyframe_recall": test_metrics["recall"],
                "test/keyframe_threshold": best_val_metrics["threshold"],
            },
            step=int(args.epochs) + 1,
        )
        wandb.finish()

    print(
        f"Test key_active_f1={test_metrics['keyframe_segment_f1_active']:.4f} "
        f"key_global_f1={test_metrics['keyframe_global_event_f1']:.4f} "
        f"(th={best_val_metrics['threshold']:.2f})"
    )


if __name__ == "__main__":
    main()
