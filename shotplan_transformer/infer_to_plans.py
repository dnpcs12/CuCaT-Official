from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera_autoshot.shotplan_bucket_utils import group_and_consolidate_by_frame

from .compact_dance import (
    compact_dance_from_full,
    compact_dance_roll_signals_from_full,
    compact_dance_root_head_rot6d_from_full,
    load_feature_manifest,
)
from .dataset import load_normalization_stats
from .model import (
    DiscreteShotPlanCMN,
    DiscreteShotPlanCMNHeadMoE,
    DiscreteShotPlanMLP,
    DiscreteShotPlanTransformer,
    ShotPlanTransformerConfig,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_ROOT = WORKSPACE_ROOT / "music_processing" / "results" / "cut_keyframe_multimodal_dataset" / "songs"
DEFAULT_LABEL_ROOT = WORKSPACE_ROOT / "camera_autoshot" / "reverses_roll_feet_5fps"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "camera_autoshot" / "results" / "shotplan_transformer_gtkey_base"
DEFAULT_CHECKPOINT = (
    WORKSPACE_ROOT
    / "shotplan_transformer"
    / "results"
    / "training_runs"
    / "shotplan_transformer_20260502_000514_base"
    / "best.pt"
)
PLAN_FIELDS = (
    "target_body",
    "shot_size",
    "angle",
    "composition",
    "roll",
)
PAD_VALUE = -100


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Predict semantic camera shot plans from a trained shotplan transformer, "
            "using GT keyframe timing and optional GT slot counts."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument(
        "--label-name",
        default="camera_poses_inferred.json",
        help="Camera pose JSON filename to read inside each cNN label folder.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--indices", default="", help="Comma-separated cNN indices. Empty means all labels found.")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--timeline-fps", type=float, default=30.0)
    parser.add_argument("--window-radius", type=int, default=-1, help="Default uses checkpoint manifest window_radius.")
    parser.add_argument("--window-stride", type=int, default=-1, help="Default uses checkpoint manifest window_stride.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--slot-count-source",
        choices=["gt", "predicted"],
        default="gt",
        help="Use GT consolidated slot counts or model-predicted slot counts.",
    )
    parser.add_argument(
        "--prev-context-source",
        choices=["label", "predicted"],
        default="label",
        help=(
            "Use previous labels from the timing JSON, or run sequentially and feed "
            "the transformer's own previous predictions back as prev2 context."
        ),
    )
    return parser


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _parse_indices(raw: str, label_root: Path, label_name: str) -> list[int]:
    if raw.strip():
        values = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            values.append(int(token[1:] if token.startswith("c") else token))
        return sorted(set(values))

    values: list[int] = []
    for path in sorted(label_root.glob(f"c*/{label_name}")):
        name = path.parent.name
        if name.startswith("c") and name[1:].isdigit():
            values.append(int(name[1:]))
    return sorted(set(values))


def _default_token_for_field(field: str) -> str:
    if field == "roll":
        return "none"
    if field == "transition":
        return "linear"
    return ""


def _load_gt_frame_buckets(label_path: Path, label_fields: list[str] | tuple[str, ...]) -> dict[int, list[dict[str, str]]]:
    payload = _load_json(label_path)
    rows = payload.get("keyframes", payload if isinstance(payload, list) else [])
    normalized: list[dict[str, str | int]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized_row = {"frame": int(row["frame"])}
        for field in label_fields:
            normalized_row[field] = str(row.get(field, _default_token_for_field(field)))
        normalized.append(normalized_row)
    grouped = group_and_consolidate_by_frame(
        normalized,
        frame_field="frame",
        signature_fields=label_fields,
    )
    return {
        int(frame): [{field: str(row[field]) for field in label_fields} for row in frame_rows]
        for frame, frame_rows in grouped.items()
    }


def _downsample_source_indices(num_frames: int, *, sample_fps: float, timeline_fps: float) -> np.ndarray:
    sample_count = int(round((num_frames - 1) * sample_fps / timeline_fps)) + 1
    return np.asarray(
        [min(int(round(sample_idx * timeline_fps / sample_fps)), num_frames - 1) for sample_idx in range(sample_count)],
        dtype=np.int32,
    )


def _window_offsets(*, radius: int, stride: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("window radius must be non-negative")
    if stride <= 0:
        raise ValueError("window stride must be positive")
    if radius == 0:
        return np.asarray([0], dtype=np.int32)
    if radius % stride != 0:
        raise ValueError(
            f"window radius must be divisible by window stride for symmetric sampling: radius={radius}, stride={stride}"
        )
    return np.arange(-radius, radius + 1, stride, dtype=np.int32)


def _window_indices(center_index: int, *, length: int, radius: int, stride: int = 1) -> np.ndarray:
    values = int(center_index) + _window_offsets(radius=int(radius), stride=int(stride))
    return np.clip(values, 0, length - 1)


def _build_model_from_checkpoint(checkpoint: dict[str, Any]) -> torch.nn.Module:
    manifest = checkpoint["manifest"]
    config = checkpoint.get("config", {})
    architecture = str(config.get("architecture", "transformer"))
    vocab_sizes = {field: len(values) for field, values in manifest["vocabularies"].items()}
    if "fusion" in config:
        fusion = str(config["fusion"])
    elif "fusion" in manifest:
        fusion = str(manifest["fusion"])
    elif any(key.startswith("modality_embedding") for key in checkpoint["model"]):
        fusion = "interleave"
    else:
        fusion = "film"
    use_context_tokens = any(
        key.startswith("cut_flag_proj") or key.startswith("prev_field_embeddings")
        for key in checkpoint["model"]
    )
    model_config = ShotPlanTransformerConfig(
        audio_dim=int(manifest["audio_feature_dim"]),
        dance_dim=int(manifest["dance_feature_dim"]),
        max_slots=int(manifest["max_slots"]),
        vocab_sizes=vocab_sizes,
        window_size=int(manifest["window_size"]),
        prev_context_size=int(manifest.get("prev_context_size", 2)),
        future_context_size=int(manifest.get("future_context_size", 0)),
        model_dim=int(config.get("model_dim", 256)),
        num_heads=int(config.get("num_heads", 8)),
        num_encoder_layers=int(config.get("num_encoder_layers", 4)),
        ffn_dim=int(config.get("ffn_dim", 512)),
        dropout=float(config.get("dropout", 0.1)),
        input_dropout=float(config.get("input_dropout", 0.0)),
        fusion=fusion,
        use_context_tokens=use_context_tokens,
        joint_vocab_sizes=config.get("joint_vocab_sizes") or None,
    )
    if architecture == "mlp":
        model = DiscreteShotPlanMLP(model_config)
    elif architecture == "cmn":
        model = DiscreteShotPlanCMN(model_config)
    elif architecture == "cmn_head_moe":
        model = DiscreteShotPlanCMNHeadMoE(model_config)
    else:
        model = DiscreteShotPlanTransformer(model_config)
    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint is not compatible with the current prev2/cut-flag model. "
            "Train a fresh shotplan_transformer checkpoint with the rebuilt dataset."
        ) from exc
    return model


def _build_prev_context_arrays(
    *,
    frames: list[int],
    frame_buckets: dict[int, list[dict[str, str]]],
    manifest: dict[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prev_context_size = int(manifest.get("prev_context_size", 2))
    future_context_size = int(manifest.get("future_context_size", 0))
    max_slots = int(manifest["max_slots"])
    field_to_id = {
        field: {value: idx for idx, value in enumerate(values)}
        for field, values in manifest["vocabularies"].items()
    }
    prev_labels = {
        field: np.full((len(frames), prev_context_size, max_slots), PAD_VALUE, dtype=np.int64)
        for field in manifest["label_fields"]
    }
    prev_slot_mask = np.zeros((len(frames), prev_context_size, max_slots), dtype=np.float32)
    prev_frame_delta = np.zeros((len(frames), prev_context_size), dtype=np.float32)
    future_frame_delta = np.zeros((len(frames), future_context_size), dtype=np.float32)
    future_frame_mask = np.zeros((len(frames), future_context_size), dtype=np.float32)
    cut_flag = np.zeros((len(frames),), dtype=np.float32)

    for sample_idx, frame in enumerate(frames):
        rows = frame_buckets[int(frame)]
        cut_flag[sample_idx] = 1.0 if len(rows) > 1 else 0.0

        previous_frames = frames[:sample_idx][-prev_context_size:]
        future_frames = frames[sample_idx + 1 : sample_idx + 1 + future_context_size]
        prev_start = prev_context_size - len(previous_frames)
        for prev_offset, prev_frame in enumerate(previous_frames, start=prev_start):
            prev_rows = frame_buckets[int(prev_frame)]
            prev_frame_delta[sample_idx, prev_offset] = float(int(frame) - int(prev_frame))
            prev_slot_mask[sample_idx, prev_offset, : len(prev_rows)] = 1.0
            for slot_idx, row in enumerate(prev_rows):
                for field in manifest["label_fields"]:
                    token = str(row.get(field, ""))
                    token_id = field_to_id[field].get(token)
                    if token_id is None:
                        raise KeyError(f"Unknown previous-context token {token!r} for field {field}")
                    prev_labels[field][sample_idx, prev_offset, slot_idx] = int(token_id)
        for future_offset, future_frame in enumerate(future_frames):
            future_frame_delta[sample_idx, future_offset] = float(int(future_frame) - int(frame))
            future_frame_mask[sample_idx, future_offset] = 1.0
    return prev_labels, prev_slot_mask, prev_frame_delta, future_frame_delta, future_frame_mask, cut_flag


def _predict_song(
    *,
    model: DiscreteShotPlanTransformer,
    checkpoint_manifest: dict[str, Any],
    norm_stats: dict[str, np.ndarray],
    feature_path: Path,
    frame_buckets: dict[int, list[dict[str, str]]],
    sample_fps: float,
    timeline_fps: float,
    window_radius: int,
    window_stride: int,
    batch_size: int,
    device: torch.device,
    slot_count_source: str,
    prev_context_source: str,
) -> list[dict[str, object]]:
    arrays = np.load(feature_path, allow_pickle=False)
    audio = np.asarray(arrays["audio_features"], dtype=np.float32)
    dance = np.asarray(arrays["dance_features"], dtype=np.float32)
    dance_feature_set = str(checkpoint_manifest.get("dance_feature_set", "full"))
    if dance_feature_set == "compact":
        feature_manifest = load_feature_manifest(feature_path.parent)
        dance, _dance_names, _dance_meta = compact_dance_from_full(
            dance,
            list(feature_manifest.get("dance_feature_names", [])),
        )
    elif dance_feature_set == "compact_root_head_rot6d":
        feature_manifest = load_feature_manifest(feature_path.parent)
        dance, _dance_names, _dance_meta = compact_dance_root_head_rot6d_from_full(
            dance,
            list(feature_manifest.get("dance_feature_names", [])),
        )
    elif dance_feature_set == "compact_root_head_wrist_rel_rot6d_vectors":
        feature_manifest = load_feature_manifest(feature_path.parent)
        dance, _dance_names, _dance_meta = compact_dance_roll_signals_from_full(
            dance,
            list(feature_manifest.get("dance_feature_names", [])),
        )
    source_indices = _downsample_source_indices(
        int(dance.shape[0]) if bool(checkpoint_manifest.get("dance_only", False)) else int(audio.shape[0]),
        sample_fps=sample_fps,
        timeline_fps=timeline_fps,
    )
    audio_5fps = (
        np.zeros((int(source_indices.shape[0]), 0), dtype=np.float32)
        if bool(checkpoint_manifest.get("dance_only", False))
        else audio[source_indices]
    )
    dance_5fps = dance[source_indices]

    audio_5fps = (audio_5fps - norm_stats["audio_mean"].reshape(1, -1)) / norm_stats["audio_std"].reshape(1, -1)
    dance_5fps = (dance_5fps - norm_stats["dance_mean"].reshape(1, -1)) / norm_stats["dance_std"].reshape(1, -1)

    frames = sorted(frame_buckets)
    windows_audio = []
    windows_dance = []
    for frame in frames:
        center = min(max(int(frame), 0), int(audio_5fps.shape[0]) - 1)
        indices = _window_indices(
            center,
            length=int(audio_5fps.shape[0]),
            radius=window_radius,
            stride=window_stride,
        )
        windows_audio.append(audio_5fps[indices])
        windows_dance.append(dance_5fps[indices])

    vocabularies = checkpoint_manifest["vocabularies"]
    label_fields = list(checkpoint_manifest["label_fields"])
    output_keyframes: list[dict[str, object]] = []

    model.eval()
    if prev_context_source == "predicted":
        prev_context_size = int(checkpoint_manifest.get("prev_context_size", 2))
        max_slots = int(checkpoint_manifest["max_slots"])
        field_to_id = {
            field: {value: idx for idx, value in enumerate(values)}
            for field, values in vocabularies.items()
        }
        predicted_buckets: dict[int, list[dict[str, str]]] = {}
        predicted_frames: list[int] = []

        with torch.no_grad():
            for sample_idx, frame in enumerate(frames):
                prev_labels_np = {
                    field: np.full((1, prev_context_size, max_slots), PAD_VALUE, dtype=np.int64)
                    for field in label_fields
                }
                prev_slot_mask_np = np.zeros((1, prev_context_size, max_slots), dtype=np.float32)
                prev_frame_delta_np = np.zeros((1, prev_context_size), dtype=np.float32)
                future_context_size = int(checkpoint_manifest.get("future_context_size", 0))
                future_frame_delta_np = np.zeros((1, future_context_size), dtype=np.float32)
                future_frame_mask_np = np.zeros((1, future_context_size), dtype=np.float32)
                cut_flag_np = np.asarray([1.0 if len(frame_buckets[int(frame)]) > 1 else 0.0], dtype=np.float32)

                previous_frames = predicted_frames[-prev_context_size:]
                prev_start = prev_context_size - len(previous_frames)
                for prev_offset, prev_frame in enumerate(previous_frames, start=prev_start):
                    prev_rows = predicted_buckets[int(prev_frame)]
                    prev_frame_delta_np[0, prev_offset] = float(int(frame) - int(prev_frame))
                    prev_slot_mask_np[0, prev_offset, : len(prev_rows)] = 1.0
                    for slot_idx, row in enumerate(prev_rows):
                        for field in label_fields:
                            token = str(row.get(field, "none" if field == "roll" else ""))
                            token_id = field_to_id[field].get(token)
                            if token_id is None:
                                raise KeyError(f"Unknown previous-context token {token!r} for field {field}")
                            prev_labels_np[field][0, prev_offset, slot_idx] = int(token_id)
                future_frames = frames[sample_idx + 1 : sample_idx + 1 + future_context_size]
                for future_offset, future_frame in enumerate(future_frames):
                    future_frame_delta_np[0, future_offset] = float(int(future_frame) - int(frame))
                    future_frame_mask_np[0, future_offset] = 1.0

                audio_batch = torch.from_numpy(np.asarray(windows_audio[sample_idx : sample_idx + 1], dtype=np.float32)).to(device)
                dance_batch = torch.from_numpy(np.asarray(windows_dance[sample_idx : sample_idx + 1], dtype=np.float32)).to(device)
                prev_labels = {
                    field: torch.from_numpy(values).to(device)
                    for field, values in prev_labels_np.items()
                }
                outputs = model(
                    audio_batch,
                    dance_batch,
                    prev_labels=prev_labels,
                    prev_slot_mask=torch.from_numpy(prev_slot_mask_np).to(device),
                    prev_frame_delta=torch.from_numpy(prev_frame_delta_np).to(device),
                    future_frame_delta=torch.from_numpy(future_frame_delta_np).to(device),
                    future_frame_mask=torch.from_numpy(future_frame_mask_np).to(device),
                    cut_flag=torch.from_numpy(cut_flag_np).to(device),
                )
                predicted_count = int(outputs["count_logits"].argmax(dim=-1).cpu().numpy()[0]) + 1
                if slot_count_source == "gt":
                    slot_count = len(frame_buckets[int(frame)])
                else:
                    slot_count = predicted_count
                slot_count = max(1, min(slot_count, max_slots))

                current_rows: list[dict[str, str]] = []
                for slot_idx in range(slot_count):
                    row: dict[str, object] = {"frame": int(frame)}
                    context_row: dict[str, str] = {}
                    for field in label_fields:
                        if field in label_fields:
                            token_id = int(outputs[f"{field}_logits"].argmax(dim=-1).cpu().numpy()[0, slot_idx])
                            token = str(vocabularies[field][token_id])
                            context_row[field] = token
                        else:
                            token = "none" if field == "roll" else frame_buckets[int(frame)][0].get(field, "")
                            context_row[field] = str(token)
                        if field in PLAN_FIELDS:
                            row[field] = token
                    for field in PLAN_FIELDS:
                        row.setdefault(field, _default_token_for_field(field))
                    row["transition"] = "linear"
                    output_keyframes.append(row)
                    current_rows.append(context_row)

                predicted_buckets[int(frame)] = current_rows
                predicted_frames.append(int(frame))

        return output_keyframes

    (
        prev_labels_np,
        prev_slot_mask_np,
        prev_frame_delta_np,
        future_frame_delta_np,
        future_frame_mask_np,
        cut_flag_np,
    ) = _build_prev_context_arrays(
        frames=frames,
        frame_buckets=frame_buckets,
        manifest=checkpoint_manifest,
    )

    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            end = min(start + batch_size, len(frames))
            audio_batch = torch.from_numpy(np.asarray(windows_audio[start:end], dtype=np.float32)).to(device)
            dance_batch = torch.from_numpy(np.asarray(windows_dance[start:end], dtype=np.float32)).to(device)
            prev_labels = {
                field: torch.from_numpy(values[start:end]).to(device)
                for field, values in prev_labels_np.items()
            }
            outputs = model(
                audio_batch,
                dance_batch,
                prev_labels=prev_labels,
                prev_slot_mask=torch.from_numpy(prev_slot_mask_np[start:end]).to(device),
                prev_frame_delta=torch.from_numpy(prev_frame_delta_np[start:end]).to(device),
                future_frame_delta=torch.from_numpy(future_frame_delta_np[start:end]).to(device),
                future_frame_mask=torch.from_numpy(future_frame_mask_np[start:end]).to(device),
                cut_flag=torch.from_numpy(cut_flag_np[start:end]).to(device),
            )
            predicted_counts = outputs["count_logits"].argmax(dim=-1).cpu().numpy() + 1

            field_predictions: dict[str, np.ndarray] = {}
            for field in label_fields:
                field_predictions[field] = outputs[f"{field}_logits"].argmax(dim=-1).cpu().numpy()

            for local_idx, frame in enumerate(frames[start:end]):
                if slot_count_source == "gt":
                    slot_count = len(frame_buckets[int(frame)])
                else:
                    slot_count = int(predicted_counts[local_idx])
                slot_count = max(1, min(slot_count, int(checkpoint_manifest["max_slots"])))

                for slot_idx in range(slot_count):
                    row: dict[str, object] = {"frame": int(frame)}
                    for field in PLAN_FIELDS:
                        if field in field_predictions:
                            token_id = int(field_predictions[field][local_idx, slot_idx])
                            row[field] = str(vocabularies[field][token_id])
                        else:
                            # Old or experimental checkpoints may not contain a newer field.
                            row[field] = "none" if field == "roll" else frame_buckets[int(frame)][0].get(field, "")
                    row["transition"] = "linear"
                    output_keyframes.append(row)

    return output_keyframes


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_path = args.checkpoint.resolve()
    feature_root = args.feature_root.resolve()
    label_root = args.label_root.resolve()
    output_root = args.output_root.resolve()
    device = _device_from_arg(str(args.device))

    checkpoint = torch.load(checkpoint_path, map_location=device)
    manifest = checkpoint["manifest"]
    run_dir = checkpoint_path.parent
    norm_path = run_dir / "normalization_stats.npz"
    if not norm_path.exists():
        raise FileNotFoundError(f"normalization stats not found next to checkpoint: {norm_path}")
    norm_stats = load_normalization_stats(norm_path)

    model = _build_model_from_checkpoint(checkpoint).to(device)
    window_radius = int(args.window_radius)
    if window_radius < 0:
        window_radius = int(manifest.get("window_radius", (int(manifest["window_size"]) - 1) // 2))
    window_stride = int(args.window_stride)
    if window_stride < 0:
        window_stride = int(manifest.get("window_stride", 1))

    label_name = str(args.label_name)
    indices = _parse_indices(str(args.indices), label_root, label_name)
    manifest_rows: list[dict[str, object]] = []
    for index in indices:
        song_id = f"c{index}"
        label_path = label_root / song_id / label_name
        feature_path = feature_root / f"{song_id}.npz"
        output_dir = output_root / song_id
        output_plan = output_dir / "camera_poses_inferred.json"
        row: dict[str, object] = {
            "index": index,
            "status": "",
            "label_path": str(label_path),
            "feature_path": str(feature_path),
            "output": str(output_plan),
        }
        if not label_path.exists():
            row["status"] = "missing-label"
            manifest_rows.append(row)
            continue
        if not feature_path.exists():
            row["status"] = "missing-feature"
            manifest_rows.append(row)
            continue

        frame_buckets = _load_gt_frame_buckets(label_path, list(manifest["label_fields"]))
        keyframes = _predict_song(
            model=model,
            checkpoint_manifest=manifest,
            norm_stats=norm_stats,
            feature_path=feature_path,
            frame_buckets=frame_buckets,
            sample_fps=float(args.sample_fps),
            timeline_fps=float(args.timeline_fps),
            window_radius=window_radius,
            window_stride=window_stride,
            batch_size=int(args.batch_size),
            device=device,
            slot_count_source=str(args.slot_count_source),
            prev_context_source=str(args.prev_context_source),
        )

        payload = {
            "sample_fps": float(args.sample_fps),
            "model_checkpoint": str(checkpoint_path),
            "keyframe_timing": "gt",
            "timing_label_root": str(label_root),
            "timing_label_name": label_name,
            "slot_count_source": str(args.slot_count_source),
            "prev_context_source": str(args.prev_context_source),
            "keyframes": keyframes,
        }
        _write_json(output_plan, payload)
        row["status"] = "ok"
        row["frame_bucket_count"] = len(frame_buckets)
        row["keyframe_count"] = len(keyframes)
        manifest_rows.append(row)
        print(f"[ok] {song_id}: frames={len(frame_buckets)} keys={len(keyframes)}")

    _write_json(output_root / "manifest.json", manifest_rows)
    print(f"output root: {output_root}")


if __name__ == "__main__":
    main()
