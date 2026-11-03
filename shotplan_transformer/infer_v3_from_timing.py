from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera_autoshot.providers import JsonFrameTargetProvider
from camera_autoshot.reverse_v3.solver_v3 import CameraSolveConfigV3, solve_camera_timeline_v3
from camera_autoshot.reverse_v3.spec_v3 import ShotKeyframeRequestV3
from camera_autoshot.vmd import write_camera_vmd

from .compact_dance import (
    compact_dance_from_full,
    compact_dance_hand_motion_from_full,
    compact_dance_roll_signals_from_full,
    compact_dance_root_head_rot6d_from_full,
    load_feature_manifest,
)
from .dataset import load_normalization_stats
from .infer_to_plans import _build_model_from_checkpoint, _device_from_arg, _window_indices


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_ROOT = WORKSPACE_ROOT / "music_processing" / "results" / "01_datasets" / "cut_keyframe_multimodal_dataset" / "songs"
DEFAULT_TIMING_ROOT = (
    WORKSPACE_ROOT
    / "camera_autoshot"
    / "results"
    / "shotplan_transformer_mlp_dropout015_cutdupw100_dcm_film_demucs_no_rms_density40_budgeted_cutgap7_6_keygap5"
)
DEFAULT_TARGETS_ROOT = WORKSPACE_ROOT / "camera_autoshot" / "reverses"
DEFAULT_RUN_DIR = (
    WORKSPACE_ROOT
    / "shotplan_transformer"
    / "results"
    / "training_runs"
    / "shotplan_transformer_20260506_224809_v3_30fps_middle_hist8_small"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "camera_autoshot"
    / "results"
    / "shotplan_transformer_20260506_224809_v3_30fps_middle_hist8_small_base_mlp_timing"
)
PAD_VALUE = -100


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_indices(raw: str, timing_root: Path, timing_name: str) -> list[int]:
    if raw.strip():
        values: set[int] = set()
        for token in raw.split(","):
            item = token.strip()
            if not item:
                continue
            values.add(int(item[1:] if item.startswith("c") else item))
        return sorted(values)

    values: list[int] = []
    for path in sorted(timing_root.glob(f"c*/{timing_name}")):
        name = path.parent.name
        if name.startswith("c") and name[1:].isdigit():
            values.append(int(name[1:]))
    return sorted(set(values))


def _timing_rows_to_source_frames(
    rows: list[dict[str, object]],
    *,
    timing_sample_fps: float,
    model_sample_fps: float,
) -> tuple[list[int], list[dict[str, object]]]:
    """Convert timing rows into model-sample frame ids.

    Legacy 5fps shot plans can contain two shots in one bucket for a cut.
    The V3 30fps model has max_slots=1, so same-bucket shots are unfolded
    into consecutive 30fps frames while preserving chronological order.
    """

    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict) or "frame" not in row:
            continue
        grouped[int(row["frame"])].append(row)

    scale = float(model_sample_fps) / max(float(timing_sample_fps), 1e-6)
    frames: list[int] = []
    timing_debug: list[dict[str, object]] = []
    used: set[int] = set()
    for timing_frame in sorted(grouped):
        base = int(round(float(timing_frame) * scale))
        for duplicate_offset, _row in enumerate(grouped[timing_frame]):
            frame = base + duplicate_offset
            while frame in used:
                frame += 1
            used.add(frame)
            frames.append(frame)
            timing_debug.append(
                {
                    "timing_frame": int(timing_frame),
                    "source_frame": int(frame),
                    "duplicate_offset": int(duplicate_offset),
                    "duplicate_count": int(len(grouped[timing_frame])),
                }
            )
    return frames, timing_debug


def _prepare_features(
    feature_path: Path,
    *,
    manifest: dict[str, Any],
    norm_stats: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    arrays = np.load(feature_path, allow_pickle=False)
    audio = np.asarray(arrays["audio_features"], dtype=np.float32)
    dance = np.asarray(arrays["dance_features"], dtype=np.float32)

    dance_feature_set = str(manifest.get("dance_feature_set", "full"))
    if dance_feature_set == "compact":
        feature_manifest = load_feature_manifest(feature_path.parent)
        dance, _names, _meta = compact_dance_from_full(
            dance,
            list(feature_manifest.get("dance_feature_names", [])),
        )
    elif dance_feature_set == "compact_hand_motion":
        feature_manifest = load_feature_manifest(feature_path.parent)
        dance, _names, _meta = compact_dance_hand_motion_from_full(
            dance,
            list(feature_manifest.get("dance_feature_names", [])),
        )
    elif dance_feature_set == "compact_root_head_rot6d":
        feature_manifest = load_feature_manifest(feature_path.parent)
        dance, _names, _meta = compact_dance_root_head_rot6d_from_full(
            dance,
            list(feature_manifest.get("dance_feature_names", [])),
        )
    elif dance_feature_set == "compact_root_head_wrist_rel_rot6d_vectors":
        feature_manifest = load_feature_manifest(feature_path.parent)
        dance, _names, _meta = compact_dance_roll_signals_from_full(
            dance,
            list(feature_manifest.get("dance_feature_names", [])),
        )

    if bool(manifest.get("dance_only", False)):
        audio = np.zeros((int(dance.shape[0]), 0), dtype=np.float32)

    frame_count = min(int(audio.shape[0]) if audio.shape[1] > 0 else int(dance.shape[0]), int(dance.shape[0]))
    audio = audio[:frame_count]
    dance = dance[:frame_count]
    audio = (audio - norm_stats["audio_mean"].reshape(1, -1)) / norm_stats["audio_std"].reshape(1, -1)
    dance = (dance - norm_stats["dance_mean"].reshape(1, -1)) / norm_stats["dance_std"].reshape(1, -1)
    return audio.astype(np.float32), dance.astype(np.float32)


def _predict_v3_keyframes(
    *,
    model: torch.nn.Module,
    manifest: dict[str, Any],
    norm_stats: dict[str, np.ndarray],
    feature_path: Path,
    frames: list[int],
    batch_device: torch.device,
) -> list[dict[str, object]]:
    audio, dance = _prepare_features(feature_path, manifest=manifest, norm_stats=norm_stats)
    if not frames:
        return []

    label_fields = [str(field) for field in manifest["label_fields"]]
    vocabularies: dict[str, list[str]] = {
        str(field): [str(value) for value in values]
        for field, values in manifest["vocabularies"].items()
    }
    field_to_id = {
        field: {value: idx for idx, value in enumerate(values)}
        for field, values in vocabularies.items()
    }
    prev_context_size = int(manifest.get("prev_context_size", 0))
    max_slots = int(manifest["max_slots"])
    window_radius = int(manifest.get("window_radius", (int(manifest["window_size"]) - 1) // 2))
    window_stride = int(manifest.get("window_stride", 1))

    predicted_rows_by_frame: dict[int, list[dict[str, str]]] = {}
    predicted_frames: list[int] = []
    output_rows: list[dict[str, object]] = []

    model.eval()
    with torch.no_grad():
        for frame in sorted(frames):
            center = min(max(int(frame), 0), int(audio.shape[0]) - 1)
            indices = _window_indices(
                center,
                length=int(audio.shape[0]),
                radius=window_radius,
                stride=window_stride,
            )
            audio_batch = torch.from_numpy(audio[indices][None, :, :]).to(batch_device)
            dance_batch = torch.from_numpy(dance[indices][None, :, :]).to(batch_device)

            prev_labels_np = {
                field: np.full((1, prev_context_size, max_slots), PAD_VALUE, dtype=np.int64)
                for field in label_fields
            }
            prev_slot_mask_np = np.zeros((1, prev_context_size, max_slots), dtype=np.float32)
            prev_frame_delta_np = np.zeros((1, prev_context_size), dtype=np.float32)
            previous_frames = predicted_frames[-prev_context_size:]
            prev_start = prev_context_size - len(previous_frames)
            for prev_offset, prev_frame in enumerate(previous_frames, start=prev_start):
                previous_rows = predicted_rows_by_frame[int(prev_frame)]
                prev_frame_delta_np[0, prev_offset] = float(int(frame) - int(prev_frame))
                prev_slot_mask_np[0, prev_offset, : len(previous_rows)] = 1.0
                for slot_idx, row in enumerate(previous_rows[:max_slots]):
                    for field in label_fields:
                        token = str(row[field])
                        prev_labels_np[field][0, prev_offset, slot_idx] = int(field_to_id[field][token])

            future_context_size = int(manifest.get("future_context_size", 0))
            future_frame_delta_np = np.zeros((1, future_context_size), dtype=np.float32)
            future_frame_mask_np = np.zeros((1, future_context_size), dtype=np.float32)
            prev_labels = {
                field: torch.from_numpy(values).to(batch_device)
                for field, values in prev_labels_np.items()
            }
            outputs = model(
                audio_batch,
                dance_batch,
                prev_labels=prev_labels,
                prev_slot_mask=torch.from_numpy(prev_slot_mask_np).to(batch_device),
                prev_frame_delta=torch.from_numpy(prev_frame_delta_np).to(batch_device),
                future_frame_delta=torch.from_numpy(future_frame_delta_np).to(batch_device),
                future_frame_mask=torch.from_numpy(future_frame_mask_np).to(batch_device),
                cut_flag=torch.zeros((1,), dtype=torch.float32, device=batch_device),
            )

            context_row: dict[str, str] = {}
            output_row: dict[str, object] = {"frame": int(frame)}
            for field in label_fields:
                token_id = int(outputs[f"{field}_logits"].argmax(dim=-1).cpu().numpy()[0, 0])
                token = str(vocabularies[field][token_id])
                context_row[field] = token
                output_row[field] = token
            output_row["transition"] = "linear"
            output_rows.append(output_row)
            predicted_rows_by_frame[int(frame)] = [context_row]
            predicted_frames.append(int(frame))

    return output_rows


def _write_vmd_from_v3_json(
    *,
    plan_path: Path,
    targets_path: Path,
    output_vmd: Path,
    debug_path: Path,
    model_name: str,
) -> None:
    payload = _load_json(plan_path)
    requests = [ShotKeyframeRequestV3.from_dict(row) for row in payload.get("keyframes", [])]
    provider = JsonFrameTargetProvider(targets_path)
    solved = solve_camera_timeline_v3(provider, requests, CameraSolveConfigV3())
    write_camera_vmd(solved, output_vmd, model_name=model_name)
    _write_json(debug_path, [asdict(key) for key in solved])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer Reverse V3 shot plans from legacy timing JSONs.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--timing-root", type=Path, default=DEFAULT_TIMING_ROOT)
    parser.add_argument("--timing-name", default="camera_poses_timing.json")
    parser.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS_ROOT)
    parser.add_argument("--targets-name", default="real_targets.json")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--indices", default="3,8,20,77")
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir.resolve()
    checkpoint_path = run_dir / str(args.checkpoint_name)
    device = _device_from_arg(str(args.device))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    manifest = checkpoint["manifest"]
    model = _build_model_from_checkpoint(checkpoint).to(device)
    norm_stats = load_normalization_stats(run_dir / "normalization_stats.npz")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    indices = _parse_indices(str(args.indices), args.timing_root.resolve(), str(args.timing_name))
    model_sample_fps = float(manifest.get("sample_fps", 30.0))

    manifest_rows: list[dict[str, object]] = []
    for index in indices:
        song_id = f"c{index}"
        timing_path = args.timing_root.resolve() / song_id / str(args.timing_name)
        feature_path = args.feature_root.resolve() / f"{song_id}.npz"
        targets_path = args.targets_root.resolve() / song_id / str(args.targets_name)
        output_dir = output_root / song_id
        plan_path = output_dir / "camera_poses_v3_inferred.json"
        timing_v3_path = output_dir / "camera_poses_timing_v3_30fps.json"
        vmd_path = output_dir / "camera_poses_v3_inferred.vmd"
        debug_path = output_dir / "camera_poses_v3_inferred_debug.json"
        solved_debug_path = output_dir / "camera_poses_v3_inferred_reconstruction_debug.json"

        row: dict[str, object] = {
            "index": int(index),
            "status": "",
            "checkpoint": str(checkpoint_path),
            "timing": str(timing_path),
            "feature": str(feature_path),
            "targets": str(targets_path),
            "plan": str(plan_path),
            "vmd": str(vmd_path),
            "mp4": str(output_dir / "camera_poses_v3_inferred.mp4"),
        }
        if not timing_path.exists():
            row["status"] = "missing-timing"
            manifest_rows.append(row)
            continue
        if not feature_path.exists():
            row["status"] = "missing-feature"
            manifest_rows.append(row)
            continue
        if not targets_path.exists():
            row["status"] = "missing-targets"
            manifest_rows.append(row)
            continue

        timing_payload = _load_json(timing_path)
        timing_rows = [
            item for item in timing_payload.get("keyframes", [])
            if isinstance(item, dict) and "frame" in item
        ]
        timing_sample_fps = float(timing_payload.get("sample_fps", 5.0))
        frames, timing_debug = _timing_rows_to_source_frames(
            timing_rows,
            timing_sample_fps=timing_sample_fps,
            model_sample_fps=model_sample_fps,
        )
        keyframes = _predict_v3_keyframes(
            model=model,
            manifest=manifest,
            norm_stats=norm_stats,
            feature_path=feature_path,
            frames=frames,
            batch_device=device,
        )

        plan_payload = {
            "schema": "camera_autoshot.reverse_v3",
            "sample_fps": model_sample_fps,
            "model_checkpoint": str(checkpoint_path),
            "timing_source": str(timing_path),
            "timing_source_sample_fps": timing_sample_fps,
            "timing_conversion": "legacy buckets scaled to 30fps; duplicate buckets unfolded to adjacent frames",
            "prev_context_source": "predicted",
            "keyframes": keyframes,
        }
        _write_json(plan_path, plan_payload)
        _write_json(
            timing_v3_path,
            {
                "sample_fps": model_sample_fps,
                "source_timing": str(timing_path),
                "converted_frames": timing_debug,
            },
        )
        _write_vmd_from_v3_json(
            plan_path=plan_path,
            targets_path=targets_path,
            output_vmd=vmd_path,
            debug_path=solved_debug_path,
            model_name=f"V3 Transformer {song_id}",
        )
        _write_json(debug_path, {"timing": timing_debug, "keyframes": keyframes})

        row["status"] = "ok"
        row["timing_keyframe_count"] = len(timing_rows)
        row["converted_frame_count"] = len(frames)
        row["predicted_keyframe_count"] = len(keyframes)
        manifest_rows.append(row)
        print(f"[ok] {song_id}: timing={len(timing_rows)} converted={len(frames)} keys={len(keyframes)}")

    _write_json(output_root / "manifest.json", manifest_rows)
    print(f"output root: {output_root}")


if __name__ == "__main__":
    main()
