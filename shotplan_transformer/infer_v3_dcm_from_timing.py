from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
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
from camera_autoshot.render_server_dcm_plans import (
    _load_render_variants_module,
    load_dcm_rows,
    resolve_render_assets,
)
from camera_autoshot.reverse_v3.solver_v3 import CameraSolveConfigV3, solve_camera_timeline_v3
from camera_autoshot.reverse_v3.spec_v3 import ShotKeyframeRequestV3
from camera_autoshot.reverse_v3.close_dense_schema import (
    EXPORTED_DISTANCE_VALUES,
    apply_close_dense_distance_schema,
    internalize_close_dense_row,
    uses_close_dense_distance_schema,
)
from camera_autoshot.vmd import write_camera_vmd

from .dataset import load_normalization_stats
from .infer_to_plans import _build_model_from_checkpoint, _device_from_arg, _window_indices
from .infer_v3_from_timing import _prepare_features


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    WORKSPACE_ROOT
    / "shotplan_transformer"
    / "results"
    / "training_runs"
    / "shotplan_transformer_20260507_011629_v3_30fps_middle_dcm_hist8_small"
)
DEFAULT_FEATURE_ROOT = (
    WORKSPACE_ROOT
    / "music_processing"
    / "results"
    / "01_datasets"
    / "cut_keyframe_multimodal_dataset_dcm"
    / "songs"
)
DEFAULT_DCM_MANIFEST = (
    WORKSPACE_ROOT
    / "music_processing"
    / "results"
    / "01_datasets"
    / "cut_keyframe_multimodal_dataset_dcm"
    / "manifest.json"
)
DEFAULT_TIMING_ROOT = (
    WORKSPACE_ROOT
    / "camera_autoshot"
    / "results"
    / "shotplan_transformer_mlp_dropout015_cutdupw100_dcm_clip_cutplusrandom_p030_seed42_20cuts_firsthalf_corrected"
)
DEFAULT_TARGETS_ROOT = WORKSPACE_ROOT / "camera_autoshot" / "reverses"
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "camera_autoshot"
    / "results"
    / "shotplan_transformer_20260507_011629_v3_30fps_middle_dcm_hist8_small_dcm_clip_timing"
)
PAD_VALUE = -100


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_indices(raw: str) -> list[int]:
    values: set[int] = set()
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        values.add(int(item[1:] if item.startswith("c") else item))
    return sorted(values)


def _clip_rows_by_source_index(dcm_manifest: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dcm_manifest.get("songs", []):
        if not isinstance(row, dict):
            continue
        grouped[int(row["index"])].append(row)
    for rows in grouped.values():
        rows.sort(
            key=lambda item: (
                str(item.get("official_split", "")),
                int(item.get("segment_index") or 0),
                int(item.get("source_start_frame") or 0),
            )
        )
    return grouped


def _select_dcm_clips_for_index(rows: list[dict[str, Any]], *, split: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if str(row.get("official_split", "")).lower() == split.lower()]
    return selected if selected else rows[:1]


def _timing_rows_to_source_frames(
    rows: list[dict[str, object]],
    *,
    timing_sample_fps: float,
    timeline_fps: float,
) -> tuple[list[int], list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict) or "frame" not in row:
            continue
        grouped[int(row["frame"])].append(row)

    scale = float(timeline_fps) / max(float(timing_sample_fps), 1e-6)
    frames: list[int] = []
    converted_rows: list[dict[str, object]] = []
    cut_pairs: list[dict[str, object]] = []
    used: set[int] = set()
    for timing_frame in sorted(grouped):
        source_frames_for_bucket: list[int] = []
        for duplicate_offset, _row in enumerate(grouped[timing_frame]):
            source_frame = int(round(float(timing_frame) * scale)) + duplicate_offset
            while source_frame in used:
                source_frame += 1
            used.add(source_frame)
            frames.append(source_frame)
            source_frames_for_bucket.append(source_frame)
            converted_rows.append(
                {
                    "timing_frame": int(timing_frame),
                    "source_frame": int(source_frame),
                    "duplicate_offset": int(duplicate_offset),
                    "duplicate_count": int(len(grouped[timing_frame])),
                }
            )
        if len(source_frames_for_bucket) >= 2:
            cut_pairs.append(
                {
                    "timing_frame": int(timing_frame),
                    "source_frames": source_frames_for_bucket[:2],
                    "duplicate_count": int(len(source_frames_for_bucket)),
                }
            )
    return frames, converted_rows, cut_pairs


def _predict_for_clip(
    *,
    model: torch.nn.Module,
    manifest: dict[str, Any],
    norm_stats: dict[str, np.ndarray],
    feature_path: Path,
    full_frames: list[int],
    source_start_frame: int,
    source_end_frame: int,
    device: torch.device,
) -> list[dict[str, object]]:
    audio, dance = _prepare_features(feature_path, manifest=manifest, norm_stats=norm_stats)
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

    selected_full_frames = [
        int(frame)
        for frame in sorted(set(full_frames))
        if int(source_start_frame) <= int(frame) <= int(source_end_frame)
    ]
    if int(source_start_frame) not in selected_full_frames:
        selected_full_frames.insert(0, int(source_start_frame))
    if int(source_end_frame) not in selected_full_frames:
        selected_full_frames.append(int(source_end_frame))
    selected_full_frames = sorted(set(selected_full_frames))

    predicted_rows_by_frame: dict[int, list[dict[str, str]]] = {}
    predicted_frames: list[int] = []
    output_rows: list[dict[str, object]] = []

    model.eval()
    with torch.no_grad():
        for full_frame in selected_full_frames:
            local_frame = int(full_frame) - int(source_start_frame)
            center = min(max(int(local_frame), 0), int(audio.shape[0]) - 1)
            indices = _window_indices(
                center,
                length=int(audio.shape[0]),
                radius=window_radius,
                stride=window_stride,
            )
            audio_batch = torch.from_numpy(audio[indices][None, :, :]).to(device)
            dance_batch = torch.from_numpy(dance[indices][None, :, :]).to(device)

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
                prev_frame_delta_np[0, prev_offset] = float(int(full_frame) - int(prev_frame))
                prev_slot_mask_np[0, prev_offset, : len(previous_rows)] = 1.0
                for slot_idx, row in enumerate(previous_rows[:max_slots]):
                    for field in label_fields:
                        token = str(row[field])
                        prev_labels_np[field][0, prev_offset, slot_idx] = int(field_to_id[field][token])

            future_context_size = int(manifest.get("future_context_size", 0))
            future_frame_delta_np = np.zeros((1, future_context_size), dtype=np.float32)
            future_frame_mask_np = np.zeros((1, future_context_size), dtype=np.float32)
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
                cut_flag=torch.zeros((1,), dtype=torch.float32, device=device),
            )

            context_row: dict[str, str] = {}
            output_row: dict[str, object] = {"frame": int(full_frame)}
            for field in label_fields:
                token_id = int(outputs[f"{field}_logits"].argmax(dim=-1).cpu().numpy()[0, 0])
                token = str(vocabularies[field][token_id])
                context_row[field] = token
                output_row[field] = token
            output_row["transition"] = "linear"
            output_rows.append(output_row)
            predicted_rows_by_frame[int(full_frame)] = [context_row]
            predicted_frames.append(int(full_frame))

    return output_rows


def _write_vmd_from_plan(plan_path: Path, targets_path: Path, output_vmd: Path, debug_path: Path) -> None:
    payload = _load_json(plan_path)
    keyframe_rows = [row for row in payload.get("keyframes", []) if isinstance(row, dict)]
    if uses_close_dense_distance_schema(payload):
        apply_close_dense_distance_schema()
        keyframe_rows = [internalize_close_dense_row(row) for row in keyframe_rows]
    requests = [ShotKeyframeRequestV3.from_dict(row) for row in keyframe_rows]
    provider = JsonFrameTargetProvider(targets_path)
    solved = solve_camera_timeline_v3(provider, requests, CameraSolveConfigV3())
    write_camera_vmd(solved, output_vmd, model_name="V3 DCM Shotplan")
    _write_json(debug_path, [asdict(key) for key in solved])


def _close_dense_distance_schema_for_manifest(manifest: dict[str, Any]) -> dict[str, object] | None:
    vocab = manifest.get("vocabularies", {})
    distance_values = vocab.get("distance", []) if isinstance(vocab, dict) else []
    if "very_close" not in {str(value) for value in distance_values}:
        return None
    return {
        "distance_values": list(EXPORTED_DISTANCE_VALUES),
        "notes": (
            "Reverse V3.1 close-dense distance labels. Exported labels are "
            "mapped to legacy internal V3 enum slots only while solving VMD."
        ),
    }


def _cut_change_stats(
    *,
    keyframes: list[dict[str, object]],
    cut_pairs: list[dict[str, object]],
    source_start_frame: int,
    source_end_frame: int,
    fields: list[str],
) -> dict[str, object]:
    by_frame = {int(row["frame"]): row for row in keyframes}
    total = 0
    changed = 0
    per_field = {field: 0 for field in fields}
    skipped = 0
    examples: list[dict[str, object]] = []
    for pair in cut_pairs:
        frames = [int(frame) for frame in pair["source_frames"][:2]]
        if not all(int(source_start_frame) <= frame <= int(source_end_frame) for frame in frames):
            skipped += 1
            continue
        left = by_frame.get(frames[0])
        right = by_frame.get(frames[1])
        if left is None or right is None:
            skipped += 1
            continue
        total += 1
        field_changes = {field: str(left.get(field)) != str(right.get(field)) for field in fields}
        if any(field_changes.values()):
            changed += 1
        for field, is_changed in field_changes.items():
            if is_changed:
                per_field[field] += 1
        if len(examples) < 20:
            examples.append(
                {
                    "timing_frame": pair["timing_frame"],
                    "source_frames": frames,
                    "changed": any(field_changes.values()),
                    "field_changes": field_changes,
                    "left": {field: left.get(field) for field in fields},
                    "right": {field: right.get(field) for field in fields},
                }
            )
    return {
        "cut_pair_count": total,
        "changed_pair_count": changed,
        "changed_pair_rate": (changed / total) if total else 0.0,
        "skipped_cut_pair_count": skipped,
        "per_field_changed_count": per_field,
        "per_field_changed_rate": {field: (count / total if total else 0.0) for field, count in per_field.items()},
        "examples": examples,
    }


def _write_manifest_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "index",
        "clip_id",
        "status",
        "official_split",
        "source_start_frame",
        "source_end_frame",
        "plan",
        "vmd",
        "mp4",
        "cut_pair_count",
        "changed_pair_count",
        "changed_pair_rate",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _compute_partial_render_stats(
    batch,
    *,
    motion: Path,
    camera: Path,
    wav: Path,
    clip_start_frame: int,
    clip_end_frame: int,
) -> dict[str, object]:
    motion_stats = batch.read_vmd_stats(motion)
    camera_stats = batch.read_vmd_stats(camera)
    wav_duration_sec = batch.ffprobe_duration_sec(wav)
    wav_duration_float_sec = batch.ffprobe_duration_float_sec(wav)

    motion_end_frame = int(motion_stats["motion_max_frame"])
    camera_end_frame = int(camera_stats["camera_max_frame"])
    music_end_frame = max(0, int(math.floor(wav_duration_float_sec * batch.FPS)) - 1)

    bounded_start = max(0, int(clip_start_frame))
    bounded_end = min(int(clip_end_frame), motion_end_frame, camera_end_frame, music_end_frame)
    if bounded_end < bounded_start:
        raise ValueError(
            f"empty render segment: start={bounded_start} end={bounded_end} "
            f"(motion={motion_end_frame}, camera={camera_end_frame}, music={music_end_frame})"
        )

    render_duration_sec = (bounded_end + 1) / batch.FPS
    segment_duration_sec = (bounded_end - bounded_start + 1) / batch.FPS
    return {
        "wav_duration_sec": wav_duration_sec,
        "motion_end_frame": motion_end_frame,
        "camera_end_frame": camera_end_frame,
        "music_end_frame": music_end_frame,
        "segment_start_frame": bounded_start,
        "segment_end_frame": bounded_end,
        "segment_frame_count": bounded_end - bounded_start + 1,
        "segment_duration_sec": segment_duration_sec,
        "render_duration_sec": render_duration_sec,
    }


def _finalize_partial_render(
    *,
    batch,
    temp_render_path: Path,
    output_path: Path,
    render_stats: dict[str, object],
    render_width: int,
    render_height: int,
) -> None:
    bounded_start = int(render_stats["segment_start_frame"])
    render_duration_sec = float(render_stats["render_duration_sec"])
    segment_duration_sec = float(render_stats["segment_duration_sec"])

    if not temp_render_path.exists() or temp_render_path.stat().st_size <= 0:
        raise RuntimeError(f"render failed for {output_path.name}: missing temp output {temp_render_path.name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    try:
        needs_postprocess = (
            bounded_start > 0
            or abs(segment_duration_sec - render_duration_sec) > 1e-6
            or int(render_width) > 0
            or int(render_height) > 0
        )
        if needs_postprocess:
            vf_args: list[str] = []
            if int(render_width) > 0 or int(render_height) > 0:
                width = int(render_width) if int(render_width) > 0 else -2
                height = int(render_height) if int(render_height) > 0 else -2
                vf_args = ["-vf", f"scale={width}:{height}"]
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{bounded_start / batch.FPS:.6f}",
                    "-i",
                    str(temp_render_path),
                    "-t",
                    f"{segment_duration_sec:.6f}",
                    *vf_args,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "23",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    str(output_path),
                ],
                check=True,
            )
            temp_render_path.unlink()
        else:
            shutil.move(str(temp_render_path), str(output_path))
    finally:
        if temp_render_path.exists():
            temp_render_path.unlink()


def _render_partial_jobs_batch(
    *,
    batch,
    jobs: list[dict[str, object]],
    default_model_path: Path,
    viewer_session_size: int,
    render_width: int,
    render_height: int,
) -> None:
    if not jobs:
        return

    render_rows: list[dict[str, str]] = []
    for job_index, job in enumerate(jobs):
        output_path = Path(str(job["output_path"]))
        temp_render_path = batch.RELEASE_DIR / f"{output_path.stem}.render_{job_index:04d}.mp4"
        job["temp_render_path"] = str(temp_render_path)
        stats = job["render_stats"]
        render_rows.append(
            {
                "status": "planned",
                "index": str(job["index"]),
                "model": str(job["model_path"]),
                "motion": str(job["motion_path"]),
                "camera": str(job["vmd_path"]),
                "wav": str(job["wav_path"]),
                "output": str(temp_render_path),
                "export_duration_sec": f"{float(stats['render_duration_sec']):.6f}",
            }
        )

    batch.render_planned_rows(
        render_rows,
        default_model_path,
        viewer_session_size=int(viewer_session_size),
        fps=int(batch.FPS),
        width=None,
        height=None,
    )

    for job, render_row in zip(jobs, render_rows):
        row = job["manifest_row"]
        try:
            if render_row.get("status") != "ok":
                raise RuntimeError(f"viewer render failed for {Path(str(job['output_path'])).name}")
            _finalize_partial_render(
                batch=batch,
                temp_render_path=Path(str(job["temp_render_path"])),
                output_path=Path(str(job["output_path"])),
                render_stats=job["render_stats"],
                render_width=int(render_width),
                render_height=int(render_height),
            )
            render_stats = job["render_stats"]
            row["status"] = "ok"
            row["clip_start_frame"] = int(render_stats["segment_start_frame"])
            row["clip_end_frame"] = int(render_stats["segment_end_frame"])
            row["segment_frame_count"] = int(render_stats["segment_frame_count"])
            row["segment_duration_sec"] = float(render_stats["segment_duration_sec"])
            print(
                f"[ok] c{job['index']}/{job['clip_id']}: {Path(str(job['output_path'])).name} "
                f"cut_change={float(row.get('changed_pair_rate', 0.0))*100:.1f}%"
            )
        except Exception as exc:
            row["status"] = "failed-render"
            row["error"] = str(exc)
            print(f"[fail] c{job['index']}/{job['clip_id']}: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer V3 DCM split model on DCM clip timing and render MP4s.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--dcm-manifest", type=Path, default=DEFAULT_DCM_MANIFEST)
    parser.add_argument("--timing-root", type=Path, default=DEFAULT_TIMING_ROOT)
    parser.add_argument("--timing-name", default="camera_poses_timing.json")
    parser.add_argument("--targets-root", type=Path, default=WORKSPACE_ROOT / "camera_autoshot" / "reverses")
    parser.add_argument("--dcm-summary-csv", type=Path, default=WORKSPACE_ROOT / "saba_mmd" / "build" / "Release" / "camera_dcm_render_result.csv")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--indices", default="3,8,20,77")
    parser.add_argument("--clip-split", default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--render-width", type=int, default=0)
    parser.add_argument("--render-height", type=int, default=0)
    parser.add_argument(
        "--viewer-session-size",
        type=int,
        default=0,
        help="Number of planned renders per saba_viewer session. 0 means one session per model group.",
    )
    parser.add_argument("--force-render", action="store_true", help="Re-render MP4s even when they already exist.")
    parser.add_argument("--vmd-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = _device_from_arg(str(args.device))
    run_dir = args.run_dir.resolve()
    checkpoint_path = run_dir / str(args.checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    manifest = checkpoint["manifest"]
    label_fields = [str(field) for field in manifest["label_fields"]]
    model = _build_model_from_checkpoint(checkpoint).to(device)
    norm_stats = load_normalization_stats(run_dir / "normalization_stats.npz")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dcm_manifest = _load_json(args.dcm_manifest.resolve())
    clip_rows_by_index = _clip_rows_by_source_index(dcm_manifest)
    dcm_render_rows = load_dcm_rows(args.dcm_summary_csv.resolve())

    variants = _load_render_variants_module()
    batch = variants.load_batch_module()
    default_model_path = Path(batch.MODEL_PATH)
    mapping = variants.load_mapping(batch)
    latest_ok_rows = variants.load_latest_ok_rows()

    manifest_rows: list[dict[str, object]] = []
    render_jobs: list[dict[str, object]] = []
    for index in _parse_indices(str(args.indices)):
        clips = _select_dcm_clips_for_index(clip_rows_by_index.get(index, []), split=str(args.clip_split))
        if not clips:
            manifest_rows.append({"index": index, "status": "missing-dcm-clip"})
            continue
        timing_path = args.timing_root.resolve() / f"c{index}" / str(args.timing_name)
        targets_path = args.targets_root.resolve() / f"c{index}" / "real_targets.json"
        if not timing_path.exists():
            manifest_rows.append({"index": index, "status": "missing-timing", "timing": str(timing_path)})
            continue
        if not targets_path.exists():
            manifest_rows.append({"index": index, "status": "missing-targets", "targets": str(targets_path)})
            continue

        timing_payload = _load_json(timing_path)
        timing_rows = [row for row in timing_payload.get("keyframes", []) if isinstance(row, dict) and "frame" in row]
        source_frames, timing_debug, cut_pairs = _timing_rows_to_source_frames(
            timing_rows,
            timing_sample_fps=float(timing_payload.get("sample_fps", 5.0)),
            timeline_fps=30.0,
        )

        for clip in clips:
            clip_id = str(clip["id"])
            source_start = int(clip["source_start_frame"])
            source_end = int(clip["source_end_frame"])
            feature_path = args.feature_root.resolve() / f"{clip_id}.npz"
            item_dir = output_root / f"c{index}"
            item_dir.mkdir(parents=True, exist_ok=True)
            suffix = "" if len(clips) == 1 else f"_{clip_id}"
            stem = f"camera_poses_v3_inferred{suffix}"
            plan_path = item_dir / f"{stem}.json"
            timing_debug_path = item_dir / f"{stem}_timing_debug.json"
            vmd_path = item_dir / f"{stem}.vmd"
            debug_path = item_dir / f"{stem}_debug.json"
            mp4_path = item_dir / f"{stem}.mp4"

            row: dict[str, object] = {
                "index": index,
                "clip_id": clip_id,
                "status": "",
                "official_split": str(clip.get("official_split", "")),
                "source_start_frame": source_start,
                "source_end_frame": source_end,
                "feature": str(feature_path),
                "timing": str(timing_path),
                "plan": str(plan_path),
                "vmd": str(vmd_path),
                "mp4": str(mp4_path),
            }
            if not feature_path.exists():
                row["status"] = "missing-feature"
                manifest_rows.append(row)
                continue

            cut_stats_path = item_dir / f"{stem}_cut_change_stats.json"
            if (
                not bool(args.force_render)
                and mp4_path.exists()
                and mp4_path.stat().st_size > 0
                and plan_path.exists()
                and vmd_path.exists()
            ):
                row["status"] = "ok-existing"
                if cut_stats_path.exists():
                    try:
                        existing_cut_stats = _load_json(cut_stats_path)
                        row.update(
                            {
                                "cut_pair_count": int(existing_cut_stats.get("cut_pair_count", 0) or 0),
                                "changed_pair_count": int(existing_cut_stats.get("changed_pair_count", 0) or 0),
                                "changed_pair_rate": float(existing_cut_stats.get("changed_pair_rate", 0.0) or 0.0),
                                "cut_stats": str(cut_stats_path),
                            }
                        )
                    except Exception as exc:
                        row["cut_stats_error"] = str(exc)
                print(f"[skip] c{index}/{clip_id}: {mp4_path.name} already exists")
                manifest_rows.append(row)
                continue

            keyframes = _predict_for_clip(
                model=model,
                manifest=manifest,
                norm_stats=norm_stats,
                feature_path=feature_path,
                full_frames=source_frames,
                source_start_frame=source_start,
                source_end_frame=source_end,
                device=device,
            )
            plan_payload = {
                "schema": (
                    "camera_autoshot.reverse_v3.close_dense_no_wide"
                    if _close_dense_distance_schema_for_manifest(manifest) is not None
                    else "camera_autoshot.reverse_v3"
                ),
                "source_model": str(checkpoint_path),
                "clip_id": clip_id,
                "source_song_id": f"c{index}",
                "source_start_frame": source_start,
                "source_end_frame": source_end,
                "sample_fps": 30.0,
                "timing_source": str(timing_path),
                "keyframes": keyframes,
            }
            distance_schema = _close_dense_distance_schema_for_manifest(manifest)
            if distance_schema is not None:
                plan_payload["distance_schema"] = distance_schema
            _write_json(plan_path, plan_payload)
            _write_json(timing_debug_path, {"converted_timing": timing_debug, "cut_pairs": cut_pairs})
            _write_vmd_from_plan(plan_path, targets_path, vmd_path, debug_path)

            cut_stats = _cut_change_stats(
                keyframes=keyframes,
                cut_pairs=cut_pairs,
                source_start_frame=source_start,
                source_end_frame=source_end,
                fields=label_fields,
            )
            _write_json(cut_stats_path, cut_stats)
            row.update(
                {
                    "cut_pair_count": int(cut_stats["cut_pair_count"]),
                    "changed_pair_count": int(cut_stats["changed_pair_count"]),
                    "changed_pair_rate": float(cut_stats["changed_pair_rate"]),
                    "cut_stats": str(cut_stats_path),
                }
            )

            if args.vmd_only:
                row["status"] = "ok-vmd-only"
                print(f"[ok] c{index}/{clip_id}: {vmd_path.name} (vmd-only)")
                manifest_rows.append(row)
                continue

            matched_rows = [
                candidate
                for candidate in dcm_render_rows.get(index, [])
                if int(candidate["segment_start_frame"]) == source_start
                and int(candidate["segment_end_frame"]) == source_end
            ]
            if not matched_rows:
                matched_rows = [
                    {
                        "segment_start_frame": source_start,
                        "segment_end_frame": source_end,
                        "motion": "",
                        "wav": "",
                        "output": "",
                    }
                ]
            try:
                model_path, motion_path, wav_path = resolve_render_assets(
                    variants=variants,
                    batch=batch,
                    mapping=mapping,
                    latest_ok_rows=latest_ok_rows,
                    index=index,
                    matched_rows=matched_rows,
                    override_model=None,
                    default_model_path=default_model_path,
                )
                batch.MODEL_PATH = model_path
                render_stats = _compute_partial_render_stats(
                    batch=batch,
                    motion=motion_path,
                    camera=vmd_path,
                    wav=wav_path,
                    clip_start_frame=source_start,
                    clip_end_frame=source_end,
                )
                row["motion"] = str(motion_path)
                row["wav"] = str(wav_path)
                row["model"] = str(model_path)
                row["clip_start_frame"] = int(render_stats["segment_start_frame"])
                row["clip_end_frame"] = int(render_stats["segment_end_frame"])
                row["segment_frame_count"] = int(render_stats["segment_frame_count"])
                row["segment_duration_sec"] = float(render_stats["segment_duration_sec"])
                if not bool(args.force_render) and mp4_path.exists() and mp4_path.stat().st_size > 0:
                    row["status"] = "ok-existing"
                    print(f"[skip] c{index}/{clip_id}: {mp4_path.name} already exists")
                else:
                    row["status"] = "planned-render"
                    render_jobs.append(
                        {
                            "manifest_row": row,
                            "index": index,
                            "clip_id": clip_id,
                            "model_path": model_path,
                            "motion_path": motion_path,
                            "wav_path": wav_path,
                            "vmd_path": vmd_path,
                            "output_path": mp4_path,
                            "render_stats": render_stats,
                        }
                    )
            except Exception as exc:
                row["status"] = "failed-render"
                row["error"] = str(exc)
                print(f"[fail] c{index}/{clip_id}: {exc}")
            manifest_rows.append(row)

    _render_partial_jobs_batch(
        batch=batch,
        jobs=render_jobs,
        default_model_path=default_model_path,
        viewer_session_size=int(args.viewer_session_size),
        render_width=int(args.render_width),
        render_height=int(args.render_height),
    )

    _write_json(output_root / "manifest.json", manifest_rows)
    _write_manifest_csv(manifest_rows, output_root / "manifest.csv")

    total = sum(int(row.get("cut_pair_count", 0) or 0) for row in manifest_rows)
    changed = sum(int(row.get("changed_pair_count", 0) or 0) for row in manifest_rows)
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "output_root": str(output_root),
        "clip_count": len([row for row in manifest_rows if str(row.get("status", "")).startswith("ok")]),
        "cut_pair_count": total,
        "changed_pair_count": changed,
        "changed_pair_rate": (changed / total) if total else 0.0,
    }
    _write_json(output_root / "cut_change_summary.json", summary)
    print(f"output root: {output_root}")
    print(f"cut change: {changed}/{total} = {summary['changed_pair_rate']*100:.2f}%")


if __name__ == "__main__":
    main()
