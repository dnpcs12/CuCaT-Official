from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from .cli import load_requests
from .plan_timeline import load_plan_payload
from .providers import JsonFrameTargetProvider
from .solver import CameraSolveConfig, solve_camera_timeline
from .vmd import (
    CameraVmdKey,
    dump_debug_json,
    evaluate_camera_vmd,
    read_camera_vmd,
    read_camera_vmd_stats,
    write_camera_vmd,
    write_camera_vmd_records,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def release_dir() -> Path:
    return repo_root() / "saba_mmd" / "build" / "Release"


def _ensure_release_import_path() -> Path:
    path = release_dir()
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
    return path


def _load_render_variants_module():
    _ensure_release_import_path()
    return importlib.import_module("render_camera_variants")


def _normalize_release_output_path(text: str) -> Path:
    candidate = Path((text or "").strip())
    if not str(candidate):
        return candidate
    local_candidate = release_dir() / candidate.name
    if local_candidate.exists():
        return local_candidate
    return candidate


def _parse_indices(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return sorted(set(values))


def _discover_indices(source_root: Path, plan_name: str) -> list[int]:
    values: list[int] = []
    for folder in sorted(source_root.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name
        if not (name.startswith("c") and name[1:].isdigit()):
            continue
        if (folder / plan_name).exists():
            values.append(int(name[1:]))
    return sorted(values)


def _dedupe_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for row in rows:
        key = (
            row.get("dcm_seq"),
            row.get("segment_start_frame"),
            row.get("segment_end_frame"),
            row.get("segment_frame_count"),
            row.get("camera"),
            row.get("output"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    deduped.sort(
        key=lambda item: (
            int(item.get("segment_start_frame", 0)),
            int(item.get("dcm_seq", 0)),
        )
    )
    return deduped


def _output_name(text: object) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    return Path(raw).name.lower()


def _normalize_task1_model_path(raw: str, fallback: Path) -> Path:
    candidate = Path((raw or "").strip())
    if str(candidate).strip() and candidate.exists():
        return candidate
    return fallback


def _reduce_segment_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = _dedupe_rows(rows)
    reduced: list[dict[str, object]] = []
    for row in rows:
        is_superset = False
        row_start = int(row["segment_start_frame"])
        row_end = int(row["segment_end_frame"])
        row_output = _output_name(row.get("output"))
        for other in rows:
            if row is other:
                continue
            other_output = _output_name(other.get("output"))
            if row_output != other_output:
                continue
            other_start = int(other["segment_start_frame"])
            other_end = int(other["segment_end_frame"])
            if row_start <= other_start and row_end >= other_end and (
                row_start < other_start or row_end > other_end
            ):
                is_superset = True
                break
        if not is_superset:
            reduced.append(row)

    seen: set[tuple[int, int, str]] = set()
    unique: list[dict[str, object]] = []
    for row in reduced:
        key = (
            int(row["segment_start_frame"]),
            int(row["segment_end_frame"]),
            _output_name(row.get("output")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)

    unique.sort(
        key=lambda item: (
            int(item["segment_start_frame"]),
            int(item["segment_end_frame"]),
            _output_name(item.get("output")),
        )
    )
    return unique


def _slice_camera_vmd_keys(
    keys: list[CameraVmdKey],
    start_frame: int,
    end_frame: int,
) -> list[CameraVmdKey]:
    if not keys:
        return []

    sliced = [key for key in keys if start_frame <= key.frame <= end_frame]

    if not sliced or sliced[0].frame > start_frame:
        evaluated = evaluate_camera_vmd(keys, float(start_frame))
        if evaluated is not None:
            interpolation = sliced[0].interpolation if sliced else keys[-1].interpolation
            sliced.insert(
                0,
                CameraVmdKey(
                    frame=start_frame,
                    distance=evaluated.distance,
                    interest=evaluated.interest,
                    rotate=evaluated.rotate,
                    view_angle_deg=int(round(evaluated.view_angle_deg)),
                    perspective=evaluated.perspective,
                    interpolation=interpolation,
                ),
            )

    if not sliced or sliced[-1].frame < end_frame:
        evaluated = evaluate_camera_vmd(keys, float(end_frame))
        if evaluated is not None:
            interpolation = sliced[-1].interpolation if sliced else keys[-1].interpolation
            sliced.append(
                CameraVmdKey(
                    frame=end_frame,
                    distance=evaluated.distance,
                    interest=evaluated.interest,
                    rotate=evaluated.rotate,
                    view_angle_deg=int(round(evaluated.view_angle_deg)),
                    perspective=evaluated.perspective,
                    interpolation=interpolation,
                )
            )

    deduped: list[CameraVmdKey] = []
    seen_frames: set[int] = set()
    for key in sorted(sliced, key=lambda item: item.frame):
        if key.frame in seen_frames:
            continue
        seen_frames.add(key.frame)
        deduped.append(key)
    return deduped


def _slice_debug_keys(keys, start_frame: int, end_frame: int):
    sliced = [key for key in keys if start_frame <= key.frame <= end_frame]
    previous = None
    for key in keys:
        if key.frame < start_frame:
            previous = key
            continue
        break

    if previous is not None and (not sliced or sliced[0].frame > start_frame):
        sliced.insert(0, replace(previous, frame=start_frame))

    if sliced and sliced[-1].frame < end_frame:
        sliced.append(replace(sliced[-1], frame=end_frame))
    elif not sliced and previous is not None:
        sliced.append(replace(previous, frame=end_frame))

    deduped = []
    seen_frames: set[int] = set()
    for key in sorted(sliced, key=lambda item: item.frame):
        if key.frame in seen_frames:
            continue
        seen_frames.add(key.frame)
        deduped.append(key)
    return deduped


def _extend_solved_keys_to_frame(keys, frame: int):
    if not keys:
        return keys

    target_frame = int(frame)
    max_key_frame = max(int(key.frame) for key in keys)
    if max_key_frame >= target_frame:
        return keys

    # Keep the camera alive through the full DCM clip. Without this hold key,
    # render_partial_clip bounds the MP4 by camera_max_frame and can cut the
    # output shorter than the original DCM evaluation segment.
    last_key = max(keys, key=lambda item: int(item.frame))
    return sorted([*keys, replace(last_key, frame=target_frame)], key=lambda item: int(item.frame))


def load_dcm_rows(summary_csv: Path) -> dict[int, list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = {}
    with summary_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("status") or "").strip().lower() != "ok":
                continue
            index_text = (row.get("index") or "").strip()
            if not index_text.isdigit():
                continue
            grouped.setdefault(int(index_text), []).append(
                {
                    "index": int(index_text),
                    "dcm_seq": int(row.get("dcm_seq") or 0),
                    "model": str(row.get("model") or ""),
                    "motion": str(row.get("motion") or ""),
                    "base_camera": str(row.get("base_camera") or ""),
                    "wav": str(row.get("wav") or ""),
                    "camera": str(row.get("camera") or ""),
                    "segment_start_frame": int(row.get("segment_start_frame") or 0),
                    "segment_end_frame": int(row.get("segment_end_frame") or 0),
                    "segment_frame_count": int(row.get("segment_frame_count") or 0),
                    "output": str(_normalize_release_output_path(row.get("output", ""))),
                }
            )
    return {index: _dedupe_rows(rows) for index, rows in grouped.items()}


def select_clip_window(
    index: int,
    payload: dict[str, object],
    dcm_rows: dict[int, list[dict[str, object]]],
) -> tuple[int, int, list[dict[str, object]]]:
    keyframes = list(payload.get("keyframes", []))
    timeline_frames = [int(item["timeline_frame"]) for item in keyframes]
    if not timeline_frames:
        raise ValueError(f"plan has no keyframes: c{index}")

    plan_min = min(timeline_frames)
    plan_max = max(timeline_frames)
    candidates = dcm_rows.get(index, [])
    overlaps = []
    for row in candidates:
        row_start = int(row["segment_start_frame"])
        row_end = int(row["segment_end_frame"])
        if any(row_start <= frame <= row_end for frame in timeline_frames):
            overlaps.append(row)

    overlaps = _reduce_segment_rows(overlaps)
    if overlaps:
        clip_start = min(int(row["segment_start_frame"]) for row in overlaps)
        clip_end = max(int(row["segment_end_frame"]) for row in overlaps)
        return clip_start, clip_end, overlaps

    return plan_min, plan_max, []


def resolve_render_assets(
    variants,
    batch,
    mapping,
    latest_ok_rows,
    index: int,
    matched_rows: list[dict[str, object]],
    override_model: Path | None,
    default_model_path: Path,
) -> tuple[Path, Path, Path]:
    latest_row = latest_ok_rows.get(index, {})
    normalized_model = override_model or _normalize_task1_model_path(
        str(latest_row.get("model", "")),
        default_model_path,
    )

    # Keep the model aligned with Task 1's latest-ok assets for the same index.
    _folder, base_model, base_motion, _base_camera, base_wav = variants.find_base_assets(
        batch,
        mapping,
        latest_ok_rows,
        index,
        normalized_model,
    )

    # c50 target dumps and labels are aligned to the 1p TDA motion. Some legacy
    # DCM segment rows point at a 5p motion for the same clip window, which makes
    # generated cameras aim at the wrong dancer during rendering.
    if index == 50 and "1p ver" in str(base_motion):
        for row in matched_rows:
            motion_text = str(row.get("motion") or "")
            if "5p ver" in motion_text:
                return base_model, base_motion, base_wav

    for row in matched_rows:
        motion_text = str(row.get("motion") or "").strip()
        wav_text = str(row.get("wav") or "").strip()
        if not motion_text or not wav_text:
            continue

        motion = Path(motion_text).resolve()
        wav = Path(wav_text).resolve()
        if motion.is_file() and wav.is_file():
            return base_model, motion, wav

    return base_model, base_motion, base_wav


def render_partial_clip(
    batch,
    motion: Path,
    camera: Path,
    wav: Path,
    output_path: Path,
    clip_start_frame: int,
    clip_end_frame: int,
    render_width: int = 0,
    render_height: int = 0,
    worker_id: str = "auto",
    worker_dir: Path | None = None,
    worker_lock_wait_sec: float = 600.0,
    worker_lock_stale_sec: float = 12 * 60 * 60,
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
    with batch.acquire_saba_worker(
        worker_id,
        worker_dir=worker_dir,
        wait_sec=worker_lock_wait_sec,
        stale_sec=worker_lock_stale_sec,
        lease_info={
            "script": "camera_autoshot.render_server_dcm_plans",
            "output": str(output_path),
        },
    ) as worker:
        temp_render_path = worker.worker_dir / f"{output_path.stem}.{os.getpid()}.render.mp4"

        if temp_render_path.exists():
            temp_render_path.unlink()
        if output_path.exists():
            output_path.unlink()

        batch.write_init_lua(
            motion,
            camera,
            wav,
            temp_render_path.name,
            render_duration_sec,
            work_dir=worker.worker_dir,
        )

        proc = None
        try:
            proc = subprocess.Popen([str(batch.VIEWER_EXE)], cwd=str(worker.worker_dir))
            worker.update(viewer_pid=proc.pid)
            ok = batch.wait_until_output_stable(proc, temp_render_path)
            batch.terminate_viewer_process(proc)
            batch.cleanup_export_frame_dirs(worker.worker_dir)

            if not ok or not temp_render_path.exists() or temp_render_path.stat().st_size <= 0:
                raise RuntimeError(f"render failed for {output_path.name}")

            output_path.parent.mkdir(parents=True, exist_ok=True)
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
            if proc is not None:
                batch.terminate_viewer_process(proc)
            batch.cleanup_export_frame_dirs(worker.worker_dir)
            worker.update(viewer_pid=0)
            if temp_render_path.exists():
                temp_render_path.unlink()

    return {
        "wav_duration_sec": wav_duration_sec,
        "motion_end_frame": motion_end_frame,
        "camera_end_frame": camera_end_frame,
        "music_end_frame": music_end_frame,
        "segment_start_frame": bounded_start,
        "segment_end_frame": bounded_end,
        "segment_frame_count": bounded_end - bounded_start + 1,
        "segment_duration_sec": segment_duration_sec,
    }


def write_manifest(rows: list[dict[str, object]], json_path: Path, csv_path: Path) -> None:
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = [
        "index",
        "part",
        "part_count",
        "status",
        "plan_src",
        "plan_dst",
        "targets",
        "vmd",
        "debug_json",
        "mp4",
        "sample_fps",
        "timeline_fps",
        "plan_timeline_start",
        "plan_timeline_end",
        "clip_start_frame",
        "clip_end_frame",
        "segment_frame_count",
        "matched_dcm_outputs",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def load_existing_manifest_rows(json_path: Path) -> list[dict[str, object]]:
    if not json_path.exists():
        return []
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    rows: list[dict[str, object]] = []
    for item in payload:
        if isinstance(item, dict):
            rows.append(dict(item))
    return rows


def _manifest_sort_key(row: dict[str, object]) -> tuple[int, int]:
    try:
        index = int(row.get("index", 0))
    except Exception:
        index = 0
    try:
        part = int(row.get("part", 0) or 0)
    except Exception:
        part = 0
    return index, part


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render server-side DCM partial shot-plan JSON files into camera VMD and clipped MP4 outputs."
        )
    )
    parser.add_argument("--source-root", required=True, help="Root containing cNN/<plan>.json folders.")
    parser.add_argument("--plan-name", required=True, help="Plan JSON filename inside each cNN folder.")
    parser.add_argument(
        "--indices",
        default="",
        help="Optional comma-separated indices. When omitted, all cNN folders containing the plan are used.",
    )
    parser.add_argument(
        "--targets-root",
        default=str(Path("camera_autoshot") / "reverses"),
        help="Root containing cNN/real_targets.json.",
    )
    parser.add_argument(
        "--dcm-summary-csv",
        default=str(Path("saba_mmd") / "build" / "Release" / "camera_dcm_render_result.csv"),
        help="CSV containing DCM segment_start_frame / segment_end_frame metadata.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory for copied plans, VMDs, debug JSON, MP4s, and manifest.",
    )
    parser.add_argument("--plan-sample-fps", type=float, default=5.0, help="Sample FPS used by the source plan.")
    parser.add_argument("--timeline-fps", type=float, default=30.0, help="Target timeline FPS.")
    parser.add_argument("--model-path", default="", help="Optional explicit model override.")
    parser.add_argument("--model-alias", default="", help="Optional model alias override.")
    parser.add_argument("--render-width", type=int, default=0, help="Optional final MP4 width after rendering.")
    parser.add_argument("--render-height", type=int, default=0, help="Optional final MP4 height after rendering.")
    parser.add_argument(
        "--worker-id",
        default="auto",
        help="Saba worker id to lease for MP4 rendering. Use auto to pick a free worker.",
    )
    parser.add_argument(
        "--worker-dir",
        default="",
        help="Optional explicit Saba worker directory. Overrides --worker-id when set.",
    )
    parser.add_argument(
        "--worker-lock-wait-sec",
        type=float,
        default=600.0,
        help="Seconds to wait for a free Saba worker before failing.",
    )
    parser.add_argument(
        "--worker-lock-stale-sec",
        type=float,
        default=12 * 60 * 60,
        help="Seconds after which an ownerless worker lock can be reclaimed.",
    )
    parser.add_argument(
        "--vmd-only",
        action="store_true",
        help="Write plan JSON, camera VMD, and debug JSON only; skip MP4 rendering.",
    )
    parser.add_argument(
        "--use-full-plan-window",
        action="store_true",
        help="Ignore DCM overlap windows and render the full expanded plan timeline.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = repo_root()
    source_root = Path(args.source_root).resolve()
    targets_root = (root / args.targets_root).resolve()
    dcm_summary_csv = (root / args.dcm_summary_csv).resolve()
    output_root = (root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not source_root.exists():
        raise FileNotFoundError(f"source root not found: {source_root}")
    if not dcm_summary_csv.exists():
        raise FileNotFoundError(f"DCM summary CSV not found: {dcm_summary_csv}")

    variants = _load_render_variants_module()
    batch = variants.load_batch_module()
    default_model_path = Path(batch.MODEL_PATH)
    mapping = variants.load_mapping(batch)
    latest_ok_rows = variants.load_latest_ok_rows()

    override_model: Path | None = None
    if args.model_path or args.model_alias:
        override_model = variants.resolve_model_choice(args.model_path, args.model_alias, batch.MODEL_PATH)
        if not override_model.exists():
            raise FileNotFoundError(f"model not found: {override_model}")

    if args.indices.strip():
        indices = _parse_indices(args.indices)
    else:
        indices = _discover_indices(source_root, args.plan_name)
    if not indices:
        raise ValueError("no input cNN folders found")

    dcm_rows = load_dcm_rows(dcm_summary_csv)
    manifest_path = output_root / "manifest.json"
    manifest_csv_path = output_root / "manifest.csv"
    previous_manifest_rows = load_existing_manifest_rows(manifest_path)
    manifest_rows: list[dict[str, object]] = []
    run_at = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        for index in indices:
            source_dir = source_root / f"c{index}"
            plan_src = source_dir / args.plan_name
            targets_path = targets_root / f"c{index}" / "real_targets.json"

            item_dir = output_root / f"c{index}"
            item_dir.mkdir(parents=True, exist_ok=True)
            plan_dst = item_dir / args.plan_name
            stem = Path(args.plan_name).stem
            vmd_path = item_dir / f"{stem}.vmd"
            debug_json_path = item_dir / f"{stem}_debug.json"
            mp4_path = item_dir / f"{stem}.mp4"

            row: dict[str, object] = {
                "run_at": run_at,
                "index": index,
                "part": "",
                "part_count": 1,
                "status": "",
                "plan_src": str(plan_src),
                "plan_dst": str(plan_dst),
                "targets": str(targets_path),
                "vmd": str(vmd_path),
                "debug_json": str(debug_json_path),
                "mp4": str(mp4_path),
                "sample_fps": float(args.plan_sample_fps),
                "timeline_fps": float(args.timeline_fps),
                "plan_timeline_start": "",
                "plan_timeline_end": "",
                "clip_start_frame": "",
                "clip_end_frame": "",
                "segment_frame_count": "",
                "matched_dcm_outputs": "",
            }

            if not plan_src.exists():
                row["status"] = "missing-plan"
                manifest_rows.append(row)
                print(f"[skip] c{index}: missing plan")
                continue
            if not targets_path.exists():
                row["status"] = "missing-targets"
                manifest_rows.append(row)
                print(f"[skip] c{index}: missing targets")
                continue

            if plan_src.resolve() != plan_dst.resolve():
                shutil.copy2(plan_src, plan_dst)

            payload = load_plan_payload(plan_src, args.plan_sample_fps, args.timeline_fps)
            timeline_frames = [int(item["timeline_frame"]) for item in payload.get("keyframes", [])]
            if not timeline_frames:
                row["status"] = "empty-plan"
                manifest_rows.append(row)
                print(f"[skip] c{index}: empty plan")
                continue

            base_row = dict(row)
            base_row["plan_timeline_start"] = min(timeline_frames)
            base_row["plan_timeline_end"] = max(timeline_frames)
            if args.use_full_plan_window:
                clip_start = int(base_row["plan_timeline_start"])
                clip_end = int(base_row["plan_timeline_end"])
                matched_rows = []
            else:
                clip_start, clip_end, matched_rows = select_clip_window(index, payload, dcm_rows)

            provider = JsonFrameTargetProvider(targets_path)
            requests = load_requests(
                plan_src,
                plan_sample_fps=args.plan_sample_fps,
                timeline_fps=args.timeline_fps,
            )
            solved = solve_camera_timeline(provider, requests, CameraSolveConfig())
            solved = _extend_solved_keys_to_frame(solved, clip_end)
            write_camera_vmd(solved, vmd_path, model_name="Autoshot Camera")
            dump_debug_json(solved, debug_json_path)
            full_camera_keys = read_camera_vmd(vmd_path)
            vmd_stats = read_camera_vmd_stats(vmd_path)
            base_row["camera_count"] = vmd_stats["camera_count"]
            base_row["camera_max_frame"] = vmd_stats["camera_max_frame"]

            segment_rows = matched_rows or [
                {
                    "segment_start_frame": clip_start,
                    "segment_end_frame": clip_end,
                    "output": "",
                }
            ]
            split_segments = len(segment_rows) > 1
            if not args.vmd_only and split_segments and mp4_path.exists():
                mp4_path.unlink()
            if not args.vmd_only:
                for stale in item_dir.glob(f"{stem}_part*.mp4"):
                    stale.unlink()
            for stale in item_dir.glob(f"{stem}_part*.vmd"):
                stale.unlink()
            for stale in item_dir.glob(f"{stem}_part*_debug.json"):
                stale.unlink()

            for part_index, segment in enumerate(segment_rows, start=1):
                part_row = dict(base_row)
                part_rows = [segment]
                part_clip_start = int(segment["segment_start_frame"])
                part_clip_end = int(segment["segment_end_frame"])
                part_mp4_path = mp4_path
                part_vmd_path = vmd_path
                part_debug_json_path = debug_json_path
                if split_segments:
                    part_mp4_path = item_dir / f"{stem}_part{part_index}.mp4"
                    part_vmd_path = item_dir / f"{stem}_part{part_index}.vmd"
                    part_debug_json_path = item_dir / f"{stem}_part{part_index}_debug.json"
                    part_row["part"] = part_index
                    part_row["part_count"] = len(segment_rows)
                    part_camera_keys = _slice_camera_vmd_keys(full_camera_keys, part_clip_start, part_clip_end)
                    if not part_camera_keys:
                        raise ValueError(
                            f"unable to build segmented VMD for c{index} part{part_index}: "
                            f"{part_clip_start}..{part_clip_end}"
                        )
                    part_debug_keys = _slice_debug_keys(solved, part_clip_start, part_clip_end)
                    write_camera_vmd_records(part_camera_keys, part_vmd_path, model_name="Autoshot Camera")
                    dump_debug_json(part_debug_keys, part_debug_json_path)
                part_row["mp4"] = str(part_mp4_path)
                part_row["vmd"] = str(part_vmd_path)
                part_row["debug_json"] = str(part_debug_json_path)
                part_row["clip_start_frame"] = part_clip_start
                part_row["clip_end_frame"] = part_clip_end
                part_row["matched_dcm_outputs"] = ";".join(str(item["output"]) for item in part_rows)
                part_vmd_stats = read_camera_vmd_stats(part_vmd_path)
                part_row["camera_count"] = part_vmd_stats["camera_count"]
                part_row["camera_max_frame"] = part_vmd_stats["camera_max_frame"]

                if args.vmd_only:
                    part_row["status"] = "ok-vmd-only"
                    part_row["segment_frame_count"] = max(0, part_clip_end - part_clip_start + 1)
                    part_row["segment_duration_sec"] = part_row["segment_frame_count"] / float(args.timeline_fps)
                    label = f"c{index}"
                    if split_segments:
                        label = f"c{index} part{part_index}/{len(segment_rows)}"
                    print(f"[ok] {label}: {part_vmd_path.name} (vmd-only)")
                    manifest_rows.append(part_row)
                    continue

                try:
                    model, motion, wav = resolve_render_assets(
                        variants=variants,
                        batch=batch,
                        mapping=mapping,
                        latest_ok_rows=latest_ok_rows,
                        index=index,
                        matched_rows=part_rows,
                        override_model=override_model,
                        default_model_path=default_model_path,
                    )
                    batch.MODEL_PATH = model
                    render_stats = render_partial_clip(
                        batch=batch,
                        motion=motion,
                        camera=part_vmd_path,
                        wav=wav,
                        output_path=part_mp4_path,
                        clip_start_frame=part_clip_start,
                        clip_end_frame=part_clip_end,
                        render_width=int(args.render_width),
                        render_height=int(args.render_height),
                        worker_id=args.worker_id,
                        worker_dir=Path(args.worker_dir).resolve() if args.worker_dir else None,
                        worker_lock_wait_sec=float(args.worker_lock_wait_sec),
                        worker_lock_stale_sec=float(args.worker_lock_stale_sec),
                    )
                    part_row["status"] = "ok"
                    part_row["model"] = str(model)
                    part_row["motion"] = str(motion)
                    part_row["wav"] = str(wav)
                    part_row["clip_start_frame"] = render_stats["segment_start_frame"]
                    part_row["clip_end_frame"] = render_stats["segment_end_frame"]
                    part_row["segment_frame_count"] = render_stats["segment_frame_count"]
                    part_row["segment_duration_sec"] = render_stats["segment_duration_sec"]
                    label = f"c{index}"
                    if split_segments:
                        label = f"c{index} part{part_index}/{len(segment_rows)}"
                    print(
                        f"[ok] {label}: {part_mp4_path.name} "
                        f"(clip {render_stats['segment_start_frame']}..{render_stats['segment_end_frame']})"
                    )
                except Exception as exc:
                    part_row["status"] = "failed"
                    part_row["error"] = str(exc)
                    label = f"c{index}"
                    if split_segments:
                        label = f"c{index} part{part_index}/{len(segment_rows)}"
                    print(f"[fail] {label}: {exc}")

                manifest_rows.append(part_row)
    finally:
        pass

    preserved_rows = [
        row
        for row in previous_manifest_rows
        if int(row.get("index", -1)) not in set(indices)
    ]
    final_rows = preserved_rows + manifest_rows
    final_rows.sort(key=_manifest_sort_key)
    write_manifest(
        final_rows,
        manifest_path,
        manifest_csv_path,
    )


if __name__ == "__main__":
    main()
