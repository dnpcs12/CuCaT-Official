from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .vmd import CameraVmdKey, EvaluatedCamera, evaluate_camera_vmd, read_camera_vmd, write_camera_vmd_records


LINEAR_CAMERA_INTERPOLATION = bytes([20, 107, 20, 107] * 6)


@dataclass(frozen=True)
class SegmentResult:
    start_frame: int
    end_frame: int
    selected_frames: list[int]
    selected_noncut_frames: list[int]
    max_interest_error: float
    max_rotate_error: float
    max_distance_error: float
    max_fov_error: float
    reachable: bool


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_int_list(text: str | None) -> list[int]:
    if not text:
        return []
    values: list[int] = []
    for item in str(text).replace(",", ";").split(";"):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(float(item)))
        except ValueError:
            continue
    return sorted(set(values))


def _load_csv_by_index(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_index: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            by_index[int(row["index"])] = row
        except (KeyError, TypeError, ValueError):
            continue
    return by_index


def _angle_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def _vec_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=1)


def _camera_to_key(frame: int, pose: EvaluatedCamera) -> CameraVmdKey:
    return CameraVmdKey(
        frame=int(frame),
        distance=float(pose.distance),
        interest=tuple(float(x) for x in pose.interest),
        rotate=tuple(float(x) for x in pose.rotate),
        view_angle_deg=int(round(float(pose.view_angle_deg))),
        perspective=bool(pose.perspective),
        interpolation=LINEAR_CAMERA_INTERPOLATION,
    )


def evaluate_trajectory(keys: list[CameraVmdKey], frames: Iterable[int]) -> dict[str, np.ndarray]:
    poses: list[EvaluatedCamera] = []
    valid_frames: list[int] = []
    for frame in frames:
        pose = evaluate_camera_vmd(keys, float(frame))
        if pose is None:
            continue
        poses.append(pose)
        valid_frames.append(int(frame))

    if not poses:
        return {
            "frames": np.asarray([], dtype=np.int32),
            "interest": np.empty((0, 3), dtype=np.float64),
            "rotate": np.empty((0, 3), dtype=np.float64),
            "distance": np.asarray([], dtype=np.float64),
            "fov": np.asarray([], dtype=np.float64),
            "perspective": np.asarray([], dtype=bool),
        }

    return {
        "frames": np.asarray(valid_frames, dtype=np.int32),
        "interest": np.asarray([pose.interest for pose in poses], dtype=np.float64),
        "rotate": np.asarray([pose.rotate for pose in poses], dtype=np.float64),
        "distance": np.asarray([pose.distance for pose in poses], dtype=np.float64),
        "fov": np.asarray([pose.view_angle_deg for pose in poses], dtype=np.float64),
        "perspective": np.asarray([pose.perspective for pose in poses], dtype=bool),
    }


def _edge_error(trajectory: dict[str, np.ndarray], left: int, right: int) -> dict[str, float | bool]:
    frames = trajectory["frames"][left : right + 1].astype(np.float64)
    start = float(frames[0])
    end = float(frames[-1])
    if end <= start:
        return {
            "max_interest_error": 0.0,
            "max_rotate_error": 0.0,
            "max_distance_error": 0.0,
            "max_fov_error": 0.0,
            "perspective_changed": False,
        }

    t = ((frames - start) / (end - start))[:, None]
    interest_interp = trajectory["interest"][left] + (
        trajectory["interest"][right] - trajectory["interest"][left]
    ) * t
    rotate_interp = trajectory["rotate"][left] + (
        trajectory["rotate"][right] - trajectory["rotate"][left]
    ) * t
    distance_interp = trajectory["distance"][left] + (
        trajectory["distance"][right] - trajectory["distance"][left]
    ) * t[:, 0]
    fov_interp = trajectory["fov"][left] + (trajectory["fov"][right] - trajectory["fov"][left]) * t[:, 0]
    perspective = trajectory["perspective"][left : right + 1]

    return {
        "max_interest_error": float(
            np.max(_vec_distance(trajectory["interest"][left : right + 1], interest_interp))
        ),
        "max_rotate_error": float(
            np.max(_vec_distance(_angle_delta(trajectory["rotate"][left : right + 1], rotate_interp), 0.0))
        ),
        "max_distance_error": float(np.max(np.abs(trajectory["distance"][left : right + 1] - distance_interp))),
        "max_fov_error": float(np.max(np.abs(trajectory["fov"][left : right + 1] - fov_interp))),
        "perspective_changed": bool(np.any(perspective != perspective[0])),
    }


def _edge_is_valid(
    trajectory: dict[str, np.ndarray],
    left: int,
    right: int,
    *,
    interest_tol: float,
    rotate_tol: float,
    distance_tol: float,
    fov_tol: float,
) -> tuple[bool, dict[str, float | bool]]:
    error = _edge_error(trajectory, left, right)
    valid = (
        not bool(error["perspective_changed"])
        and float(error["max_interest_error"]) <= interest_tol
        and float(error["max_rotate_error"]) <= rotate_tol
        and float(error["max_distance_error"]) <= distance_tol
        and float(error["max_fov_error"]) <= fov_tol
    )
    return valid, error


def _path_error(trajectory: dict[str, np.ndarray], path_indices: list[int]) -> dict[str, float]:
    max_interest = 0.0
    max_rotate = 0.0
    max_distance = 0.0
    max_fov = 0.0
    for left, right in zip(path_indices, path_indices[1:]):
        error = _edge_error(trajectory, left, right)
        max_interest = max(max_interest, float(error["max_interest_error"]))
        max_rotate = max(max_rotate, float(error["max_rotate_error"]))
        max_distance = max(max_distance, float(error["max_distance_error"]))
        max_fov = max(max_fov, float(error["max_fov_error"]))
    return {
        "max_interest_error": max_interest,
        "max_rotate_error": max_rotate,
        "max_distance_error": max_distance,
        "max_fov_error": max_fov,
    }


def _compare_trajectories(
    original: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
) -> dict[str, float | bool]:
    if len(original["frames"]) == 0 or len(test["frames"]) == 0:
        return {
            "max_interest_error": 0.0,
            "max_rotate_error": 0.0,
            "max_distance_error": 0.0,
            "max_fov_error": 0.0,
            "perspective_changed": False,
        }
    return {
        "max_interest_error": float(np.max(_vec_distance(original["interest"], test["interest"]))),
        "max_rotate_error": float(np.max(_vec_distance(_angle_delta(original["rotate"], test["rotate"]), 0.0))),
        "max_distance_error": float(np.max(np.abs(original["distance"] - test["distance"]))),
        "max_fov_error": float(np.max(np.abs(original["fov"] - test["fov"]))),
        "perspective_changed": bool(np.any(original["perspective"] != test["perspective"])),
    }


def _candidate_edge_error(
    original_trajectory: dict[str, np.ndarray],
    left_frame: int,
    right_frame: int,
    selected_keys: list[CameraVmdKey],
) -> dict[str, float | bool]:
    frames = original_trajectory["frames"]
    left = int(np.searchsorted(frames, left_frame, side="left"))
    right = int(np.searchsorted(frames, right_frame, side="right")) - 1
    if left < 0 or right < left:
        return {
            "max_interest_error": 0.0,
            "max_rotate_error": 0.0,
            "max_distance_error": 0.0,
            "max_fov_error": 0.0,
            "perspective_changed": False,
        }

    original = {key: value[left : right + 1] for key, value in original_trajectory.items()}
    test = evaluate_trajectory(selected_keys, original["frames"])
    return _compare_trajectories(original, test)


def _edge_error_is_valid(
    error: dict[str, float | bool],
    *,
    interest_tol: float,
    rotate_tol: float,
    distance_tol: float,
    fov_tol: float,
) -> bool:
    return (
        not bool(error["perspective_changed"])
        and float(error["max_interest_error"]) <= interest_tol
        and float(error["max_rotate_error"]) <= rotate_tol
        and float(error["max_distance_error"]) <= distance_tol
        and float(error["max_fov_error"]) <= fov_tol
    )


def _original_keys_in_span(keys: list[CameraVmdKey], start_frame: int, end_frame: int) -> list[CameraVmdKey]:
    candidates = [key for key in keys if start_frame <= int(key.frame) <= end_frame]
    candidates.sort(key=lambda key: int(key.frame))
    return candidates


def _key_by_frame(keys: list[CameraVmdKey]) -> dict[int, CameraVmdKey]:
    by_frame: dict[int, CameraVmdKey] = {}
    for key in keys:
        by_frame.setdefault(int(key.frame), key)
    return by_frame


def _path_error_original_bezier(
    original_trajectory: dict[str, np.ndarray],
    candidates: list[CameraVmdKey],
    path_indices: list[int],
) -> dict[str, float]:
    max_interest = 0.0
    max_rotate = 0.0
    max_distance = 0.0
    max_fov = 0.0
    for left, right in zip(path_indices, path_indices[1:]):
        left_key = candidates[left]
        right_key = candidates[right]
        error = _candidate_edge_error(
            original_trajectory,
            int(left_key.frame),
            int(right_key.frame),
            [left_key, right_key],
        )
        max_interest = max(max_interest, float(error["max_interest_error"]))
        max_rotate = max(max_rotate, float(error["max_rotate_error"]))
        max_distance = max(max_distance, float(error["max_distance_error"]))
        max_fov = max(max_fov, float(error["max_fov_error"]))
    return {
        "max_interest_error": max_interest,
        "max_rotate_error": max_rotate,
        "max_distance_error": max_distance,
        "max_fov_error": max_fov,
    }


def minimize_segment_original_bezier(
    keys: list[CameraVmdKey],
    start_frame: int,
    end_frame: int,
    cut_pair_frames: set[int],
    *,
    frame_step: int,
    max_segment_frames: int,
    interest_tol: float,
    rotate_tol: float,
    distance_tol: float,
    fov_tol: float,
) -> SegmentResult:
    candidates = _original_keys_in_span(keys, start_frame, end_frame)
    if len(candidates) <= 1:
        selected = [int(key.frame) for key in candidates] or [start_frame]
        return SegmentResult(
            start_frame=start_frame,
            end_frame=end_frame,
            selected_frames=selected,
            selected_noncut_frames=[frame for frame in selected if frame not in cut_pair_frames],
            max_interest_error=0.0,
            max_rotate_error=0.0,
            max_distance_error=0.0,
            max_fov_error=0.0,
            reachable=True,
        )

    step = max(1, int(frame_step))
    frames = list(range(int(candidates[0].frame), int(candidates[-1].frame) + 1, step))
    if frames[-1] != int(candidates[-1].frame):
        frames.append(int(candidates[-1].frame))
    original_trajectory = evaluate_trajectory(keys, frames)
    n = len(candidates)
    inf = 10**9
    dp = [inf] * n
    prev = [-1] * n
    dp[0] = 1
    max_gap = max(1, int(max_segment_frames))

    for left in range(n - 1):
        if dp[left] >= inf:
            continue
        left_key = candidates[left]
        for right in range(left + 1, n):
            right_key = candidates[right]
            if int(right_key.frame - left_key.frame) > max_gap:
                break
            error = _candidate_edge_error(
                original_trajectory,
                int(left_key.frame),
                int(right_key.frame),
                [left_key, right_key],
            )
            if not _edge_error_is_valid(
                error,
                interest_tol=interest_tol,
                rotate_tol=rotate_tol,
                distance_tol=distance_tol,
                fov_tol=fov_tol,
            ):
                continue
            candidate = dp[left] + 1
            if candidate < dp[right]:
                dp[right] = candidate
                prev[right] = left

    reachable = dp[-1] < inf
    if reachable:
        path_indices: list[int] = []
        index = n - 1
        while index >= 0:
            path_indices.append(index)
            index = prev[index]
        path_indices.reverse()
    else:
        # Keeping all original keys exactly reproduces the MMD/VMD Bezier trajectory.
        path_indices = list(range(n))

    selected = [int(candidates[index].frame) for index in path_indices]
    errors = _path_error_original_bezier(original_trajectory, candidates, path_indices)
    return SegmentResult(
        start_frame=start_frame,
        end_frame=end_frame,
        selected_frames=selected,
        selected_noncut_frames=[frame for frame in selected if frame not in cut_pair_frames],
        max_interest_error=errors["max_interest_error"],
        max_rotate_error=errors["max_rotate_error"],
        max_distance_error=errors["max_distance_error"],
        max_fov_error=errors["max_fov_error"],
        reachable=reachable,
    )


def minimize_segment(
    keys: list[CameraVmdKey],
    start_frame: int,
    end_frame: int,
    cut_pair_frames: set[int],
    *,
    frame_step: int,
    max_segment_frames: int,
    interest_tol: float,
    rotate_tol: float,
    distance_tol: float,
    fov_tol: float,
) -> SegmentResult:
    if end_frame <= start_frame:
        return SegmentResult(
            start_frame=start_frame,
            end_frame=end_frame,
            selected_frames=[start_frame],
            selected_noncut_frames=[] if start_frame in cut_pair_frames else [start_frame],
            max_interest_error=0.0,
            max_rotate_error=0.0,
            max_distance_error=0.0,
            max_fov_error=0.0,
            reachable=True,
        )

    step = max(1, int(frame_step))
    frames = list(range(start_frame, end_frame + 1, step))
    if frames[-1] != end_frame:
        frames.append(end_frame)
    trajectory = evaluate_trajectory(keys, frames)
    n = int(len(trajectory["frames"]))
    if n <= 1:
        selected = [start_frame, end_frame] if start_frame != end_frame else [start_frame]
        return SegmentResult(
            start_frame=start_frame,
            end_frame=end_frame,
            selected_frames=selected,
            selected_noncut_frames=[frame for frame in selected if frame not in cut_pair_frames],
            max_interest_error=0.0,
            max_rotate_error=0.0,
            max_distance_error=0.0,
            max_fov_error=0.0,
            reachable=True,
        )

    inf = 10**9
    dp = [inf] * n
    prev = [-1] * n
    dp[0] = 1
    max_gap = max(1, int(max_segment_frames))

    for left in range(n - 1):
        if dp[left] >= inf:
            continue
        for right in range(left + 1, n):
            if int(trajectory["frames"][right] - trajectory["frames"][left]) > max_gap:
                break
            valid, _ = _edge_is_valid(
                trajectory,
                left,
                right,
                interest_tol=interest_tol,
                rotate_tol=rotate_tol,
                distance_tol=distance_tol,
                fov_tol=fov_tol,
            )
            if not valid:
                continue
            candidate = dp[left] + 1
            if candidate < dp[right]:
                dp[right] = candidate
                prev[right] = left

    reachable = dp[-1] < inf
    if reachable:
        path_indices: list[int] = []
        index = n - 1
        while index >= 0:
            path_indices.append(index)
            index = prev[index]
        path_indices.reverse()
    else:
        # Conservative fallback: keep every sampled frame, which is always exact.
        path_indices = list(range(n))

    selected = [int(trajectory["frames"][index]) for index in path_indices]
    errors = _path_error(trajectory, path_indices)
    return SegmentResult(
        start_frame=start_frame,
        end_frame=end_frame,
        selected_frames=selected,
        selected_noncut_frames=[frame for frame in selected if frame not in cut_pair_frames],
        max_interest_error=errors["max_interest_error"],
        max_rotate_error=errors["max_rotate_error"],
        max_distance_error=errors["max_distance_error"],
        max_fov_error=errors["max_fov_error"],
        reachable=reachable,
    )


def cut_pair_frames_from_cut_frames(cut_frames: Iterable[int]) -> set[int]:
    frames: set[int] = set()
    for cut_frame in cut_frames:
        cut = int(cut_frame)
        frames.add(cut)
        if cut > 0:
            frames.add(cut - 1)
    return frames


def build_noncut_spans(start_frame: int, end_frame: int, cut_frames: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = int(start_frame)
    for cut_frame in sorted(set(int(frame) for frame in cut_frames)):
        pre_cut = max(start_frame, cut_frame - 1)
        cut = min(end_frame, cut_frame)
        if cursor < pre_cut:
            spans.append((cursor, pre_cut))
        cursor = max(cursor, cut)
    if cursor < end_frame:
        spans.append((cursor, int(end_frame)))
    return spans


def _resolve_indices(args: argparse.Namespace, keyframe_rows: dict[int, dict[str, str]]) -> list[int]:
    if args.indices:
        return sorted(set(args.indices))
    if args.start is not None or args.end is not None:
        start = int(args.start if args.start is not None else min(keyframe_rows))
        end = int(args.end if args.end is not None else max(keyframe_rows))
        return [idx for idx in range(start, end + 1) if idx in keyframe_rows]
    return sorted(keyframe_rows)


def process_song(
    index: int,
    *,
    keyframe_row: dict[str, str],
    cut_row: dict[str, str] | None,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    camera_vmd = Path(args.camera_vmd) if args.camera_vmd else Path(keyframe_row["camera_vmd"])
    keys = read_camera_vmd(camera_vmd)
    if not keys:
        raise RuntimeError(f"No camera keys found: {camera_vmd}")

    gt_keyframes = _parse_int_list(keyframe_row.get("keyframe_frames_text"))
    cut_frames = _parse_int_list(cut_row.get("cut_frames_text") if cut_row else "")
    cut_pair_frames = cut_pair_frames_from_cut_frames(cut_frames)
    gt_noncut_frames = sorted(frame for frame in gt_keyframes if frame not in cut_pair_frames)

    start_frame = min([keys[0].frame, *(gt_keyframes or [keys[0].frame])])
    end_frame = max([keys[-1].frame, *(gt_keyframes or [keys[-1].frame])])
    spans = build_noncut_spans(start_frame, end_frame, cut_frames)

    segment_results: list[SegmentResult] = []
    selected_frames: set[int] = set()
    selected_noncut_frames: set[int] = set()
    for start, end in spans:
        if args.fit_mode == "linear-sampled":
            result = minimize_segment(
                keys,
                start,
                end,
                cut_pair_frames,
                frame_step=args.frame_step,
                max_segment_frames=args.max_segment_frames,
                interest_tol=args.interest_tol,
                rotate_tol=args.rotate_tol,
                distance_tol=args.distance_tol,
                fov_tol=args.fov_tol,
            )
        else:
            result = minimize_segment_original_bezier(
                keys,
                start,
                end,
                cut_pair_frames,
                frame_step=args.frame_step,
                max_segment_frames=args.max_segment_frames,
                interest_tol=args.interest_tol,
                rotate_tol=args.rotate_tol,
                distance_tol=args.distance_tol,
                fov_tol=args.fov_tol,
            )
        segment_results.append(result)
        selected_frames.update(result.selected_frames)
        selected_noncut_frames.update(result.selected_noncut_frames)

    # Cut-pair frames are preserved for rendering/reconstruction, but not counted as optimized non-cut keys.
    reconstructed_frames = sorted(selected_frames | cut_pair_frames)
    selected_noncut_sorted = sorted(selected_noncut_frames - cut_pair_frames)
    output_dir = output_root / f"c{index}"
    output_dir.mkdir(parents=True, exist_ok=True)

    vmd_path = None
    if args.write_vmd:
        if args.fit_mode == "linear-sampled":
            materialized: list[CameraVmdKey] = []
            for frame in reconstructed_frames:
                pose = evaluate_camera_vmd(keys, float(frame))
                if pose is not None:
                    materialized.append(_camera_to_key(frame, pose))
            vmd_path = output_dir / "camera_minimal_noncut_linear.vmd"
        else:
            by_frame = _key_by_frame(keys)
            materialized = [by_frame[frame] for frame in reconstructed_frames if frame in by_frame]
            vmd_path = output_dir / "camera_minimal_noncut_bezier.vmd"
        write_camera_vmd_records(materialized, vmd_path, model_name=f"Minimal Noncut c{index}")

    summary = {
        "index": index,
        "fit_mode": args.fit_mode,
        "camera_vmd": str(camera_vmd),
        "duration_sec": float(keyframe_row.get("duration_sec", 0.0) or 0.0),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "tolerances": {
            "fit_mode": args.fit_mode,
            "interest_tol": args.interest_tol,
            "rotate_tol": args.rotate_tol,
            "distance_tol": args.distance_tol,
            "fov_tol": args.fov_tol,
            "frame_step": args.frame_step,
            "max_segment_frames": args.max_segment_frames,
        },
        "gt_keyframe_count": len(gt_keyframes),
        "gt_cut_count": len(cut_frames),
        "gt_cut_pair_frame_count": len(cut_pair_frames),
        "gt_noncut_keyframe_count": len(gt_noncut_frames),
        "selected_noncut_keyframe_count": len(selected_noncut_sorted),
        "selected_reconstruction_keyframe_count_including_cut_pairs": len(reconstructed_frames),
        "noncut_reduction_ratio": (
            1.0 - (len(selected_noncut_sorted) / len(gt_noncut_frames)) if gt_noncut_frames else 0.0
        ),
        "gt_keyframe_frames": gt_keyframes,
        "cut_frames": cut_frames,
        "cut_pair_frames": sorted(cut_pair_frames),
        "gt_noncut_keyframe_frames": gt_noncut_frames,
        "selected_noncut_keyframe_frames": selected_noncut_sorted,
        "selected_reconstruction_frames_including_cut_pairs": reconstructed_frames,
        "segments": [asdict(result) for result in segment_results],
        "unreachable_segment_count": sum(1 for result in segment_results if not result.reachable),
        "max_interest_error": max((result.max_interest_error for result in segment_results), default=0.0),
        "max_rotate_error": max((result.max_rotate_error for result in segment_results), default=0.0),
        "max_distance_error": max((result.max_distance_error for result in segment_results), default=0.0),
        "max_fov_error": max((result.max_fov_error for result in segment_results), default=0.0),
        "output_vmd": str(vmd_path) if vmd_path else None,
    }
    (output_dir / "minimal_noncut_keyframes.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find a near-minimal non-cut camera keyframe set that reconstructs GT camera trajectory."
    )
    default_analysis = repo_root() / "camera_autoshot" / "analysis"
    parser.add_argument("--indices", nargs="+", type=int, help="Song indices to process, e.g. --indices 0 14 49")
    parser.add_argument("--start", type=int, help="Inclusive start index when --indices is omitted.")
    parser.add_argument("--end", type=int, help="Inclusive end index when --indices is omitted.")
    parser.add_argument("--camera-vmd", help="Optional single camera VMD override. Use with one --indices value.")
    parser.add_argument(
        "--gt-keyframes-csv",
        type=Path,
        default=default_analysis / "gt_keyframes" / "gt_keyframes_per_song.csv",
    )
    parser.add_argument(
        "--gt-cuts-csv",
        type=Path,
        default=default_analysis / "gt_cut_frames" / "gt_cut_frames_per_song.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_analysis / "minimal_noncut_camera_keyframes",
    )
    parser.add_argument("--interest-tol", type=float, default=1.0)
    parser.add_argument("--rotate-tol", type=float, default=0.035, help="Rotation tolerance in radians.")
    parser.add_argument("--distance-tol", type=float, default=1.0)
    parser.add_argument("--fov-tol", type=float, default=1.0)
    parser.add_argument("--frame-step", type=int, default=1, help="Sample every N frames while fitting.")
    parser.add_argument("--max-segment-frames", type=int, default=180)
    parser.add_argument(
        "--fit-mode",
        choices=["original-bezier", "linear-sampled"],
        default="original-bezier",
        help=(
            "original-bezier prunes only original VMD keys and preserves their MMD interpolation; "
            "linear-sampled is the older frame-sampling linear approximation."
        ),
    )
    parser.add_argument("--exclude", nargs="*", type=int, default=[])
    parser.add_argument("--write-vmd", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    keyframe_rows = _load_csv_by_index(args.gt_keyframes_csv)
    cut_rows = _load_csv_by_index(args.gt_cuts_csv)
    indices = [idx for idx in _resolve_indices(args, keyframe_rows) if idx not in set(args.exclude)]
    if args.camera_vmd and len(indices) != 1:
        raise SystemExit("--camera-vmd can only be used with exactly one index.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for index in indices:
        try:
            summary = process_song(
                index,
                keyframe_row=keyframe_rows[index],
                cut_row=cut_rows.get(index),
                output_root=args.output_root,
                args=args,
            )
            summaries.append(summary)
            print(
                f"c{index}: GT non-cut {summary['gt_noncut_keyframe_count']} -> "
                f"minimal {summary['selected_noncut_keyframe_count']} "
                f"(reduction {summary['noncut_reduction_ratio']:.1%})"
            )
        except Exception as exc:  # keep batch runs moving
            errors.append({"index": index, "error": repr(exc)})
            print(f"c{index}: ERROR {exc}")

    csv_path = args.output_root / "summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "index",
            "fit_mode",
            "duration_sec",
            "gt_keyframe_count",
            "gt_cut_count",
            "gt_cut_pair_frame_count",
            "gt_noncut_keyframe_count",
            "selected_noncut_keyframe_count",
            "selected_reconstruction_keyframe_count_including_cut_pairs",
            "noncut_reduction_ratio",
            "unreachable_segment_count",
            "max_interest_error",
            "max_rotate_error",
            "max_distance_error",
            "max_fov_error",
            "output_vmd",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({name: summary.get(name) for name in fieldnames})

    manifest = {
        "output_root": str(args.output_root),
        "song_count": len(summaries),
        "error_count": len(errors),
        "errors": errors,
        "summary_csv": str(csv_path),
        "params": {
            "fit_mode": args.fit_mode,
            "interest_tol": args.interest_tol,
            "rotate_tol": args.rotate_tol,
            "distance_tol": args.distance_tol,
            "fov_tol": args.fov_tol,
            "frame_step": args.frame_step,
            "max_segment_frames": args.max_segment_frames,
            "write_vmd": args.write_vmd,
        },
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary: {csv_path}")


if __name__ == "__main__":
    main()
