from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .optimize_topk_candidates import (
    _candidate_sequence_from_topk,
    _crop_gt_sequence,
    _infer_schema_name,
    _load_json,
    _metadata_for_output,
    _payload_keyframes,
    _rank_summary,
    _resolve_gt_path,
    _strip_optimizer_fields,
    _write_json,
)
from .distance_fov_level import (
    adjacent_distance_values,
    adjacent_fov_values,
    distance_fov_level_index,
    distance_fov_level_label,
)
from .bodypart_proxy import visible_bodyparts
from .shot_sequence_cp_sat import fields_for_schema, optimize_shot_sequence_cp_sat


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT_ROOT = WORKSPACE_ROOT / "camera_autoshot" / "reverse_v3" / "dcm_clip_labels" / "v3_1_close_dense_temporal_delta"
DEFAULT_REAL_TARGETS_ROOT = WORKSPACE_ROOT / "camera_autoshot" / "reverses"
DEFAULT_FEATURE_ROOT = (
    WORKSPACE_ROOT
    / "music_processing"
    / "results"
    / "01_datasets"
    / "cut_keyframe_multimodal_dataset"
    / "songs"
)
DEFAULT_MOTION_BODY_ACTIVITY_ROOT = (
    WORKSPACE_ROOT
    / "music_processing"
    / "results"
    / "06_motion_body"
    / "motion_body_frame_labels"
    / "songs"
    / "npz"
)
DEFAULT_FINGER_VISIBILITY_THRESHOLD_CSV = (
    WORKSPACE_ROOT
    / "camera_autoshot"
    / "analysis"
    / "finger_missing_threshold_probe"
    / "close_distance_fov_screen_thresholds.csv"
)

FINGER_VISIBILITY_CLOSE_DISTANCES = {"extreme_close", "very_close", "close"}
FINGER_VISIBILITY_FINGERTIP_JOINTS = (25, 28, 31, 34, 37, 47, 50, 53, 56, 59)
FINGER_VISIBILITY_ANCHOR_JOINTS: dict[str, tuple[int, ...]] = {
    "feet": (11, 15),
    "lower_leg": (10, 14),
    "thigh": (9, 13),
    "hip": (0,),
    "waist": (1,),
    "chest": (2,),
    "face": (4,),
}
FINGER_VISIBILITY_YAW_DEG = {
    "front": 0.0,
    "right_45": 45.0,
    "profile_right": 90.0,
    "back_right_45": 135.0,
    "back": 180.0,
    "back_left_45": -135.0,
    "profile_left": -90.0,
    "left_45": -45.0,
}

_MOTION_BODY_LABEL_CACHE: dict[tuple[str, tuple[str, ...]], tuple[np.ndarray | None, Path | None, list[str]]] = {}
_FINGER_DANCE_FEATURE_CACHE: dict[str, tuple[np.ndarray, list[str], Path]] = {}
_FINGER_THRESHOLD_CACHE: dict[tuple[str, str], dict[tuple[str, str], tuple[float, float]]] = {}


def _wrap_delta_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _unwrap_degrees(values: list[float]) -> list[float]:
    if not values:
        return []
    out = [float(values[0])]
    for value in values[1:]:
        out.append(out[-1] + _wrap_delta_deg(float(value) - out[-1]))
    return out


def _heading_from_forward(forward: list[float]) -> float:
    if len(forward) < 3:
        return 0.0
    return math.degrees(math.atan2(float(forward[0]), float(forward[2])))


def _load_heading_track(targets_path: Path) -> tuple[list[int], list[float]]:
    payload = _load_json(targets_path)
    frames: list[int] = []
    headings: list[float] = []
    for row in payload.get("frames", []):
        if not isinstance(row, dict) or "forward" not in row:
            continue
        frames.append(int(row.get("frame", len(frames))))
        headings.append(_heading_from_forward(row.get("forward") or [0.0, 0.0, 1.0]))
    return frames, _unwrap_degrees(headings)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize arbitrary top-k candidate JSON files with CP-SAT.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--candidate-glob", default="c*/camera_poses_v3_cutctx*_top10_candidates.json")
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--gt-name", default="camera_poses_v3.json")
    parser.add_argument("--schema", choices=["auto", "v2", "v3_1"], default="auto")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-time-sec", type=float, default=30.0)
    parser.add_argument("--local-cost-scale", type=int, default=100)
    parser.add_argument("--disable-sequence-stat-objective", action="store_true")
    parser.add_argument("--allow-missing-gt", action="store_true")
    parser.add_argument("--distance-bounce-weight", type=int, default=0)
    parser.add_argument("--distance-bounce-min-rank-delta", type=int, default=80)
    parser.add_argument("--distance-bounce-max-frame-gap", type=int, default=60)
    parser.add_argument("--composition-bounce-weight", type=int, default=0)
    parser.add_argument("--composition-bounce-min-rank-delta", type=int, default=80)
    parser.add_argument("--composition-bounce-max-frame-gap", type=int, default=60)
    parser.add_argument("--vertical-bounce-weight", type=int, default=0)
    parser.add_argument("--vertical-bounce-min-rank-delta", type=int, default=35)
    parser.add_argument("--vertical-bounce-max-frame-gap", type=int, default=60)
    parser.add_argument("--top-pitch-bounce-weight", type=int, default=0)
    parser.add_argument("--top-pitch-bounce-max-frame-gap", type=int, default=30)
    parser.add_argument("--top-pitch-bounce-neighbor-pitches", default="low,level")
    parser.add_argument("--top-pitch-bounce-return-pitches", default="low,level,high")
    parser.add_argument("--add-top-pitch-escape-candidates", action="store_true")
    parser.add_argument("--top-pitch-escape-candidate-cost", type=float, default=1.5)
    parser.add_argument("--top-pitch-escape-candidate-pitches", default="high")
    parser.add_argument("--top-pitch-escape-candidate-max-per-step", type=int, default=2)
    parser.add_argument("--disallow-new-top-pitch-candidates", action="store_true")
    parser.add_argument("--turn-yaw-jitter-weight", type=int, default=0)
    parser.add_argument("--turn-yaw-jitter-targets-root", type=Path, default=DEFAULT_REAL_TARGETS_ROOT)
    parser.add_argument("--turn-yaw-jitter-fps", type=float, default=30.0)
    parser.add_argument("--turn-yaw-jitter-min-frame-gap", type=int, default=1)
    parser.add_argument("--turn-yaw-jitter-max-frame-gap", type=int, default=90)
    parser.add_argument("--turn-yaw-jitter-min-heading-delta", type=float, default=25.0)
    parser.add_argument("--turn-yaw-jitter-min-heading-speed", type=float, default=9999.0)
    parser.add_argument("--turn-yaw-jitter-min-relative-yaw-delta", type=float, default=45.0)
    parser.add_argument("--turn-yaw-jitter-min-world-yaw-delta", type=float, default=60.0)
    parser.add_argument("--turn-yaw-jitter-penalty-degrees-per-unit", type=float, default=15.0)
    parser.add_argument("--turn-yaw-jitter-allow-opposite-direction", action="store_true")
    parser.add_argument("--turn-yaw-jitter-include-cut-pairs", action="store_true")
    parser.add_argument("--reference-plan-root", type=Path, default=None)
    parser.add_argument("--reference-change-weight", type=int, default=0)
    parser.add_argument("--bodypart-duration-weight", type=int, default=0)
    parser.add_argument("--bodypart-duration-stats", type=Path, default=None)
    parser.add_argument("--bodypart-duration-fps", type=float, default=30.0)
    parser.add_argument("--bodypart-duration-proxy-mode", choices=["baseline", "calibrated"], default="calibrated")
    parser.add_argument("--bodypart-duration-upper-key", default="max_sec")
    parser.add_argument("--bodypart-duration-penalty-sec-per-unit", type=float, default=0.5)
    parser.add_argument("--bodypart-duration-upper-blend-combos", default="")
    parser.add_argument("--bodypart-duration-upper-blend-alpha", type=float, default=0.0)
    parser.add_argument("--bodypart-duration-upper-blend-from-key", default="p95_sec")
    parser.add_argument("--bodypart-duration-upper-blend-to-key", default="max_sec")
    parser.add_argument("--bodypart-duration-short-weight", type=int, default=0)
    parser.add_argument("--bodypart-duration-lower-key", default="p25_sec")
    parser.add_argument("--bodypart-duration-short-penalty-sec-per-unit", type=float, default=0.25)
    parser.add_argument("--bodypart-duration-short-include-cut-pairs", action="store_true")
    parser.add_argument("--distance-level-duration-weight", type=int, default=0)
    parser.add_argument("--distance-level-duration-stats", type=Path, default=None)
    parser.add_argument("--distance-level-duration-fps", type=float, default=30.0)
    parser.add_argument("--distance-level-duration-include-levels", default="")
    parser.add_argument("--distance-level-duration-end-mode", choices=["next_change", "midpoint"], default="next_change")
    parser.add_argument("--distance-level-duration-upper-key", default="p95_sec")
    parser.add_argument("--distance-level-duration-upper-override-levels", default="")
    parser.add_argument("--distance-level-duration-upper-override-key", default="max_sec")
    parser.add_argument("--distance-level-duration-penalty-sec-per-unit", type=float, default=0.5)
    parser.add_argument("--distance-level-duration-short-weight", type=int, default=0)
    parser.add_argument("--distance-level-duration-lower-key", default="p25_sec")
    parser.add_argument("--distance-level-duration-min-sec", type=float, default=0.7)
    parser.add_argument("--distance-level-duration-short-penalty-sec-per-unit", type=float, default=0.25)
    parser.add_argument("--distance-level-duration-short-include-cut-pairs", action="store_true")
    parser.add_argument("--distance-level-duration-short-exclude-levels", default="")
    parser.add_argument("--distance-level-direction-weight", type=int, default=0)
    parser.add_argument("--distance-level-direction-close-levels", default="tight_close,close,medium_close")
    parser.add_argument("--distance-level-direction-medium-level", default="medium")
    parser.add_argument("--distance-level-direction-far-levels", default="medium_far,far,wide")
    parser.add_argument("--distance-level-direction-hard", action="store_true")
    parser.add_argument("--distance-level-protect-source-levels", default="")
    parser.add_argument("--distance-level-protect-min-level", default="medium_far")
    parser.add_argument("--distance-level-change-only-eligible", action="store_true")
    parser.add_argument("--distance-level-eligible-close-levels", default="tight_close,close,medium_close")
    parser.add_argument("--distance-level-eligible-upper-key", default="p90_sec")
    parser.add_argument("--distance-level-eligible-require-lower-body-activity", action="store_true")
    parser.add_argument("--distance-level-activity-feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--distance-level-lower-body-joints", default="9,10,11,13,14,15")
    parser.add_argument("--distance-level-lower-body-activity-min", type=float, default=0.30)
    parser.add_argument("--distance-level-activity-window-radius", type=int, default=15)
    parser.add_argument("--distance-level-eligible-min-next-frame-gap", type=int, default=5)
    parser.add_argument("--distance-level-hold-after-change", action="store_true")
    parser.add_argument(
        "--distance-level-hold-after-change-mode",
        choices=["level", "group", "distance", "distance_fov"],
        default="level",
    )
    parser.add_argument("--distance-level-hold-after-change-min-frame-gap", type=int, default=1)
    parser.add_argument("--distance-level-hold-after-change-max-frame-gap", type=int, default=60)
    parser.add_argument("--distance-level-hold-after-change-min-keyframes", type=int, default=2)
    parser.add_argument("--distance-level-hold-after-change-min-sec", type=float, default=0.0)
    parser.add_argument("--distance-level-hold-after-change-min-sec-key", default="")
    parser.add_argument("--distance-level-hold-after-change-max-sec-key", default="")
    parser.add_argument("--distance-level-hold-after-change-disallow-cut-pairs", action="store_true")
    parser.add_argument(
        "--finger-activity-visibility-cost",
        type=float,
        default=0.0,
        help=(
            "Additional local candidate cost when hand/arm body-activity labels are active "
            "but the threshold finger-visibility proxy predicts missing fingers."
        ),
    )
    parser.add_argument(
        "--finger-activity-root",
        type=Path,
        default=DEFAULT_MOTION_BODY_ACTIVITY_ROOT,
        help="NPZ root with motion_body_frame_labels/songs/npz/cNN.npz activity labels.",
    )
    parser.add_argument(
        "--finger-activity-feature-root",
        type=Path,
        default=DEFAULT_FEATURE_ROOT,
        help="Full-song dance feature root with relpos joints for threshold finger visibility.",
    )
    parser.add_argument(
        "--finger-activity-threshold-csv",
        type=Path,
        default=DEFAULT_FINGER_VISIBILITY_THRESHOLD_CSV,
        help="CSV with close distance/FOV screen thresholds from finger_missing_threshold_probe.",
    )
    parser.add_argument("--finger-activity-threshold-schema", default="v3_1_close_dense_training")
    parser.add_argument("--finger-activity-parts", default="left_hand,right_hand,left_arm,right_arm")
    parser.add_argument("--finger-activity-window-radius", type=int, default=5)
    parser.add_argument("--finger-activity-min", type=float, default=0.5)
    parser.add_argument("--finger-activity-visible-threshold", type=float, default=0.2)
    parser.add_argument(
        "--finger-activity-window-mode",
        choices=["any", "q75"],
        default="any",
        help="How to aggregate hand/arm activity inside the local finger activity window.",
    )
    parser.add_argument(
        "--finger-activity-axis-mode",
        choices=["camera_right_approx", "world_x", "xz_radius"],
        default="camera_right_approx",
    )
    parser.add_argument("--finger-activity-close-distances", default="extreme_close,very_close,close")
    parser.add_argument("--add-bodyproxy-safe-target-anchor-candidates", action="store_true")
    parser.add_argument("--bodyproxy-safe-target-anchor-cost", type=float, default=2.5)
    parser.add_argument(
        "--bodyproxy-safe-target-anchor-values",
        default="feet,lower_leg,thigh,hip,waist,chest,face",
        help="Target anchors to try when adding body-proxy-safe synthetic candidates.",
    )
    parser.add_argument("--bodyproxy-safe-target-anchor-max-per-step", type=int, default=4)
    parser.add_argument("--bodyproxy-safe-target-anchor-proxy-mode", default="calibrated")
    parser.add_argument("--add-distance-fov-only-candidates", action="store_true")
    parser.add_argument("--distance-fov-only-candidate-cost", type=float, default=2.5)
    parser.add_argument("--distance-fov-only-candidate-radius", type=int, default=1)
    parser.add_argument("--distance-fov-only-candidate-max-per-step", type=int, default=8)
    parser.add_argument("--distance-fov-only-directional-escape", action="store_true")
    parser.add_argument("--distance-fov-only-exclude-source-levels", default="")
    parser.add_argument(
        "--distance-fov-only-finger-triggered-only",
        action="store_true",
        help=(
            "When adding distance/FOV-only synthetic candidates, only add them on "
            "source frames where active hands/fingers are predicted to be cropped."
        ),
    )
    parser.add_argument("--finger-activity-trigger-source-distances", default="very_close,close")
    parser.add_argument("--finger-activity-trigger-target-distances", default="medium_close")
    parser.add_argument("--cut-identity-weight", type=int, default=0)
    parser.add_argument("--cut-contrast-weight", type=int, default=0)
    parser.add_argument("--cut-contrast-min-score", type=int, default=3)
    parser.add_argument("--cut-contrast-medium-rank-delta", type=int, default=100)
    parser.add_argument("--cut-contrast-strong-rank-delta", type=int, default=220)
    parser.add_argument("--post-cut-distance-jump-weight", type=int, default=0)
    parser.add_argument("--post-cut-distance-jump-min-rank-delta", type=int, default=250)
    parser.add_argument("--post-cut-distance-jump-max-frame-gap", type=int, default=60)
    parser.add_argument("--cut-context-speed-weight", type=int, default=0)
    parser.add_argument("--cut-context-speed-min-speed", type=float, default=250.0)
    parser.add_argument("--cut-context-speed-max-frame-gap", type=int, default=60)
    parser.add_argument("--cut-context-speed-fps", type=float, default=30.0)
    parser.add_argument("--repeated-distance-bounce-weight", type=int, default=0)
    parser.add_argument("--repeated-distance-bounce-min-rank-delta", type=int, default=180)
    parser.add_argument("--repeated-distance-bounce-max-frame-gap", type=int, default=90)
    parser.add_argument("--manifest-name", default="cp_sat_optimization_manifest.json")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _candidate_paths(source_root: Path, pattern: str) -> list[Path]:
    ignored_markers = ("_optimized_cp_sat", "_optimizer_debug")
    return sorted(
        path
        for path in source_root.glob(pattern)
        if path.is_file() and not any(marker in path.stem for marker in ignored_markers)
    )


def _output_path_for_candidate(candidate_path: Path) -> Path:
    stem = re.sub(r"_top\d+_candidates$", "_optimized_cp_sat", candidate_path.stem)
    if stem == candidate_path.stem:
        stem = f"{candidate_path.stem}_optimized_cp_sat"
    return candidate_path.with_name(f"{stem}.json")


def _debug_path_for_output(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_optimizer_debug.json")


def _source_plan_rows(candidate_payload: dict[str, Any]) -> list[dict[str, Any]]:
    source_plan = candidate_payload.get("source_plan")
    if not source_plan:
        return []
    path = Path(str(source_plan))
    if not path.exists():
        return []
    return _payload_keyframes(_load_json(path))


def _rows_from_sequence(sequence: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    return [_strip_optimizer_fields(row, fields) for row in sequence]


def _top1_reference_sequence(candidates: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    """Build a reference sequence from each timestep's rank-1 candidate."""
    sequence: list[dict[str, Any]] = []
    for _, group in sorted(_group_candidates_by_step(candidates).items()):
        if not group:
            continue
        top1_rows = [row for row in group if int(row.get("rank", 0)) == 1]
        reference = top1_rows[0] if top1_rows else min(group, key=lambda row: float(row.get("local_cost", 0.0)))
        sequence.append(_strip_optimizer_fields(reference, fields))
    return sequence


def _group_candidates_by_step(candidates: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(int(candidate.get("frame", 0)), []).append(candidate)
    return grouped


def _reference_by_step(reference_sequence: list[dict[str, Any]] | None) -> dict[int, dict[str, Any]]:
    if not reference_sequence:
        return {}
    out: dict[int, dict[str, Any]] = {}
    for step, row in enumerate(reference_sequence):
        out[step] = row
    return out


def _reference_by_source_frame(reference_sequence: list[dict[str, Any]] | None) -> dict[int, dict[str, Any]]:
    if not reference_sequence:
        return {}
    out: dict[int, dict[str, Any]] = {}
    for row in reference_sequence:
        if not isinstance(row, dict):
            continue
        if "source_frame" not in row and "frame" not in row:
            continue
        try:
            frame = int(row.get("source_frame", row.get("frame", 0)))
        except (TypeError, ValueError):
            continue
        out[frame] = row
    return out


def _candidate_signature(row: dict[str, Any], fields: list[str]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in fields)


def _csv_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {part.strip() for part in value.split(",") if part.strip()}
    try:
        return {str(part).strip() for part in value if str(part).strip()}
    except TypeError:
        return {str(value).strip()} if str(value).strip() else set()


def _csv_int_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, str):
        parts = value.split(",")
    else:
        try:
            parts = list(value)
        except TypeError:
            parts = [value]
    out: set[int] = set()
    for part in parts:
        text = str(part).strip()
        if not text:
            continue
        try:
            out.add(int(text))
        except (TypeError, ValueError):
            continue
    return out


_FEATURE_MANIFEST_CACHE: dict[Path, dict[str, Any]] = {}
_LOWER_BODY_ACTIVITY_CACHE: dict[tuple[str, tuple[int, ...]], tuple[np.ndarray, Path, list[int]]] = {}


def _load_feature_manifest(feature_root: Path) -> dict[str, Any]:
    root = Path(feature_root).resolve()
    if root in _FEATURE_MANIFEST_CACHE:
        return _FEATURE_MANIFEST_CACHE[root]
    manifest_path = root.parent / "manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.exists() else {}
    _FEATURE_MANIFEST_CACHE[root] = manifest if isinstance(manifest, dict) else {}
    return _FEATURE_MANIFEST_CACHE[root]


def _feature_song_candidates(source_song_id: str, fallback_name: str) -> list[str]:
    names: list[str] = []
    for raw in (source_song_id, fallback_name):
        text = str(raw or "").strip()
        if text and text not in names:
            names.append(text)
        match = re.search(r"(\d+)", text)
        if match:
            clip_name = f"c{int(match.group(1))}"
            if clip_name not in names:
                names.append(clip_name)
    return names


def _load_lower_body_activity(
    feature_root: Path,
    song_ids: list[str],
    lower_body_joints: set[int],
) -> tuple[np.ndarray | None, Path | None, list[int]]:
    root = Path(feature_root).resolve()
    joints = tuple(sorted(int(joint) for joint in lower_body_joints))
    manifest = _load_feature_manifest(root)
    feature_names = list(manifest.get("dance_feature_names") or [])
    speed_columns: list[int] = []
    for joint in joints:
        name = f"dance_joint_speed_j{joint:02d}"
        if name in feature_names:
            speed_columns.append(feature_names.index(name))
    if not speed_columns:
        return None, None, []

    for song_id in song_ids:
        path = root / f"{song_id}.npz"
        if not path.exists():
            continue
        cache_key = (str(path.resolve()), joints)
        if cache_key in _LOWER_BODY_ACTIVITY_CACHE:
            activity, cached_path, cached_columns = _LOWER_BODY_ACTIVITY_CACHE[cache_key]
            return activity, cached_path, cached_columns
        with np.load(path) as payload:
            dance = np.asarray(payload["dance_features"], dtype=np.float32)
        activity = np.nanmean(dance[:, speed_columns], axis=1)
        _LOWER_BODY_ACTIVITY_CACHE[cache_key] = (activity, path, speed_columns)
        return activity, path, speed_columns
    return None, None, speed_columns


def _window_activity(activity: np.ndarray | None, frame: int, radius: int) -> float | None:
    if activity is None or len(activity) == 0:
        return None
    center = min(max(0, int(frame)), len(activity) - 1)
    start = max(0, center - int(radius))
    end = min(len(activity), center + int(radius) + 1)
    if start >= end:
        return None
    return float(np.nanquantile(activity[start:end], 0.75))


def _finger_window_activity(
    activity: np.ndarray | None,
    frame: int,
    radius: int,
    *,
    mode: str = "any",
) -> float | None:
    if activity is None or len(activity) == 0:
        return None
    center = min(max(0, int(frame)), len(activity) - 1)
    start = max(0, center - int(radius))
    end = min(len(activity), center + int(radius) + 1)
    if start >= end:
        return None
    window = np.asarray(activity[start:end], dtype=np.float32)
    if str(mode) == "q75":
        return float(np.nanquantile(window, 0.75))
    return float(np.nanmax(window))


def _csv_str_list(text: str) -> list[str]:
    return [piece.strip() for piece in str(text or "").split(",") if piece.strip()]


def _load_motion_body_activity_labels(
    activity_root: Path,
    song_ids: list[str],
    part_names: list[str],
) -> tuple[np.ndarray | None, Path | None, list[str]]:
    """Load max hand/arm activity labels for the requested song.

    The NPZ files contain per-frame binary labels and `part_keys`, e.g.
    `left_arm`, `right_arm`, `left_hand`, `right_hand`.
    """

    root = Path(activity_root).resolve()
    cache_key = (str(root), tuple(part_names), *song_ids)
    if cache_key in _MOTION_BODY_LABEL_CACHE:
        return _MOTION_BODY_LABEL_CACHE[cache_key]
    for song_id in song_ids:
        path = root / f"{song_id}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as payload:
            labels = np.asarray(payload["labels"], dtype=np.float32)
            part_keys = [str(value) for value in payload["part_keys"].tolist()]
        columns = [part_keys.index(part) for part in part_names if part in part_keys]
        if not columns:
            result = (None, path, [])
        else:
            activity = np.max(labels[:, columns], axis=1)
            result = (activity.astype(np.float32), path, [part_keys[index] for index in columns])
        _MOTION_BODY_LABEL_CACHE[cache_key] = result
        return result
    result = (None, None, [])
    _MOTION_BODY_LABEL_CACHE[cache_key] = result
    return result


def _load_finger_visibility_thresholds(
    csv_path: Path,
    schema: str,
) -> dict[tuple[str, str], tuple[float, float]]:
    path = Path(csv_path).resolve()
    cache_key = (str(path), str(schema))
    if cache_key in _FINGER_THRESHOLD_CACHE:
        return _FINGER_THRESHOLD_CACHE[cache_key]
    thresholds: dict[tuple[str, str], tuple[float, float]] = {}
    if not path.exists():
        _FINGER_THRESHOLD_CACHE[cache_key] = thresholds
        return thresholds
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("schema", "")) != str(schema):
                continue
            try:
                thresholds[(str(row["distance"]), str(row["fov"]))] = (
                    float(row["suggested_threshold_w_85pct"]),
                    float(row["suggested_threshold_h_85pct"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
    _FINGER_THRESHOLD_CACHE[cache_key] = thresholds
    return thresholds


def _load_finger_dance_features(
    feature_root: Path,
    song_ids: list[str],
) -> tuple[np.ndarray | None, list[str], Path | None]:
    root = Path(feature_root).resolve()
    manifest = _load_feature_manifest(root)
    feature_names = [str(name) for name in manifest.get("dance_feature_names") or []]
    if not feature_names:
        return None, [], None
    for song_id in song_ids:
        path = root / f"{song_id}.npz"
        if not path.exists():
            continue
        cache_key = str(path.resolve())
        if cache_key in _FINGER_DANCE_FEATURE_CACHE:
            dance, cached_feature_names, cached_path = _FINGER_DANCE_FEATURE_CACHE[cache_key]
            return dance, cached_feature_names, cached_path
        with np.load(path, allow_pickle=False) as payload:
            dance = np.asarray(payload["dance_features"], dtype=np.float32)
        _FINGER_DANCE_FEATURE_CACHE[cache_key] = (dance, feature_names, path)
        return dance, feature_names, path
    return None, feature_names, None


def _relpos_columns(feature_names: list[str], joints: tuple[int, ...] | list[int]) -> list[int]:
    by_name = {name: index for index, name in enumerate(feature_names)}
    columns: list[int] = []
    for joint in joints:
        for axis in ("x", "y", "z"):
            name = f"dance_relpos_j{int(joint):02d}_{axis}"
            if name not in by_name:
                return []
            columns.append(by_name[name])
    return columns


def _relpos_at(
    dance_features: np.ndarray,
    feature_names: list[str],
    frame: int,
    joints: tuple[int, ...] | list[int],
) -> np.ndarray | None:
    frame = min(max(int(frame), 0), int(dance_features.shape[0]) - 1)
    columns = _relpos_columns(feature_names, joints)
    if not columns:
        return None
    values = dance_features[frame, columns]
    return values.reshape(len(joints), 3).astype(np.float64)


def _normalize_xz(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    xz = np.asarray([float(vector[0]), float(vector[2])], dtype=np.float64)
    norm = float(np.linalg.norm(xz))
    if norm < 1e-6:
        return np.asarray(fallback, dtype=np.float64)
    return xz / norm


def _finger_camera_right_axis(
    dance_features: np.ndarray,
    feature_names: list[str],
    frame: int,
    yaw: str,
) -> np.ndarray:
    shoulders = _relpos_at(dance_features, feature_names, frame, [17, 39])
    if shoulders is None:
        return np.asarray([1.0, 0.0], dtype=np.float64)
    left_shoulder, right_shoulder = shoulders
    body_right = _normalize_xz(right_shoulder - left_shoulder, np.asarray([1.0, 0.0]))
    body_forward = np.asarray([body_right[1], -body_right[0]], dtype=np.float64)
    theta = math.radians(float(FINGER_VISIBILITY_YAW_DEG.get(str(yaw), 0.0)))
    axis = body_right * math.cos(theta) + body_forward * math.sin(theta)
    norm = float(np.linalg.norm(axis))
    return axis / norm if norm > 1e-6 else body_right


def _finger_visibility_score_for_candidate(
    candidate: dict[str, Any],
    dance_features: np.ndarray,
    feature_names: list[str],
    thresholds: dict[tuple[str, str], tuple[float, float]],
    *,
    axis_mode: str,
    close_distances: set[str],
) -> float | None:
    distance = str(candidate.get("distance", "medium"))
    fov = str(candidate.get("fov", "normal"))
    if distance not in close_distances:
        return 1.0
    if not thresholds:
        return None
    threshold = thresholds.get((distance, fov)) or thresholds.get((distance, "normal"))
    if threshold is None:
        return None
    threshold_w, threshold_h = threshold
    frame = int(candidate.get("source_frame", candidate.get("frame", 0)))
    anchor_joints = FINGER_VISIBILITY_ANCHOR_JOINTS.get(
        str(candidate.get("target_anchor", "waist")),
        FINGER_VISIBILITY_ANCHOR_JOINTS["waist"],
    )
    anchor_points = _relpos_at(dance_features, feature_names, frame, anchor_joints)
    tips = _relpos_at(dance_features, feature_names, frame, list(FINGER_VISIBILITY_FINGERTIP_JOINTS))
    if anchor_points is None or tips is None:
        return None
    anchor = anchor_points.mean(axis=0)
    offsets = tips - anchor.reshape(1, 3)
    dy = np.abs(offsets[:, 1])
    if axis_mode == "world_x":
        horiz = np.abs(offsets[:, 0])
    elif axis_mode == "xz_radius":
        horiz = np.sqrt(offsets[:, 0] ** 2 + offsets[:, 2] ** 2)
    else:
        axis = _finger_camera_right_axis(
            dance_features,
            feature_names,
            frame,
            str(candidate.get("yaw", "front")),
        )
        horiz = np.abs(offsets[:, 0] * axis[0] + offsets[:, 2] * axis[1])
    visible = (horiz <= float(threshold_w)) & (dy <= float(threshold_h))
    return float(np.mean(visible))


def _apply_finger_activity_visibility_cost(
    candidates: list[dict[str, Any]],
    *,
    source_song_id: str,
    fallback_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Bias CP-SAT away from active-hand candidates that hide fingers.

    This is intentionally a local cost term, not a hard constraint. It should
    preserve the base model distribution while nudging close shots toward
    nearby top-k alternatives that keep active hands/fingers visible.
    """

    cost = float(args.finger_activity_visibility_cost)
    debug: dict[str, Any] = {
        "enabled": cost > 0,
        "penalized_candidate_count": 0,
        "penalized_source_frame_count": 0,
        "penalized_source_frames": [],
        "total_added_cost": 0.0,
    }
    if cost <= 0:
        return debug

    song_ids = _feature_song_candidates(source_song_id, fallback_name)
    activity_parts = _csv_str_list(str(args.finger_activity_parts))
    activity, activity_path, loaded_parts = _load_motion_body_activity_labels(
        Path(args.finger_activity_root),
        song_ids,
        activity_parts,
    )
    dance_features, feature_names, feature_path = _load_finger_dance_features(
        Path(args.finger_activity_feature_root),
        song_ids,
    )
    thresholds = _load_finger_visibility_thresholds(
        Path(args.finger_activity_threshold_csv),
        str(args.finger_activity_threshold_schema),
    )
    debug.update(
        {
            "song_candidates": song_ids,
            "activity_path": str(activity_path) if activity_path else None,
            "activity_parts": loaded_parts,
            "feature_path": str(feature_path) if feature_path else None,
            "threshold_csv": str(args.finger_activity_threshold_csv),
            "threshold_schema": str(args.finger_activity_threshold_schema),
            "axis_mode": str(args.finger_activity_axis_mode),
            "visible_threshold": float(args.finger_activity_visible_threshold),
            "activity_min": float(args.finger_activity_min),
            "window_radius": int(args.finger_activity_window_radius),
            "window_mode": str(getattr(args, "finger_activity_window_mode", "any")),
            "close_distances": _csv_str_list(str(args.finger_activity_close_distances)),
        }
    )
    if activity is None or dance_features is None or not feature_names or not thresholds:
        debug["reason"] = "missing-activity-or-feature-or-thresholds"
        return debug

    close_distances = set(_csv_str_list(str(args.finger_activity_close_distances))) or set(
        FINGER_VISIBILITY_CLOSE_DISTANCES
    )
    visible_threshold = float(args.finger_activity_visible_threshold)
    activity_min = float(args.finger_activity_min)
    penalized_source_frames: set[int] = set()
    for candidate in candidates:
        source_frame = int(candidate.get("source_frame", candidate.get("frame", 0)))
        active = _finger_window_activity(
            activity,
            source_frame,
            int(args.finger_activity_window_radius),
            mode=str(getattr(args, "finger_activity_window_mode", "any")),
        )
        if active is None or active < activity_min:
            continue
        score = _finger_visibility_score_for_candidate(
            candidate,
            dance_features,
            feature_names,
            thresholds,
            axis_mode=str(args.finger_activity_axis_mode),
            close_distances=close_distances,
        )
        if score is None or score >= visible_threshold:
            continue
        normalized_missing = (visible_threshold - score) / max(1e-6, visible_threshold)
        penalty = cost * float(normalized_missing)
        candidate["local_cost"] = float(candidate.get("local_cost", 0.0)) + penalty
        candidate["finger_activity_visibility_penalty"] = penalty
        candidate["finger_activity_score"] = score
        candidate["finger_activity_value"] = float(active)
        penalized_source_frames.add(source_frame)
        debug["penalized_candidate_count"] += 1
        debug["total_added_cost"] = float(debug["total_added_cost"]) + penalty
    debug["penalized_source_frames"] = sorted(penalized_source_frames)
    debug["penalized_source_frame_count"] = len(penalized_source_frames)
    return debug


def _finger_activity_visibility_trigger_frames(
    candidates: list[dict[str, Any]],
    reference_sequence: list[dict[str, Any]] | None,
    *,
    source_song_id: str,
    fallback_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Return source frames where a reference close shot likely crops active fingers."""

    debug: dict[str, Any] = {
        "enabled": bool(args.distance_fov_only_finger_triggered_only),
        "frames": [],
        "frame_count": 0,
        "trigger_source_distances": _csv_str_list(str(args.finger_activity_trigger_source_distances)),
        "trigger_target_distances": _csv_str_list(str(args.finger_activity_trigger_target_distances)),
    }
    if not bool(args.distance_fov_only_finger_triggered_only):
        debug["reason"] = "disabled"
        return debug

    song_ids = _feature_song_candidates(source_song_id, fallback_name)
    activity_parts = _csv_str_list(str(args.finger_activity_parts))
    activity, activity_path, loaded_parts = _load_motion_body_activity_labels(
        Path(args.finger_activity_root),
        song_ids,
        activity_parts,
    )
    dance_features, feature_names, feature_path = _load_finger_dance_features(
        Path(args.finger_activity_feature_root),
        song_ids,
    )
    thresholds = _load_finger_visibility_thresholds(
        Path(args.finger_activity_threshold_csv),
        str(args.finger_activity_threshold_schema),
    )
    debug.update(
        {
            "song_candidates": song_ids,
            "activity_path": str(activity_path) if activity_path else None,
            "activity_parts": loaded_parts,
            "feature_path": str(feature_path) if feature_path else None,
            "threshold_csv": str(args.finger_activity_threshold_csv),
            "threshold_schema": str(args.finger_activity_threshold_schema),
            "axis_mode": str(args.finger_activity_axis_mode),
            "visible_threshold": float(args.finger_activity_visible_threshold),
            "activity_min": float(args.finger_activity_min),
            "window_radius": int(args.finger_activity_window_radius),
            "window_mode": str(getattr(args, "finger_activity_window_mode", "any")),
            "close_distances": _csv_str_list(str(args.finger_activity_close_distances)),
        }
    )
    if activity is None or dance_features is None or not feature_names or not thresholds:
        debug["reason"] = "missing-activity-or-feature-or-thresholds"
        return debug

    trigger_source_distances = _csv_set(args.finger_activity_trigger_source_distances)
    if not trigger_source_distances:
        trigger_source_distances = {"very_close", "close"}
    close_distances = set(_csv_str_list(str(args.finger_activity_close_distances))) or set(
        FINGER_VISIBILITY_CLOSE_DISTANCES
    )
    visible_threshold = float(args.finger_activity_visible_threshold)
    activity_min = float(args.finger_activity_min)
    references = _reference_by_step(reference_sequence)
    frames: set[int] = set()
    skipped_non_source_distance = 0
    inactive = 0
    visible_enough = 0
    missing_score = 0

    for step, group in sorted(_group_candidates_by_step(candidates).items()):
        if not group:
            continue
        reference = references.get(step)
        if reference is None:
            top1_rows = [row for row in group if int(row.get("rank", 0)) == 1]
            reference = top1_rows[0] if top1_rows else min(group, key=lambda row: float(row.get("local_cost", 0.0)))
        ref_distance = str(reference.get("distance", "medium"))
        if ref_distance not in trigger_source_distances:
            skipped_non_source_distance += 1
            continue
        source_frame = int(group[0].get("source_frame", reference.get("frame", 0)))
        active = _finger_window_activity(
            activity,
            source_frame,
            int(args.finger_activity_window_radius),
            mode=str(getattr(args, "finger_activity_window_mode", "any")),
        )
        if active is None or active < activity_min:
            inactive += 1
            continue
        probe = dict(reference)
        probe["source_frame"] = source_frame
        score = _finger_visibility_score_for_candidate(
            probe,
            dance_features,
            feature_names,
            thresholds,
            axis_mode=str(args.finger_activity_axis_mode),
            close_distances=close_distances,
        )
        if score is None:
            missing_score += 1
            continue
        if score >= visible_threshold:
            visible_enough += 1
            continue
        frames.add(source_frame)

    debug.update(
        {
            "frames": sorted(frames),
            "frame_count": len(frames),
            "skipped_non_source_distance": skipped_non_source_distance,
            "inactive_frame_count": inactive,
            "visible_enough_frame_count": visible_enough,
            "missing_score_frame_count": missing_score,
            "reason": "ok",
        }
    )
    return debug


def _duration_stat_bound(stats: dict[str, Any], key: str) -> float | None:
    value = stats.get(str(key))
    if value is None and str(key) != "max_sec":
        value = stats.get("max_sec")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _run_duration_end_frame(frames: list[int], end_index: int, duration_end_mode: str) -> float:
    mode = str(duration_end_mode or "next_change").strip().lower()
    if end_index + 1 < len(frames):
        if mode == "midpoint":
            return (float(frames[end_index]) + float(frames[end_index + 1])) * 0.5
        return float(frames[end_index + 1])
    return float(frames[end_index])


def _run_duration_sec(frames: list[int], start: int, end_index: int, fps: float, duration_end_mode: str) -> float:
    if float(fps) <= 0:
        return 0.0
    return max(
        0.0,
        float(_run_duration_end_frame(frames, end_index, str(duration_end_mode)) - float(frames[start]))
        / float(fps),
    )


def _distance_level_duration_classes(distance_level_duration_stats: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(distance_level_duration_stats, dict):
        return {}
    classes = distance_level_duration_stats.get("classes", distance_level_duration_stats)
    if not isinstance(classes, dict):
        return {}
    return {str(label): value for label, value in classes.items() if isinstance(value, dict)}


def _eligible_close_activity_distance_frames(
    reference_sequence: list[dict[str, Any]] | None,
    *,
    distance_level_duration_stats: dict[str, Any] | None,
    fps: float,
    close_levels: Any,
    upper_key: str,
    feature_root: Path,
    feature_song_ids: list[str],
    lower_body_joints: set[int],
    lower_body_activity_min: float,
    activity_window_radius: int,
    min_next_frame_gap: int,
    duration_end_mode: str,
    require_lower_body_activity: bool,
) -> dict[str, Any]:
    close_level_set = _csv_set(close_levels)
    classes = _distance_level_duration_classes(distance_level_duration_stats)
    if not reference_sequence or not classes or not close_level_set or float(fps) <= 0:
        return {"frames": [], "events": [], "reason": "missing-reference-or-stats"}

    activity, feature_path, activity_columns = _load_lower_body_activity(
        feature_root,
        feature_song_ids,
        lower_body_joints,
    )
    if require_lower_body_activity and activity is None:
        return {
            "frames": [],
            "events": [],
            "reason": "missing-lower-body-activity",
            "feature_song_ids": feature_song_ids,
            "activity_columns": activity_columns,
        }

    frames = [int(row.get("source_frame", row.get("frame", 0))) for row in reference_sequence]
    levels = [distance_fov_level_label(row) for row in reference_sequence]
    allowed_frames: set[int] = set()
    events: list[dict[str, Any]] = []
    start = 0
    for index in range(1, len(reference_sequence) + 1):
        changed = index == len(reference_sequence) or levels[index] != levels[start]
        if not changed:
            continue
        end_index = index - 1
        level = levels[start]
        if level not in close_level_set:
            start = index
            continue
        stats = classes.get(level, {})
        upper_sec = _duration_stat_bound(stats, str(upper_key))
        if upper_sec is None:
            start = index
            continue
        end_frame_exclusive = _run_duration_end_frame(frames, end_index, str(duration_end_mode))
        duration_sec = _run_duration_sec(frames, start, end_index, float(fps), str(duration_end_mode))
        if duration_sec <= float(upper_sec):
            start = index
            continue

        frame_entries: list[dict[str, Any]] = []
        for run_index in range(start, end_index + 1):
            if run_index + 1 >= len(reference_sequence):
                continue
            frame = int(frames[run_index])
            next_gap = int(frames[run_index + 1]) - frame
            if next_gap < int(min_next_frame_gap):
                continue
            activity_value = _window_activity(activity, frame, int(activity_window_radius))
            if require_lower_body_activity and (
                activity_value is None or activity_value < float(lower_body_activity_min)
            ):
                continue
            allowed_frames.add(frame)
            frame_entries.append(
                {
                    "index": int(run_index),
                    "frame": frame,
                    "next_frame_gap": int(next_gap),
                    "lower_body_activity": activity_value,
                }
            )
        if frame_entries:
            events.append(
                {
                    "level": level,
                    "start_index": int(start),
                    "end_index": int(end_index),
                    "frames": [int(frames[start]), int(frames[end_index])],
                    "end_frame_exclusive": float(end_frame_exclusive),
                    "duration_sec": float(duration_sec),
                    "upper_sec": float(upper_sec),
                    "upper_key": str(upper_key),
                    "eligible_frames": frame_entries,
                }
            )
        start = index

    return {
        "frames": sorted(allowed_frames),
        "events": events,
        "reason": "ok",
        "feature_path": str(feature_path) if feature_path else None,
        "feature_song_ids": feature_song_ids,
        "activity_columns": activity_columns,
    }


def _augment_distance_fov_only_candidates(
    candidates: list[dict[str, Any]],
    reference_sequence: list[dict[str, Any]] | None,
    *,
    fields: list[str],
    cost_penalty: float,
    radius: int,
    max_per_step: int,
    directional_escape: bool = False,
    direction_close_levels: Any = None,
    direction_medium_level: str = "medium",
    direction_far_levels: Any = None,
    exclude_source_levels: Any = None,
    allowed_source_frames: Any = None,
    target_distance_values: Any = None,
) -> int:
    """Add synthetic candidates that keep the reference shot but swap distance/FOV.

    This gives duration objectives a clean escape hatch when the model's top-k
    alternatives change anchor/yaw/composition along with distance.
    """
    if not reference_sequence or int(max_per_step) <= 0:
        return 0
    grouped = _group_candidates_by_step(candidates)
    references = _reference_by_step(reference_sequence)
    allowed_source_frame_set = None if allowed_source_frames is None else _csv_int_set(allowed_source_frames)
    added = 0
    for step, group in grouped.items():
        reference = references.get(step)
        if not reference:
            continue
        source_frame = int(group[0].get("source_frame", reference.get("frame", 0)))
        if allowed_source_frame_set is not None and source_frame not in allowed_source_frame_set:
            continue
        slot_index = int(group[0].get("slot_index", 0))
        existing = {_candidate_signature(row, fields) for row in group}
        ref_distance = str(reference.get("distance", "medium"))
        ref_fov = str(reference.get("fov", "normal"))
        ref_level = distance_fov_level_label(reference)
        if ref_level in _csv_set(exclude_source_levels):
            continue
        medium_level_index = distance_fov_level_index(str(direction_medium_level or "medium"))
        close_levels = _csv_set(direction_close_levels)
        far_levels = _csv_set(direction_far_levels)
        pairs: list[tuple[str, str]] = []
        target_distances = _csv_str_list(str(target_distance_values or ""))
        if target_distances:
            for distance in target_distances:
                if str(distance) != ref_distance:
                    pairs.append((str(distance), ref_fov))
        else:
            for candidate in group:
                distance = str(candidate.get("distance", ref_distance))
                fov = str(candidate.get("fov", ref_fov))
                if (distance, fov) != (ref_distance, ref_fov):
                    pairs.append((distance, fov))
            for distance in adjacent_distance_values(ref_distance, radius=radius):
                pairs.append((distance, ref_fov))
            for fov in adjacent_fov_values(ref_fov, radius=radius):
                pairs.append((ref_distance, fov))
        best_cost = min(float(row.get("local_cost", 0.0)) for row in group) if group else 0.0
        per_step_added = 0
        seen_pairs: set[tuple[str, str]] = set()
        for distance, fov in pairs:
            pair = (str(distance), str(fov))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            synthetic = dict(reference)
            synthetic["frame"] = step
            synthetic["source_frame"] = source_frame
            synthetic["slot_index"] = slot_index
            synthetic["distance"] = pair[0]
            synthetic["fov"] = pair[1]
            for key, value in group[0].items():
                if key not in synthetic and key not in fields:
                    synthetic[key] = value
            synthetic["rank"] = int(1000 + per_step_added)
            synthetic["model_logprob"] = -float(best_cost + float(cost_penalty))
            synthetic["model_prob"] = 0.0
            synthetic["local_cost"] = float(best_cost + float(cost_penalty))
            synthetic["synthetic_distance_fov_only"] = True
            synthetic["synthetic_distance_fov_level"] = distance_fov_level_label(synthetic)
            if bool(directional_escape) and ref_level in close_levels:
                synthetic_level_index = distance_fov_level_index(synthetic["synthetic_distance_fov_level"])
                if synthetic_level_index < medium_level_index:
                    continue
            if bool(directional_escape) and ref_level in far_levels:
                synthetic_level_index = distance_fov_level_index(synthetic["synthetic_distance_fov_level"])
                if synthetic_level_index > medium_level_index:
                    continue
            signature = _candidate_signature(synthetic, fields)
            if signature in existing:
                continue
            candidates.append(synthetic)
            existing.add(signature)
            per_step_added += 1
            added += 1
            if per_step_added >= int(max_per_step):
                break
    return added


def _augment_bodyproxy_safe_target_anchor_candidates(
    candidates: list[dict[str, Any]],
    reference_sequence: list[dict[str, Any]] | None,
    *,
    fields: list[str],
    cost_penalty: float,
    target_anchor_values: Any,
    max_per_step: int,
    proxy_mode: str,
) -> dict[str, Any]:
    """Add target-anchor-only synthetic candidates that preserve body-proxy coverage.

    A synthetic target_anchor candidate is allowed only when the body parts
    predicted visible by the reference shot remain visible after changing the
    anchor. In set terms:

        visible_bodyparts(new) >= visible_bodyparts(reference)

    This gives the optimizer room to shift the anchor for hand/finger recovery
    without accidentally losing body parts that the original shot plan already
    kept in frame.
    """

    debug: dict[str, Any] = {
        "enabled": bool(reference_sequence) and int(max_per_step) > 0,
        "added": 0,
        "rejected_lost_bodyparts": 0,
        "tried": 0,
        "proxy_mode": str(proxy_mode),
    }
    if not reference_sequence or int(max_per_step) <= 0:
        return debug

    grouped = _group_candidates_by_step(candidates)
    references = _reference_by_step(reference_sequence)
    anchor_values = [value for value in _csv_str_list(str(target_anchor_values)) if value]
    added = 0
    rejected_lost = 0
    tried = 0
    for step, group in grouped.items():
        reference = references.get(step)
        if not reference or not group:
            continue
        source_frame = int(group[0].get("source_frame", reference.get("frame", 0)))
        slot_index = int(group[0].get("slot_index", 0))
        existing = {_candidate_signature(row, fields) for row in group}
        ref_anchor = str(reference.get("target_anchor", "waist"))
        ref_visible = set(visible_bodyparts(reference, proxy_mode=proxy_mode))
        best_cost = min(float(row.get("local_cost", 0.0)) for row in group)
        per_step_added = 0
        for anchor in anchor_values:
            anchor = str(anchor).strip()
            if not anchor or anchor == ref_anchor:
                continue
            tried += 1
            synthetic = dict(reference)
            synthetic["frame"] = step
            synthetic["source_frame"] = source_frame
            synthetic["slot_index"] = slot_index
            synthetic["target_anchor"] = anchor
            for key, value in group[0].items():
                if key not in synthetic and key not in fields:
                    synthetic[key] = value
            new_visible = set(visible_bodyparts(synthetic, proxy_mode=proxy_mode))
            lost_parts = sorted(ref_visible - new_visible)
            if lost_parts:
                rejected_lost += 1
                continue
            synthetic["rank"] = int(3000 + per_step_added)
            synthetic["model_logprob"] = -float(best_cost + float(cost_penalty))
            synthetic["model_prob"] = 0.0
            synthetic["local_cost"] = float(best_cost + float(cost_penalty))
            synthetic["synthetic_bodyproxy_safe_target_anchor"] = True
            synthetic["bodyproxy_reference_visible"] = sorted(ref_visible)
            synthetic["bodyproxy_candidate_visible"] = sorted(new_visible)
            signature = _candidate_signature(synthetic, fields)
            if signature in existing:
                continue
            candidates.append(synthetic)
            existing.add(signature)
            per_step_added += 1
            added += 1
            if per_step_added >= int(max_per_step):
                break

    debug["added"] = int(added)
    debug["rejected_lost_bodyparts"] = int(rejected_lost)
    debug["tried"] = int(tried)
    return debug


def _augment_top_pitch_escape_candidates(
    candidates: list[dict[str, Any]],
    reference_sequence: list[dict[str, Any]] | None,
    *,
    fields: list[str],
    cost_penalty: float,
    escape_pitches: Any = None,
    max_per_step: int = 2,
) -> int:
    """Add synthetic candidates that keep the reference shot but lower pitch from top."""
    if not reference_sequence or int(max_per_step) <= 0:
        return 0
    grouped = _group_candidates_by_step(candidates)
    references = _reference_by_step(reference_sequence)
    escape_pitch_values = list(_csv_set(escape_pitches or "high"))
    added = 0
    for step, group in grouped.items():
        reference = references.get(step)
        if not reference or str(reference.get("pitch", "level")).strip() != "top":
            continue
        source_frame = int(group[0].get("source_frame", reference.get("frame", 0)))
        slot_index = int(group[0].get("slot_index", 0))
        existing = {_candidate_signature(row, fields) for row in group}
        best_cost = min(float(row.get("local_cost", 0.0)) for row in group) if group else 0.0
        per_step_added = 0
        for pitch in escape_pitch_values:
            pitch_value = str(pitch).strip()
            if not pitch_value or pitch_value == "top":
                continue
            synthetic = dict(reference)
            synthetic["frame"] = step
            synthetic["source_frame"] = source_frame
            synthetic["slot_index"] = slot_index
            synthetic["pitch"] = pitch_value
            for key, value in group[0].items():
                if key not in synthetic and key not in fields:
                    synthetic[key] = value
            synthetic["rank"] = int(2000 + per_step_added)
            synthetic["model_logprob"] = -float(best_cost + float(cost_penalty))
            synthetic["model_prob"] = 0.0
            synthetic["local_cost"] = float(best_cost + float(cost_penalty))
            synthetic["synthetic_top_pitch_escape"] = True
            signature = _candidate_signature(synthetic, fields)
            if signature in existing:
                continue
            candidates.append(synthetic)
            existing.add(signature)
            per_step_added += 1
            added += 1
            if per_step_added >= int(max_per_step):
                break
    return added


def _filter_new_top_pitch_candidates(
    candidates: list[dict[str, Any]],
    reference_sequence: list[dict[str, Any]] | None,
) -> int:
    """Drop candidates that create a new top pitch where the reference was not top."""
    if not reference_sequence:
        return 0
    grouped = _group_candidates_by_step(candidates)
    references = _reference_by_step(reference_sequence)
    references_by_source_frame = _reference_by_source_frame(reference_sequence)
    remove_ids: set[int] = set()
    for step, group in grouped.items():
        source_frame = int(group[0].get("source_frame", group[0].get("frame", step))) if group else step
        reference = references_by_source_frame.get(source_frame) or references.get(step)
        if not reference or str(reference.get("pitch", "level")).strip() == "top":
            continue
        keep = [row for row in group if str(row.get("pitch", "level")).strip() != "top"]
        if not keep:
            continue
        for row in group:
            if str(row.get("pitch", "level")).strip() == "top":
                remove_ids.add(id(row))
    if not remove_ids:
        return 0
    before = len(candidates)
    candidates[:] = [row for row in candidates if id(row) not in remove_ids]
    return before - len(candidates)


def _change_summary(base_rows: list[dict[str, Any]], optimized_rows: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    base_by_frame = {int(row.get("frame", 0)): row for row in base_rows if isinstance(row, dict) and "frame" in row}
    changed_rows: list[dict[str, Any]] = []
    field_counts = {field: 0 for field in fields}
    for row in optimized_rows:
        frame = int(row.get("frame", 0))
        base = base_by_frame.get(frame)
        if base is None:
            continue
        changes = {
            field: {"from": base.get(field), "to": row.get(field)}
            for field in fields
            if str(base.get(field)) != str(row.get(field))
        }
        if not changes:
            continue
        for field in changes:
            field_counts[field] += 1
        changed_rows.append({"frame": frame, "changes": changes})
    denom = max(1, len([row for row in optimized_rows if int(row.get("frame", 0)) in base_by_frame]))
    return {
        "base_frame_count": len(base_by_frame),
        "optimized_frame_count": len(optimized_rows),
        "changed_keyframe_count": len(changed_rows),
        "changed_keyframe_rate": float(len(changed_rows) / denom),
        "field_change_count": int(sum(field_counts.values())),
        "field_change_counts": field_counts,
        "changed_keyframes": changed_rows,
    }


def _optimize_candidate_path(
    *,
    candidate_path: Path,
    gt_root: Path,
    gt_name: str,
    schema: str,
    top_k: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = _load_json(candidate_path)
    schema_name = _infer_schema_name(payload, schema)
    optimized_fields = fields_for_schema(schema_name)
    optimizer_candidates, step_meta = _candidate_sequence_from_topk(payload, int(top_k))
    if not optimizer_candidates:
        return {"candidate_path": str(candidate_path), "status": "empty-candidates"}

    source_song_id = str(payload.get("source_song_id", candidate_path.parent.name))
    finger_activity_visibility_debug: dict[str, Any] = {
        "enabled": float(args.finger_activity_visibility_cost) > 0,
        "penalized_candidate_count": 0,
        "total_added_cost": 0.0,
        "reason": "pending-until-after-synthetic-candidates",
    }
    gt_path = _resolve_gt_path(
        gt_root=gt_root,
        song_id=source_song_id,
        gt_name=gt_name,
        candidate_payload=payload,
    )
    source_frames = [int(meta["source_frame"]) for meta in step_meta]
    min_frame = min(source_frames)
    max_frame = max(source_frames)
    if gt_path.exists():
        gt_full = _payload_keyframes(_load_json(gt_path))
        gt_used = _crop_gt_sequence(gt_full, min_frame, max_frame)
        if not gt_used:
            gt_used = gt_full
        gt_stats_fallback = ""
    elif bool(args.allow_missing_gt) and bool(args.disable_sequence_stat_objective):
        gt_full = [
            _strip_optimizer_fields(candidate, optimized_fields)
            for candidate in optimizer_candidates
            if int(candidate.get("rank", 0)) == 1
        ]
        gt_used = list(gt_full)
        gt_stats_fallback = "top1_candidates"
    else:
        return {"candidate_path": str(candidate_path), "status": "missing-gt", "gt_path": str(gt_path)}

    weights = None
    if bool(args.disable_sequence_stat_objective):
        weights = {
            "marginal": {field: 0 for field in optimized_fields},
            "transition": {field: 0 for field in optimized_fields},
            "change": {field: 0 for field in optimized_fields},
            "local": 1,
        }

    turn_yaw_heading_frames: list[int] | None = None
    turn_yaw_heading_values: list[float] | None = None
    turn_yaw_targets_path: Path | None = None
    if int(args.turn_yaw_jitter_weight) > 0:
        turn_yaw_targets_path = Path(args.turn_yaw_jitter_targets_root) / source_song_id / "real_targets.json"
        if turn_yaw_targets_path.exists():
            turn_yaw_heading_frames, turn_yaw_heading_values = _load_heading_track(turn_yaw_targets_path)

    reference_path: Path | None = None
    reference_sequence: list[dict[str, Any]] | None = None
    reference_source = ""
    source_plan_sequence = _source_plan_rows(payload)
    needs_reference_plan = (
        int(args.reference_change_weight) > 0
        or bool(args.add_bodyproxy_safe_target_anchor_candidates)
        or bool(args.add_distance_fov_only_candidates)
        or bool(args.add_top_pitch_escape_candidates)
        or bool(args.distance_level_change_only_eligible)
        or bool(args.disallow_new_top_pitch_candidates)
    )
    if args.reference_plan_root and needs_reference_plan:
        source_root = Path(args.source_root).resolve()
        reference_candidate_path = Path(args.reference_plan_root).resolve() / candidate_path.relative_to(source_root)
        reference_path = _output_path_for_candidate(reference_candidate_path)
        if reference_path.exists():
            reference_sequence = _payload_keyframes(_load_json(reference_path))
            reference_source = "reference_plan_root"
    if needs_reference_plan and not reference_sequence:
        reference_sequence = _top1_reference_sequence(optimizer_candidates, optimized_fields)
        reference_source = "top1_candidates"

    new_top_pitch_reference_sequence = source_plan_sequence or reference_sequence
    new_top_pitch_reference_source = (
        "source_plan"
        if source_plan_sequence
        else (reference_source if reference_sequence else "")
    )

    bodypart_duration_stats = _load_json(args.bodypart_duration_stats) if args.bodypart_duration_stats else None
    distance_level_duration_stats = (
        _load_json(args.distance_level_duration_stats) if args.distance_level_duration_stats else None
    )
    distance_level_eligible_debug: dict[str, Any] = {"frames": [], "events": [], "reason": "disabled"}
    if bool(args.distance_level_change_only_eligible):
        feature_song_ids = _feature_song_candidates(source_song_id, candidate_path.parent.name)
        distance_level_eligible_debug = _eligible_close_activity_distance_frames(
            reference_sequence,
            distance_level_duration_stats=distance_level_duration_stats,
            fps=float(args.distance_level_duration_fps),
            close_levels=str(args.distance_level_eligible_close_levels),
            upper_key=str(args.distance_level_eligible_upper_key),
            feature_root=Path(args.distance_level_activity_feature_root),
            feature_song_ids=feature_song_ids,
            lower_body_joints=_csv_int_set(args.distance_level_lower_body_joints),
            lower_body_activity_min=float(args.distance_level_lower_body_activity_min),
            activity_window_radius=int(args.distance_level_activity_window_radius),
            min_next_frame_gap=int(args.distance_level_eligible_min_next_frame_gap),
            duration_end_mode=str(args.distance_level_duration_end_mode),
            require_lower_body_activity=bool(args.distance_level_eligible_require_lower_body_activity),
        )
    distance_change_allowed_frames = (
        distance_level_eligible_debug.get("frames", [])
        if bool(args.distance_level_change_only_eligible)
        else None
    )
    finger_activity_trigger_debug: dict[str, Any] = {
        "enabled": bool(args.distance_fov_only_finger_triggered_only),
        "frames": [],
        "frame_count": 0,
        "reason": "pending",
    }

    synthetic_bodyproxy_safe_target_anchor_debug: dict[str, Any] = {"enabled": False, "added": 0}
    if bool(args.add_bodyproxy_safe_target_anchor_candidates) and reference_sequence:
        synthetic_bodyproxy_safe_target_anchor_debug = _augment_bodyproxy_safe_target_anchor_candidates(
            optimizer_candidates,
            reference_sequence,
            fields=optimized_fields,
            cost_penalty=float(args.bodyproxy_safe_target_anchor_cost),
            target_anchor_values=str(args.bodyproxy_safe_target_anchor_values),
            max_per_step=int(args.bodyproxy_safe_target_anchor_max_per_step),
            proxy_mode=str(args.bodyproxy_safe_target_anchor_proxy_mode),
        )

    synthetic_distance_fov_only_added = 0
    if bool(args.add_distance_fov_only_candidates) and reference_sequence:
        distance_fov_allowed_frames = distance_change_allowed_frames
        distance_fov_target_distances = None
        distance_fov_reference_sequence = reference_sequence
        if bool(args.distance_fov_only_finger_triggered_only):
            distance_fov_reference_sequence = source_plan_sequence or reference_sequence
            finger_activity_trigger_debug["reference_source"] = (
                "source_plan" if source_plan_sequence else reference_source
            )
            finger_activity_trigger_debug = _finger_activity_visibility_trigger_frames(
                optimizer_candidates,
                distance_fov_reference_sequence,
                source_song_id=source_song_id,
                fallback_name=candidate_path.parent.name,
                args=args,
            )
            finger_activity_trigger_debug["reference_source"] = (
                "source_plan" if source_plan_sequence else reference_source
            )
            trigger_frames = set(int(frame) for frame in finger_activity_trigger_debug.get("frames", []))
            if distance_fov_allowed_frames is not None:
                allowed_frames = set(int(frame) for frame in distance_fov_allowed_frames)
                trigger_frames &= allowed_frames
            distance_fov_allowed_frames = sorted(trigger_frames)
            distance_fov_target_distances = str(args.finger_activity_trigger_target_distances)
        synthetic_distance_fov_only_added = _augment_distance_fov_only_candidates(
            optimizer_candidates,
            distance_fov_reference_sequence,
            fields=optimized_fields,
            cost_penalty=float(args.distance_fov_only_candidate_cost),
            radius=int(args.distance_fov_only_candidate_radius),
            max_per_step=int(args.distance_fov_only_candidate_max_per_step),
            directional_escape=bool(args.distance_fov_only_directional_escape),
            direction_close_levels=str(args.distance_level_direction_close_levels),
            direction_medium_level=str(args.distance_level_direction_medium_level),
            direction_far_levels=str(args.distance_level_direction_far_levels),
            exclude_source_levels=str(args.distance_fov_only_exclude_source_levels),
            allowed_source_frames=distance_fov_allowed_frames,
            target_distance_values=distance_fov_target_distances,
        )

    new_top_pitch_candidates_removed = 0
    if bool(args.disallow_new_top_pitch_candidates) and new_top_pitch_reference_sequence:
        new_top_pitch_candidates_removed = _filter_new_top_pitch_candidates(
            optimizer_candidates,
            new_top_pitch_reference_sequence,
        )

    synthetic_top_pitch_escape_added = 0
    if bool(args.add_top_pitch_escape_candidates) and reference_sequence:
        synthetic_top_pitch_escape_added = _augment_top_pitch_escape_candidates(
            optimizer_candidates,
            reference_sequence,
            fields=optimized_fields,
            cost_penalty=float(args.top_pitch_escape_candidate_cost),
            escape_pitches=str(args.top_pitch_escape_candidate_pitches),
            max_per_step=int(args.top_pitch_escape_candidate_max_per_step),
        )

    finger_activity_visibility_debug = _apply_finger_activity_visibility_cost(
        optimizer_candidates,
        source_song_id=source_song_id,
        fallback_name=candidate_path.parent.name,
        args=args,
    )

    result = optimize_shot_sequence_cp_sat(
        optimizer_candidates,
        gt_used,
        fields=optimized_fields,
        weights=weights,
        max_time_sec=float(args.max_time_sec),
        local_cost_scale=int(args.local_cost_scale),
        distance_bounce_weight=int(args.distance_bounce_weight),
        distance_bounce_min_rank_delta=int(args.distance_bounce_min_rank_delta),
        distance_bounce_max_frame_gap=int(args.distance_bounce_max_frame_gap),
        composition_bounce_weight=int(args.composition_bounce_weight),
        composition_bounce_min_rank_delta=int(args.composition_bounce_min_rank_delta),
        composition_bounce_max_frame_gap=int(args.composition_bounce_max_frame_gap),
        vertical_bounce_weight=int(args.vertical_bounce_weight),
        vertical_bounce_min_rank_delta=int(args.vertical_bounce_min_rank_delta),
        vertical_bounce_max_frame_gap=int(args.vertical_bounce_max_frame_gap),
        top_pitch_bounce_weight=int(args.top_pitch_bounce_weight),
        top_pitch_bounce_max_frame_gap=int(args.top_pitch_bounce_max_frame_gap),
        top_pitch_bounce_neighbor_pitches=str(args.top_pitch_bounce_neighbor_pitches),
        top_pitch_bounce_return_pitches=str(args.top_pitch_bounce_return_pitches),
        turn_yaw_jitter_weight=int(args.turn_yaw_jitter_weight),
        turn_yaw_jitter_heading_frames=turn_yaw_heading_frames,
        turn_yaw_jitter_heading_values=turn_yaw_heading_values,
        turn_yaw_jitter_fps=float(args.turn_yaw_jitter_fps),
        turn_yaw_jitter_min_frame_gap=int(args.turn_yaw_jitter_min_frame_gap),
        turn_yaw_jitter_max_frame_gap=int(args.turn_yaw_jitter_max_frame_gap),
        turn_yaw_jitter_min_heading_delta=float(args.turn_yaw_jitter_min_heading_delta),
        turn_yaw_jitter_min_heading_speed=float(args.turn_yaw_jitter_min_heading_speed),
        turn_yaw_jitter_min_relative_yaw_delta=float(args.turn_yaw_jitter_min_relative_yaw_delta),
        turn_yaw_jitter_min_world_yaw_delta=float(args.turn_yaw_jitter_min_world_yaw_delta),
        turn_yaw_jitter_penalty_degrees_per_unit=float(args.turn_yaw_jitter_penalty_degrees_per_unit),
        turn_yaw_jitter_require_same_direction=not bool(args.turn_yaw_jitter_allow_opposite_direction),
        turn_yaw_jitter_exclude_cut_pairs=not bool(args.turn_yaw_jitter_include_cut_pairs),
        reference_sequence=reference_sequence,
        reference_change_weight=int(args.reference_change_weight),
        reference_change_fields=optimized_fields,
        bodypart_duration_weight=int(args.bodypart_duration_weight),
        bodypart_duration_stats=bodypart_duration_stats,
        bodypart_duration_fps=float(args.bodypart_duration_fps),
        bodypart_duration_proxy_mode=str(args.bodypart_duration_proxy_mode),
        bodypart_duration_upper_key=str(args.bodypart_duration_upper_key),
        bodypart_duration_penalty_sec_per_unit=float(args.bodypart_duration_penalty_sec_per_unit),
        bodypart_duration_upper_blend_combos=str(args.bodypart_duration_upper_blend_combos),
        bodypart_duration_upper_blend_alpha=float(args.bodypart_duration_upper_blend_alpha),
        bodypart_duration_upper_blend_from_key=str(args.bodypart_duration_upper_blend_from_key),
        bodypart_duration_upper_blend_to_key=str(args.bodypart_duration_upper_blend_to_key),
        bodypart_duration_short_weight=int(args.bodypart_duration_short_weight),
        bodypart_duration_lower_key=str(args.bodypart_duration_lower_key),
        bodypart_duration_short_penalty_sec_per_unit=float(args.bodypart_duration_short_penalty_sec_per_unit),
        bodypart_duration_short_exclude_cut_pairs=not bool(args.bodypart_duration_short_include_cut_pairs),
        distance_level_duration_weight=int(args.distance_level_duration_weight),
        distance_level_duration_stats=distance_level_duration_stats,
        distance_level_duration_fps=float(args.distance_level_duration_fps),
        distance_level_duration_include_levels=str(args.distance_level_duration_include_levels),
        distance_level_duration_end_mode=str(args.distance_level_duration_end_mode),
        distance_level_duration_upper_key=str(args.distance_level_duration_upper_key),
        distance_level_duration_upper_override_levels=str(args.distance_level_duration_upper_override_levels),
        distance_level_duration_upper_override_key=str(args.distance_level_duration_upper_override_key),
        distance_level_duration_penalty_sec_per_unit=float(args.distance_level_duration_penalty_sec_per_unit),
        distance_level_duration_short_weight=int(args.distance_level_duration_short_weight),
        distance_level_duration_lower_key=str(args.distance_level_duration_lower_key),
        distance_level_duration_min_sec=float(args.distance_level_duration_min_sec),
        distance_level_duration_short_penalty_sec_per_unit=float(
            args.distance_level_duration_short_penalty_sec_per_unit
        ),
        distance_level_duration_short_exclude_cut_pairs=not bool(args.distance_level_duration_short_include_cut_pairs),
        distance_level_duration_short_exclude_levels=str(args.distance_level_duration_short_exclude_levels),
        distance_level_direction_weight=int(args.distance_level_direction_weight),
        distance_level_direction_close_levels=str(args.distance_level_direction_close_levels),
        distance_level_direction_medium_level=str(args.distance_level_direction_medium_level),
        distance_level_direction_far_levels=str(args.distance_level_direction_far_levels),
        distance_level_direction_hard=bool(args.distance_level_direction_hard),
        distance_level_protect_source_levels=str(args.distance_level_protect_source_levels),
        distance_level_protect_min_level=str(args.distance_level_protect_min_level),
        distance_change_allowed_frames=distance_change_allowed_frames,
        distance_level_hold_after_change=bool(args.distance_level_hold_after_change),
        distance_level_hold_after_change_mode=str(args.distance_level_hold_after_change_mode),
        distance_level_hold_after_change_min_frame_gap=int(args.distance_level_hold_after_change_min_frame_gap),
        distance_level_hold_after_change_max_frame_gap=int(args.distance_level_hold_after_change_max_frame_gap),
        distance_level_hold_after_change_min_keyframes=int(args.distance_level_hold_after_change_min_keyframes),
        distance_level_hold_after_change_min_sec=float(args.distance_level_hold_after_change_min_sec),
        distance_level_hold_after_change_min_sec_key=str(args.distance_level_hold_after_change_min_sec_key),
        distance_level_hold_after_change_max_sec_key=str(args.distance_level_hold_after_change_max_sec_key),
        distance_level_hold_after_change_disallow_cut_pairs=bool(
            args.distance_level_hold_after_change_disallow_cut_pairs
        ),
        cut_identity_weight=int(args.cut_identity_weight),
        cut_contrast_weight=int(args.cut_contrast_weight),
        cut_contrast_min_score=int(args.cut_contrast_min_score),
        cut_contrast_medium_rank_delta=int(args.cut_contrast_medium_rank_delta),
        cut_contrast_strong_rank_delta=int(args.cut_contrast_strong_rank_delta),
        post_cut_distance_jump_weight=int(args.post_cut_distance_jump_weight),
        post_cut_distance_jump_min_rank_delta=int(args.post_cut_distance_jump_min_rank_delta),
        post_cut_distance_jump_max_frame_gap=int(args.post_cut_distance_jump_max_frame_gap),
        cut_context_speed_weight=int(args.cut_context_speed_weight),
        cut_context_speed_min_speed=float(args.cut_context_speed_min_speed),
        cut_context_speed_max_frame_gap=int(args.cut_context_speed_max_frame_gap),
        cut_context_speed_fps=float(args.cut_context_speed_fps),
        repeated_distance_bounce_weight=int(args.repeated_distance_bounce_weight),
        repeated_distance_bounce_min_rank_delta=int(args.repeated_distance_bounce_min_rank_delta),
        repeated_distance_bounce_max_frame_gap=int(args.repeated_distance_bounce_max_frame_gap),
        verbose=bool(args.verbose),
    )
    output_path = _output_path_for_candidate(candidate_path)
    debug_path = _debug_path_for_output(output_path)
    if result["status"] not in {"OPTIMAL", "FEASIBLE"}:
        _write_json(debug_path, result)
        return {
            "candidate_path": str(candidate_path),
            "status": result["status"],
            "objective": result.get("objective"),
            "output_path": str(output_path),
            "debug_path": str(debug_path),
        }

    optimized_rows = [_strip_optimizer_fields(selected, optimized_fields) for selected in result["selected_sequence"]]
    comparison_source_plan = str(reference_path) if reference_sequence and reference_path else str(payload.get("source_plan", ""))
    output_payload = _metadata_for_output(payload, schema_name)
    output_payload.update(
        {
            "source_topk_candidates": str(candidate_path),
            "source_plan": comparison_source_plan,
            "gt_stats_source": str(gt_path) if gt_path.exists() else None,
            "gt_stats_fallback": gt_stats_fallback,
            "gt_stats_frame_span": [min_frame, max_frame],
            "optimizer": {
                "name": "ortools_cp_sat_sequence_stats",
                "schema": schema_name,
                "fields": optimized_fields,
                "top_k": int(top_k),
                "max_time_sec": float(args.max_time_sec),
                "local_cost_scale": int(args.local_cost_scale),
                "disable_sequence_stat_objective": bool(args.disable_sequence_stat_objective),
                "distance_bounce_weight": int(args.distance_bounce_weight),
                "distance_bounce_min_rank_delta": int(args.distance_bounce_min_rank_delta),
                "distance_bounce_max_frame_gap": int(args.distance_bounce_max_frame_gap),
                "composition_bounce_weight": int(args.composition_bounce_weight),
                "composition_bounce_min_rank_delta": int(args.composition_bounce_min_rank_delta),
                "composition_bounce_max_frame_gap": int(args.composition_bounce_max_frame_gap),
                "vertical_bounce_weight": int(args.vertical_bounce_weight),
                "vertical_bounce_min_rank_delta": int(args.vertical_bounce_min_rank_delta),
                "vertical_bounce_max_frame_gap": int(args.vertical_bounce_max_frame_gap),
                "top_pitch_bounce_weight": int(args.top_pitch_bounce_weight),
                "top_pitch_bounce_max_frame_gap": int(args.top_pitch_bounce_max_frame_gap),
                "top_pitch_bounce_neighbor_pitches": str(args.top_pitch_bounce_neighbor_pitches),
                "top_pitch_bounce_return_pitches": str(args.top_pitch_bounce_return_pitches),
                "turn_yaw_jitter_weight": int(args.turn_yaw_jitter_weight),
                "turn_yaw_jitter_targets_path": str(turn_yaw_targets_path) if turn_yaw_targets_path else None,
                "turn_yaw_jitter_enabled": bool(turn_yaw_heading_frames and turn_yaw_heading_values),
                "turn_yaw_jitter_fps": float(args.turn_yaw_jitter_fps),
                "turn_yaw_jitter_min_frame_gap": int(args.turn_yaw_jitter_min_frame_gap),
                "turn_yaw_jitter_max_frame_gap": int(args.turn_yaw_jitter_max_frame_gap),
                "turn_yaw_jitter_min_heading_delta": float(args.turn_yaw_jitter_min_heading_delta),
                "turn_yaw_jitter_min_heading_speed": float(args.turn_yaw_jitter_min_heading_speed),
                "turn_yaw_jitter_min_relative_yaw_delta": float(args.turn_yaw_jitter_min_relative_yaw_delta),
                "turn_yaw_jitter_min_world_yaw_delta": float(args.turn_yaw_jitter_min_world_yaw_delta),
                "turn_yaw_jitter_penalty_degrees_per_unit": float(args.turn_yaw_jitter_penalty_degrees_per_unit),
                "turn_yaw_jitter_require_same_direction": not bool(args.turn_yaw_jitter_allow_opposite_direction),
                "turn_yaw_jitter_exclude_cut_pairs": not bool(args.turn_yaw_jitter_include_cut_pairs),
                "reference_plan_path": str(reference_path) if reference_path else None,
                "reference_source": reference_source,
                "reference_change_weight": int(args.reference_change_weight),
                "reference_change_enabled": bool(reference_sequence),
                "bodypart_duration_weight": int(args.bodypart_duration_weight),
                "bodypart_duration_stats": str(args.bodypart_duration_stats) if args.bodypart_duration_stats else None,
                "bodypart_duration_enabled": bool(
                    bodypart_duration_stats
                    and (int(args.bodypart_duration_weight) > 0 or int(args.bodypart_duration_short_weight) > 0)
                ),
                "bodypart_duration_fps": float(args.bodypart_duration_fps),
                "bodypart_duration_proxy_mode": str(args.bodypart_duration_proxy_mode),
                "bodypart_duration_upper_key": str(args.bodypart_duration_upper_key),
                "bodypart_duration_penalty_sec_per_unit": float(args.bodypart_duration_penalty_sec_per_unit),
                "bodypart_duration_upper_blend_combos": str(args.bodypart_duration_upper_blend_combos),
                "bodypart_duration_upper_blend_alpha": float(args.bodypart_duration_upper_blend_alpha),
                "bodypart_duration_upper_blend_from_key": str(args.bodypart_duration_upper_blend_from_key),
                "bodypart_duration_upper_blend_to_key": str(args.bodypart_duration_upper_blend_to_key),
                "bodypart_duration_short_weight": int(args.bodypart_duration_short_weight),
                "bodypart_duration_short_enabled": bool(
                    bodypart_duration_stats and int(args.bodypart_duration_short_weight) > 0
                ),
                "bodypart_duration_lower_key": str(args.bodypart_duration_lower_key),
                "bodypart_duration_short_penalty_sec_per_unit": float(
                    args.bodypart_duration_short_penalty_sec_per_unit
                ),
                "bodypart_duration_short_exclude_cut_pairs": not bool(args.bodypart_duration_short_include_cut_pairs),
                "distance_level_duration_weight": int(args.distance_level_duration_weight),
                "distance_level_duration_stats": (
                    str(args.distance_level_duration_stats) if args.distance_level_duration_stats else None
                ),
                "distance_level_duration_enabled": bool(
                    distance_level_duration_stats
                    and (
                        int(args.distance_level_duration_weight) > 0
                        or int(args.distance_level_duration_short_weight) > 0
                    )
                ),
                "distance_level_duration_fps": float(args.distance_level_duration_fps),
                "distance_level_duration_include_levels": str(args.distance_level_duration_include_levels),
                "distance_level_duration_end_mode": str(args.distance_level_duration_end_mode),
                "distance_level_duration_upper_key": str(args.distance_level_duration_upper_key),
                "distance_level_duration_upper_override_levels": str(
                    args.distance_level_duration_upper_override_levels
                ),
                "distance_level_duration_upper_override_key": str(args.distance_level_duration_upper_override_key),
                "distance_level_duration_penalty_sec_per_unit": float(
                    args.distance_level_duration_penalty_sec_per_unit
                ),
                "distance_level_duration_short_weight": int(args.distance_level_duration_short_weight),
                "distance_level_duration_lower_key": str(args.distance_level_duration_lower_key),
                "distance_level_duration_min_sec": float(args.distance_level_duration_min_sec),
                "distance_level_duration_short_penalty_sec_per_unit": float(
                    args.distance_level_duration_short_penalty_sec_per_unit
                ),
                "distance_level_duration_short_exclude_cut_pairs": not bool(
                    args.distance_level_duration_short_include_cut_pairs
                ),
                "distance_level_duration_short_exclude_levels": str(args.distance_level_duration_short_exclude_levels),
                "distance_level_change_only_eligible": bool(args.distance_level_change_only_eligible),
                "distance_level_eligible_close_levels": str(args.distance_level_eligible_close_levels),
                "distance_level_eligible_upper_key": str(args.distance_level_eligible_upper_key),
                "distance_level_eligible_require_lower_body_activity": bool(
                    args.distance_level_eligible_require_lower_body_activity
                ),
                "distance_level_activity_feature_root": str(args.distance_level_activity_feature_root),
                "distance_level_lower_body_joints": str(args.distance_level_lower_body_joints),
                "distance_level_lower_body_activity_min": float(args.distance_level_lower_body_activity_min),
                "distance_level_activity_window_radius": int(args.distance_level_activity_window_radius),
                "distance_level_eligible_min_next_frame_gap": int(args.distance_level_eligible_min_next_frame_gap),
                "distance_change_allowed_frame_count": len(distance_change_allowed_frames or []),
                "distance_level_eligible_debug": distance_level_eligible_debug,
                "distance_level_direction_weight": int(args.distance_level_direction_weight),
                "distance_level_direction_close_levels": str(args.distance_level_direction_close_levels),
                "distance_level_direction_medium_level": str(args.distance_level_direction_medium_level),
                "distance_level_direction_far_levels": str(args.distance_level_direction_far_levels),
                "distance_level_direction_hard": bool(args.distance_level_direction_hard),
                "distance_level_protect_source_levels": str(args.distance_level_protect_source_levels),
                "distance_level_protect_min_level": str(args.distance_level_protect_min_level),
                "distance_level_hold_after_change": bool(args.distance_level_hold_after_change),
                "distance_level_hold_after_change_mode": str(args.distance_level_hold_after_change_mode),
                "distance_level_hold_after_change_min_frame_gap": int(
                    args.distance_level_hold_after_change_min_frame_gap
                ),
                "distance_level_hold_after_change_max_frame_gap": int(
                    args.distance_level_hold_after_change_max_frame_gap
                ),
                "distance_level_hold_after_change_min_keyframes": int(
                    args.distance_level_hold_after_change_min_keyframes
                ),
                "distance_level_hold_after_change_min_sec": float(args.distance_level_hold_after_change_min_sec),
                "distance_level_hold_after_change_min_sec_key": str(
                    args.distance_level_hold_after_change_min_sec_key
                ),
                "distance_level_hold_after_change_max_sec_key": str(
                    args.distance_level_hold_after_change_max_sec_key
                ),
                "distance_level_hold_after_change_disallow_cut_pairs": bool(
                    args.distance_level_hold_after_change_disallow_cut_pairs
                ),
                "synthetic_bodyproxy_safe_target_anchor_debug": synthetic_bodyproxy_safe_target_anchor_debug,
                "synthetic_bodyproxy_safe_target_anchor_enabled": bool(
                    args.add_bodyproxy_safe_target_anchor_candidates
                ),
                "synthetic_bodyproxy_safe_target_anchor_cost": float(args.bodyproxy_safe_target_anchor_cost),
                "synthetic_bodyproxy_safe_target_anchor_values": str(args.bodyproxy_safe_target_anchor_values),
                "synthetic_bodyproxy_safe_target_anchor_proxy_mode": str(
                    args.bodyproxy_safe_target_anchor_proxy_mode
                ),
                "finger_activity_visibility_cost": float(args.finger_activity_visibility_cost),
                "finger_activity_visibility_debug": finger_activity_visibility_debug,
                "finger_activity_trigger_debug": finger_activity_trigger_debug,
                "finger_activity_trigger_frame_count": int(
                    finger_activity_trigger_debug.get("frame_count", 0)
                ),
                "distance_fov_only_finger_triggered_only": bool(
                    args.distance_fov_only_finger_triggered_only
                ),
                "finger_activity_trigger_source_distances": str(args.finger_activity_trigger_source_distances),
                "finger_activity_trigger_target_distances": str(args.finger_activity_trigger_target_distances),
                "synthetic_distance_fov_only_candidates": int(synthetic_distance_fov_only_added),
                "synthetic_distance_fov_only_enabled": bool(args.add_distance_fov_only_candidates),
                "synthetic_distance_fov_only_cost": float(args.distance_fov_only_candidate_cost),
                "synthetic_distance_fov_only_radius": int(args.distance_fov_only_candidate_radius),
                "synthetic_distance_fov_only_max_per_step": int(args.distance_fov_only_candidate_max_per_step),
                "synthetic_distance_fov_only_directional_escape": bool(args.distance_fov_only_directional_escape),
                "synthetic_distance_fov_only_exclude_source_levels": str(args.distance_fov_only_exclude_source_levels),
                "synthetic_top_pitch_escape_candidates": int(synthetic_top_pitch_escape_added),
                "synthetic_top_pitch_escape_enabled": bool(args.add_top_pitch_escape_candidates),
                "synthetic_top_pitch_escape_cost": float(args.top_pitch_escape_candidate_cost),
                "synthetic_top_pitch_escape_pitches": str(args.top_pitch_escape_candidate_pitches),
                "synthetic_top_pitch_escape_max_per_step": int(args.top_pitch_escape_candidate_max_per_step),
                "disallow_new_top_pitch_candidates": bool(args.disallow_new_top_pitch_candidates),
                "new_top_pitch_reference_source": new_top_pitch_reference_source,
                "new_top_pitch_candidates_removed": int(new_top_pitch_candidates_removed),
                "cut_context_speed_weight": int(args.cut_context_speed_weight),
                "cut_context_speed_min_speed": float(args.cut_context_speed_min_speed),
                "cut_context_speed_max_frame_gap": int(args.cut_context_speed_max_frame_gap),
                "cut_context_speed_fps": float(args.cut_context_speed_fps),
                "repeated_distance_bounce_weight": int(args.repeated_distance_bounce_weight),
                "repeated_distance_bounce_min_rank_delta": int(args.repeated_distance_bounce_min_rank_delta),
                "repeated_distance_bounce_max_frame_gap": int(args.repeated_distance_bounce_max_frame_gap),
                "status": result["status"],
                "objective": result["objective"],
                "objective_breakdown": result["objective_breakdown"],
                "rank_summary": _rank_summary(result["selected_sequence"]),
            },
            "keyframes": optimized_rows,
        }
    )
    base_rows = (
        source_plan_sequence
        if source_plan_sequence
        else (_rows_from_sequence(reference_sequence, optimized_fields) if reference_sequence else [])
    )
    changes = _change_summary(base_rows, optimized_rows, optimized_fields)
    output_payload["optimization_change_summary"] = changes
    debug_payload = {
        "output": output_payload,
        "step_meta": step_meta,
        "selected_indices": result["selected_indices"],
        "selected_sequence_with_scores": result["selected_sequence"],
        "stats": result["stats"],
    }
    _write_json(output_path, output_payload)
    _write_json(debug_path, debug_payload)
    return {
        "candidate_path": str(candidate_path),
        "source_plan": comparison_source_plan,
        "output_path": str(output_path),
        "debug_path": str(debug_path),
        "gt_path": str(gt_path),
        "status": result["status"],
        "schema": schema_name,
        "objective": result["objective"],
        "frame_bucket_count": len(payload.get("frames", [])),
        "slot_count": len(optimized_rows),
        "rank_summary_rank_gt_1": _rank_summary(result["selected_sequence"])["rank_gt_1"],
        "rank_summary_avg_rank": _rank_summary(result["selected_sequence"])["avg_rank"],
        "changed_keyframe_count": changes["changed_keyframe_count"],
        "changed_keyframe_rate": changes["changed_keyframe_rate"],
        "field_change_count": changes["field_change_count"],
        "finger_activity_visibility_penalized_candidate_count": int(
            finger_activity_visibility_debug.get("penalized_candidate_count", 0)
        ),
        "finger_activity_visibility_penalized_source_frame_count": int(
            finger_activity_visibility_debug.get("penalized_source_frame_count", 0)
        ),
        "finger_activity_trigger_frame_count": int(finger_activity_trigger_debug.get("frame_count", 0)),
        "finger_activity_visibility_total_added_cost": float(
            finger_activity_visibility_debug.get("total_added_cost", 0.0)
        ),
        "distance_change_allowed_frame_count": len(distance_change_allowed_frames or []),
        "synthetic_bodyproxy_safe_target_anchor_candidates": int(
            synthetic_bodyproxy_safe_target_anchor_debug.get("added", 0)
        ),
        "synthetic_bodyproxy_safe_target_anchor_rejected_lost_bodyparts": int(
            synthetic_bodyproxy_safe_target_anchor_debug.get("rejected_lost_bodyparts", 0)
        ),
        "synthetic_distance_fov_only_candidates": int(synthetic_distance_fov_only_added),
        "synthetic_top_pitch_escape_candidates": int(synthetic_top_pitch_escape_added),
        "new_top_pitch_reference_source": new_top_pitch_reference_source,
        "new_top_pitch_candidates_removed": int(new_top_pitch_candidates_removed),
        **{f"objective_{key}": value for key, value in result["objective_breakdown"].items()},
    }


def main() -> None:
    args = build_parser().parse_args()
    source_root = args.source_root.resolve()
    gt_root = args.gt_root.resolve()
    paths = _candidate_paths(source_root, str(args.candidate_glob))
    if not paths:
        raise FileNotFoundError(f"No candidates under {source_root} matching {args.candidate_glob!r}")

    rows: list[dict[str, Any]] = []
    for path in paths:
        row = _optimize_candidate_path(
            candidate_path=path,
            gt_root=gt_root,
            gt_name=str(args.gt_name),
            schema=str(args.schema),
            top_k=int(args.top_k),
            args=args,
        )
        rows.append(row)
        print(
            f"[{row.get('status')}] {path.parent.name}/{path.name}: "
            f"changed={row.get('changed_keyframe_count', 0)} fields={row.get('field_change_count', 0)} "
            f"rank>1={row.get('rank_summary_rank_gt_1', 0)}"
        )

    manifest_path = source_root / str(args.manifest_name)
    _write_json(manifest_path, rows)
    csv_path = manifest_path.with_suffix(".csv")
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
