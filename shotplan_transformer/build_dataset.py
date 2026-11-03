from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera_autoshot.shotplan_bucket_utils import group_and_consolidate_by_frame
from shotplan_transformer.compact_dance import (
    compact_dance_from_full,
    compact_dance_hand_motion_from_full,
    compact_dance_roll_signals_from_full,
    compact_dance_root_head_rot6d_from_full,
    load_feature_manifest,
)
from shotplan_transformer.bodypart_proxy import (
    EXPORTED_DISTANCE_MULTIPLIERS,
    FOV_DEGREES,
    PARTS as BODYPART_PROXY_PARTS,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_ROOT = WORKSPACE_ROOT / "music_processing" / "results" / "cut_keyframe_multimodal_dataset_dcm" / "songs"
DEFAULT_LABEL_ROOT = WORKSPACE_ROOT / "camera_autoshot" / "reverse_v3" / "dcm_clip_labels" / "v3_1_close_dense_temporal_delta"
DEFAULT_SPLIT_MANIFEST = (
    Path(__file__).resolve().parent / "results" / "dcm_clip_split_manifest_v3_1_close_dense_temporal_delta.json"
)
DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parent / "results" / "discrete_v3_1_30fps_close_dense_temporal_delta_dcm_split_dataset"
)
DEFAULT_DCA_BONEMASK_ROOT = WORKSPACE_ROOT / "local_assets" / "DCM++"

LABEL_FIELDS = (
    "target_body",
    "shot_size",
    "angle",
    "composition",
    "roll",
)
V3_LABEL_FIELDS = (
    "target_anchor",
    "distance",
    "yaw",
    "pitch",
    "composition",
    "roll",
    "fov",
)
V3_INTERPOLATION_LABEL_FIELDS = (
    *V3_LABEL_FIELDS,
    "interpolation",
)
PAD_VALUE = -100
PREV_CONTEXT_SIZE = 2
BODYPART_BONEMASK_GROUPS = {
    "head": [3, 4, 5, 6, 7],
    "arms_upper_body": [16, 17, 18, 19, 20, 21, 22, 38, 39, 40, 41, 42, 43, 44],
    "fingers": list(range(23, 38)) + list(range(45, 60)),
    "waist": [0, 1, 2],
    "legs": [8, 9, 12, 13],
    "feet": [10, 11, 14, 15],
}
FINGER_POSE_HINT_NAMES = [
    "left_finger_y_norm",
    "right_finger_y_norm",
    "left_finger_x_abs_norm",
    "right_finger_x_abs_norm",
    "both_finger_width_x_norm",
    "both_finger_height_y_norm",
]
FINGER_POSE_LEFT_TIPS = (25, 28, 31, 34, 37)
FINGER_POSE_RIGHT_TIPS = (47, 50, 53, 56, 59)
FINGER_POSE_HEAD_JOINT = 4
FINGER_POSE_LEFT_TOE_JOINT = 11
FINGER_POSE_RIGHT_TOE_JOINT = 15
FINGER_THRESHOLD_REQUIRED_FIELDS = ("target_anchor", "distance", "fov", "yaw")
FINGER_THRESHOLD_CLOSE_DISTANCES = {"extreme_close", "very_close", "close"}
FINGER_THRESHOLD_SCREEN_FRACTION = 0.85
FINGER_THRESHOLD_ASPECT_RATIO = 16.0 / 9.0
FINGER_THRESHOLD_TARGET_ANCHOR_JOINTS = {
    "feet": (11, 15),
    "lower_leg": (10, 14),
    "thigh": (9, 13),
    "hip": (0,),
    "waist": (1,),
    "chest": (2,),
    "face": (4,),
}
FINGER_THRESHOLD_YAW_DEGREES = {
    "front": 0.0,
    "right_45": 45.0,
    "profile_right": 90.0,
    "back_right_45": 135.0,
    "back": 180.0,
    "back_left_45": -135.0,
    "profile_left": -90.0,
    "left_45": -45.0,
}
FINGER_THRESHOLD_SHOULDER_JOINTS = (17, 39)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a 5fps multimodal dataset for discrete camera shot-plan prediction "
            "from music+dance features and reverse shot-plan labels."
        )
    )
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument(
        "--label-filename",
        type=str,
        default="camera_poses_inferred.json",
        help="Per-song shot-plan JSON filename under --label-root/song_id.",
    )
    parser.add_argument(
        "--label-fields",
        type=str,
        default=",".join(LABEL_FIELDS),
        help=(
            "Comma-separated categorical label fields. "
            "Use 'v3' for target_anchor,distance,yaw,pitch,composition,roll,fov, "
            "or 'v3_interpolation' to also predict interpolation."
        ),
    )
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--timeline-fps", type=float, default=30.0)
    parser.add_argument("--window-radius", type=int, default=4)
    parser.add_argument(
        "--window-stride",
        type=int,
        default=1,
        help=(
            "Feature-window sampling stride in sampled feature frames. "
            "For example, --window-radius 24 --window-stride 6 uses offsets "
            "-24,-18,...,+24, preserving a +-24-frame span with only 9 tokens."
        ),
    )
    parser.add_argument(
        "--prev-context-size",
        type=int,
        default=PREV_CONTEXT_SIZE,
        help="Number of previous consolidated shot-plan buckets to store as model context.",
    )
    parser.add_argument(
        "--future-context-size",
        type=int,
        default=0,
        help="Number of future keyframe timing deltas to expose, without future shot-plan labels.",
    )
    parser.add_argument(
        "--dance-feature-set",
        choices=[
            "compact",
            "compact_hand_motion",
            "compact_root_head_rot6d",
            "compact_root_head_wrist_rel_rot6d_vectors",
            "full",
        ],
        default="compact",
        help="Use compact dance features, compact plus roll-related additions, or the original full dance feature.",
    )
    parser.add_argument(
        "--dca-bonemask-root",
        type=Path,
        default=None,
        help=(
            "Optional DCA DCM++ root containing Train/Validation/Test/BoneMask. "
            "When set, saves bodypart_visible_target for each sample."
        ),
    )
    parser.add_argument(
        "--dance-only",
        action="store_true",
        help="Drop audio/music features from the saved dataset so the model receives dance windows only.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite without --force: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_npz(path: Path, *, force: bool, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite without --force: {path}")
    np.savez_compressed(path, **arrays)


def _load_keyframes(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if isinstance(payload, dict):
        rows = payload.get("keyframes", [])
        if not isinstance(rows, list):
            raise ValueError(f"Unexpected keyframe format: {path}")
        return [row for row in rows if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise ValueError(f"Unsupported keyframe file format: {path}")


def _parse_label_fields(raw: str) -> tuple[str, ...]:
    normalized = raw.strip().lower()
    if normalized == "v3":
        return V3_LABEL_FIELDS
    if normalized in {"v3_interpolation", "v3_interp", "v3+interpolation", "v3+interp"}:
        return V3_INTERPOLATION_LABEL_FIELDS
    fields = tuple(token.strip() for token in raw.split(",") if token.strip())
    if not fields:
        raise ValueError("--label-fields must contain at least one field")
    return fields


def _default_label_value(field: str) -> str:
    if field == "interpolation":
        return "linear"
    if field == "transition":
        return "linear"
    raise KeyError(field)


def _group_labels(rows: list[dict[str, Any]], *, label_fields: tuple[str, ...]) -> dict[int, list[dict[str, str]]]:
    normalized_rows = []
    for row in rows:
        values: dict[str, str] = {}
        for field in label_fields:
            if field in row:
                values[field] = str(row[field])
            else:
                values[field] = _default_label_value(field)
        normalized_rows.append(
            {
                "frame": int(row["frame"]),
                **values,
            }
        )
    grouped = group_and_consolidate_by_frame(
        normalized_rows,
        frame_field="frame",
        signature_fields=label_fields,
    )
    return {
        int(frame): [{field: str(row[field]) for field in label_fields} for row in frame_rows]
        for frame, frame_rows in grouped.items()
    }


def _build_vocabularies(
    song_groups: dict[str, dict[int, list[dict[str, str]]]],
    *,
    label_fields: tuple[str, ...],
) -> tuple[dict[str, list[str]], int, Counter[int]]:
    values = {field: set() for field in label_fields}
    max_slots = 0
    slot_count_dist: Counter[int] = Counter()
    for grouped in song_groups.values():
        for rows in grouped.values():
            slot_count = len(rows)
            max_slots = max(max_slots, slot_count)
            slot_count_dist[slot_count] += 1
            for row in rows:
                for field in label_fields:
                    values[field].add(str(row[field]))
    vocab = {field: sorted(values[field]) for field in label_fields}
    return vocab, max_slots, slot_count_dist


def _fill_label_slots(
    label_arrays: dict[str, np.ndarray],
    *,
    sample_idx: int,
    rows: list[dict[str, str]],
    field_to_id: dict[str, dict[str, int]],
    label_fields: tuple[str, ...],
    prefix_index: tuple[int, ...] = (),
) -> None:
    for slot_idx, row in enumerate(rows):
        for field in label_fields:
            token = row[field]
            token_id = field_to_id[field].get(token)
            if token_id is None:
                raise KeyError(f"Unknown token {token!r} for field {field}")
            label_arrays[field][(sample_idx, *prefix_index, slot_idx)] = int(token_id)


def _downsample_source_indices(num_frames: int, *, sample_fps: float, timeline_fps: float) -> np.ndarray:
    if num_frames <= 0:
        raise ValueError("feature sequence is empty")
    if sample_fps <= 0.0 or timeline_fps <= 0.0:
        raise ValueError("sample_fps and timeline_fps must be positive")
    sample_count = int(round((num_frames - 1) * sample_fps / timeline_fps)) + 1
    indices = [min(int(round(sample_idx * timeline_fps / sample_fps)), num_frames - 1) for sample_idx in range(sample_count)]
    return np.asarray(indices, dtype=np.int32)


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
    if length <= 0:
        raise ValueError("window length must be positive")
    indices = int(center_index) + _window_offsets(radius=int(radius), stride=int(stride))
    return np.clip(indices, 0, length - 1)


def _resolve_song_ids(split_manifest: dict[str, Any]) -> list[str]:
    available = split_manifest.get("available_song_ids")
    if isinstance(available, list) and available:
        return [str(song_id) for song_id in available]
    values: list[str] = []
    for split_name in ("train", "val", "test"):
        values.extend(str(song_id) for song_id in split_manifest.get(split_name, []))
    seen = set()
    ordered: list[str] = []
    for song_id in values:
        if song_id in seen:
            continue
        seen.add(song_id)
        ordered.append(song_id)
    return ordered


def _split_for_song(song_id: str, splits: dict[str, list[str]]) -> str | None:
    for split_name, values in splits.items():
        if song_id in values:
            return split_name
    return None


def _dca_split_dir_name(split_name: str | None) -> str:
    mapping = {"train": "Train", "val": "Validation", "validation": "Validation", "test": "Test"}
    return mapping.get(str(split_name or "").lower(), str(split_name or ""))


def _bonemask_stem_from_song_id(song_id: str) -> str:
    pieces = [piece for piece in str(song_id).split("_") if piece]
    numeric = []
    for piece in pieces:
        if piece.isdigit():
            numeric.append(str(int(piece)))
    if not numeric:
        raise ValueError(f"Cannot derive BoneMask filename from song_id={song_id!r}")
    return "bm" + "_".join(numeric)


def _bonemask_path_for_song(
    *,
    bonemask_root: Path,
    song_id: str,
    split_name: str | None,
) -> tuple[Path, bool]:
    """Return BoneMask path and whether local sample frames need source_start offset."""

    stem = _bonemask_stem_from_song_id(song_id)
    pieces = [piece for piece in str(song_id).split("_") if piece]
    song_num = next((str(int(piece)) for piece in pieces if piece.isdigit()), "")
    part_num = None
    numeric_seen = 0
    for piece in pieces:
        if piece.isdigit():
            numeric_seen += 1
            if numeric_seen == 2:
                part_num = int(piece)
                break
    candidates = []
    split_dir = _dca_split_dir_name(split_name)
    if split_dir:
        candidates.append(bonemask_root / split_dir / "BoneMask" / f"{stem}.json")
    candidates.append(bonemask_root / "BoneMask" / f"{stem}.json")
    candidates.append(bonemask_root / f"{stem}.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate, False
    if split_dir and song_num:
        ranged_root = bonemask_root / split_dir / "BoneMask"
        for candidate in sorted(ranged_root.glob(f"bm{song_num}_*.json")):
            suffix = candidate.stem.removeprefix(f"bm{song_num}_")
            if "~" not in suffix or part_num is None:
                continue
            start_raw, end_raw = suffix.split("~", 1)
            if start_raw.isdigit() and end_raw.isdigit() and int(start_raw) <= part_num <= int(end_raw):
                return candidate, True
    return candidates[0], False


def _load_bonemask_array(path: Path) -> np.ndarray:
    payload = _load_json(path)
    rows = payload.get("bone_mask") if isinstance(payload, dict) else payload
    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 60:
        raise ValueError(f"Unexpected BoneMask shape in {path}: {arr.shape}")
    return arr


def _bodypart_visible_targets_from_bonemask(
    bonemask: np.ndarray,
    source_frames: np.ndarray,
    *,
    frame_offset: int = 0,
) -> np.ndarray:
    targets = np.zeros((int(source_frames.shape[0]), len(BODYPART_PROXY_PARTS)), dtype=np.float32)
    if bonemask.shape[0] <= 0:
        return targets
    for sample_idx, source_frame in enumerate(source_frames.tolist()):
        frame = min(max(int(source_frame) + int(frame_offset), 0), int(bonemask.shape[0]) - 1)
        row = bonemask[frame]
        for part_idx, part in enumerate(BODYPART_PROXY_PARTS):
            columns = BODYPART_BONEMASK_GROUPS.get(part.name, [])
            if columns:
                targets[sample_idx, part_idx] = float(np.mean(row[columns]))
    return targets


def _dance_columns_for_names(all_names: list[str], wanted_names: list[str]) -> list[int]:
    index_by_name = {name: idx for idx, name in enumerate(all_names)}
    missing = [name for name in wanted_names if name not in index_by_name]
    if missing:
        raise ValueError(f"Missing expected dance feature names: {missing[:8]}")
    return [index_by_name[name] for name in wanted_names]


def _dance_relpos_xyz(
    dance_features: np.ndarray,
    joint_indices: tuple[int, ...] | list[int],
    full_dance_feature_names: list[str],
) -> np.ndarray:
    columns = _dance_columns_for_names(
        full_dance_feature_names,
        [
            f"dance_relpos_j{joint_idx:02d}_{axis}"
            for joint_idx in joint_indices
            for axis in ("x", "y", "z")
        ],
    )
    return dance_features[:, columns].reshape(dance_features.shape[0], len(joint_indices), 3)


def _finger_pose_hints_from_full_dance(
    dance_features: np.ndarray,
    full_dance_feature_names: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return per-frame hand/finger pose hints normalized by body height.

    The static body-part proxy cannot know if the dancer's hands are inside a
    close shot. These hints preserve just enough pose information to make the
    finger proxy conditional on the current hand location without changing the
    model input itself.
    """

    try:
        head = _dance_relpos_xyz(dance_features, [FINGER_POSE_HEAD_JOINT], full_dance_feature_names)[:, 0, :]
        toes = _dance_relpos_xyz(
            dance_features,
            [FINGER_POSE_LEFT_TOE_JOINT, FINGER_POSE_RIGHT_TOE_JOINT],
            full_dance_feature_names,
        )
        left_tips = _dance_relpos_xyz(dance_features, list(FINGER_POSE_LEFT_TIPS), full_dance_feature_names)
        right_tips = _dance_relpos_xyz(dance_features, list(FINGER_POSE_RIGHT_TIPS), full_dance_feature_names)
    except ValueError as exc:
        hints = np.zeros((int(dance_features.shape[0]), len(FINGER_POSE_HINT_NAMES)), dtype=np.float32)
        return hints, {"enabled": False, "reason": str(exc), "feature_names": list(FINGER_POSE_HINT_NAMES)}

    foot_y = toes[:, :, 1].min(axis=1)
    body_height = np.maximum(head[:, 1] - foot_y, 1e-3)
    left_mean = left_tips.mean(axis=1)
    right_mean = right_tips.mean(axis=1)
    all_tips = np.concatenate([left_tips, right_tips], axis=1)
    hints = np.stack(
        [
            (left_mean[:, 1] - foot_y) / body_height,
            (right_mean[:, 1] - foot_y) / body_height,
            np.abs(left_mean[:, 0]) / body_height,
            np.abs(right_mean[:, 0]) / body_height,
            (all_tips[:, :, 0].max(axis=1) - all_tips[:, :, 0].min(axis=1)) / body_height,
            (all_tips[:, :, 1].max(axis=1) - all_tips[:, :, 1].min(axis=1)) / body_height,
        ],
        axis=1,
    ).astype(np.float32)
    hints[:, 0:2] = np.clip(hints[:, 0:2], -0.5, 1.5)
    hints[:, 2:6] = np.clip(hints[:, 2:6], 0.0, 2.0)
    metadata = {
        "enabled": True,
        "feature_names": list(FINGER_POSE_HINT_NAMES),
        "normalization": "finger relpos normalized by head-to-toe body height",
        "joints": {
            "head": FINGER_POSE_HEAD_JOINT,
            "toes": [FINGER_POSE_LEFT_TOE_JOINT, FINGER_POSE_RIGHT_TOE_JOINT],
            "left_fingertips": list(FINGER_POSE_LEFT_TIPS),
            "right_fingertips": list(FINGER_POSE_RIGHT_TIPS),
        },
    }
    return hints, metadata


def _normalized_xz(vectors: np.ndarray, *, fallback: tuple[float, float]) -> np.ndarray:
    xz = vectors[:, [0, 2]].astype(np.float32, copy=False)
    norm = np.linalg.norm(xz, axis=1, keepdims=True)
    safe = norm[:, 0] > 1e-6
    out = np.zeros_like(xz, dtype=np.float32)
    out[safe] = xz[safe] / norm[safe]
    if not np.all(safe):
        out[~safe] = np.asarray(fallback, dtype=np.float32)
    return out


def _finger_visibility_threshold_for_distance_fov(
    *,
    body_height: float,
    distance: str,
    fov: str,
) -> tuple[float, float]:
    distance_multiplier = float(
        EXPORTED_DISTANCE_MULTIPLIERS.get(str(distance), EXPORTED_DISTANCE_MULTIPLIERS["medium"])
    )
    fov_degrees = float(FOV_DEGREES.get(str(fov), FOV_DEGREES["normal"]))
    half_height = max(float(body_height), 1e-3) * distance_multiplier * math.tan(math.radians(fov_degrees) * 0.5)
    half_width = half_height * FINGER_THRESHOLD_ASPECT_RATIO
    return (
        float(half_width * FINGER_THRESHOLD_SCREEN_FRACTION),
        float(half_height * FINGER_THRESHOLD_SCREEN_FRACTION),
    )


def _finger_threshold_visibility_table_from_full_dance(
    dance_features: np.ndarray,
    full_dance_feature_names: list[str],
    source_frames: np.ndarray,
    vocab: dict[str, list[str]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate finger visibility for every V3 anchor/distance/FOV/yaw choice.

    The table is intentionally precomputed during dataset build because the
    compact dance feature used by the training model does not keep every finger
    joint.  Axis order is:
    sample, target_anchor, distance, fov, yaw.
    """

    missing_fields = [field for field in FINGER_THRESHOLD_REQUIRED_FIELDS if field not in vocab]
    if missing_fields:
        empty = np.zeros((int(source_frames.shape[0]), 0, 0, 0, 0), dtype=np.float16)
        return empty, {
            "enabled": False,
            "reason": f"missing label fields: {missing_fields}",
            "axis_order": list(FINGER_THRESHOLD_REQUIRED_FIELDS),
        }

    target_anchor_vocab = [str(value) for value in vocab["target_anchor"]]
    distance_vocab = [str(value) for value in vocab["distance"]]
    fov_vocab = [str(value) for value in vocab["fov"]]
    yaw_vocab = [str(value) for value in vocab["yaw"]]
    try:
        all_anchor_joints = sorted(
            {joint for joints in FINGER_THRESHOLD_TARGET_ANCHOR_JOINTS.values() for joint in joints}
        )
        anchor_joint_pos = _dance_relpos_xyz(dance_features, all_anchor_joints, full_dance_feature_names)
        anchor_joint_to_col = {joint: idx for idx, joint in enumerate(all_anchor_joints)}
        fingertip_pos = _dance_relpos_xyz(
            dance_features,
            list(FINGER_POSE_LEFT_TIPS + FINGER_POSE_RIGHT_TIPS),
            full_dance_feature_names,
        )
        shoulders = _dance_relpos_xyz(dance_features, list(FINGER_THRESHOLD_SHOULDER_JOINTS), full_dance_feature_names)
        head = _dance_relpos_xyz(dance_features, [FINGER_POSE_HEAD_JOINT], full_dance_feature_names)[:, 0, :]
        toes = _dance_relpos_xyz(
            dance_features,
            [FINGER_POSE_LEFT_TOE_JOINT, FINGER_POSE_RIGHT_TOE_JOINT],
            full_dance_feature_names,
        )
    except ValueError as exc:
        empty = np.zeros((int(source_frames.shape[0]), 0, 0, 0, 0), dtype=np.float16)
        return empty, {
            "enabled": False,
            "reason": str(exc),
            "axis_order": list(FINGER_THRESHOLD_REQUIRED_FIELDS),
        }

    clamped_frames = np.clip(
        source_frames.astype(np.int64, copy=False),
        0,
        max(0, int(dance_features.shape[0]) - 1),
    )
    sample_count = int(clamped_frames.shape[0])
    table = np.ones(
        (
            sample_count,
            len(target_anchor_vocab),
            len(distance_vocab),
            len(fov_vocab),
            len(yaw_vocab),
        ),
        dtype=np.float32,
    )
    if sample_count <= 0:
        return table.astype(np.float16), {
            "enabled": True,
            "axis_order": list(FINGER_THRESHOLD_REQUIRED_FIELDS),
            "score": "mean visible fingertip ratio",
            "close_distances": sorted(FINGER_THRESHOLD_CLOSE_DISTANCES),
        }

    foot_y = toes[:, :, 1].min(axis=1)
    body_height = np.maximum(head[:, 1] - foot_y, 1e-3)
    # Use a clip-level median height so crouching/lying poses do not shrink the
    # visibility box and accidentally label hands as invisible.
    clip_body_height = float(np.median(body_height))
    tips = fingertip_pos[clamped_frames]
    shoulder_delta = shoulders[:, 1, :] - shoulders[:, 0, :]
    body_right = _normalized_xz(shoulder_delta, fallback=(1.0, 0.0))[clamped_frames]
    body_forward = np.stack([body_right[:, 1], -body_right[:, 0]], axis=1).astype(np.float32)
    anchor_positions: list[np.ndarray] = []
    for anchor in target_anchor_vocab:
        joints = FINGER_THRESHOLD_TARGET_ANCHOR_JOINTS.get(anchor, FINGER_THRESHOLD_TARGET_ANCHOR_JOINTS["waist"])
        cols = [anchor_joint_to_col[joint] for joint in joints if joint in anchor_joint_to_col]
        if not cols:
            cols = [anchor_joint_to_col[FINGER_THRESHOLD_TARGET_ANCHOR_JOINTS["waist"][0]]]
        anchor_positions.append(anchor_joint_pos[:, cols, :].mean(axis=1)[clamped_frames])

    for anchor_idx, anchor in enumerate(anchor_positions):
        rel = tips - anchor[:, None, :]
        vertical_abs = np.abs(rel[:, :, 1])
        for yaw_idx, yaw in enumerate(yaw_vocab):
            yaw_degrees = float(FINGER_THRESHOLD_YAW_DEGREES.get(yaw, 0.0))
            yaw_rad = math.radians(yaw_degrees)
            camera_right = body_right * math.cos(yaw_rad) + body_forward * math.sin(yaw_rad)
            horizontal_abs = np.abs(rel[:, :, 0] * camera_right[:, None, 0] + rel[:, :, 2] * camera_right[:, None, 1])
            for distance_idx, distance in enumerate(distance_vocab):
                if distance not in FINGER_THRESHOLD_CLOSE_DISTANCES:
                    table[:, anchor_idx, distance_idx, :, yaw_idx] = 1.0
                    continue
                for fov_idx, fov in enumerate(fov_vocab):
                    width_threshold, height_threshold = _finger_visibility_threshold_for_distance_fov(
                        body_height=clip_body_height,
                        distance=distance,
                        fov=fov,
                    )
                    visible = (horizontal_abs <= width_threshold) & (vertical_abs <= height_threshold)
                    table[:, anchor_idx, distance_idx, fov_idx, yaw_idx] = visible.mean(axis=1)

    metadata = {
        "enabled": True,
        "axis_order": list(FINGER_THRESHOLD_REQUIRED_FIELDS),
        "score": "mean visible fingertip ratio",
        "horizontal_axis": "camera_right_approx_from_shoulder_direction_and_character_relative_yaw",
        "vertical_axis": "world_y",
        "threshold_source": "0.85 * projected close-distance half extents",
        "body_height": "clip median head-to-toe relpos height",
        "close_distances": sorted(FINGER_THRESHOLD_CLOSE_DISTANCES),
        "target_anchor_joints": {key: list(value) for key, value in FINGER_THRESHOLD_TARGET_ANCHOR_JOINTS.items()},
        "fingertip_joints": list(FINGER_POSE_LEFT_TIPS + FINGER_POSE_RIGHT_TIPS),
    }
    return table.astype(np.float16), metadata


def main() -> None:
    args = build_parser().parse_args()
    label_fields = _parse_label_fields(str(args.label_fields))
    feature_root = args.feature_root.resolve()
    label_root = args.label_root.resolve()
    split_manifest_path = args.split_manifest.resolve()
    output_root = args.output_root.resolve()
    songs_output_root = output_root / "songs"
    feature_manifest = load_feature_manifest(feature_root)
    source_audio_feature_names = [] if bool(args.dance_only) else list(feature_manifest.get("audio_feature_names", []))
    source_dance_feature_names = list(feature_manifest.get("dance_feature_names", []))
    if not source_dance_feature_names:
        raise ValueError(f"Feature manifest does not contain dance_feature_names near {feature_root}")

    split_manifest = _load_json(split_manifest_path)
    excluded = set(str(song_id) for song_id in split_manifest.get("excluded", []))
    splits = {
        "train": [str(song_id) for song_id in split_manifest.get("train", [])],
        "val": [str(song_id) for song_id in split_manifest.get("val", [])],
        "test": [str(song_id) for song_id in split_manifest.get("test", [])],
    }
    ordered_song_ids = _resolve_song_ids(split_manifest)
    clip_lookup = {
        str(row.get("clip_id")): row
        for row in split_manifest.get("clips", [])
        if isinstance(row, dict) and row.get("clip_id") is not None
    }

    song_groups: dict[str, dict[int, list[dict[str, str]]]] = {}
    skipped_missing_label: list[str] = []
    skipped_empty_label: list[str] = []
    for song_id in ordered_song_ids:
        if song_id in excluded:
            continue
        label_path = label_root / song_id / str(args.label_filename)
        if not label_path.exists():
            skipped_missing_label.append(song_id)
            continue
        rows = _load_keyframes(label_path)
        grouped = _group_labels(rows, label_fields=label_fields)
        if not grouped:
            skipped_empty_label.append(song_id)
            continue
        song_groups[song_id] = grouped

    vocab, max_slots, slot_count_dist = _build_vocabularies(song_groups, label_fields=label_fields)
    field_to_id = {field: {value: idx for idx, value in enumerate(values)} for field, values in vocab.items()}

    feature_paths = {path.stem: path for path in feature_root.glob("*.npz")}
    skipped_missing_feature: list[str] = []
    skipped_missing_field: list[dict[str, Any]] = []
    skipped_clamped_frame: list[dict[str, Any]] = []
    skipped_missing_bonemask: list[str] = []
    bonemask_root = args.dca_bonemask_root.resolve() if args.dca_bonemask_root else None

    song_entries: list[dict[str, Any]] = []
    total_samples = 0
    audio_feature_dim = None
    dance_feature_dim = None
    output_dance_feature_names: list[str] | None = None
    compact_dance_meta: dict[str, object] | None = None
    finger_pose_hint_meta: dict[str, Any] = {"enabled": False, "feature_names": list(FINGER_POSE_HINT_NAMES)}
    finger_threshold_visibility_meta: dict[str, Any] = {
        "enabled": False,
        "axis_order": list(FINGER_THRESHOLD_REQUIRED_FIELDS),
    }

    window_radius = max(0, int(args.window_radius))
    window_stride = max(1, int(args.window_stride))
    window_offsets = _window_offsets(radius=window_radius, stride=window_stride)
    window_size = int(window_offsets.shape[0])
    prev_context_size = max(0, int(args.prev_context_size))
    future_context_size = max(0, int(args.future_context_size))

    for song_id in ordered_song_ids:
        if song_id in excluded or song_id not in song_groups:
            continue

        feature_path = feature_paths.get(song_id)
        if feature_path is None:
            skipped_missing_feature.append(song_id)
            continue

        npz = np.load(feature_path, allow_pickle=False)
        if "audio_features" not in npz.files or "dance_features" not in npz.files:
            skipped_missing_feature.append(song_id)
            continue

        audio = np.asarray(npz["audio_features"], dtype=np.float32)
        dance = np.asarray(npz["dance_features"], dtype=np.float32)
        full_dance = dance
        if audio.ndim != 2 or dance.ndim != 2 or audio.shape[0] != dance.shape[0]:
            raise ValueError(f"Unexpected feature shapes in {feature_path}")
        finger_pose_hints_source, finger_pose_hint_meta = _finger_pose_hints_from_full_dance(
            full_dance,
            source_dance_feature_names,
        )
        if str(args.dance_feature_set) == "compact":
            dance, output_dance_feature_names, compact_dance_meta = compact_dance_from_full(
                dance,
                source_dance_feature_names,
            )
        elif str(args.dance_feature_set) == "compact_hand_motion":
            dance, output_dance_feature_names, compact_dance_meta = compact_dance_hand_motion_from_full(
                dance,
                source_dance_feature_names,
            )
        elif str(args.dance_feature_set) == "compact_root_head_rot6d":
            dance, output_dance_feature_names, compact_dance_meta = compact_dance_root_head_rot6d_from_full(
                dance,
                source_dance_feature_names,
            )
        elif str(args.dance_feature_set) == "compact_root_head_wrist_rel_rot6d_vectors":
            dance, output_dance_feature_names, compact_dance_meta = compact_dance_roll_signals_from_full(
                dance,
                source_dance_feature_names,
            )
        elif output_dance_feature_names is None:
            output_dance_feature_names = source_dance_feature_names

        source_length = int(dance.shape[0]) if bool(args.dance_only) else int(audio.shape[0])
        source_indices = _downsample_source_indices(
            source_length,
            sample_fps=float(args.sample_fps),
            timeline_fps=float(args.timeline_fps),
        )
        audio_5fps = np.zeros((int(source_indices.shape[0]), 0), dtype=np.float32) if bool(args.dance_only) else audio[source_indices]
        dance_5fps = dance[source_indices]
        finger_pose_hints_5fps = finger_pose_hints_source[source_indices]

        if audio_feature_dim is None:
            audio_feature_dim = int(audio_5fps.shape[1])
        if dance_feature_dim is None:
            dance_feature_dim = int(dance_5fps.shape[1])

        grouped = song_groups[song_id]
        sample_frames = np.asarray(sorted(grouped.keys()), dtype=np.int32)
        num_samples = int(sample_frames.shape[0])
        bonemask = None
        bonemask_frame_offset = 0
        if bonemask_root is not None:
            split_name = _split_for_song(song_id, splits)
            bonemask_path, bonemask_uses_global_frames = _bonemask_path_for_song(
                bonemask_root=bonemask_root,
                song_id=song_id,
                split_name=split_name,
            )
            if not bonemask_path.exists():
                skipped_missing_bonemask.append(f"{song_id}:{bonemask_path}")
                continue
            bonemask = _load_bonemask_array(bonemask_path)
            if bonemask_uses_global_frames:
                bonemask_frame_offset = int(clip_lookup.get(song_id, {}).get("source_start_frame", 0))

        audio_windows = np.zeros((num_samples, window_size, audio_5fps.shape[1]), dtype=np.float32)
        dance_windows = np.zeros((num_samples, window_size, dance_5fps.shape[1]), dtype=np.float32)
        source_center_frames = np.zeros((num_samples,), dtype=np.int32)
        slot_count = np.zeros((num_samples,), dtype=np.int16)
        slot_mask = np.zeros((num_samples, max_slots), dtype=np.float32)
        cut_flag = np.zeros((num_samples,), dtype=np.float32)
        prev_frame_delta = np.zeros((num_samples, prev_context_size), dtype=np.float32)
        prev_slot_mask = np.zeros((num_samples, prev_context_size, max_slots), dtype=np.float32)
        future_frame_delta = np.zeros((num_samples, future_context_size), dtype=np.float32)
        future_frame_mask = np.zeros((num_samples, future_context_size), dtype=np.float32)
        bodypart_visible_target = (
            np.zeros((num_samples, len(BODYPART_PROXY_PARTS)), dtype=np.float32)
            if bonemask is not None
            else None
        )
        finger_pose_hint = np.zeros((num_samples, len(FINGER_POSE_HINT_NAMES)), dtype=np.float32)
        label_arrays = {
            field: np.full((num_samples, max_slots), PAD_VALUE, dtype=np.int16)
            for field in label_fields
        }
        prev_label_arrays = {
            field: np.full((num_samples, prev_context_size, max_slots), PAD_VALUE, dtype=np.int16)
            for field in label_fields
        }

        for sample_idx, sampled_frame in enumerate(sample_frames.tolist()):
            clamped_center = min(max(int(sampled_frame), 0), int(audio_5fps.shape[0]) - 1)
            if clamped_center != int(sampled_frame) and len(skipped_clamped_frame) < 200:
                skipped_clamped_frame.append(
                    {
                        "song_id": song_id,
                        "sampled_frame": int(sampled_frame),
                        "clamped_center": int(clamped_center),
                    }
                )
            indices = _window_indices(
                clamped_center,
                length=int(audio_5fps.shape[0]),
                radius=window_radius,
                stride=window_stride,
            )
            audio_windows[sample_idx] = audio_5fps[indices]
            dance_windows[sample_idx] = dance_5fps[indices]
            source_center_frames[sample_idx] = int(source_indices[clamped_center])
            finger_pose_hint[sample_idx] = finger_pose_hints_5fps[clamped_center]
            if bodypart_visible_target is not None and bonemask is not None:
                bodypart_visible_target[sample_idx] = _bodypart_visible_targets_from_bonemask(
                    bonemask,
                    np.asarray([source_center_frames[sample_idx]], dtype=np.int32),
                    frame_offset=bonemask_frame_offset,
                )[0]

            rows = grouped[int(sampled_frame)]
            slot_count[sample_idx] = len(rows)
            slot_mask[sample_idx, : len(rows)] = 1.0
            cut_flag[sample_idx] = 1.0 if len(rows) > 1 else 0.0
            _fill_label_slots(
                label_arrays,
                sample_idx=sample_idx,
                rows=rows,
                field_to_id=field_to_id,
                label_fields=label_fields,
            )

            previous_frames = sample_frames[:sample_idx].tolist()[-prev_context_size:] if prev_context_size else []
            future_frames = (
                sample_frames[sample_idx + 1 : sample_idx + 1 + future_context_size].tolist()
                if future_context_size
                else []
            )
            prev_start = prev_context_size - len(previous_frames)
            for prev_offset, prev_frame in enumerate(previous_frames, start=prev_start):
                prev_rows = grouped[int(prev_frame)]
                prev_frame_delta[sample_idx, prev_offset] = float(int(sampled_frame) - int(prev_frame))
                prev_slot_mask[sample_idx, prev_offset, : len(prev_rows)] = 1.0
                _fill_label_slots(
                    prev_label_arrays,
                    sample_idx=sample_idx,
                    rows=prev_rows,
                    field_to_id=field_to_id,
                    label_fields=label_fields,
                    prefix_index=(prev_offset,),
                )
            for future_offset, future_frame in enumerate(future_frames):
                future_frame_delta[sample_idx, future_offset] = float(int(future_frame) - int(sampled_frame))
                future_frame_mask[sample_idx, future_offset] = 1.0

        finger_threshold_visibility_table, finger_threshold_visibility_meta = (
            _finger_threshold_visibility_table_from_full_dance(
                full_dance,
                source_dance_feature_names,
                source_center_frames,
                vocab,
            )
        )

        output_npz = songs_output_root / f"{song_id}.npz"
        _write_npz(
            output_npz,
            force=bool(args.force),
            audio_windows=audio_windows,
            dance_windows=dance_windows,
            sampled_frames=sample_frames,
            source_center_frames=source_center_frames,
            slot_count=slot_count,
            slot_mask=slot_mask,
            cut_flag=cut_flag,
            prev_frame_delta=prev_frame_delta,
            prev_slot_mask=prev_slot_mask,
            future_frame_delta=future_frame_delta,
            future_frame_mask=future_frame_mask,
            finger_pose_hint=finger_pose_hint,
            finger_threshold_visibility_table=finger_threshold_visibility_table,
            **{f"label_{field}": values for field, values in label_arrays.items()},
            **{f"prev_label_{field}": values for field, values in prev_label_arrays.items()},
            **(
                {"bodypart_visible_target": bodypart_visible_target}
                if bodypart_visible_target is not None
                else {}
            ),
        )

        total_samples += num_samples
        song_entries.append(
            {
                "song_id": song_id,
                "path": str(output_npz),
                "num_samples": num_samples,
                "feature_frames_30fps": int(source_length),
                "feature_frames_5fps": int(audio_5fps.shape[0]),
                "audio_feature_dim": int(audio_5fps.shape[1]),
                "dance_feature_dim": int(dance_5fps.shape[1]),
                "max_slots_in_song": int(max(len(rows) for rows in grouped.values())),
            }
        )

    default_fusion = "film" if str(args.dance_feature_set).startswith("compact") else "interleave"
    manifest = {
        "dataset_type": "discrete_shotplan_transformer",
        "version": 1,
        "feature_root": str(feature_root),
        "label_root": str(label_root),
        "label_filename": str(args.label_filename),
        "split_manifest": str(split_manifest_path),
        "output_root": str(output_root),
        "sample_fps": float(args.sample_fps),
        "timeline_fps": float(args.timeline_fps),
        "window_radius": window_radius,
        "window_stride": window_stride,
        "window_offsets": [int(value) for value in window_offsets.tolist()],
        "window_size": window_size,
        "audio_feature_dim": int(audio_feature_dim or 0),
        "dance_feature_dim": int(dance_feature_dim or 0),
        "audio_feature_names": source_audio_feature_names,
        "input_modalities": ["dance"] if bool(args.dance_only) else ["audio", "dance"],
        "dance_only": bool(args.dance_only),
        "dance_feature_set": str(args.dance_feature_set),
        "source_dance_feature_dim": len(source_dance_feature_names),
        "dance_feature_names": output_dance_feature_names or [],
        "compact_dance_metadata": compact_dance_meta if str(args.dance_feature_set).startswith("compact") else None,
        "finger_pose_hint": finger_pose_hint_meta,
        "finger_threshold_visibility_table": finger_threshold_visibility_meta,
        "bodypart_visible_target": (
            {
                "enabled": True,
                "source": "DCA DCM++ BoneMask",
                "bonemask_root": str(bonemask_root),
                "parts": [part.name for part in BODYPART_PROXY_PARTS],
                "groups": BODYPART_BONEMASK_GROUPS,
            }
            if bonemask_root is not None
            else {"enabled": False}
        ),
        "fusion": default_fusion,
        "label_fields": list(label_fields),
        "pad_value": PAD_VALUE,
        "max_slots": int(max_slots),
        "prev_context_size": prev_context_size,
        "future_context_size": future_context_size,
        "input_context": {
            "previous_keyframes": prev_context_size,
            "previous_keyframe_source": "gt_consolidated_labels",
            "future_keyframes": future_context_size,
            "future_keyframe_source": "gt_consolidated_timing_only",
            "future_keyframe_values": ["frame_delta", "mask"],
            "cut_flag": "gt_slot_count_greater_than_one",
        },
        "vocabularies": vocab,
        "slot_count_distribution": {str(key): int(value) for key, value in sorted(slot_count_dist.items())},
        "total_song_count": len(song_entries),
        "total_sample_count": int(total_samples),
        "splits": splits,
        "songs": song_entries,
        "skipped": {
            "excluded": sorted(excluded),
            "missing_label": skipped_missing_label,
            "empty_label": skipped_empty_label,
            "missing_feature": skipped_missing_feature,
            "clamped_frame_count": len(skipped_clamped_frame),
            "clamped_frame_samples": skipped_clamped_frame,
            "missing_field": skipped_missing_field,
            "missing_bonemask": skipped_missing_bonemask,
        },
    }
    _write_json(output_root / "manifest.json", manifest, force=bool(args.force))

    print(f"output root: {output_root}")
    print(f"songs built: {len(song_entries)}")
    print(f"total samples: {total_samples}")
    print(f"max slots: {max_slots}")
    print(f"window radius: {window_radius}")
    print(f"window stride: {window_stride}")
    print(f"window size: {window_size}")
    print(f"manifest: {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
