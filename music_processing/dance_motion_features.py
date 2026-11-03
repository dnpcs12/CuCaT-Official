from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_MOTION_ROOT = Path(__file__).resolve().parents[1] / "local_assets" / "Simplified_MotionGlobalTransform"
ROOT_HEAD_WORLD_ROTATION_JOINTS: tuple[tuple[str, int], ...] = (
    ("root", 0),
    ("head", 4),
)
WRIST_RELATIVE_ROTATION_JOINTS: tuple[tuple[str, int, int], ...] = (
    ("left_wrist", 19, 20),
    ("right_wrist", 41, 42),
)
FOOT_ROOT_HEAD_VECTOR_JOINTS = {
    "root": 0,
    "head": 4,
    "left_foot": 10,
    "right_foot": 14,
}


@dataclass(frozen=True)
class BodyPartSpec:
    key: str
    display_name: str
    joint_names: tuple[str, ...]
    joint_speed_reducer: str = "mean"
    threshold_group: str | None = None
    threshold_scale: float = 1.0


BODY_PART_SPECS: tuple[BodyPartSpec, ...] = (
    BodyPartSpec(
        key="head",
        display_name="Head",
        joint_names=("首", "頭", "両目", "左目", "右目"),
    ),
    BodyPartSpec(
        key="left_arm",
        display_name="Left arm",
        joint_names=("左肩P", "左肩", "左腕", "左ひじ", "左腕捩", "左手首", "左手捩"),
        joint_speed_reducer="p75",
        threshold_group="arm",
    ),
    BodyPartSpec(
        key="right_arm",
        display_name="Right arm",
        joint_names=("右肩P", "右肩", "右腕", "右ひじ", "右腕捩", "右手首", "右手捩"),
        joint_speed_reducer="p75",
        threshold_group="arm",
    ),
    BodyPartSpec(
        key="left_leg",
        display_name="Left leg",
        joint_names=("左足", "左ひざ", "左足首", "左つま先"),
        joint_speed_reducer="max",
        threshold_group="leg",
    ),
    BodyPartSpec(
        key="right_leg",
        display_name="Right leg",
        joint_names=("右足", "右ひざ", "右足首", "右つま先"),
        joint_speed_reducer="max",
        threshold_group="leg",
    ),
    BodyPartSpec(
        key="left_hand",
        display_name="Left hand",
        threshold_group="hand",
        joint_names=(
            "左手首",
            "左手捩",
            "左親指０",
            "左親指１",
            "左親指２",
            "左人指１",
            "左人指２",
            "左人指３",
            "左中指１",
            "左中指２",
            "左中指３",
            "左薬指１",
            "左薬指２",
            "左薬指３",
            "左小指１",
            "左小指２",
            "左小指３",
        ),
    ),
    BodyPartSpec(
        key="right_hand",
        display_name="Right hand",
        threshold_group="hand",
        joint_names=(
            "右手首",
            "右手捩",
            "右親指０",
            "右親指１",
            "右親指２",
            "右人指１",
            "右人指２",
            "右人指３",
            "右中指１",
            "右中指２",
            "右中指３",
            "右薬指１",
            "右薬指２",
            "右薬指３",
            "右小指１",
            "右小指２",
            "右小指３",
        ),
    ),
    BodyPartSpec(
        key="full_body",
        display_name="Full body",
        joint_names=("下半身", "上半身", "上半身2", "左足", "右足"),
    ),
)


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_motion_path(index: int, motion_root: Path = DEFAULT_MOTION_ROOT) -> Path:
    return Path(motion_root) / f"m{int(index)}_GlobalTransform.json"


def load_motion_transforms(motion_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    payload = load_json(Path(motion_path))
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected motion JSON format: {motion_path}")

    bone_names_raw = payload.get("BoneList")
    if not isinstance(bone_names_raw, list) or not bone_names_raw:
        raise ValueError(f"No BoneList found: {motion_path}")
    bone_names = [str(name) for name in bone_names_raw]

    records = payload.get("BoneKeyFrameTransformRecord")
    if not isinstance(records, list) or not records:
        raise ValueError(f"No BoneKeyFrameTransformRecord found: {motion_path}")

    sorted_records = sorted(records, key=lambda row: int(row["FrameTime"]))
    frame_numbers: list[int] = []
    positions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    joint_count: int | None = None

    for row in sorted_records:
        frame_numbers.append(int(row["FrameTime"]))
        transform = np.asarray(row["Transform"], dtype=np.float32)
        if transform.ndim != 1 or transform.size % 16 != 0:
            raise ValueError(f"Unexpected transform size in {motion_path}")

        current_joint_count = int(transform.size // 16)
        if joint_count is None:
            joint_count = current_joint_count
        elif current_joint_count != joint_count:
            raise ValueError(f"Inconsistent joint count in {motion_path}")

        matrices = transform.reshape(current_joint_count, 4, 4)
        positions.append(matrices[:, 3, :3])
        rotations.append(matrices[:, :3, :3])

    if joint_count is None:
        raise ValueError(f"No joints found in {motion_path}")
    if joint_count != len(bone_names):
        raise ValueError(
            f"Joint count / BoneList mismatch in {motion_path}: {joint_count} transforms vs {len(bone_names)} bones"
        )

    return (
        np.asarray(frame_numbers, dtype=np.int32),
        np.stack(positions, axis=0).astype(np.float32),
        np.stack(rotations, axis=0).astype(np.float32),
        bone_names,
    )


def load_motion_positions(motion_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    frame_numbers, positions, _rotations, bone_names = load_motion_transforms(motion_path)
    return frame_numbers, positions, bone_names


def align_motion_array(values: np.ndarray, *, target_frames: int, label: str) -> tuple[np.ndarray, str]:
    current_frames = int(values.shape[0])
    if current_frames == target_frames:
        return values, "matched"
    if current_frames > target_frames:
        return values[:target_frames], f"cropped_from_{current_frames}"
    if current_frames <= 0:
        raise ValueError(f"Motion {label} are empty.")
    pad_count = target_frames - current_frames
    padding = np.repeat(values[-1:], pad_count, axis=0)
    return np.concatenate([values, padding], axis=0), f"edge_padded_from_{current_frames}"


def align_motion_positions(positions: np.ndarray, *, target_frames: int) -> tuple[np.ndarray, str]:
    return align_motion_array(positions, target_frames=target_frames, label="positions")


def align_motion_rotations(rotations: np.ndarray, *, target_frames: int) -> tuple[np.ndarray, str]:
    return align_motion_array(rotations, target_frames=target_frames, label="rotations")


def build_world_rotation_6d_feature_names(
    rotation_joints: tuple[tuple[str, int], ...] = ROOT_HEAD_WORLD_ROTATION_JOINTS,
) -> list[str]:
    names: list[str] = []
    for alias, _joint_idx in rotation_joints:
        for column in ("c0", "c1"):
            for axis in ("x", "y", "z"):
                names.append(f"dance_world_rot6d_{alias}_{column}_{axis}")
    return names


def build_world_rotation_6d_features(
    rotations: np.ndarray,
    rotation_joints: tuple[tuple[str, int], ...] = ROOT_HEAD_WORLD_ROTATION_JOINTS,
) -> tuple[np.ndarray, list[str]]:
    if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotations shape (frames, joints, 3, 3), got {rotations.shape}")
    selected = []
    for _alias, joint_idx in rotation_joints:
        if joint_idx < 0 or joint_idx >= rotations.shape[1]:
            raise ValueError(f"Rotation joint index out of range: {joint_idx} for {rotations.shape[1]} joints")
        # The first two world-rotation matrix columns form a continuous 6D representation.
        selected.append(np.transpose(rotations[:, joint_idx, :, :2], (0, 2, 1)).reshape(rotations.shape[0], 6))
    if not selected:
        return np.zeros((rotations.shape[0], 0), dtype=np.float32), []
    return np.concatenate(selected, axis=1).astype(np.float32), build_world_rotation_6d_feature_names(rotation_joints)


def build_wrist_relative_rotation_6d_feature_names(
    rotation_joints: tuple[tuple[str, int, int], ...] = WRIST_RELATIVE_ROTATION_JOINTS,
) -> list[str]:
    names: list[str] = []
    for alias, _parent_idx, _child_idx in rotation_joints:
        for column in ("c0", "c1"):
            for axis in ("x", "y", "z"):
                names.append(f"dance_rel_rot6d_{alias}_from_parent_{column}_{axis}")
    return names


def build_wrist_relative_rotation_6d_features(
    rotations: np.ndarray,
    rotation_joints: tuple[tuple[str, int, int], ...] = WRIST_RELATIVE_ROTATION_JOINTS,
) -> tuple[np.ndarray, list[str]]:
    if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotations shape (frames, joints, 3, 3), got {rotations.shape}")
    selected = []
    for _alias, parent_idx, child_idx in rotation_joints:
        for joint_idx in (parent_idx, child_idx):
            if joint_idx < 0 or joint_idx >= rotations.shape[1]:
                raise ValueError(f"Rotation joint index out of range: {joint_idx} for {rotations.shape[1]} joints")
        parent_world = rotations[:, parent_idx]
        child_world = rotations[:, child_idx]
        # Global transforms use row-vector convention, so child-local = child-world * inverse(parent-world).
        relative = np.matmul(child_world, np.swapaxes(parent_world, -1, -2))
        selected.append(np.transpose(relative[:, :, :2], (0, 2, 1)).reshape(rotations.shape[0], 6))
    if not selected:
        return np.zeros((rotations.shape[0], 0), dtype=np.float32), []
    return (
        np.concatenate(selected, axis=1).astype(np.float32),
        build_wrist_relative_rotation_6d_feature_names(rotation_joints),
    )


def build_foot_root_head_vector_feature_names() -> list[str]:
    names: list[str] = []
    for vector_name in ("left_foot_to_root", "right_foot_to_root", "root_to_head"):
        for axis in ("x", "y", "z"):
            names.append(f"dance_vector_{vector_name}_{axis}")
    return names


def build_foot_root_head_vector_features(positions: np.ndarray) -> tuple[np.ndarray, list[str]]:
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"Expected positions shape (frames, joints, 3), got {positions.shape}")
    max_required = max(FOOT_ROOT_HEAD_VECTOR_JOINTS.values())
    if positions.shape[1] <= max_required:
        raise ValueError(f"Need at least {max_required + 1} joints for foot/root/head vectors, got {positions.shape[1]}")
    root = positions[:, FOOT_ROOT_HEAD_VECTOR_JOINTS["root"]]
    head = positions[:, FOOT_ROOT_HEAD_VECTOR_JOINTS["head"]]
    left_foot = positions[:, FOOT_ROOT_HEAD_VECTOR_JOINTS["left_foot"]]
    right_foot = positions[:, FOOT_ROOT_HEAD_VECTOR_JOINTS["right_foot"]]
    features = np.concatenate(
        [
            root - left_foot,
            root - right_foot,
            head - root,
        ],
        axis=1,
    ).astype(np.float32)
    return features, build_foot_root_head_vector_feature_names()


def build_motion_feature_names(
    joint_count: int,
    *,
    include_root_head_world_rot6d: bool = False,
    include_wrist_relative_rot6d: bool = False,
    include_foot_root_head_vectors: bool = False,
) -> list[str]:
    names: list[str] = []
    for prefix in ("dance_relpos", "dance_relvel"):
        for joint_idx in range(joint_count):
            for axis in ("x", "y", "z"):
                names.append(f"{prefix}_j{joint_idx:02d}_{axis}")
    for axis in ("x", "y", "z"):
        names.append(f"dance_root_pos_{axis}")
    for axis in ("x", "y", "z"):
        names.append(f"dance_root_vel_{axis}")
    for joint_idx in range(joint_count):
        names.append(f"dance_joint_speed_j{joint_idx:02d}")
    names.append("dance_motion_energy")
    if include_root_head_world_rot6d:
        names.extend(build_world_rotation_6d_feature_names())
    if include_wrist_relative_rot6d:
        names.extend(build_wrist_relative_rotation_6d_feature_names())
    if include_foot_root_head_vectors:
        names.extend(build_foot_root_head_vector_feature_names())
    return names


def build_motion_features(
    positions: np.ndarray,
    *,
    rotations: np.ndarray | None = None,
    include_root_head_world_rot6d: bool = False,
    include_wrist_relative_rot6d: bool = False,
    include_foot_root_head_vectors: bool = False,
) -> tuple[np.ndarray, list[str]]:
    frames, joint_count, _ = positions.shape
    root = positions[:, :1, :]
    root_relative = positions - root
    position_velocity = np.diff(positions, axis=0, prepend=positions[:1])
    root_relative_velocity = np.diff(root_relative, axis=0, prepend=root_relative[:1])
    root_velocity = position_velocity[:, 0, :]
    joint_speeds = np.linalg.norm(root_relative_velocity, axis=-1)
    motion_energy = np.mean(joint_speeds, axis=1, keepdims=True)

    features = np.concatenate(
        [
            root_relative.reshape(frames, joint_count * 3),
            root_relative_velocity.reshape(frames, joint_count * 3),
            root[:, 0, :],
            root_velocity,
            joint_speeds,
            motion_energy,
        ],
        axis=1,
    ).astype(np.float32)
    if include_root_head_world_rot6d:
        if rotations is None:
            raise ValueError("rotations must be provided when include_root_head_world_rot6d=True")
        if rotations.shape[:2] != positions.shape[:2]:
            raise ValueError(f"Position/rotation shape mismatch: {positions.shape} vs {rotations.shape}")
        rotation_features, _rotation_feature_names = build_world_rotation_6d_features(rotations)
        features = np.concatenate([features, rotation_features], axis=1).astype(np.float32)
    if include_wrist_relative_rot6d:
        if rotations is None:
            raise ValueError("rotations must be provided when include_wrist_relative_rot6d=True")
        if rotations.shape[:2] != positions.shape[:2]:
            raise ValueError(f"Position/rotation shape mismatch: {positions.shape} vs {rotations.shape}")
        wrist_features, _wrist_feature_names = build_wrist_relative_rotation_6d_features(rotations)
        features = np.concatenate([features, wrist_features], axis=1).astype(np.float32)
    if include_foot_root_head_vectors:
        vector_features, _vector_feature_names = build_foot_root_head_vector_features(positions)
        features = np.concatenate([features, vector_features], axis=1).astype(np.float32)
    return features, build_motion_feature_names(
        joint_count,
        include_root_head_world_rot6d=include_root_head_world_rot6d,
        include_wrist_relative_rot6d=include_wrist_relative_rot6d,
        include_foot_root_head_vectors=include_foot_root_head_vectors,
    )


def build_window_starts(length: int, window: int, stride: int) -> list[int]:
    if window <= 0:
        raise ValueError("window must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    last_start = length - window
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def reduce_joint_series(values: np.ndarray, reducer: str) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError(f"Expected 2D joint-speed array, got shape {values.shape}")
    if values.shape[1] == 0:
        return np.zeros(values.shape[0], dtype=np.float32)
    if reducer == "mean":
        return np.mean(values, axis=1).astype(np.float32)
    if reducer == "max":
        return np.max(values, axis=1).astype(np.float32)
    if reducer == "p75":
        return np.percentile(values, 75, axis=1).astype(np.float32)
    if reducer == "p90":
        return np.percentile(values, 90, axis=1).astype(np.float32)
    raise ValueError(f"Unsupported reducer: {reducer}")


def compute_global_joint_speed_series(
    dance_features: np.ndarray,
    feature_names: list[str],
    bone_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    feature_index = {name: idx for idx, name in enumerate(feature_names)}
    speed_names = [f"dance_joint_speed_j{joint_idx:02d}" for joint_idx in range(1, len(bone_names))]
    missing_columns = [name for name in speed_names if name not in feature_index]
    if missing_columns:
        raise ValueError(f"Missing global speed features: {missing_columns}")
    columns = [feature_index[name] for name in speed_names]
    return reduce_joint_series(dance_features[:, columns], "mean"), speed_names


def resolve_body_part_metadata(bone_names: list[str]) -> dict[str, dict]:
    index_by_name = {name: idx for idx, name in enumerate(bone_names)}
    metadata: dict[str, dict] = {}
    missing_parts: dict[str, list[str]] = {}

    for spec in BODY_PART_SPECS:
        joint_indices = [index_by_name[name] for name in spec.joint_names if name in index_by_name]
        missing_names = [name for name in spec.joint_names if name not in index_by_name]
        if not joint_indices:
            missing_parts[spec.key] = missing_names
            continue
        metadata[spec.key] = {
            "display_name": spec.display_name,
            "joint_names": list(spec.joint_names),
            "joint_indices": joint_indices,
            "missing_joint_names": missing_names,
            "joint_speed_reducer": spec.joint_speed_reducer,
            "threshold_group": spec.threshold_group or spec.key,
            "threshold_scale": float(spec.threshold_scale),
        }

    if missing_parts:
        raise ValueError(f"Unable to resolve body-part joints from BoneList: {missing_parts}")
    return metadata


def compute_body_part_speed_series(
    dance_features: np.ndarray,
    feature_names: list[str],
    bone_names: list[str],
) -> dict[str, dict]:
    metadata = resolve_body_part_metadata(bone_names)
    feature_index = {name: idx for idx, name in enumerate(feature_names)}
    global_reference_series, global_reference_feature_names = compute_global_joint_speed_series(
        dance_features,
        feature_names,
        bone_names,
    )
    positive_global_reference = global_reference_series[global_reference_series > 0.0]
    global_reference_floor = (
        max(float(np.percentile(positive_global_reference, 5)) * 0.05, 1e-6)
        if positive_global_reference.size
        else 1e-6
    )
    score_series: dict[str, dict] = {}

    for key, info in metadata.items():
        speed_names = [f"dance_joint_speed_j{joint_idx:02d}" for joint_idx in info["joint_indices"]]
        missing_columns = [name for name in speed_names if name not in feature_index]
        if missing_columns:
            raise ValueError(f"Missing speed features for {key}: {missing_columns}")
        columns = [feature_index[name] for name in speed_names]
        joint_speeds = dance_features[:, columns]
        series = reduce_joint_series(joint_speeds, str(info["joint_speed_reducer"]))
        score_series[key] = {
            **info,
            "speed_feature_names": speed_names,
            "global_reference_feature_names": global_reference_feature_names,
            "global_reference_reducer": "mean",
            "global_reference_floor": global_reference_floor,
            "raw_series": series,
            "global_reference_series": global_reference_series,
            "series": series,
        }
    return score_series


def reduce_window(values: np.ndarray, reducer: str) -> float:
    if values.size == 0:
        return 0.0
    if reducer == "mean":
        return float(np.mean(values))
    if reducer == "max":
        return float(np.max(values))
    if reducer == "p75":
        return float(np.percentile(values, 75))
    if reducer == "p90":
        return float(np.percentile(values, 90))
    raise ValueError(f"Unsupported reducer: {reducer}")
