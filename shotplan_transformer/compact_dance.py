from __future__ import annotations

import json
from pathlib import Path

import numpy as np


COMPACT_ROOT_JOINT_INDEX = 0
COMPACT_DANCE_JOINT_INDICES = [
    0,
    1,
    2,
    4,
    9,
    10,
    11,
    13,
    14,
    15,
    19,
    20,
    25,
    28,
    31,
    34,
    37,
    41,
    42,
    47,
    50,
    53,
    56,
    59,
]
COMPACT_DANCE_JOINT_NAMES = {
    0: "lower_body",
    1: "upper_body",
    2: "upper_body2",
    4: "head",
    9: "left_knee",
    10: "left_ankle",
    11: "left_toe",
    13: "right_knee",
    14: "right_ankle",
    15: "right_toe",
    19: "left_elbow",
    20: "left_wrist",
    25: "left_thumb_tip",
    28: "left_index_tip",
    31: "left_middle_tip",
    34: "left_ring_tip",
    37: "left_pinky_tip",
    41: "right_elbow",
    42: "right_wrist",
    47: "right_thumb_tip",
    50: "right_index_tip",
    53: "right_middle_tip",
    56: "right_ring_tip",
    59: "right_pinky_tip",
}
ROOT_HEAD_WORLD_ROT6D_FEATURE_NAMES = [
    f"dance_world_rot6d_{alias}_{column}_{axis}"
    for alias in ("root", "head")
    for column in ("c0", "c1")
    for axis in ("x", "y", "z")
]
WRIST_RELATIVE_ROT6D_FEATURE_NAMES = [
    f"dance_rel_rot6d_{alias}_from_parent_{column}_{axis}"
    for alias in ("left_wrist", "right_wrist")
    for column in ("c0", "c1")
    for axis in ("x", "y", "z")
]
FOOT_ROOT_HEAD_VECTOR_FEATURE_NAMES = [
    f"dance_vector_{vector_name}_{axis}"
    for vector_name in ("left_foot_to_root", "right_foot_to_root", "root_to_head")
    for axis in ("x", "y", "z")
]
ROLL_SIGNAL_FEATURE_NAMES = (
    ROOT_HEAD_WORLD_ROT6D_FEATURE_NAMES
    + WRIST_RELATIVE_ROT6D_FEATURE_NAMES
    + FOOT_ROOT_HEAD_VECTOR_FEATURE_NAMES
)
HAND_MOTION_LEFT_WRIST = 20
HAND_MOTION_RIGHT_WRIST = 42
HAND_MOTION_LEFT_TIPS = (25, 28, 31, 34, 37)
HAND_MOTION_RIGHT_TIPS = (47, 50, 53, 56, 59)
HAND_MOTION_EXTRA_FEATURE_NAMES = [
    "dance_left_hand_spread_mean",
    "dance_right_hand_spread_mean",
    "dance_left_hand_speed_mean",
    "dance_right_hand_speed_mean",
    "dance_left_hand_speed_max",
    "dance_right_hand_speed_max",
    "dance_left_finger_x_abs_mean",
    "dance_right_finger_x_abs_mean",
    "dance_left_finger_y_mean",
    "dance_right_finger_y_mean",
    "dance_left_finger_z_abs_mean",
    "dance_right_finger_z_abs_mean",
    "dance_both_hand_width_x",
    "dance_both_hand_height_y",
    "dance_hand_motion_energy",
]


def _columns_for_names(all_names: list[str], wanted_names: list[str]) -> list[int]:
    index_by_name = {name: idx for idx, name in enumerate(all_names)}
    missing = [name for name in wanted_names if name not in index_by_name]
    if missing:
        raise ValueError(f"Missing expected dance feature names: {missing[:8]}")
    return [index_by_name[name] for name in wanted_names]


def compact_dance_feature_names() -> list[str]:
    non_root_indices = [idx for idx in COMPACT_DANCE_JOINT_INDICES if idx != COMPACT_ROOT_JOINT_INDEX]
    relpos_names = [
        f"dance_relpos_j{joint_idx:02d}_{axis}"
        for joint_idx in non_root_indices
        for axis in ("x", "y", "z")
    ]
    relvel_names = [
        f"dance_relvel_j{joint_idx:02d}_{axis}"
        for joint_idx in non_root_indices
        for axis in ("x", "y", "z")
    ]
    root_pos_names = [f"dance_root_pos_{axis}" for axis in ("x", "y", "z")]
    root_vel_names = [f"dance_root_vel_{axis}" for axis in ("x", "y", "z")]
    speed_names = [f"dance_joint_speed_j{joint_idx:02d}" for joint_idx in non_root_indices]
    return relpos_names + relvel_names + root_pos_names + root_vel_names + speed_names + ["dance_compact_motion_energy"]


def compact_dance_root_head_rot6d_feature_names() -> list[str]:
    return compact_dance_feature_names() + list(ROOT_HEAD_WORLD_ROT6D_FEATURE_NAMES)


def compact_dance_roll_signals_feature_names() -> list[str]:
    return compact_dance_feature_names() + list(ROLL_SIGNAL_FEATURE_NAMES)


def compact_dance_hand_motion_feature_names() -> list[str]:
    return compact_dance_feature_names() + list(HAND_MOTION_EXTRA_FEATURE_NAMES)


def compact_dance_metadata() -> dict[str, object]:
    non_root_indices = [idx for idx in COMPACT_DANCE_JOINT_INDICES if idx != COMPACT_ROOT_JOINT_INDEX]
    return {
        "compact_root_joint_index": COMPACT_ROOT_JOINT_INDEX,
        "compact_joint_indices": COMPACT_DANCE_JOINT_INDICES,
        "compact_non_root_joint_indices": non_root_indices,
        "compact_joint_names": {str(idx): COMPACT_DANCE_JOINT_NAMES[idx] for idx in COMPACT_DANCE_JOINT_INDICES},
        "compact_dance_feature_dim": len(compact_dance_feature_names()),
        "compact_motion_energy": "mean speed over selected non-root joints",
    }


def compact_dance_root_head_rot6d_metadata() -> dict[str, object]:
    metadata = compact_dance_metadata()
    metadata.update(
        {
            "compact_dance_feature_dim": len(compact_dance_root_head_rot6d_feature_names()),
            "world_rotation_feature": "root and head world rotations encoded as 6D matrix columns",
            "world_rotation_joints": {"root": 0, "head": 4},
            "world_rotation_feature_names": list(ROOT_HEAD_WORLD_ROT6D_FEATURE_NAMES),
        }
    )
    return metadata


def compact_dance_roll_signals_metadata() -> dict[str, object]:
    metadata = compact_dance_root_head_rot6d_metadata()
    metadata.update(
        {
            "compact_dance_feature_dim": len(compact_dance_roll_signals_feature_names()),
            "wrist_relative_rotation_feature": "left/right wrist rotations relative to parent elbow encoded as 6D",
            "wrist_relative_rotation_joints": {
                "left_wrist": {"parent": 19, "child": 20},
                "right_wrist": {"parent": 41, "child": 42},
            },
            "foot_root_head_vector_feature": "left/right foot-to-root and root-to-head world-space vectors",
            "foot_root_head_vector_joints": {"root": 0, "head": 4, "left_foot": 10, "right_foot": 14},
            "wrist_relative_rotation_feature_names": list(WRIST_RELATIVE_ROT6D_FEATURE_NAMES),
            "foot_root_head_vector_feature_names": list(FOOT_ROOT_HEAD_VECTOR_FEATURE_NAMES),
        }
    )
    return metadata


def compact_dance_hand_motion_metadata() -> dict[str, object]:
    metadata = compact_dance_metadata()
    metadata.update(
        {
            "compact_dance_feature_dim": len(compact_dance_hand_motion_feature_names()),
            "hand_motion_feature": (
                "Extra wrist/fingertip spread, speed, and span cues intended to help "
                "camera labels reason about whether fingers/hands are visible."
            ),
            "hand_motion_joints": {
                "left_wrist": HAND_MOTION_LEFT_WRIST,
                "right_wrist": HAND_MOTION_RIGHT_WRIST,
                "left_fingertips": list(HAND_MOTION_LEFT_TIPS),
                "right_fingertips": list(HAND_MOTION_RIGHT_TIPS),
            },
            "hand_motion_feature_names": list(HAND_MOTION_EXTRA_FEATURE_NAMES),
        }
    )
    return metadata


def compact_dance_from_full(
    dance_features: np.ndarray,
    full_dance_feature_names: list[str],
) -> tuple[np.ndarray, list[str], dict[str, object]]:
    non_root_indices = [idx for idx in COMPACT_DANCE_JOINT_INDICES if idx != COMPACT_ROOT_JOINT_INDEX]
    names = compact_dance_feature_names()
    selected_names = names[:-1]
    speed_names = [f"dance_joint_speed_j{joint_idx:02d}" for joint_idx in non_root_indices]
    selected_columns = _columns_for_names(full_dance_feature_names, selected_names)
    speed_columns = _columns_for_names(full_dance_feature_names, speed_names)
    compact_features = np.concatenate(
        [
            dance_features[:, selected_columns],
            np.mean(dance_features[:, speed_columns], axis=1, keepdims=True),
        ],
        axis=1,
    ).astype(np.float32)
    return compact_features, names, compact_dance_metadata()


def _relpos_columns(joint_indices: tuple[int, ...] | list[int], axis: str, full_names: list[str]) -> list[int]:
    return _columns_for_names(full_names, [f"dance_relpos_j{joint_idx:02d}_{axis}" for joint_idx in joint_indices])


def _relpos_xyz(dance_features: np.ndarray, joint_indices: tuple[int, ...] | list[int], full_names: list[str]) -> np.ndarray:
    columns = _columns_for_names(
        full_names,
        [
            f"dance_relpos_j{joint_idx:02d}_{axis}"
            for joint_idx in joint_indices
            for axis in ("x", "y", "z")
        ],
    )
    return dance_features[:, columns].reshape(dance_features.shape[0], len(joint_indices), 3)


def _speed_columns(joint_indices: tuple[int, ...] | list[int], full_names: list[str]) -> list[int]:
    return _columns_for_names(full_names, [f"dance_joint_speed_j{joint_idx:02d}" for joint_idx in joint_indices])


def compact_dance_hand_motion_from_full(
    dance_features: np.ndarray,
    full_dance_feature_names: list[str],
) -> tuple[np.ndarray, list[str], dict[str, object]]:
    """Return compact 168-D dance features plus explicit hand/finger motion cues."""

    compact_features, _compact_names, _compact_metadata = compact_dance_from_full(
        dance_features,
        full_dance_feature_names,
    )
    left_wrist = _relpos_xyz(dance_features, [HAND_MOTION_LEFT_WRIST], full_dance_feature_names)[:, 0, :]
    right_wrist = _relpos_xyz(dance_features, [HAND_MOTION_RIGHT_WRIST], full_dance_feature_names)[:, 0, :]
    left_tips = _relpos_xyz(dance_features, list(HAND_MOTION_LEFT_TIPS), full_dance_feature_names)
    right_tips = _relpos_xyz(dance_features, list(HAND_MOTION_RIGHT_TIPS), full_dance_feature_names)
    left_speeds = dance_features[:, _speed_columns([HAND_MOTION_LEFT_WRIST, *HAND_MOTION_LEFT_TIPS], full_dance_feature_names)]
    right_speeds = dance_features[:, _speed_columns([HAND_MOTION_RIGHT_WRIST, *HAND_MOTION_RIGHT_TIPS], full_dance_feature_names)]
    all_tips = np.concatenate([left_tips, right_tips], axis=1)

    left_spread = np.linalg.norm(left_tips - left_wrist[:, None, :], axis=-1).mean(axis=1, keepdims=True)
    right_spread = np.linalg.norm(right_tips - right_wrist[:, None, :], axis=-1).mean(axis=1, keepdims=True)
    left_speed_mean = left_speeds.mean(axis=1, keepdims=True)
    right_speed_mean = right_speeds.mean(axis=1, keepdims=True)
    left_speed_max = left_speeds.max(axis=1, keepdims=True)
    right_speed_max = right_speeds.max(axis=1, keepdims=True)
    left_x_abs = np.abs(left_tips[:, :, 0]).mean(axis=1, keepdims=True)
    right_x_abs = np.abs(right_tips[:, :, 0]).mean(axis=1, keepdims=True)
    left_y = left_tips[:, :, 1].mean(axis=1, keepdims=True)
    right_y = right_tips[:, :, 1].mean(axis=1, keepdims=True)
    left_z_abs = np.abs(left_tips[:, :, 2]).mean(axis=1, keepdims=True)
    right_z_abs = np.abs(right_tips[:, :, 2]).mean(axis=1, keepdims=True)
    hand_width = (all_tips[:, :, 0].max(axis=1) - all_tips[:, :, 0].min(axis=1)).reshape(-1, 1)
    hand_height = (all_tips[:, :, 1].max(axis=1) - all_tips[:, :, 1].min(axis=1)).reshape(-1, 1)
    hand_energy = np.concatenate([left_speeds, right_speeds], axis=1).mean(axis=1, keepdims=True)
    hand_features = np.concatenate(
        [
            left_spread,
            right_spread,
            left_speed_mean,
            right_speed_mean,
            left_speed_max,
            right_speed_max,
            left_x_abs,
            right_x_abs,
            left_y,
            right_y,
            left_z_abs,
            right_z_abs,
            hand_width,
            hand_height,
            hand_energy,
        ],
        axis=1,
    ).astype(np.float32)
    features = np.concatenate([compact_features, hand_features], axis=1).astype(np.float32)
    return features, compact_dance_hand_motion_feature_names(), compact_dance_hand_motion_metadata()


def compact_dance_root_head_rot6d_from_full(
    dance_features: np.ndarray,
    full_dance_feature_names: list[str],
) -> tuple[np.ndarray, list[str], dict[str, object]]:
    compact_features, _compact_names, _compact_metadata = compact_dance_from_full(
        dance_features,
        full_dance_feature_names,
    )
    rotation_columns = _columns_for_names(full_dance_feature_names, ROOT_HEAD_WORLD_ROT6D_FEATURE_NAMES)
    features = np.concatenate([compact_features, dance_features[:, rotation_columns]], axis=1).astype(np.float32)
    return features, compact_dance_root_head_rot6d_feature_names(), compact_dance_root_head_rot6d_metadata()


def compact_dance_roll_signals_from_full(
    dance_features: np.ndarray,
    full_dance_feature_names: list[str],
) -> tuple[np.ndarray, list[str], dict[str, object]]:
    compact_features, _compact_names, _compact_metadata = compact_dance_from_full(
        dance_features,
        full_dance_feature_names,
    )
    signal_columns = _columns_for_names(full_dance_feature_names, ROLL_SIGNAL_FEATURE_NAMES)
    features = np.concatenate([compact_features, dance_features[:, signal_columns]], axis=1).astype(np.float32)
    return features, compact_dance_roll_signals_feature_names(), compact_dance_roll_signals_metadata()


def load_feature_manifest(feature_root: Path) -> dict[str, object]:
    feature_root = Path(feature_root)
    candidates = [
        feature_root / "manifest.json",
        feature_root.parent / "manifest.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Could not find feature manifest near {feature_root}")
