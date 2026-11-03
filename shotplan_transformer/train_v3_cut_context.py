from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import wandb
except ImportError:
    wandb = None

from shotplan_transformer.dataset import (
    compute_normalization_stats,
    label_fields_from_manifest,
    load_manifest,
    save_normalization_stats,
)
from shotplan_transformer.v3_cut_context_model import V3CutContextConfig, V3CutContextShotPlanModel
from shotplan_transformer.semantic_loss import (
    build_semantic_target_distributions,
    parse_semantic_fields,
    semantic_cross_entropy,
)
from shotplan_transformer.bodypart_proxy import (
    EXPORTED_DISTANCE_MULTIPLIERS,
    FOV_DEGREES,
    PARTS as BODYPART_PROXY_PARTS,
    TARGET_ANCHOR_Y_FRACTIONS,
    proxy_calibration_for_mode,
    proxy_visibility_score,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (
    WORKSPACE_ROOT
    / "shotplan_transformer"
    / "results"
    / "discrete_v3_30fps_middle_candidate_dcm_split_dataset"
)
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "shotplan_transformer" / "results" / "v3_cut_context_training_runs"
DEFAULT_WANDB_KEY_PATH = WORKSPACE_ROOT / "music_processing" / ".wandb_api_key"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ReverseV3 discrete shot-plan model with precomputed cut context.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoints/epoch_XXX.pt every N epochs; 0 disables periodic epoch checkpoints.")
    parser.add_argument("--save-epochs", default="", help="Comma-separated extra epoch numbers to save.")
    parser.add_argument("--keep-epoch-checkpoints", type=int, default=0, help="Keep only the most recent N epoch checkpoints; 0 keeps all.")
    parser.add_argument(
        "--architecture",
        choices=["transformer", "transformer_decoder", "transformer_decoder_full", "mlp"],
        default="transformer",
    )
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--mlp-extra-layers", type=int, default=0, help="Additional hidden blocks for the MLP backbone.")
    parser.add_argument(
        "--stage-specific-heads",
        action="store_true",
        help="Use separate output heads for cut and ordinary keyframe stages after the shared trunk.",
    )
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--input-dropout", type=float, default=0.15)
    parser.add_argument(
        "--fusion",
        choices=("film", "dance_film", "concat"),
        default="film",
        help=(
            "Audio/dance fusion mode. 'film' uses dance-conditioned FiLM over audio; "
            "'dance_film' uses audio-conditioned FiLM over dance; 'concat' projects "
            "concatenated audio+dance tokens."
        ),
    )
    parser.add_argument("--max-position-tokens", type=int, default=16)
    parser.add_argument(
        "--position-context-mode",
        choices=("radius", "prevnext"),
        default="radius",
        help=(
            "How key/cut position tokens are selected. 'radius' keeps the legacy "
            "nearest tokens within the feature window; 'prevnext' uses previous N "
            "and next N positions without a frame-radius limit."
        ),
    )
    parser.add_argument(
        "--position-context-side-count",
        type=int,
        default=8,
        help="Number of previous and next key/cut positions used when --position-context-mode=prevnext.",
    )
    parser.add_argument("--cut-sample-weight", type=float, default=1.5)
    parser.add_argument("--keyframe-sample-weight", type=float, default=1.0)
    parser.add_argument(
        "--semantic-loss-weight",
        type=float,
        default=0.0,
        help="Blend hard CE with semantic soft-target CE for ordered/circular V3 fields.",
    )
    parser.add_argument("--semantic-loss-temperature", type=float, default=0.35)
    parser.add_argument(
        "--semantic-loss-fields",
        default="target_anchor,distance,yaw,pitch,composition,roll,fov",
    )
    parser.add_argument(
        "--precut-adjacent-contrast-weight",
        type=float,
        default=0.5,
        help=(
            "For the frame immediately before a cut transition, penalize "
            "predicting the same value as the next cut-transition frame on "
            "fields that actually changed in GT."
        ),
    )
    parser.add_argument(
        "--precut-adjacent-contrast-fields",
        default="",
        help=(
            "Comma-separated fields used by --precut-adjacent-contrast-weight. "
            "Empty keeps the legacy behavior and uses all label fields."
        ),
    )
    parser.add_argument(
        "--distance-fov-joint-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary CE weight for an explicit distance+fov joint head.",
    )
    parser.add_argument(
        "--predict-distance-fov-joint-as-field",
        action="store_true",
        help=(
            "Use a distance_fov joint head as the primary supervision for distance/FOV, "
            "instead of supervising distance and fov with independent CE heads. "
            "Separate distance/fov logits are still derived from the joint logits for "
            "auxiliary differentiable losses and compatibility metrics."
        ),
    )
    parser.add_argument(
        "--visual-scale-contrast-weight",
        type=float,
        default=0.0,
        help=(
            "GT-aware pre-cut contrast loss on the perceived distance+FOV scale. "
            "This treats diagonal distance/FOV combinations, such as "
            "extreme_close+wide and very_close+normal, as visually similar."
        ),
    )
    parser.add_argument(
        "--visual-scale-contrast-margin",
        type=float,
        default=1.0,
        help="Minimum expected visual-scale rank gap required when GT has a distance/FOV scale change at a cut.",
    )
    parser.add_argument(
        "--noncut-smoothness-loss-weight",
        type=float,
        default=0.0,
        help="Penalty weight for non-cut predictions that jump too far from nearest history labels.",
    )
    parser.add_argument(
        "--noncut-smoothness-fields",
        default="distance,fov,yaw",
        help="Comma-separated fields used by non-cut smoothness loss.",
    )
    parser.add_argument(
        "--noncut-smoothness-max-delta",
        type=float,
        default=45.0,
        help="Only apply non-cut smoothness when nearest history keyframe is within this many frames.",
    )
    parser.add_argument(
        "--noncut-smoothness-margin",
        type=float,
        default=1.0,
        help="Allowed expected class-rank movement before smoothness penalty starts.",
    )
    parser.add_argument(
        "--bodypart-proxy-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary loss that matches the GT/predicted body-part visibility "
            "proxy implied by target_anchor+distance+fov+pitch. 0 disables it."
        ),
    )
    parser.add_argument(
        "--bodypart-proxy-mode",
        choices=("baseline", "calibrated", "pose_dynamic_y"),
        default="calibrated",
        help=(
            "Calibration mode used by the body-part visibility proxy loss. "
            "'pose_dynamic_y' derives body-part vertical ranges from the current dance pose."
        ),
    )
    parser.add_argument(
        "--bodypart-proxy-loss-type",
        choices=("mse", "bce"),
        default="mse",
        help="Loss type for the auxiliary body-part proxy target.",
    )
    parser.add_argument(
        "--bodypart-proxy-loss-stage",
        choices=("all", "cut", "keyframe"),
        default="all",
        help="Which cut-context stage receives the body-part proxy auxiliary loss.",
    )
    parser.add_argument(
        "--pose-conditioned-finger-proxy",
        action="store_true",
        help=(
            "Use dataset finger_pose_hint to make the fingers body-part proxy depend on "
            "the current hand/finger pose, while keeping other body-part proxies static."
        ),
    )
    parser.add_argument(
        "--bodypart-proxy-finger-source",
        choices=("proxy", "threshold"),
        default="proxy",
        help=(
            "How the fingers dimension is supervised inside bodypart_proxy_loss. "
            "'proxy' keeps the legacy coarse body-part proxy; 'threshold' replaces "
            "only the fingers dimension with dataset finger_threshold_visibility_table "
            "and bodypart_visible_target."
        ),
    )
    parser.add_argument(
        "--bodypart-proxy-finger-pred-threshold",
        type=float,
        default=0.20,
        help="Diagnostic hard-score threshold for threshold-sourced fingers inside bodypart_proxy_loss.",
    )
    parser.add_argument(
        "--visible-bodypart-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary loss against dataset-provided bodypart_visible_target, "
            "usually DCA BoneMask visibility aggregated into the proxy body parts."
        ),
    )
    parser.add_argument(
        "--visible-bodypart-loss-type",
        choices=("mse", "bce"),
        default="mse",
        help="Loss type for bodypart_visible_target auxiliary supervision.",
    )
    parser.add_argument(
        "--visible-bodypart-loss-stage",
        choices=("all", "cut", "keyframe"),
        default="all",
        help="Which cut-context stage receives bodypart_visible_target auxiliary loss.",
    )
    parser.add_argument(
        "--finger-absence-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary loss for samples where GT BoneMask says fingers are absent. "
            "It penalizes high predicted finger visibility without forcing other body parts."
        ),
    )
    parser.add_argument(
        "--finger-absence-loss-stage",
        choices=("all", "cut", "keyframe"),
        default="keyframe",
        help="Which cut-context stage receives the finger absence auxiliary loss.",
    )
    parser.add_argument(
        "--finger-absence-target-threshold",
        type=float,
        default=0.20,
        help="Treat GT finger visibility below this BoneMask score as absent.",
    )
    parser.add_argument(
        "--finger-missing-threshold-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary loss for samples where GT BoneMask says fingers are visible. "
            "It uses the dataset-provided finger_threshold_visibility_table and "
            "penalizes predicted shot combinations that would crop the fingers out."
        ),
    )
    parser.add_argument(
        "--finger-missing-threshold-loss-stage",
        choices=("all", "cut", "keyframe"),
        default="keyframe",
        help="Which cut-context stage receives the threshold-based missing-finger loss.",
    )
    parser.add_argument(
        "--finger-missing-threshold-target-threshold",
        type=float,
        default=0.20,
        help="Treat GT finger visibility at or above this BoneMask score as visible.",
    )
    parser.add_argument(
        "--finger-missing-threshold-pred-threshold",
        type=float,
        default=0.20,
        help="Diagnostic hard-score threshold for missing-finger false-negative metrics.",
    )
    parser.add_argument(
        "--finger-visibility-threshold-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Symmetric auxiliary loss for finger visibility. It matches GT "
            "BoneMask finger visibility to the expected visibility implied by "
            "target_anchor+distance+fov+yaw, penalizing both missing and extra fingers."
        ),
    )
    parser.add_argument(
        "--finger-visibility-threshold-loss-stage",
        choices=("all", "cut", "keyframe"),
        default="keyframe",
        help="Which cut-context stage receives the symmetric threshold-based finger visibility loss.",
    )
    parser.add_argument(
        "--finger-visibility-threshold-pred-threshold",
        type=float,
        default=0.20,
        help="Diagnostic hard-score threshold for symmetric finger visibility metrics.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--wandb-project", default="shotplan-v3-cut-context")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-key-file", type=Path, default=DEFAULT_WANDB_KEY_PATH)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--tag", default="")
    return parser


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _build_run_dir(output_root: Path, tag: str) -> Path:
    suffix = f"_{tag}" if tag else ""
    run_dir = output_root / f"v3_cut_context_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _parse_epoch_set(value: str) -> set[int]:
    epochs: set[int] = set()
    for piece in str(value or "").split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            start_text, end_text = piece.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if end < start:
                start, end = end, start
            epochs.update(range(start, end + 1))
        else:
            epochs.add(int(piece))
    return epochs


def _build_joint_vocab_sizes(args: argparse.Namespace, vocab_sizes: dict[str, int]) -> dict[str, int]:
    if (
        float(getattr(args, "distance_fov_joint_loss_weight", 0.0)) <= 0.0
        and not bool(getattr(args, "predict_distance_fov_joint_as_field", False))
    ):
        return {}
    if "distance" not in vocab_sizes or "fov" not in vocab_sizes:
        return {}
    return {"distance_fov": int(vocab_sizes["distance"] * vocab_sizes["fov"])}


def _nearest_indices(frames: np.ndarray, flags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = int(frames.shape[0])
    prev_idx = np.full((n,), -1, dtype=np.int32)
    next_idx = np.full((n,), -1, dtype=np.int32)
    last = -1
    for i in range(n):
        prev_idx[i] = last
        if flags[i]:
            last = i
    last = -1
    for i in range(n - 1, -1, -1):
        next_idx[i] = last
        if flags[i]:
            last = i
    return prev_idx, next_idx


def _position_context(
    frames: np.ndarray,
    centers: np.ndarray,
    flags: np.ndarray | None,
    *,
    radius: int,
    max_count: int,
    mode: str = "radius",
    side_count: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(centers.shape[0])
    delta = np.zeros((n, max_count), dtype=np.float32)
    mask = np.zeros((n, max_count), dtype=np.float32)
    candidate_indices = np.arange(n, dtype=np.int32) if flags is None else np.flatnonzero(flags)
    for i, center in enumerate(centers):
        if candidate_indices.size <= 0:
            continue
        cand_frames = frames[candidate_indices]
        if str(mode) == "prevnext":
            # Fixed layout: previous slots first, future slots second. This
            # gives long-range timing context without tying it to the feature
            # window radius. The current frame itself is intentionally omitted.
            prev_slots = min(max(0, int(side_count)), int(max_count))
            next_slots = min(max(0, int(side_count)), max(0, int(max_count) - prev_slots))
            previous = candidate_indices[cand_frames < int(center)]
            future = candidate_indices[cand_frames > int(center)]
            previous = previous[-prev_slots:] if prev_slots > 0 else previous[:0]
            future = future[:next_slots] if next_slots > 0 else future[:0]
            prev_count = int(previous.size)
            next_count = int(future.size)
            if prev_count:
                start = prev_slots - prev_count
                end = prev_slots
                delta[i, start:end] = frames[previous].astype(np.float32) - float(center)
                mask[i, start:end] = 1.0
            if next_count:
                start = prev_slots
                end = prev_slots + next_count
                delta[i, start:end] = frames[future].astype(np.float32) - float(center)
                mask[i, start:end] = 1.0
        else:
            inside = np.abs(cand_frames - int(center)) <= int(radius)
            selected = candidate_indices[inside]
            if selected.size <= 0:
                continue
            selected = selected[np.argsort(np.abs(frames[selected] - int(center)))]
            selected = selected[:max_count]
            count = int(selected.size)
            delta[i, :count] = frames[selected].astype(np.float32) - float(center)
            mask[i, :count] = 1.0
    return delta, mask


POSE_PROXY_TARGET_ANCHOR_JOINTS: dict[str, tuple[int, ...]] = {
    "feet": (11, 15),
    "lower_leg": (10, 14),
    "thigh": (9, 13),
    "hip": (1,),
    "waist": (1,),
    "chest": (2,),
    "face": (4,),
}

POSE_PROXY_PART_JOINTS: dict[str, tuple[int, ...]] = {
    "head": (4,),
    "arms_upper_body": (2, 4, 19, 20, 25, 28, 31, 34, 37, 41, 42, 47, 50, 53, 56, 59),
    "fingers": (25, 28, 31, 34, 37, 47, 50, 53, 56, 59),
    "waist": (1, 2),
    "legs": (9, 10, 13, 14),
    "feet": (10, 11, 14, 15),
}

POSE_PROXY_PART_MARGINS: dict[str, tuple[float, float]] = {
    "head": (0.16, 0.08),
    "arms_upper_body": (0.04, 0.04),
    "fingers": (0.04, 0.04),
    "waist": (0.03, 0.03),
    "legs": (0.03, 0.03),
    "feet": (0.08, 0.06),
}

POSE_DYNAMIC_PROXY_THRESHOLDS: dict[str, float] = {
    "head": 0.80,
    "arms_upper_body": 0.50,
    "fingers": 0.50,
    "waist": 0.50,
    "legs": 0.50,
    "feet": 0.35,
}


def _compact_relpos_y_indices(feature_names: list[str]) -> dict[int, int]:
    """Map compact dance joint ids to their y-feature column."""

    out: dict[int, int] = {}
    for index, name in enumerate(feature_names):
        if not name.startswith("dance_relpos_j") or not name.endswith("_y"):
            continue
        joint_text = name[len("dance_relpos_j") : -len("_y")]
        try:
            out[int(joint_text)] = int(index)
        except ValueError:
            continue
    return out


def _compute_pose_bodypart_proxy_hints(
    dance_windows: np.ndarray,
    dance_feature_names: list[str],
    target_anchor_vocab: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pose-aware vertical anchor/body-part ranges from raw compact dance windows."""

    sample_count = int(dance_windows.shape[0])
    anchor_y = np.zeros((sample_count, len(target_anchor_vocab)), dtype=np.float32)
    part_bounds = np.zeros((sample_count, len(BODYPART_PROXY_PARTS), 2), dtype=np.float32)
    if sample_count <= 0 or dance_windows.ndim != 3:
        return anchor_y, part_bounds

    center = int(dance_windows.shape[1] // 2)
    center_pose = np.asarray(dance_windows[:, center, :], dtype=np.float32)
    y_indices = _compact_relpos_y_indices(list(dance_feature_names))
    if not y_indices:
        for anchor_index, target_anchor in enumerate(target_anchor_vocab):
            anchor_y[:, anchor_index] = float(
                TARGET_ANCHOR_Y_FRACTIONS.get(str(target_anchor), TARGET_ANCHOR_Y_FRACTIONS["waist"])
            )
        for part_index, part in enumerate(BODYPART_PROXY_PARTS):
            part_bounds[:, part_index, 0] = float(part.y0)
            part_bounds[:, part_index, 1] = float(part.y1)
        return anchor_y, part_bounds

    def _joint_y(joint: int) -> np.ndarray | None:
        column = y_indices.get(int(joint))
        if column is None:
            return None
        return center_pose[:, column]

    head_y = _joint_y(4)
    toe_candidates = [values for values in (_joint_y(11), _joint_y(15)) if values is not None]
    if head_y is None or not toe_candidates:
        foot_y = np.zeros((sample_count,), dtype=np.float32)
        body_height = 1.0
    else:
        foot_y = np.min(np.stack(toe_candidates, axis=1), axis=1)
        heights = np.asarray(head_y - foot_y, dtype=np.float32)
        valid_heights = heights[np.isfinite(heights) & (heights > 1e-4)]
        body_height = float(np.median(valid_heights)) if valid_heights.size else 1.0
        if not math.isfinite(body_height) or body_height <= 1e-4:
            body_height = 1.0

    def _normalized_joint_stack(joints: tuple[int, ...]) -> np.ndarray | None:
        values = [_joint_y(joint) for joint in joints]
        valid_values = [value for value in values if value is not None]
        if not valid_values:
            return None
        stacked = np.stack(valid_values, axis=1)
        return (stacked - foot_y[:, None]) / float(body_height)

    for anchor_index, target_anchor in enumerate(target_anchor_vocab):
        stacked = _normalized_joint_stack(POSE_PROXY_TARGET_ANCHOR_JOINTS.get(str(target_anchor), ()))
        if stacked is None:
            anchor_y[:, anchor_index] = float(
                TARGET_ANCHOR_Y_FRACTIONS.get(str(target_anchor), TARGET_ANCHOR_Y_FRACTIONS["waist"])
            )
        else:
            anchor_y[:, anchor_index] = np.mean(stacked, axis=1).astype(np.float32)

    for part_index, part in enumerate(BODYPART_PROXY_PARTS):
        stacked = _normalized_joint_stack(POSE_PROXY_PART_JOINTS.get(part.name, ()))
        down_margin, up_margin = POSE_PROXY_PART_MARGINS.get(part.name, (0.03, 0.03))
        if stacked is None:
            part_bounds[:, part_index, 0] = float(part.y0)
            part_bounds[:, part_index, 1] = float(part.y1)
        else:
            part_bounds[:, part_index, 0] = np.min(stacked, axis=1).astype(np.float32) - float(down_margin)
            part_bounds[:, part_index, 1] = np.max(stacked, axis=1).astype(np.float32) + float(up_margin)
    return anchor_y.astype(np.float32), part_bounds.astype(np.float32)


class V3CutContextDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        split_name: str,
        *,
        normalization_stats: dict[str, np.ndarray],
        max_position_tokens: int,
        position_context_mode: str = "radius",
        position_context_side_count: int = 8,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split_name = split_name
        self.manifest = load_manifest(self.dataset_root)
        self.fields = label_fields_from_manifest(self.manifest)
        self.vocab_sizes = {field: len(self.manifest["vocabularies"][field]) for field in self.fields}
        self.max_position_tokens = int(max_position_tokens)
        self.position_context_mode = str(position_context_mode)
        self.position_context_side_count = int(position_context_side_count)
        self.window_radius = int(self.manifest.get("window_radius", self.manifest.get("window_size", 49) // 2))
        song_lookup = {entry["song_id"]: entry for entry in self.manifest["songs"]}

        audio_chunks = []
        dance_chunks = []
        frame_chunks = []
        cut_flag_chunks = []
        key_delta_chunks = []
        key_mask_chunks = []
        cut_delta_chunks = []
        cut_mask_chunks = []
        prev_cut_delta_chunks = []
        prev_cut_mask_chunks = []
        next_cut_delta_chunks = []
        next_cut_mask_chunks = []
        history_delta_chunks = []
        history_mask_chunks = []
        bodypart_visible_chunks = []
        finger_pose_hint_chunks = []
        finger_threshold_visibility_chunks = []
        pose_bodypart_anchor_y_chunks = []
        pose_bodypart_part_bounds_chunks = []
        label_chunks = {field: [] for field in self.fields}
        prev_cut_label_chunks = {field: [] for field in self.fields}
        next_cut_label_chunks = {field: [] for field in self.fields}
        pre_cut_label_chunks = {field: [] for field in self.fields}
        post_cut_label_chunks = {field: [] for field in self.fields}
        history_label_chunks = {field: [] for field in self.fields}
        pre_cut_mask_chunks = []
        post_cut_mask_chunks = []
        song_ids: list[str] = []

        for song_id in self.manifest["splits"][split_name]:
            entry = song_lookup.get(song_id)
            if entry is None:
                continue
            arrays = np.load(entry["path"], allow_pickle=False)
            audio = np.asarray(arrays["audio_windows"], dtype=np.float32)
            dance = np.asarray(arrays["dance_windows"], dtype=np.float32)
            frames = np.asarray(arrays["sampled_frames"], dtype=np.int32)
            if audio.shape[0] <= 0:
                continue
            labels = {field: np.asarray(arrays[f"label_{field}"], dtype=np.int64)[:, 0] for field in self.fields}
            prev_labels = {field: np.asarray(arrays[f"prev_label_{field}"], dtype=np.int64)[:, :, 0] for field in self.fields}
            prev_frame_delta = np.asarray(arrays["prev_frame_delta"], dtype=np.float32)
            prev_slot_mask = np.asarray(arrays["prev_slot_mask"], dtype=np.float32)[:, :, 0]
            bodypart_visible = (
                np.asarray(arrays["bodypart_visible_target"], dtype=np.float32)
                if "bodypart_visible_target" in arrays.files
                else np.zeros((audio.shape[0], 0), dtype=np.float32)
            )
            finger_pose_hint = (
                np.asarray(arrays["finger_pose_hint"], dtype=np.float32)
                if "finger_pose_hint" in arrays.files
                else np.zeros((audio.shape[0], 0), dtype=np.float32)
            )
            finger_threshold_visibility = (
                np.asarray(arrays["finger_threshold_visibility_table"], dtype=np.float32)
                if "finger_threshold_visibility_table" in arrays.files
                else np.zeros((audio.shape[0], 0, 0, 0, 0), dtype=np.float32)
            )
            pose_anchor_y, pose_part_bounds = _compute_pose_bodypart_proxy_hints(
                dance,
                list(self.manifest.get("dance_feature_names", [])),
                [str(value) for value in self.manifest["vocabularies"].get("target_anchor", [])],
            )
            cut_flags = np.zeros((frames.shape[0],), dtype=np.float32)
            cut_flags[1:] = (np.diff(frames) == 1).astype(np.float32)
            pre_cut_mask = cut_flags.copy()
            post_cut_mask = np.zeros((frames.shape[0],), dtype=np.float32)
            post_cut_mask[:-1] = (np.diff(frames) == 1).astype(np.float32)
            prev_cut_idx, next_cut_idx = _nearest_indices(frames, cut_flags > 0.5)
            key_delta, key_mask = _position_context(
                frames,
                frames,
                None,
                radius=self.window_radius,
                max_count=self.max_position_tokens,
                mode=self.position_context_mode,
                side_count=self.position_context_side_count,
            )
            cut_delta, cut_mask = _position_context(
                frames,
                frames,
                cut_flags > 0.5,
                radius=self.window_radius,
                max_count=self.max_position_tokens,
                mode=self.position_context_mode,
                side_count=self.position_context_side_count,
            )

            audio_chunks.append(audio)
            dance_chunks.append(dance)
            frame_chunks.append(frames)
            cut_flag_chunks.append(cut_flags)
            key_delta_chunks.append(key_delta)
            key_mask_chunks.append(key_mask)
            cut_delta_chunks.append(cut_delta)
            cut_mask_chunks.append(cut_mask)
            pre_cut_mask_chunks.append(pre_cut_mask)
            post_cut_mask_chunks.append(post_cut_mask)
            history_delta_chunks.append(prev_frame_delta)
            history_mask_chunks.append(prev_slot_mask)
            bodypart_visible_chunks.append(bodypart_visible)
            finger_pose_hint_chunks.append(finger_pose_hint)
            finger_threshold_visibility_chunks.append(finger_threshold_visibility)
            pose_bodypart_anchor_y_chunks.append(pose_anchor_y)
            pose_bodypart_part_bounds_chunks.append(pose_part_bounds)
            song_ids.extend([song_id] * int(frames.shape[0]))

            prev_mask = (prev_cut_idx >= 0).astype(np.float32)
            next_mask = (next_cut_idx >= 0).astype(np.float32)
            prev_delta = np.zeros_like(cut_flags, dtype=np.float32)
            next_delta = np.zeros_like(cut_flags, dtype=np.float32)
            valid_prev = prev_cut_idx >= 0
            valid_next = next_cut_idx >= 0
            prev_delta[valid_prev] = frames[valid_prev].astype(np.float32) - frames[prev_cut_idx[valid_prev]].astype(np.float32)
            next_delta[valid_next] = frames[next_cut_idx[valid_next]].astype(np.float32) - frames[valid_next].astype(np.float32)
            prev_cut_delta_chunks.append(prev_delta)
            prev_cut_mask_chunks.append(prev_mask)
            next_cut_delta_chunks.append(next_delta)
            next_cut_mask_chunks.append(next_mask)

            for field in self.fields:
                label_chunks[field].append(labels[field])
                history_label_chunks[field].append(prev_labels[field])
                prev_values = np.full((frames.shape[0],), -100, dtype=np.int64)
                next_values = np.full((frames.shape[0],), -100, dtype=np.int64)
                pre_cut_values = np.full((frames.shape[0],), -100, dtype=np.int64)
                post_cut_values = np.full((frames.shape[0],), -100, dtype=np.int64)
                prev_values[valid_prev] = labels[field][prev_cut_idx[valid_prev]]
                next_values[valid_next] = labels[field][next_cut_idx[valid_next]]
                cut_indices = np.flatnonzero(cut_flags > 0.5)
                pre_cut_values[cut_indices] = labels[field][cut_indices - 1]
                pre_transition_indices = np.flatnonzero(post_cut_mask > 0.5)
                post_cut_values[pre_transition_indices] = labels[field][pre_transition_indices + 1]
                prev_cut_label_chunks[field].append(prev_values)
                next_cut_label_chunks[field].append(next_values)
                pre_cut_label_chunks[field].append(pre_cut_values)
                post_cut_label_chunks[field].append(post_cut_values)

        if not audio_chunks:
            raise ValueError(f"No samples for split {split_name}")
        audio_mean = normalization_stats["audio_mean"].reshape(1, 1, -1)
        audio_std = normalization_stats["audio_std"].reshape(1, 1, -1)
        dance_mean = normalization_stats["dance_mean"].reshape(1, 1, -1)
        dance_std = normalization_stats["dance_std"].reshape(1, 1, -1)
        self.audio = (np.concatenate(audio_chunks, axis=0) - audio_mean) / audio_std
        self.dance = (np.concatenate(dance_chunks, axis=0) - dance_mean) / dance_std
        self.frames = np.concatenate(frame_chunks, axis=0)
        self.cut_flag = np.concatenate(cut_flag_chunks, axis=0)
        self.key_delta = np.concatenate(key_delta_chunks, axis=0)
        self.key_mask = np.concatenate(key_mask_chunks, axis=0)
        self.cut_delta = np.concatenate(cut_delta_chunks, axis=0)
        self.cut_mask = np.concatenate(cut_mask_chunks, axis=0)
        self.pre_cut_mask = np.concatenate(pre_cut_mask_chunks, axis=0)
        self.post_cut_mask = np.concatenate(post_cut_mask_chunks, axis=0)
        self.prev_cut_delta = np.concatenate(prev_cut_delta_chunks, axis=0)
        self.prev_cut_mask = np.concatenate(prev_cut_mask_chunks, axis=0)
        self.next_cut_delta = np.concatenate(next_cut_delta_chunks, axis=0)
        self.next_cut_mask = np.concatenate(next_cut_mask_chunks, axis=0)
        self.history_delta = np.concatenate(history_delta_chunks, axis=0)
        self.history_mask = np.concatenate(history_mask_chunks, axis=0)
        self.bodypart_visible_target = np.concatenate(bodypart_visible_chunks, axis=0)
        self.finger_pose_hint = np.concatenate(finger_pose_hint_chunks, axis=0)
        self.finger_threshold_visibility_table = np.concatenate(finger_threshold_visibility_chunks, axis=0)
        self.pose_bodypart_anchor_y = np.concatenate(pose_bodypart_anchor_y_chunks, axis=0)
        self.pose_bodypart_part_bounds = np.concatenate(pose_bodypart_part_bounds_chunks, axis=0)
        self.labels = {field: np.concatenate(values, axis=0) for field, values in label_chunks.items()}
        self.prev_cut_labels = {field: np.concatenate(values, axis=0) for field, values in prev_cut_label_chunks.items()}
        self.next_cut_labels = {field: np.concatenate(values, axis=0) for field, values in next_cut_label_chunks.items()}
        self.pre_cut_labels = {field: np.concatenate(values, axis=0) for field, values in pre_cut_label_chunks.items()}
        self.post_cut_labels = {field: np.concatenate(values, axis=0) for field, values in post_cut_label_chunks.items()}
        self.history_labels = {field: np.concatenate(values, axis=0) for field, values in history_label_chunks.items()}
        self.song_ids = song_ids

    def __len__(self) -> int:
        return int(self.audio.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            "audio": torch.from_numpy(self.audio[idx]).float(),
            "dance": torch.from_numpy(self.dance[idx]).float(),
            "frame": torch.tensor(int(self.frames[idx]), dtype=torch.long),
            "cut_flag": torch.tensor(float(self.cut_flag[idx]), dtype=torch.float32),
            "key_position_delta": torch.from_numpy(self.key_delta[idx]).float(),
            "key_position_mask": torch.from_numpy(self.key_mask[idx]).float(),
            "cut_position_delta": torch.from_numpy(self.cut_delta[idx]).float(),
            "cut_position_mask": torch.from_numpy(self.cut_mask[idx]).float(),
            "pre_cut_mask": torch.tensor(float(self.pre_cut_mask[idx]), dtype=torch.float32),
            "post_cut_mask": torch.tensor(float(self.post_cut_mask[idx]), dtype=torch.float32),
            "prev_cut_delta": torch.tensor(float(self.prev_cut_delta[idx]), dtype=torch.float32),
            "prev_cut_mask": torch.tensor(float(self.prev_cut_mask[idx]), dtype=torch.float32),
            "next_cut_delta": torch.tensor(float(self.next_cut_delta[idx]), dtype=torch.float32),
            "next_cut_mask": torch.tensor(float(self.next_cut_mask[idx]), dtype=torch.float32),
            "history_delta": torch.from_numpy(self.history_delta[idx]).float(),
            "history_mask": torch.from_numpy(self.history_mask[idx]).float(),
            "bodypart_visible_target": torch.from_numpy(self.bodypart_visible_target[idx]).float(),
            "finger_pose_hint": torch.from_numpy(self.finger_pose_hint[idx]).float(),
            "finger_threshold_visibility_table": torch.from_numpy(self.finger_threshold_visibility_table[idx]).float(),
            "pose_bodypart_anchor_y": torch.from_numpy(self.pose_bodypart_anchor_y[idx]).float(),
            "pose_bodypart_part_bounds": torch.from_numpy(self.pose_bodypart_part_bounds[idx]).float(),
            "song_id": self.song_ids[idx],
        }
        for field in self.fields:
            item[f"label_{field}"] = torch.tensor(int(self.labels[field][idx]), dtype=torch.long)
            item[f"prev_cut_label_{field}"] = torch.tensor(int(self.prev_cut_labels[field][idx]), dtype=torch.long)
            item[f"next_cut_label_{field}"] = torch.tensor(int(self.next_cut_labels[field][idx]), dtype=torch.long)
            item[f"pre_cut_label_{field}"] = torch.tensor(int(self.pre_cut_labels[field][idx]), dtype=torch.long)
            item[f"post_cut_label_{field}"] = torch.tensor(int(self.post_cut_labels[field][idx]), dtype=torch.long)
            item[f"history_label_{field}"] = torch.from_numpy(self.history_labels[field][idx]).long()
        return item


def _batch_labels(batch: dict[str, Any], prefix: str, fields: tuple[str, ...], device: torch.device) -> dict[str, torch.Tensor]:
    return {field: batch[f"{prefix}{field}"].to(device) for field in fields}


def _field_loss_and_metrics(
    outputs: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    fields: tuple[str, ...],
    *,
    semantic_soft_targets: dict[str, torch.Tensor],
    semantic_loss_weight: float,
    predict_distance_fov_joint_as_field: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses = []
    accs = []
    metrics: dict[str, float] = {}
    all_correct = None
    joint_distance_correct = None
    joint_fov_correct = None
    for field in fields:
        if bool(predict_distance_fov_joint_as_field) and field in {"distance", "fov"}:
            continue
        logits = outputs[f"{field}_logits"]
        target = labels[field].to(logits.device).long()
        hard_loss = F.cross_entropy(logits, target)
        semantic_weight = max(0.0, min(1.0, float(semantic_loss_weight)))
        if semantic_weight > 0.0 and field in semantic_soft_targets:
            semantic_raw = semantic_cross_entropy(
                logits,
                target,
                soft_targets=semantic_soft_targets[field],
                ignore_index=-100,
            )
            semantic_loss = semantic_raw.mean()
            loss = (1.0 - semantic_weight) * hard_loss + semantic_weight * semantic_loss
        else:
            loss = hard_loss
        pred = logits.argmax(dim=-1)
        correct = pred.eq(target)
        losses.append(loss)
        acc = correct.float().mean().item()
        metrics[f"{field}_acc"] = float(acc)
        accs.append(acc)
        all_correct = correct if all_correct is None else (all_correct & correct)
    if bool(predict_distance_fov_joint_as_field) and "distance_fov_logits" in outputs and "distance" in labels and "fov" in labels:
        joint_loss, joint_metrics = _distance_fov_joint_loss_and_metrics(outputs, labels)
        losses.append(joint_loss)
        logits = outputs["distance_fov_logits"]
        fov_vocab_size = int(outputs["fov_logits"].shape[-1])
        distance_target = labels["distance"].to(logits.device).long()
        fov_target = labels["fov"].to(logits.device).long()
        valid = (distance_target >= 0) & (fov_target >= 0)
        if torch.any(valid):
            joint_pred = logits.argmax(dim=-1)
            joint_distance = joint_pred // max(1, fov_vocab_size)
            joint_fov = joint_pred % max(1, fov_vocab_size)
            joint_distance_correct = joint_distance.eq(distance_target)
            joint_fov_correct = joint_fov.eq(fov_target)
            distance_acc = float(joint_distance_correct[valid].float().mean().detach().cpu().item())
            fov_acc = float(joint_fov_correct[valid].float().mean().detach().cpu().item())
            metrics["distance_acc"] = distance_acc
            metrics["fov_acc"] = fov_acc
            accs.extend([distance_acc, fov_acc])
            pair_correct = joint_distance_correct & joint_fov_correct
            all_correct = pair_correct if all_correct is None else (all_correct & pair_correct)
        metrics.update(joint_metrics)
    metrics["mean_field_acc"] = float(np.mean(accs)) if accs else 0.0
    metrics["all_field_acc"] = float(all_correct.float().mean().item()) if all_correct is not None else 0.0
    return torch.stack(losses).mean(), metrics


def _with_distance_fov_marginal_logits(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Expose distance/fov logits as differentiable marginals of the joint head.

    The model may predict distance+FOV with one categorical head.  Existing
    visual-scale/body-proxy auxiliary losses read ``distance_logits`` and
    ``fov_logits`` independently, so for the joint experiment we replace those
    logits with log-sum-exp marginals from ``distance_fov_logits``.
    """

    if "distance_fov_logits" not in outputs or "distance_logits" not in outputs or "fov_logits" not in outputs:
        return outputs
    distance_count = int(outputs["distance_logits"].shape[-1])
    fov_count = int(outputs["fov_logits"].shape[-1])
    if distance_count <= 0 or fov_count <= 0:
        return outputs
    joint_logits = outputs["distance_fov_logits"]
    if int(joint_logits.shape[-1]) != distance_count * fov_count:
        return outputs
    joint_grid = joint_logits.reshape(*joint_logits.shape[:-1], distance_count, fov_count)
    updated = dict(outputs)
    updated["distance_logits"] = torch.logsumexp(joint_grid, dim=-1)
    updated["fov_logits"] = torch.logsumexp(joint_grid, dim=-2)
    return updated


def _distance_fov_joint_loss_and_metrics(
    outputs: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Auxiliary CE for an explicit distance+fov joint head."""

    if "distance_fov_logits" not in outputs or "distance" not in labels or "fov" not in labels:
        first = next(iter(outputs.values()))
        return first.new_tensor(0.0), {"distance_fov_joint_acc": 0.0}
    logits = outputs["distance_fov_logits"]
    fov_vocab_size = int(outputs["fov_logits"].shape[-1])
    target = labels["distance"].to(logits.device).long() * fov_vocab_size + labels["fov"].to(logits.device).long()
    valid = (labels["distance"].to(logits.device) >= 0) & (labels["fov"].to(logits.device) >= 0)
    if not torch.any(valid):
        return logits.new_tensor(0.0), {"distance_fov_joint_acc": 0.0}
    loss = F.cross_entropy(logits[valid], target[valid])
    pred = logits[valid].argmax(dim=-1)
    return loss, {"distance_fov_joint_acc": float(pred.eq(target[valid]).float().mean().detach().cpu().item())}


_BODYPART_PROXY_SCORE_CACHE: dict[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], str], torch.Tensor] = {}


def _bodypart_proxy_score_table(
    *,
    target_anchor_vocab: list[str],
    distance_vocab: list[str],
    fov_vocab: list[str],
    pitch_vocab: list[str],
    proxy_mode: str,
    device: torch.device,
) -> torch.Tensor:
    """Return score table [part, anchor, distance, fov, pitch] for proxy loss."""

    key = (
        tuple(str(value) for value in target_anchor_vocab),
        tuple(str(value) for value in distance_vocab),
        tuple(str(value) for value in fov_vocab),
        tuple(str(value) for value in pitch_vocab),
        str(proxy_mode),
    )
    cached = _BODYPART_PROXY_SCORE_CACHE.get(key)
    if cached is None:
        calibration = proxy_calibration_for_mode(str(proxy_mode))
        table = torch.zeros(
            (
                len(BODYPART_PROXY_PARTS),
                len(target_anchor_vocab),
                len(distance_vocab),
                len(fov_vocab),
                len(pitch_vocab),
            ),
            dtype=torch.float32,
        )
        for part_index, part in enumerate(BODYPART_PROXY_PARTS):
            part_calibration = calibration[part.name]
            for anchor_index, target_anchor in enumerate(target_anchor_vocab):
                for distance_index, distance in enumerate(distance_vocab):
                    for fov_index, fov in enumerate(fov_vocab):
                        for pitch_index, pitch in enumerate(pitch_vocab):
                            table[part_index, anchor_index, distance_index, fov_index, pitch_index] = float(
                                proxy_visibility_score(
                                    target_anchor=str(target_anchor),
                                    distance=str(distance),
                                    fov=str(fov),
                                    pitch=str(pitch),
                                    part=part,
                                    calibration=part_calibration,
                                )
                            )
        cached = table
        _BODYPART_PROXY_SCORE_CACHE[key] = cached
    return cached.to(device=device)


def _bodypart_proxy_threshold_tensor(proxy_mode: str, device: torch.device) -> torch.Tensor:
    """Return diagnostic hard-visibility thresholds in bodypart proxy part order."""

    if str(proxy_mode) == "pose_dynamic_y":
        values = [POSE_DYNAMIC_PROXY_THRESHOLDS.get(part.name, 0.50) for part in BODYPART_PROXY_PARTS]
    else:
        calibration = proxy_calibration_for_mode(str(proxy_mode))
        values = [calibration[part.name].threshold for part in BODYPART_PROXY_PARTS]
    return torch.tensor(values, dtype=torch.float32, device=device).reshape(1, -1)


def _pose_dynamic_bodypart_score_table(
    *,
    pose_bodypart_anchor_y: torch.Tensor | None,
    pose_bodypart_part_bounds: torch.Tensor | None,
    distance_vocab: list[str],
    fov_vocab: list[str],
    pitch_vocab: list[str],
    device: torch.device,
) -> torch.Tensor | None:
    """Return per-sample score table [B, part, anchor, distance, fov, pitch]."""

    if pose_bodypart_anchor_y is None or pose_bodypart_part_bounds is None:
        return None
    if pose_bodypart_anchor_y.ndim != 2 or pose_bodypart_part_bounds.ndim != 3:
        return None
    if int(pose_bodypart_part_bounds.shape[1]) != len(BODYPART_PROXY_PARTS):
        return None

    anchor_y = pose_bodypart_anchor_y.to(device=device, dtype=torch.float32)
    part_bounds = pose_bodypart_part_bounds.to(device=device, dtype=torch.float32)
    distance_values = torch.tensor(
        [EXPORTED_DISTANCE_MULTIPLIERS.get(str(value), EXPORTED_DISTANCE_MULTIPLIERS["medium"]) for value in distance_vocab],
        dtype=torch.float32,
        device=device,
    ).reshape(1, 1, 1, -1, 1, 1)
    fov_values = torch.tensor(
        [FOV_DEGREES.get(str(value), FOV_DEGREES["normal"]) for value in fov_vocab],
        dtype=torch.float32,
        device=device,
    ).reshape(1, 1, 1, 1, -1, 1)
    pitch_scale = torch.tensor(
        [1.25 if str(value) == "top" else 1.05 if str(value) in {"high", "low"} else 1.0 for value in pitch_vocab],
        dtype=torch.float32,
        device=device,
    ).reshape(1, 1, 1, 1, 1, -1)

    half_span = distance_values * torch.tan(torch.deg2rad(fov_values) * 0.5) * pitch_scale
    anchor = anchor_y.reshape(int(anchor_y.shape[0]), 1, int(anchor_y.shape[1]), 1, 1, 1)
    visible_y0 = anchor - half_span
    visible_y1 = anchor + half_span
    part_y0 = part_bounds[:, :, 0].reshape(int(anchor_y.shape[0]), len(BODYPART_PROXY_PARTS), 1, 1, 1, 1)
    part_y1 = part_bounds[:, :, 1].reshape(int(anchor_y.shape[0]), len(BODYPART_PROXY_PARTS), 1, 1, 1, 1)
    overlap = torch.clamp(torch.minimum(visible_y1, part_y1) - torch.maximum(visible_y0, part_y0), min=0.0)
    part_height = torch.clamp(part_y1 - part_y0, min=1e-6)
    return torch.clamp(overlap / part_height, min=0.0, max=1.0)


def _pose_conditioned_finger_table(
    *,
    finger_pose_hint: torch.Tensor | None,
    static_finger_table: torch.Tensor,
    target_anchor_vocab: list[str],
    distance_vocab: list[str],
    fov_vocab: list[str],
    pitch_vocab: list[str],
    proxy_mode: str,
) -> torch.Tensor | None:
    """Return per-sample [B, anchor, distance, fov, pitch] finger visibility.

    The static proxy treats fingers as a broad vertical band. For close shots,
    this often over-penalizes or under-penalizes because hand pose changes
    whether fingers are actually in frame. The pose-conditioned version keeps
    the static proxy for sufficiently wide shots, and gates close-shot finger
    visibility by normalized left/right finger y and x location.
    """

    if finger_pose_hint is None or finger_pose_hint.ndim != 2 or int(finger_pose_hint.shape[1]) < 6:
        return None
    device = static_finger_table.device
    hint = finger_pose_hint.to(device=device, dtype=torch.float32)
    batch_size = int(hint.shape[0])
    calibration = proxy_calibration_for_mode(str(proxy_mode))["fingers"]

    anchor_values = torch.tensor(
        [TARGET_ANCHOR_Y_FRACTIONS.get(str(value), TARGET_ANCHOR_Y_FRACTIONS["waist"]) for value in target_anchor_vocab],
        dtype=torch.float32,
        device=device,
    ).reshape(1, -1, 1, 1, 1)
    distance_values = torch.tensor(
        [EXPORTED_DISTANCE_MULTIPLIERS.get(str(value), EXPORTED_DISTANCE_MULTIPLIERS["medium"]) for value in distance_vocab],
        dtype=torch.float32,
        device=device,
    ).reshape(1, 1, -1, 1, 1)
    fov_values = torch.tensor(
        [FOV_DEGREES.get(str(value), FOV_DEGREES["normal"]) for value in fov_vocab],
        dtype=torch.float32,
        device=device,
    ).reshape(1, 1, 1, -1, 1)
    pitch_scale = torch.tensor(
        [1.25 if str(value) == "top" else 1.05 if str(value) in {"high", "low"} else 1.0 for value in pitch_vocab],
        dtype=torch.float32,
        device=device,
    ).reshape(1, 1, 1, 1, -1)

    half_span = distance_values * torch.tan(torch.deg2rad(fov_values) * 0.5) * pitch_scale
    half_span = half_span * float(calibration.span_scale) + float(calibration.span_margin)
    center_y = anchor_values - float(calibration.down_shift)
    visible_y0 = center_y - half_span
    visible_y1 = center_y + half_span

    left_y = hint[:, 0].reshape(batch_size, 1, 1, 1, 1)
    right_y = hint[:, 1].reshape(batch_size, 1, 1, 1, 1)
    left_x_abs = hint[:, 2].reshape(batch_size, 1, 1, 1, 1)
    right_x_abs = hint[:, 3].reshape(batch_size, 1, 1, 1, 1)
    hand_width = hint[:, 4].reshape(batch_size, 1, 1, 1, 1)

    y_tau = 0.04
    x_tau = 0.08
    y_margin = 0.04
    x_margin = 0.06
    horizontal_half_span = half_span * 1.6

    def _hand_score(hand_y: torch.Tensor, hand_x_abs: torch.Tensor) -> torch.Tensor:
        y_inside = torch.sigmoid((hand_y - visible_y0 + y_margin) / y_tau) * torch.sigmoid(
            (visible_y1 + y_margin - hand_y) / y_tau
        )
        # If both hands are very spread out, be a little less strict about x
        # because one finger cluster can enter even if the average is near edge.
        spread_bonus = torch.clamp(hand_width * 0.25, min=0.0, max=0.15)
        x_inside = torch.sigmoid((horizontal_half_span + x_margin + spread_bonus - hand_x_abs) / x_tau)
        return y_inside * x_inside

    pose_score = torch.maximum(_hand_score(left_y, left_x_abs), _hand_score(right_y, right_x_abs))
    # Once a shot is wide enough, actual hand pose matters much less: fingers
    # are usually in frame if the static body band says they should be.
    wide_gate = torch.sigmoid((half_span - 0.70) / 0.08)
    static = static_finger_table.reshape(1, *static_finger_table.shape)
    conditioned = static * ((1.0 - wide_gate) * pose_score + wide_gate)
    return conditioned.clamp(0.0, 1.0)


def _bodypart_proxy_loss_and_metrics(
    outputs: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    *,
    vocabularies: dict[str, list[str]],
    proxy_mode: str,
    loss_type: str,
    finger_pose_hint: torch.Tensor | None = None,
    pose_conditioned_finger_proxy: bool = False,
    pose_bodypart_anchor_y: torch.Tensor | None = None,
    pose_bodypart_part_bounds: torch.Tensor | None = None,
    bodypart_visible_target: torch.Tensor | None = None,
    finger_threshold_visibility_table: torch.Tensor | None = None,
    finger_source: str = "proxy",
    finger_pred_threshold: float = 0.20,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Match predicted and GT body-part visibility proxies.

    The proxy is intentionally coarse and differentiable here: predicted class
    softmaxes define an expected body-part visibility score, while GT labels
    index the same proxy table directly. This discourages predictions that are
    class-wise plausible but imply the wrong visible body region.
    """

    required = ("target_anchor", "distance", "fov", "pitch")
    if any(f"{field}_logits" not in outputs or field not in labels or field not in vocabularies for field in required):
        first = next(iter(outputs.values()))
        return first.new_tensor(0.0), {"bodypart_proxy_active_rate": 0.0}

    first_logits = outputs["target_anchor_logits"]
    device = first_logits.device
    target_anchor = labels["target_anchor"].to(device).long()
    distance = labels["distance"].to(device).long()
    fov = labels["fov"].to(device).long()
    pitch = labels["pitch"].to(device).long()
    vocab_target_anchor = [str(value) for value in vocabularies["target_anchor"]]
    vocab_distance = [str(value) for value in vocabularies["distance"]]
    vocab_fov = [str(value) for value in vocabularies["fov"]]
    vocab_pitch = [str(value) for value in vocabularies["pitch"]]
    valid = (
        (target_anchor >= 0)
        & (target_anchor < len(vocab_target_anchor))
        & (distance >= 0)
        & (distance < len(vocab_distance))
        & (fov >= 0)
        & (fov < len(vocab_fov))
        & (pitch >= 0)
        & (pitch < len(vocab_pitch))
    )
    if not torch.any(valid):
        return first_logits.new_tensor(0.0), {"bodypart_proxy_active_rate": 0.0}

    table = _bodypart_proxy_score_table(
        target_anchor_vocab=vocab_target_anchor,
        distance_vocab=vocab_distance,
        fov_vocab=vocab_fov,
        pitch_vocab=vocab_pitch,
        proxy_mode=str(proxy_mode),
        device=device,
    )
    dynamic_table = None
    if str(proxy_mode) == "pose_dynamic_y":
        dynamic_table = _pose_dynamic_bodypart_score_table(
            pose_bodypart_anchor_y=pose_bodypart_anchor_y[valid] if pose_bodypart_anchor_y is not None else None,
            pose_bodypart_part_bounds=pose_bodypart_part_bounds[valid] if pose_bodypart_part_bounds is not None else None,
            distance_vocab=vocab_distance,
            fov_vocab=vocab_fov,
            pitch_vocab=vocab_pitch,
            device=device,
        )
    finger_index = next((idx for idx, part in enumerate(BODYPART_PROXY_PARTS) if part.name == "fingers"), -1)
    if pose_conditioned_finger_proxy and finger_index >= 0:
        dynamic_finger = _pose_conditioned_finger_table(
            finger_pose_hint=finger_pose_hint[valid] if finger_pose_hint is not None else None,
            static_finger_table=table[finger_index],
            target_anchor_vocab=vocab_target_anchor,
            distance_vocab=vocab_distance,
            fov_vocab=vocab_fov,
            pitch_vocab=vocab_pitch,
            proxy_mode=str(proxy_mode),
        )
        if dynamic_finger is not None:
            if dynamic_table is None:
                dynamic_table = table.unsqueeze(0).expand(int(dynamic_finger.shape[0]), -1, -1, -1, -1, -1).clone()
            else:
                dynamic_table = dynamic_table.clone()
            dynamic_table[:, finger_index] = dynamic_finger
    target_anchor_prob = F.softmax(outputs["target_anchor_logits"][valid], dim=-1)
    distance_prob = F.softmax(outputs["distance_logits"][valid], dim=-1)
    fov_prob = F.softmax(outputs["fov_logits"][valid], dim=-1)
    pitch_prob = F.softmax(outputs["pitch_logits"][valid], dim=-1)
    # Expected proxy score for each body part under independent field marginals.
    if dynamic_table is not None:
        pred_scores = torch.einsum(
            "ba,bd,bf,bp,bqadfp->bq",
            target_anchor_prob,
            distance_prob,
            fov_prob,
            pitch_prob,
            dynamic_table,
        )
        batch_index = torch.arange(int(dynamic_table.shape[0]), device=device)
        gt_scores = dynamic_table[
            batch_index,
            :,
            target_anchor[valid],
            distance[valid],
            fov[valid],
            pitch[valid],
        ]
    else:
        pred_scores = torch.einsum(
            "ba,bd,bf,bp,qadfp->bq",
            target_anchor_prob,
            distance_prob,
            fov_prob,
            pitch_prob,
            table,
        )
        gt_scores = table[
            :,
            target_anchor[valid],
            distance[valid],
            fov[valid],
            pitch[valid],
        ].transpose(0, 1)

    threshold_finger_active = False
    threshold_finger_pred_scores = None
    threshold_finger_gt_scores = None
    if (
        str(finger_source) == "threshold"
        and finger_index >= 0
        and bodypart_visible_target is not None
        and finger_threshold_visibility_table is not None
        and "yaw_logits" in outputs
    ):
        target_scores = bodypart_visible_target.to(device=device, dtype=torch.float32)
        visibility_table = finger_threshold_visibility_table.to(device=device, dtype=torch.float32)
        if (
            target_scores.ndim == 2
            and int(target_scores.shape[-1]) == len(BODYPART_PROXY_PARTS)
            and visibility_table.ndim == 5
            and int(visibility_table.shape[0]) == int(first_logits.shape[0])
            and int(visibility_table.shape[1]) == int(outputs["target_anchor_logits"].shape[-1])
            and int(visibility_table.shape[2]) == int(outputs["distance_logits"].shape[-1])
            and int(visibility_table.shape[3]) == int(outputs["fov_logits"].shape[-1])
            and int(visibility_table.shape[4]) == int(outputs["yaw_logits"].shape[-1])
        ):
            visibility_table = visibility_table[valid].clamp(0.0, 1.0)
            yaw_prob = F.softmax(outputs["yaw_logits"][valid], dim=-1)
            threshold_finger_pred_scores = torch.einsum(
                "ba,bd,bf,by,badfy->b",
                target_anchor_prob,
                distance_prob,
                fov_prob,
                yaw_prob,
                visibility_table,
            ).clamp(0.0, 1.0)
            threshold_finger_gt_scores = target_scores[valid, finger_index].clamp(0.0, 1.0)
            # Keep the original body proxy for the torso/legs/head dimensions,
            # but use the pose-aware threshold labels for fingers. This lets the
            # bodypart proxy loss care about missing/extra fingers without making
            # every other body part depend on the hand-pose heuristic.
            pred_scores = pred_scores.clone()
            gt_scores = gt_scores.clone()
            pred_scores[:, finger_index] = threshold_finger_pred_scores
            gt_scores[:, finger_index] = threshold_finger_gt_scores
            threshold_finger_active = True

    if str(loss_type) == "bce":
        thresholds = _bodypart_proxy_threshold_tensor(str(proxy_mode), device)
        gt_binary = (gt_scores >= thresholds).float()
        loss = F.binary_cross_entropy(pred_scores.clamp(min=1e-6, max=1.0 - 1e-6), gt_binary)
    else:
        loss = F.mse_loss(pred_scores, gt_scores)

    hard_anchor = outputs["target_anchor_logits"][valid].argmax(dim=-1)
    hard_distance = outputs["distance_logits"][valid].argmax(dim=-1)
    hard_fov = outputs["fov_logits"][valid].argmax(dim=-1)
    hard_pitch = outputs["pitch_logits"][valid].argmax(dim=-1)
    if dynamic_table is not None:
        batch_index = torch.arange(int(dynamic_table.shape[0]), device=device)
        hard_scores = dynamic_table[batch_index, :, hard_anchor, hard_distance, hard_fov, hard_pitch]
    else:
        hard_scores = table[:, hard_anchor, hard_distance, hard_fov, hard_pitch].transpose(0, 1)
    threshold_finger_hard_scores = None
    if threshold_finger_active:
        threshold_table = finger_threshold_visibility_table.to(device=device, dtype=torch.float32)[valid].clamp(0.0, 1.0)
        hard_yaw = outputs["yaw_logits"][valid].argmax(dim=-1)
        batch_index = torch.arange(int(threshold_table.shape[0]), device=device)
        threshold_finger_hard_scores = threshold_table[batch_index, hard_anchor, hard_distance, hard_fov, hard_yaw]
        hard_scores = hard_scores.clone()
        hard_scores[:, finger_index] = threshold_finger_hard_scores
    thresholds = _bodypart_proxy_threshold_tensor(str(proxy_mode), device)
    pred_binary = hard_scores >= thresholds
    gt_binary = gt_scores >= thresholds
    mismatch_rate = pred_binary.ne(gt_binary).float().mean()
    metrics = {
        "bodypart_proxy_loss": float(loss.detach().cpu().item()),
        "bodypart_proxy_score_mae": float(torch.abs(pred_scores - gt_scores).mean().detach().cpu().item()),
        "bodypart_proxy_hard_mismatch_rate": float(mismatch_rate.detach().cpu().item()),
        "bodypart_proxy_active_rate": float(valid.float().mean().detach().cpu().item()),
        "bodypart_proxy_pose_finger_active_rate": float(1.0 if dynamic_table is not None else 0.0),
        "bodypart_proxy_threshold_finger_active_rate": float(1.0 if threshold_finger_active else 0.0),
    }
    if (
        threshold_finger_active
        and threshold_finger_pred_scores is not None
        and threshold_finger_gt_scores is not None
        and threshold_finger_hard_scores is not None
    ):
        gt_visible = threshold_finger_gt_scores >= float(finger_pred_threshold)
        hard_visible = threshold_finger_hard_scores >= float(finger_pred_threshold)
        if torch.any(gt_visible):
            finger_fn_rate = (~hard_visible[gt_visible]).float().mean()
        else:
            finger_fn_rate = threshold_finger_gt_scores.new_tensor(0.0)
        if torch.any(~gt_visible):
            finger_fp_rate = hard_visible[~gt_visible].float().mean()
        else:
            finger_fp_rate = threshold_finger_gt_scores.new_tensor(0.0)
        metrics.update(
            {
                "bodypart_proxy_threshold_finger_score_mae": float(
                    torch.abs(threshold_finger_pred_scores - threshold_finger_gt_scores).mean().detach().cpu().item()
                ),
                "bodypart_proxy_threshold_finger_pred_score": float(
                    threshold_finger_pred_scores.mean().detach().cpu().item()
                ),
                "bodypart_proxy_threshold_finger_gt_score": float(
                    threshold_finger_gt_scores.mean().detach().cpu().item()
                ),
                "bodypart_proxy_threshold_finger_hard_mismatch_rate": float(
                    gt_visible.ne(hard_visible).float().mean().detach().cpu().item()
                ),
                "bodypart_proxy_threshold_finger_hard_fn_rate": float(finger_fn_rate.detach().cpu().item()),
                "bodypart_proxy_threshold_finger_hard_fp_rate": float(finger_fp_rate.detach().cpu().item()),
            }
        )
    return loss, metrics


def _visible_bodypart_target_loss_and_metrics(
    outputs: dict[str, torch.Tensor],
    target_scores: torch.Tensor,
    *,
    vocabularies: dict[str, list[str]],
    proxy_mode: str,
    loss_type: str,
    finger_pose_hint: torch.Tensor | None = None,
    pose_conditioned_finger_proxy: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Match predicted shot-plan-implied visibility to external GT BoneMask targets."""

    required = ("target_anchor", "distance", "fov", "pitch")
    if (
        target_scores.ndim != 2
        or target_scores.shape[-1] != len(BODYPART_PROXY_PARTS)
        or any(f"{field}_logits" not in outputs or field not in vocabularies for field in required)
    ):
        first = next(iter(outputs.values()))
        return first.new_tensor(0.0), {"visible_bodypart_active_rate": 0.0}

    first_logits = outputs["target_anchor_logits"]
    device = first_logits.device
    target_scores = target_scores.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    vocab_target_anchor = [str(value) for value in vocabularies["target_anchor"]]
    vocab_distance = [str(value) for value in vocabularies["distance"]]
    vocab_fov = [str(value) for value in vocabularies["fov"]]
    vocab_pitch = [str(value) for value in vocabularies["pitch"]]
    table = _bodypart_proxy_score_table(
        target_anchor_vocab=vocab_target_anchor,
        distance_vocab=vocab_distance,
        fov_vocab=vocab_fov,
        pitch_vocab=vocab_pitch,
        proxy_mode=str(proxy_mode),
        device=device,
    )
    dynamic_table = None
    finger_index = next((idx for idx, part in enumerate(BODYPART_PROXY_PARTS) if part.name == "fingers"), -1)
    if pose_conditioned_finger_proxy and finger_index >= 0:
        dynamic_finger = _pose_conditioned_finger_table(
            finger_pose_hint=finger_pose_hint,
            static_finger_table=table[finger_index],
            target_anchor_vocab=vocab_target_anchor,
            distance_vocab=vocab_distance,
            fov_vocab=vocab_fov,
            pitch_vocab=vocab_pitch,
            proxy_mode=str(proxy_mode),
        )
        if dynamic_finger is not None:
            dynamic_table = table.unsqueeze(0).expand(int(dynamic_finger.shape[0]), -1, -1, -1, -1, -1).clone()
            dynamic_table[:, finger_index] = dynamic_finger
    target_anchor_prob = F.softmax(outputs["target_anchor_logits"], dim=-1)
    distance_prob = F.softmax(outputs["distance_logits"], dim=-1)
    fov_prob = F.softmax(outputs["fov_logits"], dim=-1)
    pitch_prob = F.softmax(outputs["pitch_logits"], dim=-1)
    if dynamic_table is not None:
        pred_scores = torch.einsum(
            "ba,bd,bf,bp,bqadfp->bq",
            target_anchor_prob,
            distance_prob,
            fov_prob,
            pitch_prob,
            dynamic_table,
        )
    else:
        pred_scores = torch.einsum(
            "ba,bd,bf,bp,qadfp->bq",
            target_anchor_prob,
            distance_prob,
            fov_prob,
            pitch_prob,
            table,
        )

    if str(loss_type) == "bce":
        thresholds = torch.tensor(
            [proxy_calibration_for_mode(str(proxy_mode))[part.name].threshold for part in BODYPART_PROXY_PARTS],
            dtype=torch.float32,
            device=device,
        ).reshape(1, -1)
        target_binary = (target_scores >= thresholds).float()
        loss = F.binary_cross_entropy(pred_scores.clamp(min=1e-6, max=1.0 - 1e-6), target_binary)
    else:
        loss = F.mse_loss(pred_scores, target_scores)

    thresholds = torch.tensor(
        [proxy_calibration_for_mode(str(proxy_mode))[part.name].threshold for part in BODYPART_PROXY_PARTS],
        dtype=torch.float32,
        device=device,
    ).reshape(1, -1)
    hard_anchor = outputs["target_anchor_logits"].argmax(dim=-1)
    hard_distance = outputs["distance_logits"].argmax(dim=-1)
    hard_fov = outputs["fov_logits"].argmax(dim=-1)
    hard_pitch = outputs["pitch_logits"].argmax(dim=-1)
    if dynamic_table is not None:
        batch_index = torch.arange(int(dynamic_table.shape[0]), device=device)
        hard_scores = dynamic_table[batch_index, :, hard_anchor, hard_distance, hard_fov, hard_pitch]
    else:
        hard_scores = table[:, hard_anchor, hard_distance, hard_fov, hard_pitch].transpose(0, 1)
    mismatch_rate = (hard_scores >= thresholds).ne(target_scores >= thresholds).float().mean()
    metrics = {
        "visible_bodypart_loss": float(loss.detach().cpu().item()),
        "visible_bodypart_score_mae": float(torch.abs(pred_scores - target_scores).mean().detach().cpu().item()),
        "visible_bodypart_hard_mismatch_rate": float(mismatch_rate.detach().cpu().item()),
        "visible_bodypart_active_rate": 1.0,
        "visible_bodypart_pose_finger_active_rate": float(1.0 if dynamic_table is not None else 0.0),
    }
    return loss, metrics


def _finger_absence_loss_and_metrics(
    outputs: dict[str, torch.Tensor],
    target_scores: torch.Tensor,
    *,
    vocabularies: dict[str, list[str]],
    proxy_mode: str,
    target_threshold: float,
    finger_pose_hint: torch.Tensor | None = None,
    pose_conditioned_finger_proxy: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Penalize predicted finger visibility when GT BoneMask says fingers are absent.

    This is intentionally narrower than ``visible_bodypart_loss``: it only uses
    the fingers column from the external BoneMask target, so it can target LCD
    false positives around hands without changing the supervision for head,
    torso, legs, or feet.
    """

    required = ("target_anchor", "distance", "fov", "pitch")
    first = next(iter(outputs.values()))
    finger_index = next((idx for idx, part in enumerate(BODYPART_PROXY_PARTS) if part.name == "fingers"), -1)
    if (
        finger_index < 0
        or target_scores.ndim != 2
        or target_scores.shape[-1] != len(BODYPART_PROXY_PARTS)
        or any(f"{field}_logits" not in outputs or field not in vocabularies for field in required)
    ):
        return first.new_tensor(0.0), {"finger_absence_active_rate": 0.0}

    first_logits = outputs["target_anchor_logits"]
    device = first_logits.device
    target_scores = target_scores.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    absent = target_scores[:, finger_index] < float(target_threshold)
    if not torch.any(absent):
        return first_logits.new_tensor(0.0), {"finger_absence_active_rate": 0.0}

    vocab_target_anchor = [str(value) for value in vocabularies["target_anchor"]]
    vocab_distance = [str(value) for value in vocabularies["distance"]]
    vocab_fov = [str(value) for value in vocabularies["fov"]]
    vocab_pitch = [str(value) for value in vocabularies["pitch"]]
    table = _bodypart_proxy_score_table(
        target_anchor_vocab=vocab_target_anchor,
        distance_vocab=vocab_distance,
        fov_vocab=vocab_fov,
        pitch_vocab=vocab_pitch,
        proxy_mode=str(proxy_mode),
        device=device,
    )
    finger_table = table[finger_index]
    if pose_conditioned_finger_proxy:
        dynamic_finger = _pose_conditioned_finger_table(
            finger_pose_hint=finger_pose_hint if finger_pose_hint is not None else None,
            static_finger_table=finger_table,
            target_anchor_vocab=vocab_target_anchor,
            distance_vocab=vocab_distance,
            fov_vocab=vocab_fov,
            pitch_vocab=vocab_pitch,
            proxy_mode=str(proxy_mode),
        )
    else:
        dynamic_finger = None

    target_anchor_prob = F.softmax(outputs["target_anchor_logits"], dim=-1)
    distance_prob = F.softmax(outputs["distance_logits"], dim=-1)
    fov_prob = F.softmax(outputs["fov_logits"], dim=-1)
    pitch_prob = F.softmax(outputs["pitch_logits"], dim=-1)
    if dynamic_finger is not None:
        pred_scores = torch.einsum(
            "ba,bd,bf,bp,badfp->b",
            target_anchor_prob,
            distance_prob,
            fov_prob,
            pitch_prob,
            dynamic_finger,
        )
    else:
        pred_scores = torch.einsum(
            "ba,bd,bf,bp,adfp->b",
            target_anchor_prob,
            distance_prob,
            fov_prob,
            pitch_prob,
            finger_table,
        )

    absent_scores = pred_scores[absent].clamp(min=1e-6, max=1.0 - 1e-6)
    # Equivalent to BCE target=0, but explicit here to make the intent clear:
    # when hands/fingers are absent in GT, high predicted visibility is costly.
    loss = -torch.log1p(-absent_scores).mean()

    threshold = torch.tensor(
        float(proxy_calibration_for_mode(str(proxy_mode))["fingers"].threshold),
        dtype=torch.float32,
        device=device,
    )
    hard_anchor = outputs["target_anchor_logits"].argmax(dim=-1)
    hard_distance = outputs["distance_logits"].argmax(dim=-1)
    hard_fov = outputs["fov_logits"].argmax(dim=-1)
    hard_pitch = outputs["pitch_logits"].argmax(dim=-1)
    if dynamic_finger is not None:
        batch_index = torch.arange(int(dynamic_finger.shape[0]), device=device)
        hard_scores = dynamic_finger[batch_index, hard_anchor, hard_distance, hard_fov, hard_pitch]
    else:
        hard_scores = finger_table[hard_anchor, hard_distance, hard_fov, hard_pitch]
    false_positive_rate = (hard_scores[absent] >= threshold).float().mean()
    metrics = {
        "finger_absence_loss": float(loss.detach().cpu().item()),
        "finger_absence_active_rate": float(absent.float().mean().detach().cpu().item()),
        "finger_absence_pred_score": float(absent_scores.mean().detach().cpu().item()),
        "finger_absence_gt_score": float(target_scores[absent, finger_index].mean().detach().cpu().item()),
        "finger_absence_hard_fp_rate": float(false_positive_rate.detach().cpu().item()),
        "finger_absence_pose_finger_active_rate": float(1.0 if dynamic_finger is not None else 0.0),
    }
    return loss, metrics


def _finger_missing_threshold_loss_and_metrics(
    outputs: dict[str, torch.Tensor],
    target_scores: torch.Tensor,
    visibility_table: torch.Tensor,
    *,
    target_threshold: float,
    pred_threshold: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Penalize predicted crops that would hide fingers visible in GT.

    ``visibility_table`` is precomputed by ``build_dataset.py`` from full dance
    joints.  Its axis order is
    ``sample, target_anchor, distance, fov, yaw`` and each value is the mean
    visible fingertip ratio under that shot choice.
    """

    required = ("target_anchor", "distance", "fov", "yaw")
    first = next(iter(outputs.values()))
    finger_index = next((idx for idx, part in enumerate(BODYPART_PROXY_PARTS) if part.name == "fingers"), -1)
    if (
        finger_index < 0
        or target_scores.ndim != 2
        or target_scores.shape[-1] != len(BODYPART_PROXY_PARTS)
        or visibility_table.ndim != 5
        or int(visibility_table.shape[-1]) <= 0
        or any(f"{field}_logits" not in outputs for field in required)
    ):
        return first.new_tensor(0.0), {"finger_missing_threshold_active_rate": 0.0}

    device = outputs["target_anchor_logits"].device
    target_scores = target_scores.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    visibility_table = visibility_table.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    if (
        int(visibility_table.shape[1]) != int(outputs["target_anchor_logits"].shape[-1])
        or int(visibility_table.shape[2]) != int(outputs["distance_logits"].shape[-1])
        or int(visibility_table.shape[3]) != int(outputs["fov_logits"].shape[-1])
        or int(visibility_table.shape[4]) != int(outputs["yaw_logits"].shape[-1])
    ):
        return first.new_tensor(0.0), {"finger_missing_threshold_active_rate": 0.0}

    visible = target_scores[:, finger_index] >= float(target_threshold)
    if not torch.any(visible):
        return outputs["target_anchor_logits"].new_tensor(0.0), {"finger_missing_threshold_active_rate": 0.0}

    target_anchor_prob = F.softmax(outputs["target_anchor_logits"], dim=-1)
    distance_prob = F.softmax(outputs["distance_logits"], dim=-1)
    fov_prob = F.softmax(outputs["fov_logits"], dim=-1)
    yaw_prob = F.softmax(outputs["yaw_logits"], dim=-1)
    pred_scores = torch.einsum(
        "ba,bd,bf,by,badfy->b",
        target_anchor_prob,
        distance_prob,
        fov_prob,
        yaw_prob,
        visibility_table,
    )

    visible_scores = pred_scores[visible].clamp(min=1e-6, max=1.0)
    # BCE target=1 for GT-visible fingers. This targets missing-finger LCD
    # failures directly, without forcing anything on GT-absent finger samples.
    loss = -torch.log(visible_scores).mean()

    hard_anchor = outputs["target_anchor_logits"].argmax(dim=-1)
    hard_distance = outputs["distance_logits"].argmax(dim=-1)
    hard_fov = outputs["fov_logits"].argmax(dim=-1)
    hard_yaw = outputs["yaw_logits"].argmax(dim=-1)
    batch_index = torch.arange(int(visibility_table.shape[0]), device=device)
    hard_scores = visibility_table[batch_index, hard_anchor, hard_distance, hard_fov, hard_yaw]
    false_negative_rate = (hard_scores[visible] < float(pred_threshold)).float().mean()
    metrics = {
        "finger_missing_threshold_loss": float(loss.detach().cpu().item()),
        "finger_missing_threshold_active_rate": float(visible.float().mean().detach().cpu().item()),
        "finger_missing_threshold_pred_score": float(visible_scores.mean().detach().cpu().item()),
        "finger_missing_threshold_gt_score": float(target_scores[visible, finger_index].mean().detach().cpu().item()),
        "finger_missing_threshold_hard_fn_rate": float(false_negative_rate.detach().cpu().item()),
    }
    return loss, metrics


def _finger_visibility_threshold_loss_and_metrics(
    outputs: dict[str, torch.Tensor],
    target_scores: torch.Tensor,
    visibility_table: torch.Tensor,
    *,
    pred_threshold: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Match GT finger visibility with predicted shot-combination visibility.

    Unlike ``_finger_missing_threshold_loss_and_metrics``, this is symmetric:
    it penalizes both missing fingers and extra fingers by minimizing
    ``abs(gt_visibility - predicted_visibility)``.
    """

    required = ("target_anchor", "distance", "fov", "yaw")
    first = next(iter(outputs.values()))
    finger_index = next((idx for idx, part in enumerate(BODYPART_PROXY_PARTS) if part.name == "fingers"), -1)
    if (
        finger_index < 0
        or target_scores.ndim != 2
        or target_scores.shape[-1] != len(BODYPART_PROXY_PARTS)
        or visibility_table.ndim != 5
        or int(visibility_table.shape[-1]) <= 0
        or any(f"{field}_logits" not in outputs for field in required)
    ):
        return first.new_tensor(0.0), {"finger_visibility_threshold_active_rate": 0.0}

    device = outputs["target_anchor_logits"].device
    target_scores = target_scores.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    visibility_table = visibility_table.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    if (
        int(visibility_table.shape[1]) != int(outputs["target_anchor_logits"].shape[-1])
        or int(visibility_table.shape[2]) != int(outputs["distance_logits"].shape[-1])
        or int(visibility_table.shape[3]) != int(outputs["fov_logits"].shape[-1])
        or int(visibility_table.shape[4]) != int(outputs["yaw_logits"].shape[-1])
    ):
        return first.new_tensor(0.0), {"finger_visibility_threshold_active_rate": 0.0}

    target_anchor_prob = F.softmax(outputs["target_anchor_logits"], dim=-1)
    distance_prob = F.softmax(outputs["distance_logits"], dim=-1)
    fov_prob = F.softmax(outputs["fov_logits"], dim=-1)
    yaw_prob = F.softmax(outputs["yaw_logits"], dim=-1)
    pred_scores = torch.einsum(
        "ba,bd,bf,by,badfy->b",
        target_anchor_prob,
        distance_prob,
        fov_prob,
        yaw_prob,
        visibility_table,
    ).clamp(0.0, 1.0)
    gt_scores = target_scores[:, finger_index].clamp(0.0, 1.0)
    loss = torch.abs(pred_scores - gt_scores).mean()

    hard_anchor = outputs["target_anchor_logits"].argmax(dim=-1)
    hard_distance = outputs["distance_logits"].argmax(dim=-1)
    hard_fov = outputs["fov_logits"].argmax(dim=-1)
    hard_yaw = outputs["yaw_logits"].argmax(dim=-1)
    batch_index = torch.arange(int(visibility_table.shape[0]), device=device)
    hard_scores = visibility_table[batch_index, hard_anchor, hard_distance, hard_fov, hard_yaw]
    gt_visible = gt_scores >= float(pred_threshold)
    hard_visible = hard_scores >= float(pred_threshold)
    mismatch_rate = (gt_visible != hard_visible).float().mean()
    if torch.any(gt_visible):
        false_negative_rate = (~hard_visible[gt_visible]).float().mean()
    else:
        false_negative_rate = gt_scores.new_tensor(0.0)
    if torch.any(~gt_visible):
        false_positive_rate = hard_visible[~gt_visible].float().mean()
    else:
        false_positive_rate = gt_scores.new_tensor(0.0)

    metrics = {
        "finger_visibility_threshold_loss": float(loss.detach().cpu().item()),
        "finger_visibility_threshold_active_rate": 1.0,
        "finger_visibility_threshold_score_mae": float(torch.abs(pred_scores - gt_scores).mean().detach().cpu().item()),
        "finger_visibility_threshold_pred_score": float(pred_scores.mean().detach().cpu().item()),
        "finger_visibility_threshold_gt_score": float(gt_scores.mean().detach().cpu().item()),
        "finger_visibility_threshold_hard_mismatch_rate": float(mismatch_rate.detach().cpu().item()),
        "finger_visibility_threshold_hard_fn_rate": float(false_negative_rate.detach().cpu().item()),
        "finger_visibility_threshold_hard_fp_rate": float(false_positive_rate.detach().cpu().item()),
    }
    return loss, metrics


def _parse_optional_fields(raw: str, available: tuple[str, ...]) -> tuple[str, ...]:
    allowed = set(available)
    fields = []
    for item in str(raw or "").split(","):
        name = item.strip()
        if name and name in allowed:
            fields.append(name)
    return tuple(fields)


def _field_rank_values(field: str, vocab: list[str]) -> list[float]:
    """Map categorical vocab entries to semantic ranks for smoothness."""

    orders = {
        "distance": ["extreme_close", "very_close", "close", "medium_close", "medium", "medium_far", "far", "wide"],
        "fov": ["narrow", "normal", "wide"],
        "pitch": ["low", "level", "high", "top"],
        "composition": ["left_third", "center", "right_third"],
        "target_anchor": ["feet", "lower_leg", "thigh", "hip", "waist", "chest", "face"],
        "roll": ["strong_left", "left", "slight_left", "none", "slight_right", "right", "strong_right"],
    }
    if field == "yaw":
        # Circular order around the character.
        order = [
            "front",
            "left_45",
            "profile_left",
            "back_left_45",
            "back",
            "back_right_45",
            "profile_right",
            "right_45",
        ]
    else:
        order = orders.get(field, list(vocab))
    rank = {value: float(index) for index, value in enumerate(order)}
    fallback = float(len(order))
    return [rank.get(str(value), fallback + float(index)) for index, value in enumerate(vocab)]


def _noncut_smoothness_loss_and_metrics(
    outputs: dict[str, torch.Tensor],
    history_labels: dict[str, torch.Tensor],
    history_mask: torch.Tensor,
    history_delta: torch.Tensor,
    *,
    fields: tuple[str, ...],
    vocabularies: dict[str, list[str]],
    max_delta: float,
    margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Discourage ordinary keyframes from making abrupt semantic jumps."""

    if history_mask.ndim < 2 or history_mask.shape[1] <= 0:
        first = next(iter(outputs.values()))
        return first.new_tensor(0.0), {"noncut_smoothness_active_rate": 0.0}

    device = history_mask.device
    nearest = history_mask.shape[1] - 1
    active = (history_mask[:, nearest].to(device) > 0.5) & (history_delta[:, nearest].to(device).abs() <= float(max_delta))
    if not torch.any(active):
        first = next(iter(outputs.values()))
        return first.new_tensor(0.0), {"noncut_smoothness_active_rate": 0.0}

    losses = []
    expected_jumps = []
    for field in fields:
        if f"{field}_logits" not in outputs or field not in history_labels:
            continue
        logits = outputs[f"{field}_logits"]
        vocab = [str(value) for value in vocabularies.get(field, [])]
        if not vocab:
            continue
        prev_label = history_labels[field][:, nearest].to(device).long()
        valid = active & (prev_label >= 0) & (prev_label < len(vocab))
        if not torch.any(valid):
            continue
        ranks = torch.tensor(_field_rank_values(field, vocab), dtype=torch.float32, device=device)
        prev_rank = ranks[prev_label[valid]].unsqueeze(1)
        class_rank = ranks.reshape(1, -1)
        if field == "yaw":
            cycle = float(len(vocab))
            rank_diff = torch.abs(class_rank - prev_rank)
            rank_diff = torch.minimum(rank_diff, torch.clamp(torch.tensor(cycle, device=device) - rank_diff, min=0.0))
        else:
            rank_diff = torch.abs(class_rank - prev_rank)
        probs = F.softmax(logits[valid], dim=-1)
        expected_jump = torch.sum(probs * rank_diff, dim=-1)
        losses.append(F.relu(expected_jump - float(margin)).mean())
        expected_jumps.append(expected_jump.mean())

    if not losses:
        first = next(iter(outputs.values()))
        return first.new_tensor(0.0), {"noncut_smoothness_active_rate": 0.0}
    loss = torch.stack(losses).mean()
    metrics = {
        "noncut_smoothness_loss": float(loss.detach().cpu().item()),
        "noncut_smoothness_expected_jump": float(torch.stack(expected_jumps).mean().detach().cpu().item()),
        "noncut_smoothness_active_rate": float(active.float().mean().detach().cpu().item()),
    }
    return loss, metrics


def _precut_adjacent_contrast_loss_and_metrics(
    outputs: dict[str, torch.Tensor],
    current_labels: dict[str, torch.Tensor],
    post_cut_labels: dict[str, torch.Tensor],
    post_cut_mask: torch.Tensor,
    fields: tuple[str, ...],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Penalize pre-cut keyframe predictions that collapse to the next cut shot.

    This is GT-aware: the penalty is active only for fields where the current
    frame and the immediately following cut-transition frame have different GT
    labels. Fields that should remain equal are not pushed apart.
    """

    losses = []
    same_pred_count = 0.0
    active_count = 0.0
    sample_count = max(1, int(post_cut_mask.numel()))
    base = post_cut_mask.new_tensor(0.0)
    for field in fields:
        logits = outputs[f"{field}_logits"]
        next_target = post_cut_labels[field].to(logits.device).long()
        current_target = current_labels[field].to(logits.device).long()
        valid = (
            (post_cut_mask.to(logits.device) > 0.5)
            & (next_target >= 0)
            & (current_target >= 0)
            & next_target.ne(current_target)
        )
        if not torch.any(valid):
            continue
        probs = F.softmax(logits[valid], dim=-1)
        next_prob = probs.gather(1, next_target[valid].unsqueeze(1)).squeeze(1)
        losses.append(-torch.log1p(-next_prob.clamp(max=1.0 - 1e-6)).mean())
        pred = logits[valid].argmax(dim=-1)
        same_pred_count += float(pred.eq(next_target[valid]).float().sum().detach().cpu().item())
        active_count += float(valid.float().sum().detach().cpu().item())
    if losses:
        loss = torch.stack(losses).mean()
    else:
        loss = base
    metrics = {
        "precut_contrast_loss": float(loss.detach().cpu().item()),
        "precut_contrast_active_rate": float(active_count / max(1.0, sample_count * len(fields))),
        "precut_contrast_same_pred_rate": float(same_pred_count / max(1.0, active_count)),
    }
    return loss, metrics


_VISUAL_SCALE_DISTANCE_ORDER = [
    "extreme_close",
    "very_close",
    "close",
    "medium_close",
    "medium",
    "medium_far",
    "far",
    "wide",
]
_VISUAL_SCALE_FOV_ORDER = ["narrow", "normal", "wide"]


def _visual_scale_rank_grid(
    *,
    distance_vocab: list[str],
    fov_vocab: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Return perceived scale ranks for distance x FOV combinations.

    The rank follows the reference grid where diagonal combinations look
    similar:

    - extreme_close + normal ~= very_close + narrow
    - extreme_close + wide ~= very_close + normal ~= close + narrow

    Larger rank means visually farther / smaller framing. This is intentionally
    categorical and coarse; it is used only as a contrast signal, not as a camera
    metric.
    """

    distance_rank = {
        value: index for index, value in enumerate(_VISUAL_SCALE_DISTANCE_ORDER)
    }
    fov_rank = {value: index for index, value in enumerate(_VISUAL_SCALE_FOV_ORDER)}
    default_distance = distance_rank["medium"]
    default_fov = fov_rank["normal"]
    d = torch.tensor(
        [float(distance_rank.get(str(value), default_distance)) for value in distance_vocab],
        dtype=torch.float32,
        device=device,
    )
    f = torch.tensor(
        [float(fov_rank.get(str(value), default_fov)) for value in fov_vocab],
        dtype=torch.float32,
        device=device,
    )
    return d[:, None] + f[None, :]


def _precut_visual_scale_contrast_loss_and_metrics(
    outputs: dict[str, torch.Tensor],
    current_labels: dict[str, torch.Tensor],
    post_cut_labels: dict[str, torch.Tensor],
    post_cut_mask: torch.Tensor,
    *,
    vocabularies: dict[str, list[str]],
    margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Encourage cut pairs to differ in perceived distance/FOV scale.

    The ordinary field-wise cut contrast can see ``distance`` or ``fov`` changes
    as different even when the diagonal distance/FOV combination looks almost
    identical on screen. This loss compares the joint visual-scale rank instead.
    It is GT-aware: it is active only when the GT pre-cut and post-cut shots have
    different visual-scale ranks.
    """

    if "distance_logits" not in outputs or "fov_logits" not in outputs:
        first = next(iter(outputs.values()))
        return first.new_tensor(0.0), {"visual_scale_contrast_active_rate": 0.0}
    if "distance" not in current_labels or "fov" not in current_labels:
        first = next(iter(outputs.values()))
        return first.new_tensor(0.0), {"visual_scale_contrast_active_rate": 0.0}
    if "distance" not in post_cut_labels or "fov" not in post_cut_labels:
        first = next(iter(outputs.values()))
        return first.new_tensor(0.0), {"visual_scale_contrast_active_rate": 0.0}

    distance_logits = outputs["distance_logits"]
    fov_logits = outputs["fov_logits"]
    device = distance_logits.device
    distance_vocab = [str(value) for value in vocabularies.get("distance", [])]
    fov_vocab = [str(value) for value in vocabularies.get("fov", [])]
    if not distance_vocab or not fov_vocab:
        return distance_logits.new_tensor(0.0), {"visual_scale_contrast_active_rate": 0.0}

    scale_grid = _visual_scale_rank_grid(
        distance_vocab=distance_vocab,
        fov_vocab=fov_vocab,
        device=device,
    )
    current_distance = current_labels["distance"].to(device).long()
    current_fov = current_labels["fov"].to(device).long()
    next_distance = post_cut_labels["distance"].to(device).long()
    next_fov = post_cut_labels["fov"].to(device).long()
    valid = (
        (post_cut_mask.to(device) > 0.5)
        & (current_distance >= 0)
        & (current_distance < len(distance_vocab))
        & (current_fov >= 0)
        & (current_fov < len(fov_vocab))
        & (next_distance >= 0)
        & (next_distance < len(distance_vocab))
        & (next_fov >= 0)
        & (next_fov < len(fov_vocab))
    )
    if not torch.any(valid):
        return distance_logits.new_tensor(0.0), {"visual_scale_contrast_active_rate": 0.0}

    current_gt_scale = scale_grid[current_distance[valid], current_fov[valid]]
    next_gt_scale = scale_grid[next_distance[valid], next_fov[valid]]
    gt_diff = torch.abs(current_gt_scale - next_gt_scale)
    active = gt_diff > 0.0
    if not torch.any(active):
        return distance_logits.new_tensor(0.0), {
            "visual_scale_contrast_active_rate": 0.0,
            "visual_scale_contrast_gt_diff": 0.0,
        }

    distance_probs = F.softmax(distance_logits[valid][active], dim=-1)
    fov_probs = F.softmax(fov_logits[valid][active], dim=-1)
    joint_probs = distance_probs[:, :, None] * fov_probs[:, None, :]
    next_scale = next_gt_scale[active].reshape(-1, 1, 1)
    diff_grid = torch.abs(scale_grid.reshape(1, len(distance_vocab), len(fov_vocab)) - next_scale)
    expected_diff = torch.sum(joint_probs * diff_grid, dim=(1, 2))
    target_diff = torch.maximum(
        gt_diff[active],
        expected_diff.new_full(expected_diff.shape, float(margin)),
    )
    loss = F.relu(target_diff - expected_diff).mean()

    pred_distance = distance_logits[valid][active].argmax(dim=-1)
    pred_fov = fov_logits[valid][active].argmax(dim=-1)
    pred_scale = scale_grid[pred_distance, pred_fov]
    pred_diff = torch.abs(pred_scale - next_gt_scale[active])
    weak_pred = pred_diff < target_diff
    metrics = {
        "visual_scale_contrast_loss": float(loss.detach().cpu().item()),
        "visual_scale_contrast_active_rate": float(active.float().mean().detach().cpu().item()),
        "visual_scale_contrast_gt_diff": float(gt_diff[active].mean().detach().cpu().item()),
        "visual_scale_contrast_expected_diff": float(expected_diff.mean().detach().cpu().item()),
        "visual_scale_contrast_weak_pred_rate": float(weak_pred.float().mean().detach().cpu().item()),
    }
    return loss, metrics


def _run_epoch(
    model: V3CutContextShotPlanModel,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    fields: tuple[str, ...],
    vocabularies: dict[str, list[str]],
    semantic_soft_targets: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    sample_total = 0
    max_batches = int(args.max_train_batches if training else args.max_eval_batches)
    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches and batch_idx > max_batches:
            break
        audio = batch["audio"].to(device)
        dance = batch["dance"].to(device)
        cut_mask_bool = batch["cut_flag"].to(device) > 0.5
        labels = _batch_labels(batch, "label_", fields, device)
        prev_cut_labels = _batch_labels(batch, "prev_cut_label_", fields, device)
        next_cut_labels = _batch_labels(batch, "next_cut_label_", fields, device)
        pre_cut_labels = _batch_labels(batch, "pre_cut_label_", fields, device)
        post_cut_labels = _batch_labels(batch, "post_cut_label_", fields, device)
        history_labels = _batch_labels(batch, "history_label_", fields, device)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        batch_loss = audio.new_tensor(0.0)
        stage_records: list[tuple[str, torch.Tensor, float]] = [
            ("cut", cut_mask_bool, float(args.cut_sample_weight)),
            ("keyframe", ~cut_mask_bool, float(args.keyframe_sample_weight)),
        ]
        stage_metric_accum: dict[str, float] = {}
        stage_count = 0
        with torch.set_grad_enabled(training):
            for stage, mask, weight in stage_records:
                if not torch.any(mask):
                    continue
                idx = torch.nonzero(mask, as_tuple=False).flatten()
                outputs = model(
                    audio[idx],
                    dance[idx],
                    stage=stage,  # type: ignore[arg-type]
                    prev_cut_labels={field: value[idx] for field, value in prev_cut_labels.items()},
                    prev_cut_mask=batch["prev_cut_mask"].to(device)[idx],
                    prev_cut_delta=batch["prev_cut_delta"].to(device)[idx],
                    next_cut_labels={field: value[idx] for field, value in next_cut_labels.items()},
                    next_cut_mask=batch["next_cut_mask"].to(device)[idx],
                    next_cut_delta=batch["next_cut_delta"].to(device)[idx],
                    key_position_delta=batch["key_position_delta"].to(device)[idx],
                    key_position_mask=batch["key_position_mask"].to(device)[idx],
                    cut_position_delta=batch["cut_position_delta"].to(device)[idx],
                    cut_position_mask=batch["cut_position_mask"].to(device)[idx],
                    history_labels={field: value[idx] for field, value in history_labels.items()},
                    history_mask=batch["history_mask"].to(device)[idx],
                    history_delta=batch["history_delta"].to(device)[idx],
                )
                if bool(args.predict_distance_fov_joint_as_field):
                    outputs = _with_distance_fov_marginal_logits(outputs)
                loss, metrics = _field_loss_and_metrics(
                    outputs,
                    {field: value[idx] for field, value in labels.items()},
                    fields,
                    semantic_soft_targets=semantic_soft_targets,
                    semantic_loss_weight=float(args.semantic_loss_weight),
                    predict_distance_fov_joint_as_field=bool(args.predict_distance_fov_joint_as_field),
                )
                if (
                    float(args.distance_fov_joint_loss_weight) > 0
                    and not bool(args.predict_distance_fov_joint_as_field)
                ):
                    joint_loss, joint_metrics = _distance_fov_joint_loss_and_metrics(
                        outputs,
                        {field: value[idx] for field, value in labels.items()},
                    )
                    loss = loss + float(args.distance_fov_joint_loss_weight) * joint_loss
                    metrics.update(joint_metrics)
                proxy_stage = str(args.bodypart_proxy_loss_stage)
                if float(args.bodypart_proxy_loss_weight) > 0 and proxy_stage in ("all", stage):
                    proxy_loss, proxy_metrics = _bodypart_proxy_loss_and_metrics(
                        outputs,
                        {field: value[idx] for field, value in labels.items()},
                        vocabularies=vocabularies,
                        proxy_mode=str(args.bodypart_proxy_mode),
                        loss_type=str(args.bodypart_proxy_loss_type),
                        finger_pose_hint=batch.get("finger_pose_hint").to(device)[idx]
                        if batch.get("finger_pose_hint") is not None
                        else None,
                        pose_conditioned_finger_proxy=bool(args.pose_conditioned_finger_proxy),
                        pose_bodypart_anchor_y=batch.get("pose_bodypart_anchor_y").to(device)[idx]
                        if batch.get("pose_bodypart_anchor_y") is not None
                        else None,
                        pose_bodypart_part_bounds=batch.get("pose_bodypart_part_bounds").to(device)[idx]
                        if batch.get("pose_bodypart_part_bounds") is not None
                        else None,
                        bodypart_visible_target=batch.get("bodypart_visible_target").to(device)[idx]
                        if batch.get("bodypart_visible_target") is not None
                        else None,
                        finger_threshold_visibility_table=batch.get("finger_threshold_visibility_table").to(device)[idx]
                        if batch.get("finger_threshold_visibility_table") is not None
                        else None,
                        finger_source=str(args.bodypart_proxy_finger_source),
                        finger_pred_threshold=float(args.bodypart_proxy_finger_pred_threshold),
                    )
                    loss = loss + float(args.bodypart_proxy_loss_weight) * proxy_loss
                    metrics.update(proxy_metrics)
                visible_stage = str(args.visible_bodypart_loss_stage)
                visible_targets = batch.get("bodypart_visible_target")
                if (
                    float(args.visible_bodypart_loss_weight) > 0
                    and visible_stage in ("all", stage)
                    and visible_targets is not None
                    and int(visible_targets.shape[-1]) == len(BODYPART_PROXY_PARTS)
                ):
                    visible_loss, visible_metrics = _visible_bodypart_target_loss_and_metrics(
                        outputs,
                        visible_targets.to(device)[idx],
                        vocabularies=vocabularies,
                        proxy_mode=str(args.bodypart_proxy_mode),
                        loss_type=str(args.visible_bodypart_loss_type),
                        finger_pose_hint=batch.get("finger_pose_hint").to(device)[idx]
                        if batch.get("finger_pose_hint") is not None
                        else None,
                        pose_conditioned_finger_proxy=bool(args.pose_conditioned_finger_proxy),
                    )
                    loss = loss + float(args.visible_bodypart_loss_weight) * visible_loss
                    metrics.update(visible_metrics)
                finger_absence_stage = str(args.finger_absence_loss_stage)
                if (
                    float(args.finger_absence_loss_weight) > 0
                    and finger_absence_stage in ("all", stage)
                    and visible_targets is not None
                    and int(visible_targets.shape[-1]) == len(BODYPART_PROXY_PARTS)
                ):
                    finger_loss, finger_metrics = _finger_absence_loss_and_metrics(
                        outputs,
                        visible_targets.to(device)[idx],
                        vocabularies=vocabularies,
                        proxy_mode=str(args.bodypart_proxy_mode),
                        target_threshold=float(args.finger_absence_target_threshold),
                        finger_pose_hint=batch.get("finger_pose_hint").to(device)[idx]
                        if batch.get("finger_pose_hint") is not None
                        else None,
                        pose_conditioned_finger_proxy=bool(args.pose_conditioned_finger_proxy),
                    )
                    loss = loss + float(args.finger_absence_loss_weight) * finger_loss
                    metrics.update(finger_metrics)
                finger_missing_stage = str(args.finger_missing_threshold_loss_stage)
                finger_threshold_table = batch.get("finger_threshold_visibility_table")
                if (
                    float(args.finger_missing_threshold_loss_weight) > 0
                    and finger_missing_stage in ("all", stage)
                    and visible_targets is not None
                    and finger_threshold_table is not None
                    and int(visible_targets.shape[-1]) == len(BODYPART_PROXY_PARTS)
                    and int(finger_threshold_table.ndim) == 5
                    and int(finger_threshold_table.shape[-1]) > 0
                ):
                    finger_missing_loss, finger_missing_metrics = _finger_missing_threshold_loss_and_metrics(
                        outputs,
                        visible_targets.to(device)[idx],
                        finger_threshold_table.to(device)[idx],
                        target_threshold=float(args.finger_missing_threshold_target_threshold),
                        pred_threshold=float(args.finger_missing_threshold_pred_threshold),
                    )
                    loss = loss + float(args.finger_missing_threshold_loss_weight) * finger_missing_loss
                    metrics.update(finger_missing_metrics)
                finger_visibility_stage = str(args.finger_visibility_threshold_loss_stage)
                if (
                    float(args.finger_visibility_threshold_loss_weight) > 0
                    and finger_visibility_stage in ("all", stage)
                    and visible_targets is not None
                    and finger_threshold_table is not None
                    and int(visible_targets.shape[-1]) == len(BODYPART_PROXY_PARTS)
                    and int(finger_threshold_table.ndim) == 5
                    and int(finger_threshold_table.shape[-1]) > 0
                ):
                    finger_visibility_loss, finger_visibility_metrics = _finger_visibility_threshold_loss_and_metrics(
                        outputs,
                        visible_targets.to(device)[idx],
                        finger_threshold_table.to(device)[idx],
                        pred_threshold=float(args.finger_visibility_threshold_pred_threshold),
                    )
                    loss = loss + float(args.finger_visibility_threshold_loss_weight) * finger_visibility_loss
                    metrics.update(finger_visibility_metrics)
                if stage == "keyframe" and float(args.precut_adjacent_contrast_weight) > 0:
                    contrast_fields = _parse_optional_fields(str(args.precut_adjacent_contrast_fields), fields)
                    if not contrast_fields:
                        contrast_fields = fields
                    contrast_loss, contrast_metrics = _precut_adjacent_contrast_loss_and_metrics(
                        outputs,
                        {field: value[idx] for field, value in labels.items()},
                        {field: value[idx] for field, value in post_cut_labels.items()},
                        batch["post_cut_mask"].to(device)[idx],
                        contrast_fields,
                    )
                    loss = loss + float(args.precut_adjacent_contrast_weight) * contrast_loss
                    metrics.update(contrast_metrics)
                if stage == "keyframe" and float(args.visual_scale_contrast_weight) > 0:
                    scale_contrast_loss, scale_contrast_metrics = _precut_visual_scale_contrast_loss_and_metrics(
                        outputs,
                        {field: value[idx] for field, value in labels.items()},
                        {field: value[idx] for field, value in post_cut_labels.items()},
                        batch["post_cut_mask"].to(device)[idx],
                        vocabularies=vocabularies,
                        margin=float(args.visual_scale_contrast_margin),
                    )
                    loss = loss + float(args.visual_scale_contrast_weight) * scale_contrast_loss
                    metrics.update(scale_contrast_metrics)
                if stage == "keyframe" and float(args.noncut_smoothness_loss_weight) > 0:
                    smooth_fields = _parse_optional_fields(str(args.noncut_smoothness_fields), fields)
                    smooth_loss, smooth_metrics = _noncut_smoothness_loss_and_metrics(
                        outputs,
                        {field: value[idx] for field, value in history_labels.items()},
                        batch["history_mask"].to(device)[idx],
                        batch["history_delta"].to(device)[idx],
                        fields=smooth_fields,
                        vocabularies=vocabularies,
                        max_delta=float(args.noncut_smoothness_max_delta),
                        margin=float(args.noncut_smoothness_margin),
                    )
                    loss = loss + float(args.noncut_smoothness_loss_weight) * smooth_loss
                    metrics.update(smooth_metrics)
                batch_loss = batch_loss + float(weight) * loss
                for key, value in metrics.items():
                    stage_metric_accum[f"{stage}_{key}"] = stage_metric_accum.get(f"{stage}_{key}", 0.0) + float(value) * int(idx.numel())
                stage_metric_accum[f"{stage}_samples"] = stage_metric_accum.get(f"{stage}_samples", 0.0) + int(idx.numel())
                stage_count += 1
            if stage_count > 0:
                batch_loss = batch_loss / float(stage_count)
            if training:
                batch_loss.backward()
                if float(args.grad_clip) > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                optimizer.step()
        batch_size = int(audio.shape[0])
        sample_total += batch_size
        totals["loss"] = totals.get("loss", 0.0) + float(batch_loss.detach().item()) * batch_size
        for key, value in stage_metric_accum.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    if sample_total <= 0:
        return {"loss": math.inf}
    out = {"loss": totals["loss"] / sample_total}
    for key, value in totals.items():
        if key == "loss" or key.endswith("_samples"):
            continue
        if key.startswith("cut_"):
            denom = max(1.0, totals.get("cut_samples", 0.0))
        elif key.startswith("keyframe_"):
            denom = max(1.0, totals.get("keyframe_samples", 0.0))
        else:
            denom = sample_total
        out[key] = value / denom
    out["cut_sample_rate"] = totals.get("cut_samples", 0.0) / max(1, sample_total)
    return out


def _init_wandb(args: argparse.Namespace, run_dir: Path, config: dict[str, Any]) -> Any | None:
    if args.disable_wandb:
        return None
    if wandb is None:
        raise SystemExit("wandb is not installed. Use --disable-wandb or install wandb.")
    key_path = Path(args.wandb_key_file)
    if not key_path.exists():
        raise SystemExit(f"W&B key file not found: {key_path}")
    key = key_path.read_text(encoding="utf-8").strip()
    os.environ["WANDB_API_KEY"] = key
    wandb.login(key=key, relogin=True)
    return wandb.init(
        project=str(args.wandb_project),
        name=str(args.wandb_run_name or run_dir.name),
        dir=str(run_dir),
        config=_to_jsonable(config),
    )


def main() -> None:
    args = build_parser().parse_args()
    _set_seed(int(args.seed))
    device = _device_from_arg(str(args.device))
    dataset_root = Path(args.dataset_root)
    manifest = load_manifest(dataset_root)
    fields = label_fields_from_manifest(manifest)
    try:
        semantic_field_names = parse_semantic_fields(str(args.semantic_loss_fields), fields)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    stats = compute_normalization_stats(dataset_root, "train")
    run_dir = _build_run_dir(Path(args.output_root), str(args.tag))
    save_normalization_stats(run_dir / "normalization_stats.npz", stats)

    train_ds = V3CutContextDataset(
        dataset_root,
        "train",
        normalization_stats=stats,
        max_position_tokens=int(args.max_position_tokens),
        position_context_mode=str(args.position_context_mode),
        position_context_side_count=int(args.position_context_side_count),
    )
    val_ds = V3CutContextDataset(
        dataset_root,
        "val",
        normalization_stats=stats,
        max_position_tokens=int(args.max_position_tokens),
        position_context_mode=str(args.position_context_mode),
        position_context_side_count=int(args.position_context_side_count),
    )
    test_ds = V3CutContextDataset(
        dataset_root,
        "test",
        normalization_stats=stats,
        max_position_tokens=int(args.max_position_tokens),
        position_context_mode=str(args.position_context_mode),
        position_context_side_count=int(args.position_context_side_count),
    )
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers))
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    semantic_soft_targets = {
        field: build_semantic_target_distributions(
            field=field,
            vocab=list(manifest["vocabularies"][field]),
            temperature=float(args.semantic_loss_temperature),
            device=device,
        )
        for field in semantic_field_names
    }

    config = V3CutContextConfig(
        audio_dim=int(train_ds.audio.shape[-1]),
        dance_dim=int(train_ds.dance.shape[-1]),
        window_size=int(train_ds.audio.shape[1]),
        vocab_sizes=train_ds.vocab_sizes,
        architecture=str(args.architecture),
        model_dim=int(args.model_dim),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        ffn_dim=int(args.ffn_dim),
        mlp_extra_layers=int(args.mlp_extra_layers),
        stage_specific_heads=bool(args.stage_specific_heads),
        dropout=float(args.dropout),
        input_dropout=float(args.input_dropout),
        max_position_tokens=int(args.max_position_tokens),
        fusion=str(args.fusion),
        joint_vocab_sizes=_build_joint_vocab_sizes(args, train_ds.vocab_sizes),
    )
    model = V3CutContextShotPlanModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    saved_config = {
        "args": vars(args),
        "run_dir": str(run_dir),
        "dataset_root": str(dataset_root),
        "label_fields": list(fields),
        "semantic_loss_fields_resolved": list(semantic_field_names),
        "vocabularies": manifest["vocabularies"],
        "model_config": config.__dict__,
        "joint_vocab_sizes": _build_joint_vocab_sizes(args, train_ds.vocab_sizes),
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_test": len(test_ds),
        "note": "ReverseV3 discrete shot-plan labels; cut stage predicts only second frame of adjacent cut pair.",
    }
    _save_json(run_dir / "config.json", saved_config)
    wandb_run = _init_wandb(args, run_dir, saved_config)
    best_val = math.inf
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    checkpoint_dir = run_dir / "checkpoints"
    explicit_save_epochs = _parse_epoch_set(str(args.save_epochs))
    saved_epoch_checkpoints: list[str] = []
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            fields=fields,
            vocabularies={field: list(manifest["vocabularies"][field]) for field in fields},
            semantic_soft_targets=semantic_soft_targets,
            args=args,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            device=device,
            optimizer=None,
            fields=fields,
            vocabularies={field: list(manifest["vocabularies"][field]) for field in fields},
            semantic_soft_targets=semantic_soft_targets,
            args=args,
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        payload = {
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "fields": fields,
            "vocabularies": manifest["vocabularies"],
            "stats": stats,
            "epoch": epoch,
            "val_loss": val_metrics["loss"],
            "args": vars(args),
        }
        torch.save(payload, last_path)
        if val_metrics["loss"] < best_val:
            best_val = float(val_metrics["loss"])
            torch.save(payload, best_path)
        should_save_epoch = (
            epoch in explicit_save_epochs
            or (int(args.save_every) > 0 and epoch % int(args.save_every) == 0)
        )
        if should_save_epoch:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            epoch_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            torch.save(payload, epoch_path)
            saved_epoch_checkpoints.append(str(epoch_path))
            keep_count = int(args.keep_epoch_checkpoints)
            if keep_count > 0:
                epoch_paths = sorted(checkpoint_dir.glob("epoch_*.pt"))
                for old_path in epoch_paths[:-keep_count]:
                    old_path.unlink(missing_ok=True)
        if wandb_run is not None:
            wandb_run.log({f"train/{k}": v for k, v in train_metrics.items()} | {f"val/{k}": v for k, v in val_metrics.items()} | {"epoch": epoch})
        _save_json(run_dir / "history.json", history)
        print(
            f"epoch {epoch:03d} | train loss={train_metrics['loss']:.4f} "
            f"cut_all={train_metrics.get('cut_all_field_acc', 0):.3f} key_all={train_metrics.get('keyframe_all_field_acc', 0):.3f} | "
            f"val loss={val_metrics['loss']:.4f} cut_all={val_metrics.get('cut_all_field_acc', 0):.3f} "
            f"key_all={val_metrics.get('keyframe_all_field_acc', 0):.3f}"
        )
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = _run_epoch(
        model,
        test_loader,
        device=device,
        optimizer=None,
        fields=fields,
        vocabularies={field: list(manifest["vocabularies"][field]) for field in fields},
        semantic_soft_targets=semantic_soft_targets,
        args=args,
    )
    metrics = {
        "best_val_loss": best_val,
        "best_epoch": int(checkpoint["epoch"]),
        "saved_epoch_checkpoints": (
            [str(path) for path in sorted(checkpoint_dir.glob("epoch_*.pt"))]
            if checkpoint_dir.exists()
            else saved_epoch_checkpoints
        ),
        "test": test_metrics,
        "history": history,
    }
    _save_json(run_dir / "metrics.json", metrics)
    if wandb_run is not None:
        wandb_run.log({f"test/{k}": v for k, v in test_metrics.items()})
        wandb_run.finish()
    print(f"run_dir: {run_dir}")
    print(f"best: {best_path}")
    print(f"test loss={test_metrics['loss']:.4f} cut_all={test_metrics.get('cut_all_field_acc', 0):.3f} key_all={test_metrics.get('keyframe_all_field_acc', 0):.3f}")


if __name__ == "__main__":
    main()
