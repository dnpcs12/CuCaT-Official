from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


_ORDINAL_POSITIONS: dict[str, dict[str, float]] = {
    "target_anchor": {
        "feet": 0.0,
        "lower_leg": 1.0,
        "thigh": 2.0,
        "hip": 3.0,
        "waist": 4.0,
        "chest": 5.0,
        "face": 6.0,
    },
    "distance": {
        "extreme_close": 0.0,
        "close": 1.0,
        "medium_close": 2.0,
        "medium": 3.0,
        "medium_far": 4.0,
        "far": 5.0,
        "wide": 6.0,
    },
    "pitch": {
        "low": 0.0,
        "level": 1.0,
        "high": 2.0,
        "top": 3.0,
    },
    "composition": {
        "left_third": 0.0,
        "center": 1.0,
        "right_third": 2.0,
    },
    "roll": {
        "strong_left": -3.0,
        "left": -2.0,
        "slight_left": -1.0,
        "none": 0.0,
        "slight_right": 1.0,
        "right": 2.0,
        "strong_right": 3.0,
    },
    "fov": {
        "narrow": 0.0,
        "normal": 1.0,
        "wide": 2.0,
    },
}

_YAW_DEGREES: dict[str, float] = {
    "front": 0.0,
    "left_45": -45.0,
    "profile_left": -90.0,
    "back_left_45": -135.0,
    "back": 180.0,
    "back_right_45": 135.0,
    "profile_right": 90.0,
    "right_45": 45.0,
}


def parse_semantic_fields(raw: str, available_fields: tuple[str, ...]) -> tuple[str, ...]:
    """Parse a comma-separated semantic-loss field list."""

    if str(raw).strip().lower() in {"", "none", "off", "0"}:
        return ()
    available = set(available_fields)
    fields = tuple(token.strip() for token in str(raw).split(",") if token.strip())
    invalid = [field for field in fields if field not in available]
    if invalid:
        raise ValueError(
            f"Invalid semantic loss fields {invalid}. Available: {', '.join(available_fields)}"
        )
    return fields


def _circular_yaw_distance(a: str, b: str) -> float:
    a_deg = _YAW_DEGREES.get(a)
    b_deg = _YAW_DEGREES.get(b)
    if a_deg is None or b_deg is None:
        return 0.0 if a == b else 1.0
    diff = abs(a_deg - b_deg) % 360.0
    return min(diff, 360.0 - diff) / 180.0


def _ordinal_distance(field: str, a: str, b: str) -> float:
    positions = _ORDINAL_POSITIONS.get(field)
    if not positions:
        return 0.0 if a == b else 1.0
    a_pos = positions.get(a)
    b_pos = positions.get(b)
    if a_pos is None or b_pos is None:
        return 0.0 if a == b else 1.0
    span = max(1.0, max(positions.values()) - min(positions.values()))
    return abs(a_pos - b_pos) / span


def semantic_distance_matrix(field: str, vocab: list[Any], *, device: torch.device) -> torch.Tensor:
    """Return a normalized class-distance matrix for a shot-plan field.

    The yaw field is circular, so front/back-left/right adjacency is measured
    on a 360-degree ring rather than with a plain index distance.
    """

    values = [str(value) for value in vocab]
    size = len(values)
    matrix = torch.zeros(size, size, dtype=torch.float32, device=device)
    for i, left in enumerate(values):
        for j, right in enumerate(values):
            if field == "yaw":
                distance = _circular_yaw_distance(left, right)
            else:
                distance = _ordinal_distance(field, left, right)
            matrix[i, j] = float(distance)
    return matrix


def build_semantic_target_distributions(
    *,
    field: str,
    vocab: list[Any],
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    """Build soft target rows where nearby semantic classes get more mass."""

    distances = semantic_distance_matrix(field, vocab, device=device)
    temp = max(1e-4, float(temperature))
    targets = torch.softmax(-distances / temp, dim=-1)
    return targets


def semantic_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    soft_targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross entropy with semantic soft targets and per-item reduction."""

    original_shape = target.shape
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_target = target.reshape(-1)
    valid = flat_target != int(ignore_index)
    losses = flat_logits.new_zeros(flat_target.shape)
    if torch.any(valid):
        log_probs = F.log_softmax(flat_logits[valid], dim=-1)
        selected_targets = soft_targets.to(flat_logits.device)[flat_target[valid].long()]
        losses[valid] = -(selected_targets * log_probs).sum(dim=-1)
    return losses.reshape(original_shape)

