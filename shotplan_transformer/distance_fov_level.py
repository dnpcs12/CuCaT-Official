from __future__ import annotations

from typing import Any


V3_DISTANCE_ORDER = [
    "extreme_close",
    "very_close",
    "close",
    "medium_close",
    "medium",
    "medium_far",
    "far",
    "wide",
]

V3_DISTANCE_RANKS = {
    "extreme_close": 60,
    "very_close": 85,
    "close": 120,
    "medium_close": 175,
    "medium": 255,
    "medium_far": 370,
    "far": 580,
    "wide": 720,
}

V3_FOV_ORDER = ["narrow", "normal", "wide"]

V3_FOV_RANK_OFFSETS = {
    "narrow": -25,
    "normal": 0,
    "wide": 25,
}

# Coarse bins: close variants are merged, but distance+FOV still nudges a shot
# across a boundary when it changes the perceived camera scale.
DISTANCE_FOV_LEVEL_THRESHOLDS: list[tuple[int, str]] = [
    (110, "tight_close"),
    (150, "close"),
    (215, "medium_close"),
    (315, "medium"),
    (475, "medium_far"),
    (650, "far"),
    (10_000, "wide"),
]

DISTANCE_FOV_LEVEL_ORDER = [
    "tight_close",
    "close",
    "medium_close",
    "medium",
    "medium_far",
    "far",
    "wide",
]

DISTANCE_FOV_GROUPS = {
    "tight_close": "close_group",
    "close": "close_group",
    "medium_close": "medium_group",
    "medium": "medium_group",
    "medium_far": "far_group",
    "far": "far_group",
    "wide": "far_group",
}


def distance_fov_rank(row: dict[str, Any]) -> int:
    distance = str(row.get("distance", "medium")).strip()
    fov = str(row.get("fov", "normal")).strip()
    return int(
        V3_DISTANCE_RANKS.get(distance, V3_DISTANCE_RANKS["medium"])
        + V3_FOV_RANK_OFFSETS.get(fov, 0)
    )


def distance_fov_level_label(row: dict[str, Any]) -> str:
    rank = distance_fov_rank(row)
    for upper, label in DISTANCE_FOV_LEVEL_THRESHOLDS:
        if rank < upper:
            return label
    return "wide"


def distance_fov_group_label(row: dict[str, Any]) -> str:
    return DISTANCE_FOV_GROUPS.get(distance_fov_level_label(row), "medium_group")


def distance_fov_level_index(level: str) -> int:
    value = str(level or "medium").strip()
    if value in DISTANCE_FOV_LEVEL_ORDER:
        return DISTANCE_FOV_LEVEL_ORDER.index(value)
    return DISTANCE_FOV_LEVEL_ORDER.index("medium")


def adjacent_distance_values(distance: str, radius: int = 1) -> list[str]:
    value = str(distance or "medium").strip()
    if value not in V3_DISTANCE_ORDER:
        value = "medium"
    index = V3_DISTANCE_ORDER.index(value)
    out: list[str] = []
    for offset in range(-int(radius), int(radius) + 1):
        if offset == 0:
            continue
        candidate_index = index + offset
        if 0 <= candidate_index < len(V3_DISTANCE_ORDER):
            out.append(V3_DISTANCE_ORDER[candidate_index])
    return out


def adjacent_fov_values(fov: str, radius: int = 1) -> list[str]:
    value = str(fov or "normal").strip()
    if value not in V3_FOV_ORDER:
        value = "normal"
    index = V3_FOV_ORDER.index(value)
    out: list[str] = []
    for offset in range(-int(radius), int(radius) + 1):
        if offset == 0:
            continue
        candidate_index = index + offset
        if 0 <= candidate_index < len(V3_FOV_ORDER):
            out.append(V3_FOV_ORDER[candidate_index])
    return out
