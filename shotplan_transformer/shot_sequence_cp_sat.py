from __future__ import annotations

import math
from collections import Counter, defaultdict
from pprint import pprint
from typing import Any

from ortools.sat.python import cp_model

try:
    from .bodypart_proxy import bodypart_combo_label
    from .distance_fov_level import distance_fov_group_label, distance_fov_level_index, distance_fov_level_label
except ImportError:  # pragma: no cover - supports direct script execution
    from bodypart_proxy import bodypart_combo_label
    from distance_fov_level import distance_fov_group_label, distance_fov_level_index, distance_fov_level_label


V2_FIELDS = ["target_body", "shot_size", "angle", "composition", "roll"]
V3_1_FIELDS = ["target_anchor", "distance", "yaw", "pitch", "composition", "roll", "fov"]
FIELDS = V2_FIELDS

DEFAULT_WEIGHTS: dict[str, Any] = {
    "marginal": {
        "target_body": 10,
        "shot_size": 10,
        "angle": 10,
        "composition": 5,
        "roll": 3,
        "target_anchor": 10,
        "distance": 10,
        "yaw": 10,
        "pitch": 8,
        "fov": 4,
    },
    "transition": {
        "target_body": 10,
        "shot_size": 10,
        "angle": 10,
        "composition": 5,
        "roll": 3,
        "target_anchor": 10,
        "distance": 10,
        "yaw": 10,
        "pitch": 8,
        "fov": 4,
    },
    "change": {
        "target_body": 5,
        "shot_size": 8,
        "angle": 8,
        "composition": 3,
        "roll": 2,
        "target_anchor": 5,
        "distance": 8,
        "yaw": 8,
        "pitch": 6,
        "fov": 3,
    },
    "local": 1,
    "distance_bounce": 0,
    "cut_identity": 0,
    "cut_contrast": 0,
    "post_cut_distance_jump": 0,
    "cut_context_speed": 0,
    "repeated_distance_bounce": 0,
    "composition_bounce": 0,
    "vertical_bounce": 0,
    "top_pitch_bounce": 0,
    "turn_yaw_jitter": 0,
    "bodypart_duration": 0,
    "bodypart_duration_short": 0,
    "distance_level_duration": 0,
    "distance_level_duration_short": 0,
    "distance_level_direction": 0,
    "rare_duration": 0,
}

BODY_HEIGHT_RATIOS = {
    "face": 0.16,
    "feet": 0.22,
    "bust_up": 0.27,
    "abdomen": 0.32,
    "waist_up": 0.50,
    "knee_up": 0.68,
    "legs": 0.68,
    "full_body": 1.00,
}

SHOT_HEIGHT_FRACTIONS = {
    "far": 0.46,
    "medium": 0.66,
    "close": 0.84,
}

SHOT_FOV_DEGREES = {
    "far": 34.0,
    "medium": 30.0,
    "close": 24.0,
}

DISTANCE_SCALES = {
    ("face", "far"): 1.32,
    ("face", "medium"): 1.13,
    ("face", "close"): 1.12,
    ("bust_up", "far"): 0.96,
    ("bust_up", "medium"): 0.90,
    ("bust_up", "close"): 0.96,
    ("waist_up", "far"): 1.07,
    ("waist_up", "medium"): 1.07,
    ("waist_up", "close"): 1.07,
    ("knee_up", "far"): 0.95,
    ("knee_up", "medium"): 0.95,
    ("knee_up", "close"): 0.95,
    ("abdomen", "medium"): 1.25,
    ("abdomen", "close"): 0.90,
    ("feet", "far"): 1.06,
    ("feet", "medium"): 1.02,
    ("feet", "close"): 0.98,
}

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

V3_TARGET_ANCHOR_RANK_OFFSETS = {
    "face": -20,
    "chest": -8,
    "waist": 0,
    "hip": 0,
    "thigh": 10,
    "lower_leg": 22,
    "feet": 34,
}

V3_FOV_RANK_OFFSETS = {
    "narrow": -25,
    "normal": 0,
    "wide": 25,
}

COMPOSITION_RANKS = {
    "far_left": -160,
    "left": -100,
    "left_third": -100,
    "center": 0,
    "middle": 0,
    "right_third": 100,
    "right": 100,
    "far_right": 160,
}

V3_TARGET_ANCHOR_VERTICAL_RANKS = {
    "feet": 0,
    "lower_leg": 18,
    "thigh": 34,
    "hip": 46,
    "waist": 58,
    "chest": 78,
    "face": 100,
}

YAW_OFFSETS_DEG = {
    "front": 0.0,
    "left_45": 45.0,
    "profile_left": 90.0,
    "back_left_45": 135.0,
    "back": 180.0,
    "back_right_45": -135.0,
    "profile_right": -90.0,
    "right_45": -45.0,
}

V2_TARGET_BODY_VERTICAL_RANKS = {
    "feet": 0,
    "legs": 22,
    "knee_up": 38,
    "full_body": 50,
    "abdomen": 54,
    "waist_up": 62,
    "bust_up": 82,
    "face": 100,
}

SOLVER_STATUS_NAMES = {
    cp_model.OPTIMAL: "OPTIMAL",
    cp_model.FEASIBLE: "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
    cp_model.MODEL_INVALID: "MODEL_INVALID",
    cp_model.UNKNOWN: "UNKNOWN",
}


def _field_value(row: dict, field: str) -> Any:
    """Return a categorical field value, treating missing keys as None."""
    return row.get(field, None)


def _counter_tree_to_dict(value: Any) -> Any:
    """Convert nested Counter/defaultdict values into plain dicts."""
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, defaultdict):
        return {key: _counter_tree_to_dict(child) for key, child in value.items()}
    if isinstance(value, dict):
        return {key: _counter_tree_to_dict(child) for key, child in value.items()}
    return value


def _safe_len_denominator(length: int) -> int:
    """Return a non-zero denominator for sequence-level rescaling."""
    return max(1, int(length))


def _safe_transition_denominator(length: int) -> int:
    """Return a non-zero denominator for adjacent transition rescaling."""
    return max(1, int(length) - 1)


def _int_weight(value: Any) -> int:
    """Convert user-provided objective weights into CP-SAT integer weights."""
    return int(round(float(value)))


def fields_for_schema(schema: str) -> list[str]:
    """Return optimized semantic fields for a known shot-plan schema."""
    normalized = str(schema or "").strip().lower().replace("-", "_")
    if normalized in {"v3", "v3_1", "v31", "reverse_v3", "camera_autoshot.reverse_v3"}:
        return list(V3_1_FIELDS)
    return list(V2_FIELDS)


def _resolve_weights(weights: dict | None, fields: list[str]) -> dict[str, Any]:
    """Merge partial user weights with defaults and drop fields not optimized."""
    resolved = {
        "marginal": dict(DEFAULT_WEIGHTS["marginal"]),
        "transition": dict(DEFAULT_WEIGHTS["transition"]),
        "change": dict(DEFAULT_WEIGHTS["change"]),
        "local": DEFAULT_WEIGHTS["local"],
        "distance_bounce": DEFAULT_WEIGHTS["distance_bounce"],
        "cut_identity": DEFAULT_WEIGHTS["cut_identity"],
        "cut_contrast": DEFAULT_WEIGHTS["cut_contrast"],
        "post_cut_distance_jump": DEFAULT_WEIGHTS["post_cut_distance_jump"],
        "cut_context_speed": DEFAULT_WEIGHTS["cut_context_speed"],
        "repeated_distance_bounce": DEFAULT_WEIGHTS["repeated_distance_bounce"],
        "composition_bounce": DEFAULT_WEIGHTS["composition_bounce"],
        "vertical_bounce": DEFAULT_WEIGHTS["vertical_bounce"],
        "top_pitch_bounce": DEFAULT_WEIGHTS["top_pitch_bounce"],
        "turn_yaw_jitter": DEFAULT_WEIGHTS["turn_yaw_jitter"],
        "bodypart_duration": DEFAULT_WEIGHTS["bodypart_duration"],
        "bodypart_duration_short": DEFAULT_WEIGHTS["bodypart_duration_short"],
        "distance_level_duration": DEFAULT_WEIGHTS["distance_level_duration"],
        "distance_level_duration_short": DEFAULT_WEIGHTS["distance_level_duration_short"],
        "distance_level_direction": DEFAULT_WEIGHTS["distance_level_direction"],
        "rare_duration": DEFAULT_WEIGHTS["rare_duration"],
    }
    if weights:
        for section in ("marginal", "transition", "change"):
            if isinstance(weights.get(section), dict):
                resolved[section].update(weights[section])
        if "local" in weights:
            resolved["local"] = weights["local"]
        if "distance_bounce" in weights:
            resolved["distance_bounce"] = weights["distance_bounce"]
        if "cut_identity" in weights:
            resolved["cut_identity"] = weights["cut_identity"]
        if "cut_contrast" in weights:
            resolved["cut_contrast"] = weights["cut_contrast"]
        if "post_cut_distance_jump" in weights:
            resolved["post_cut_distance_jump"] = weights["post_cut_distance_jump"]
        if "cut_context_speed" in weights:
            resolved["cut_context_speed"] = weights["cut_context_speed"]
        if "repeated_distance_bounce" in weights:
            resolved["repeated_distance_bounce"] = weights["repeated_distance_bounce"]
        if "composition_bounce" in weights:
            resolved["composition_bounce"] = weights["composition_bounce"]
        if "vertical_bounce" in weights:
            resolved["vertical_bounce"] = weights["vertical_bounce"]
        if "top_pitch_bounce" in weights:
            resolved["top_pitch_bounce"] = weights["top_pitch_bounce"]
        if "turn_yaw_jitter" in weights:
            resolved["turn_yaw_jitter"] = weights["turn_yaw_jitter"]
        if "bodypart_duration" in weights:
            resolved["bodypart_duration"] = weights["bodypart_duration"]
        if "bodypart_duration_short" in weights:
            resolved["bodypart_duration_short"] = weights["bodypart_duration_short"]
        if "distance_level_duration" in weights:
            resolved["distance_level_duration"] = weights["distance_level_duration"]
        if "distance_level_duration_short" in weights:
            resolved["distance_level_duration_short"] = weights["distance_level_duration_short"]
        if "distance_level_direction" in weights:
            resolved["distance_level_direction"] = weights["distance_level_direction"]
        if "rare_duration" in weights:
            resolved["rare_duration"] = weights["rare_duration"]

    field_set = set(fields)
    for section in ("marginal", "transition", "change"):
        resolved[section] = {
            field: _int_weight(weight)
            for field, weight in resolved[section].items()
            if field in field_set
        }
    resolved["local"] = _int_weight(resolved["local"])
    resolved["distance_bounce"] = _int_weight(resolved["distance_bounce"])
    resolved["cut_identity"] = _int_weight(resolved["cut_identity"])
    resolved["cut_contrast"] = _int_weight(resolved["cut_contrast"])
    resolved["post_cut_distance_jump"] = _int_weight(resolved["post_cut_distance_jump"])
    resolved["cut_context_speed"] = _int_weight(resolved["cut_context_speed"])
    resolved["repeated_distance_bounce"] = _int_weight(resolved["repeated_distance_bounce"])
    resolved["composition_bounce"] = _int_weight(resolved["composition_bounce"])
    resolved["vertical_bounce"] = _int_weight(resolved["vertical_bounce"])
    resolved["top_pitch_bounce"] = _int_weight(resolved["top_pitch_bounce"])
    resolved["turn_yaw_jitter"] = _int_weight(resolved["turn_yaw_jitter"])
    resolved["bodypart_duration"] = _int_weight(resolved["bodypart_duration"])
    resolved["bodypart_duration_short"] = _int_weight(resolved["bodypart_duration_short"])
    resolved["distance_level_duration"] = _int_weight(resolved["distance_level_duration"])
    resolved["distance_level_duration_short"] = _int_weight(resolved["distance_level_duration_short"])
    resolved["distance_level_direction"] = _int_weight(resolved["distance_level_direction"])
    resolved["rare_duration"] = _int_weight(resolved["rare_duration"])
    return resolved


def _status_name(status: int) -> str:
    """Convert a CP-SAT status code into a stable readable string."""
    return SOLVER_STATUS_NAMES.get(status, "UNKNOWN")


def group_candidates_by_frame(candidates: list[dict]) -> list[list[dict]]:
    """Group a flat candidate list by frame and sort groups by ascending frame."""
    grouped: dict[Any, list[dict]] = defaultdict(list)
    for candidate in candidates:
        if "frame" not in candidate:
            raise KeyError("Every candidate must contain a 'frame' key.")
        grouped[candidate["frame"]].append(candidate)
    return [grouped[frame] for frame in sorted(grouped)]


def compute_marginal_counts(seq: list[dict], fields: list[str]) -> dict:
    """Compute per-field categorical occurrence counts."""
    counts: dict[str, Counter] = {field: Counter() for field in fields}
    for row in seq:
        for field in fields:
            counts[field][_field_value(row, field)] += 1
    return _counter_tree_to_dict(counts)


def compute_transition_counts(seq: list[dict], fields: list[str]) -> dict:
    """Compute per-field adjacent value transition counts."""
    counts: dict[str, defaultdict] = {
        field: defaultdict(Counter) for field in fields
    }
    for prev, current in zip(seq, seq[1:]):
        for field in fields:
            source = _field_value(prev, field)
            target = _field_value(current, field)
            counts[field][source][target] += 1
    return _counter_tree_to_dict(counts)


def compute_change_counts(seq: list[dict], fields: list[str]) -> dict:
    """Compute per-field adjacent value change counts."""
    counts = {field: 0 for field in fields}
    for prev, current in zip(seq, seq[1:]):
        for field in fields:
            if _field_value(prev, field) != _field_value(current, field):
                counts[field] += 1
    return counts


def compute_gt_stats(gt_sequence: list[dict], fields: list[str]) -> dict:
    """Bundle raw GT sequence statistics for later rescaling."""
    return {
        "marginal": compute_marginal_counts(gt_sequence, fields),
        "transition": compute_transition_counts(gt_sequence, fields),
        "change": compute_change_counts(gt_sequence, fields),
    }


def _frame_values(candidates_by_t: list[list[dict]], fields: list[str]) -> dict[str, set[Any]]:
    """Collect possible candidate values for each optimized field."""
    values: dict[str, set[Any]] = {field: set() for field in fields}
    for candidates_at_t in candidates_by_t:
        for candidate in candidates_at_t:
            for field in fields:
                values[field].add(_field_value(candidate, field))
    return values


def _gt_values(gt_stats: dict, fields: list[str]) -> dict[str, set[Any]]:
    """Collect values that appear in raw GT marginal statistics."""
    values: dict[str, set[Any]] = {field: set() for field in fields}
    for field in fields:
        values[field].update(gt_stats.get("marginal", {}).get(field, {}).keys())
    return values


def _candidate_transition_values(
    candidates_by_t: list[list[dict]],
    fields: list[str],
) -> dict[str, set[tuple[Any, Any]]]:
    """Collect possible adjacent field transitions from candidate pairs."""
    values: dict[str, set[tuple[Any, Any]]] = {field: set() for field in fields}
    for prev_candidates, current_candidates in zip(candidates_by_t, candidates_by_t[1:]):
        for prev in prev_candidates:
            for current in current_candidates:
                for field in fields:
                    values[field].add((_field_value(prev, field), _field_value(current, field)))
    return values


def _gt_transition_values(gt_stats: dict, fields: list[str]) -> dict[str, set[tuple[Any, Any]]]:
    """Collect transitions that appear in raw GT transition statistics."""
    values: dict[str, set[tuple[Any, Any]]] = {field: set() for field in fields}
    for field in fields:
        for source, targets in gt_stats.get("transition", {}).get(field, {}).items():
            for target in targets:
                values[field].add((source, target))
    return values


def _merge_value_sets(left: dict[str, set[Any]], right: dict[str, set[Any]]) -> dict[str, set[Any]]:
    """Merge two field-to-set mappings."""
    return {field: set(left.get(field, set())) | set(right.get(field, set())) for field in set(left) | set(right)}


def _rescale_gt_stats(
    *,
    gt_stats: dict,
    gt_length: int,
    candidate_length: int,
    fields: list[str],
    marginal_values: dict[str, set[Any]],
    transition_values: dict[str, set[tuple[Any, Any]]],
) -> dict:
    """Rescale raw GT counts to the candidate sequence length."""
    gt_len_den = _safe_len_denominator(gt_length)
    gt_trans_den = _safe_transition_denominator(gt_length)
    T = int(candidate_length)
    transition_horizon = max(1, T - 1)

    target_marginal: dict[str, dict[Any, int]] = {}
    for field in fields:
        field_counts = gt_stats.get("marginal", {}).get(field, {})
        target_marginal[field] = {
            value: int(round(int(field_counts.get(value, 0)) / gt_len_den * T))
            for value in marginal_values.get(field, set())
        }

    target_transition: dict[str, dict[Any, dict[Any, int]]] = {}
    for field in fields:
        target_transition[field] = {}
        raw_field_counts = gt_stats.get("transition", {}).get(field, {})
        for source, target in transition_values.get(field, set()):
            raw_count = int(raw_field_counts.get(source, {}).get(target, 0))
            rescaled = int(round(raw_count / gt_trans_den * transition_horizon))
            target_transition[field].setdefault(source, {})[target] = rescaled

    target_change = {
        field: int(round(int(gt_stats.get("change", {}).get(field, 0)) / gt_trans_den * transition_horizon))
        for field in fields
    }

    return {
        "marginal": target_marginal,
        "transition": target_transition,
        "change": target_change,
    }


def _add_abs_deviation(
    model: cp_model.CpModel,
    *,
    name: str,
    expression: Any,
    target: int,
    upper_bound: int,
) -> cp_model.IntVar:
    """Create an IntVar equal to abs(expression - target)."""
    dev = model.NewIntVar(0, max(0, int(upper_bound)), name)
    try:
        model.AddAbsEquality(dev, expression - int(target))
    except TypeError:
        model.Add(dev >= expression - int(target))
        model.Add(dev >= int(target) - expression)
    return dev


def _distance_rank(candidate: dict) -> int:
    """Approximate camera distance rank from the active shot schema.

    Smaller values are visually/physically closer shots; larger values are wider,
    farther shots. For V2 the proxy mirrors the old fit-distance terms; for
    V3/V3.1 it uses distance plus small anchor/FOV corrections.
    """
    if "distance" in candidate or "target_anchor" in candidate or "fov" in candidate:
        distance = str(candidate.get("distance", "medium")).strip()
        target_anchor = str(candidate.get("target_anchor", "hip")).strip()
        fov = str(candidate.get("fov", "normal")).strip()
        base = V3_DISTANCE_RANKS.get(distance, V3_DISTANCE_RANKS["medium"])
        anchor_offset = V3_TARGET_ANCHOR_RANK_OFFSETS.get(target_anchor, 0)
        fov_offset = V3_FOV_RANK_OFFSETS.get(fov, 0)
        return int(base + anchor_offset + fov_offset)

    target_body = str(candidate.get("target_body", "")).strip()
    shot_size = str(candidate.get("shot_size", "")).strip()
    body_height = BODY_HEIGHT_RATIOS.get(target_body, 0.50)
    shot_height = SHOT_HEIGHT_FRACTIONS.get(shot_size, SHOT_HEIGHT_FRACTIONS["medium"])
    fov_deg = SHOT_FOV_DEGREES.get(shot_size, SHOT_FOV_DEGREES["medium"])
    distance_scale = DISTANCE_SCALES.get((target_body, shot_size), 1.0)
    fov_term = math.tan(math.radians(fov_deg) * 0.5)
    if fov_term <= 1e-8 or shot_height <= 1e-8:
        return 0
    return int(round(100.0 * body_height * distance_scale / (shot_height * fov_term)))


def _composition_rank(candidate: dict) -> int:
    """Map composition labels onto a left/right screen-position axis."""
    value = str(candidate.get("composition", "center")).strip()
    return int(COMPOSITION_RANKS.get(value, 0))


def _vertical_rank(candidate: dict) -> int:
    """Map target labels onto an approximate bottom/top axis."""
    if "target_anchor" in candidate:
        target = str(candidate.get("target_anchor", "waist")).strip()
        return int(V3_TARGET_ANCHOR_VERTICAL_RANKS.get(target, V3_TARGET_ANCHOR_VERTICAL_RANKS["waist"]))

    target_body = str(candidate.get("target_body", "waist_up")).strip()
    return int(V2_TARGET_BODY_VERTICAL_RANKS.get(target_body, V2_TARGET_BODY_VERTICAL_RANKS["waist_up"]))


def _wrap_delta_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _unwrap_degrees(values: list[float]) -> list[float]:
    if not values:
        return []
    out = [float(values[0])]
    for value in values[1:]:
        out.append(out[-1] + _wrap_delta_deg(float(value) - out[-1]))
    return out


def _interpolate_track(frames: list[int], values: list[float], frame: int) -> float:
    if not frames or not values:
        return 0.0
    if frame <= int(frames[0]):
        return float(values[0])
    if frame >= int(frames[-1]):
        return float(values[-1])
    lo = 0
    hi = len(frames) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if int(frames[mid]) <= int(frame):
            lo = mid
        else:
            hi = mid
    left_frame = int(frames[lo])
    right_frame = int(frames[hi])
    if right_frame == left_frame:
        return float(values[lo])
    alpha = (float(frame) - float(left_frame)) / float(right_frame - left_frame)
    return float(values[lo]) * (1.0 - alpha) + float(values[hi]) * alpha


def _yaw_offset(candidate: dict) -> float:
    return float(YAW_OFFSETS_DEG.get(str(candidate.get("yaw", "front")).strip(), 0.0))


def _sign(value: float, *, eps: float = 1e-6) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _turn_yaw_jitter_pair_info(
    left: dict,
    right: dict,
    *,
    left_heading: float,
    right_heading: float,
    fps: float,
    min_heading_delta: float,
    min_heading_speed: float,
    min_relative_yaw_delta: float,
    min_world_yaw_delta: float,
    require_same_direction: bool,
) -> dict | None:
    left_frame = int(left.get("source_frame", left.get("frame", 0)))
    right_frame = int(right.get("source_frame", right.get("frame", 0)))
    frame_gap = right_frame - left_frame
    if frame_gap <= 0:
        return None

    heading_delta = float(right_heading) - float(left_heading)
    seconds = frame_gap / max(1e-6, float(fps))
    heading_speed = abs(heading_delta) / max(1e-6, seconds)
    if abs(heading_delta) < float(min_heading_delta) and heading_speed < float(min_heading_speed):
        return None

    relative_delta = _wrap_delta_deg(_yaw_offset(right) - _yaw_offset(left))
    if abs(relative_delta) < float(min_relative_yaw_delta):
        return None

    world_delta = _wrap_delta_deg((float(right_heading) + _yaw_offset(right)) - (float(left_heading) + _yaw_offset(left)))
    if abs(world_delta) < float(min_world_yaw_delta):
        return None

    same_direction = _sign(heading_delta) == _sign(relative_delta) and _sign(heading_delta) != 0
    if bool(require_same_direction) and not same_direction:
        return None

    return {
        "heading_delta_deg": heading_delta,
        "heading_speed_deg_s": heading_speed,
        "relative_yaw_delta_deg": relative_delta,
        "world_yaw_delta_deg": world_delta,
        "same_direction": same_direction,
    }


def _step_source_frame(candidates_at_t: list[dict]) -> int:
    """Return the original source frame for a candidate timestep."""
    if not candidates_at_t:
        return 0
    first = candidates_at_t[0]
    return int(first.get("source_frame", first.get("frame", 0)))


def _distance_bounce_events(
    sequence: list[dict],
    *,
    min_rank_delta: int,
    max_frame_gap: int,
) -> list[dict]:
    """Find short V/逆V distance-rank bounces in a selected sequence."""
    events: list[dict] = []
    ranks = [_distance_rank(row) for row in sequence]
    frames = [int(row.get("source_frame", row.get("frame", 0))) for row in sequence]
    for center in range(1, len(sequence) - 1):
        gap_prev = frames[center] - frames[center - 1]
        gap_next = frames[center + 1] - frames[center]
        if gap_prev <= 0 or gap_next <= 0:
            continue
        if gap_prev > max_frame_gap or gap_next > max_frame_gap:
            continue
        left_delta = ranks[center] - ranks[center - 1]
        right_delta = ranks[center + 1] - ranks[center]
        if abs(left_delta) < min_rank_delta or abs(right_delta) < min_rank_delta:
            continue
        if left_delta * right_delta >= 0:
            continue
        events.append(
            {
                "center_index": center,
                "frames": [frames[center - 1], frames[center], frames[center + 1]],
                "distance_ranks": [ranks[center - 1], ranks[center], ranks[center + 1]],
                "pattern": "far_close_far" if left_delta < 0 else "close_far_close",
            }
        )
    return events


def _composition_bounce_events(
    sequence: list[dict],
    *,
    min_rank_delta: int,
    max_frame_gap: int,
) -> list[dict]:
    """Find short left/right composition reversals in a selected sequence."""
    events: list[dict] = []
    ranks = [_composition_rank(row) for row in sequence]
    frames = [int(row.get("source_frame", row.get("frame", 0))) for row in sequence]
    values = [str(row.get("composition", "center")) for row in sequence]
    for center in range(1, len(sequence) - 1):
        gap_prev = frames[center] - frames[center - 1]
        gap_next = frames[center + 1] - frames[center]
        if gap_prev <= 0 or gap_next <= 0:
            continue
        if gap_prev > max_frame_gap or gap_next > max_frame_gap:
            continue
        left_delta = ranks[center] - ranks[center - 1]
        right_delta = ranks[center + 1] - ranks[center]
        if abs(left_delta) < min_rank_delta or abs(right_delta) < min_rank_delta:
            continue
        if left_delta * right_delta >= 0:
            continue
        events.append(
            {
                "center_index": center,
                "frames": [frames[center - 1], frames[center], frames[center + 1]],
                "compositions": values[center - 1 : center + 2],
                "composition_ranks": ranks[center - 1 : center + 2],
                "pattern": "leftward_rightward" if left_delta < 0 else "rightward_leftward",
            }
        )
    return events


def _vertical_bounce_events(
    sequence: list[dict],
    *,
    min_rank_delta: int,
    max_frame_gap: int,
) -> list[dict]:
    """Find short high-target spikes at a cut slot."""
    events: list[dict] = []
    ranks = [_vertical_rank(row) for row in sequence]
    frames = [int(row.get("source_frame", row.get("frame", 0))) for row in sequence]
    distances = [str(row.get("distance", "")) for row in sequence]
    labels = [
        {
            "target_anchor": row.get("target_anchor"),
            "target_body": row.get("target_body"),
            "pitch": row.get("pitch"),
            "angle": row.get("angle"),
        }
        for row in sequence
    ]
    for center in range(1, len(sequence) - 1):
        gap_prev = frames[center] - frames[center - 1]
        gap_next = frames[center + 1] - frames[center]
        if gap_prev != 1 or gap_next <= 0:
            continue
        if gap_next > max_frame_gap:
            continue
        if str(sequence[center].get("stage", "")).strip().lower() != "cut":
            continue
        if not (distances[center - 1] == distances[center] == distances[center + 1]):
            continue
        left_delta = ranks[center] - ranks[center - 1]
        right_delta = ranks[center + 1] - ranks[center]
        if left_delta < min_rank_delta or right_delta > -min_rank_delta:
            continue
        events.append(
            {
                "center_index": center,
                "frames": [frames[center - 1], frames[center], frames[center + 1]],
                "labels": labels[center - 1 : center + 2],
                "distances": distances[center - 1 : center + 2],
                "vertical_ranks": ranks[center - 1 : center + 2],
                "pattern": "cut_high_spike",
            }
        )
    return events


def _top_pitch_bounce_events(
    sequence: list[dict],
    *,
    neighbor_pitches: Any = None,
    return_pitches: Any = None,
    max_frame_gap: int,
) -> list[dict]:
    """Find short low/level -> top -> non-top pitch spikes."""
    events: list[dict] = []
    frames = [int(row.get("source_frame", row.get("frame", 0))) for row in sequence]
    pitches = [str(row.get("pitch", "level")).strip() for row in sequence]
    entry_pitches = _normalize_combo_set(neighbor_pitches or "low,level")
    exit_pitches = _normalize_combo_set(return_pitches or "low,level,high")
    for center in range(1, len(sequence) - 1):
        gap_prev = frames[center] - frames[center - 1]
        gap_next = frames[center + 1] - frames[center]
        if gap_prev <= 0 or gap_next <= 0:
            continue
        if gap_prev > int(max_frame_gap) or gap_next > int(max_frame_gap):
            continue
        if pitches[center] != "top":
            continue
        if pitches[center - 1] not in entry_pitches or pitches[center + 1] not in exit_pitches:
            continue
        events.append(
            {
                "center_index": center,
                "frames": [frames[center - 1], frames[center], frames[center + 1]],
                "pitches": pitches[center - 1 : center + 2],
                "pattern": "top_pitch_spike",
            }
        )
    return events


def _turn_yaw_jitter_events(
    sequence: list[dict],
    *,
    heading_frames: list[int] | None,
    heading_values: list[float] | None,
    fps: float,
    min_frame_gap: int,
    max_frame_gap: int,
    min_heading_delta: float,
    min_heading_speed: float,
    min_relative_yaw_delta: float,
    min_world_yaw_delta: float,
    require_same_direction: bool,
    exclude_cut_pairs: bool,
) -> list[dict]:
    """Find adjacent shot yaw changes during strong character turns."""
    if not heading_frames or not heading_values:
        return []
    frames = [int(row.get("source_frame", row.get("frame", 0))) for row in sequence]
    headings = [_interpolate_track(heading_frames, heading_values, frame) for frame in frames]
    events: list[dict] = []
    for index in range(1, len(sequence)):
        gap = frames[index] - frames[index - 1]
        if gap < int(min_frame_gap) or gap > int(max_frame_gap):
            continue
        if bool(exclude_cut_pairs) and _is_cut_pair_transition(sequence[index - 1], sequence[index]):
            continue
        info = _turn_yaw_jitter_pair_info(
            sequence[index - 1],
            sequence[index],
            left_heading=headings[index - 1],
            right_heading=headings[index],
            fps=float(fps),
            min_heading_delta=float(min_heading_delta),
            min_heading_speed=float(min_heading_speed),
            min_relative_yaw_delta=float(min_relative_yaw_delta),
            min_world_yaw_delta=float(min_world_yaw_delta),
            require_same_direction=bool(require_same_direction),
        )
        if info is None:
            continue
        events.append(
            {
                "index": index,
                "frames": [frames[index - 1], frames[index]],
                "frame_gap": gap,
                "yaw": [sequence[index - 1].get("yaw"), sequence[index].get("yaw")],
                "stage": [sequence[index - 1].get("stage"), sequence[index].get("stage")],
                "heading_deg": [round(headings[index - 1], 3), round(headings[index], 3)],
                "heading_delta_deg": round(float(info["heading_delta_deg"]), 3),
                "heading_speed_deg_s": round(float(info["heading_speed_deg_s"]), 3),
                "relative_yaw_delta_deg": round(float(info["relative_yaw_delta_deg"]), 3),
                "world_yaw_delta_deg": round(float(info["world_yaw_delta_deg"]), 3),
                "same_direction": bool(info["same_direction"]),
            }
        )
    return events


def _repeated_distance_bounce_events(
    sequence: list[dict],
    *,
    min_rank_delta: int,
    max_frame_gap: int,
) -> list[dict]:
    """Find repeated V/逆V distance motifs over five adjacent shot slots.

    Unlike `_distance_bounce_events`, this intentionally allows same-frame
    transitions because the bad pattern often starts at a cut slot pair, e.g.
    close -> far at the same frame, then quickly returns close afterwards.
    """
    events: list[dict] = []
    ranks = [_distance_rank(row) for row in sequence]
    frames = [int(row.get("source_frame", row.get("frame", 0))) for row in sequence]
    for start in range(0, len(sequence) - 4):
        frame_gaps = [frames[start + offset + 1] - frames[start + offset] for offset in range(4)]
        if any(gap < 0 or gap > max_frame_gap for gap in frame_gaps):
            continue
        deltas = [ranks[start + offset + 1] - ranks[start + offset] for offset in range(4)]
        if any(abs(delta) < min_rank_delta for delta in deltas):
            continue
        signs = [1 if delta > 0 else -1 for delta in deltas]
        if signs == [1, -1, 1, -1]:
            pattern = "close_far_close_far_close"
        elif signs == [-1, 1, -1, 1]:
            pattern = "far_close_far_close_far"
        else:
            continue
        events.append(
            {
                "start_index": start,
                "end_index": start + 4,
                "frames": frames[start : start + 5],
                "frame_gaps": frame_gaps,
                "distance_ranks": ranks[start : start + 5],
                "deltas": deltas,
                "pattern": pattern,
            }
        )
    return events


def _same_shot(left: dict, right: dict, fields: list[str]) -> bool:
    """Return True when two selected shots are identical for optimized fields."""
    return all(_field_value(left, field) == _field_value(right, field) for field in fields)


def _cut_pair_id(row: dict) -> str:
    value = row.get("cut_pair_id")
    if value is None:
        return ""
    return str(value)


def _is_explicit_cut_pair_transition(left: dict, right: dict) -> bool:
    left_pair = _cut_pair_id(left)
    right_pair = _cut_pair_id(right)
    if left_pair and left_pair == right_pair:
        return True

    left_timing = left.get("timing_frame")
    right_timing = right.get("timing_frame")
    if left_timing is not None and right_timing is not None and str(left_timing) == str(right_timing):
        try:
            duplicate_count = max(int(left.get("duplicate_count", 1)), int(right.get("duplicate_count", 1)))
            left_offset = int(left.get("duplicate_offset", -1))
            right_offset = int(right.get("duplicate_offset", -1))
        except (TypeError, ValueError):
            return False
        return duplicate_count >= 2 and right_offset == left_offset + 1
    return False


def _is_cut_pair_transition(left: dict, right: dict) -> bool:
    """Return whether adjacent selected rows represent a cut pair."""
    left_frame = int(left.get("source_frame", left.get("frame", 0)))
    right_frame = int(right.get("source_frame", right.get("frame", 0)))
    return left_frame == right_frame or _is_explicit_cut_pair_transition(left, right)


def _is_cut_pair_transition_groups(left_group: list[dict], right_group: list[dict]) -> bool:
    if not left_group or not right_group:
        return False
    return _is_cut_pair_transition(left_group[0], right_group[0])


def _shot_summary(row: dict, fields: list[str]) -> dict:
    return {field: _field_value(row, field) for field in fields}


def _cut_identity_events(sequence: list[dict], fields: list[str]) -> list[dict]:
    """Find same-source-frame cut slots that selected identical shot labels."""
    events: list[dict] = []
    frames = [int(row.get("source_frame", row.get("frame", 0))) for row in sequence]
    for index in range(1, len(sequence)):
        if not _is_cut_pair_transition(sequence[index - 1], sequence[index]):
            continue
        if not _same_shot(sequence[index - 1], sequence[index], fields):
            continue
        events.append(
            {
                "index": index,
                "frames": [frames[index - 1], frames[index]],
                "shot": {field: _field_value(sequence[index], field) for field in fields},
            }
        )
    return events


def _cut_contrast_score(
    left: dict,
    right: dict,
    *,
    fields: list[str],
    medium_rank_delta: int,
    strong_rank_delta: int,
) -> dict:
    """Score how visually strong a cut-pair transition is."""
    left_rank = _distance_rank(left)
    right_rank = _distance_rank(right)
    rank_delta = abs(right_rank - left_rank)
    field_changes = {
        field: _field_value(left, field) != _field_value(right, field)
        for field in fields
    }

    score = sum(1 for changed in field_changes.values() if changed)
    if rank_delta >= strong_rank_delta:
        score += 2
    elif rank_delta >= medium_rank_delta:
        score += 1

    major_fields = {"target_body", "shot_size", "target_anchor", "distance", "yaw", "pitch"}
    has_major_change = (
        any(field_changes.get(field, False) for field in major_fields)
        or rank_delta >= medium_rank_delta
    )
    return {
        "score": score,
        "has_major_change": has_major_change,
        "distance_ranks": [left_rank, right_rank],
        "rank_delta": rank_delta,
        "field_changes": field_changes,
    }


def _cut_contrast_events(
    sequence: list[dict],
    *,
    fields: list[str],
    min_score: int,
    medium_rank_delta: int,
    strong_rank_delta: int,
) -> list[dict]:
    """Find cut-pair slots whose visual contrast is too weak."""
    events: list[dict] = []
    frames = [int(row.get("source_frame", row.get("frame", 0))) for row in sequence]
    for index in range(1, len(sequence)):
        if not _is_cut_pair_transition(sequence[index - 1], sequence[index]):
            continue
        score_info = _cut_contrast_score(
            sequence[index - 1],
            sequence[index],
            fields=fields,
            medium_rank_delta=medium_rank_delta,
            strong_rank_delta=strong_rank_delta,
        )
        if score_info["has_major_change"] and int(score_info["score"]) >= int(min_score):
            continue
        events.append(
            {
                "index": index,
                "frames": [frames[index - 1], frames[index]],
                "penalty_units": max(1, int(min_score) - int(score_info["score"])),
                **score_info,
                "left": _shot_summary(sequence[index - 1], fields),
                "right": _shot_summary(sequence[index], fields),
            }
        )
    return events


def _post_cut_distance_jump_events(
    sequence: list[dict],
    *,
    min_rank_delta: int,
    max_frame_gap: int,
) -> list[dict]:
    """Find large distance jumps immediately after a cut slot pair."""
    events: list[dict] = []
    frames = [int(row.get("source_frame", row.get("frame", 0))) for row in sequence]
    ranks = [_distance_rank(row) for row in sequence]
    for index in range(2, len(sequence)):
        if not _is_cut_pair_transition(sequence[index - 2], sequence[index - 1]):
            continue
        gap = frames[index] - frames[index - 1]
        if gap <= 0 or gap > max_frame_gap:
            continue
        delta = ranks[index] - ranks[index - 1]
        if abs(delta) < min_rank_delta:
            continue
        events.append(
            {
                "index": index,
                "frames": [frames[index - 1], frames[index]],
                "distance_ranks": [ranks[index - 1], ranks[index]],
                "direction": "farther" if delta > 0 else "closer",
            }
        )
    return events


def _cut_context_speed_events(
    sequence: list[dict],
    *,
    min_speed: float,
    max_frame_gap: int,
    fps: float,
) -> list[dict]:
    """Find fast distance-rank moves immediately before or after a cut pair."""
    events: list[dict] = []
    if fps <= 0:
        return events
    frames = [int(row.get("source_frame", row.get("frame", 0))) for row in sequence]
    ranks = [_distance_rank(row) for row in sequence]

    for index in range(1, len(sequence)):
        gap = frames[index] - frames[index - 1]
        if gap <= 0 or gap > max_frame_gap:
            continue

        before_cut = index + 1 < len(sequence) and _is_cut_pair_transition(sequence[index], sequence[index + 1])
        after_cut = index >= 2 and _is_cut_pair_transition(sequence[index - 2], sequence[index - 1])
        if not before_cut and not after_cut:
            continue

        delta = ranks[index] - ranks[index - 1]
        speed = abs(delta) * float(fps) / float(gap)
        if speed < float(min_speed):
            continue
        events.append(
            {
                "index": index,
                "context": "pre_cut" if before_cut else "post_cut",
                "frames": [frames[index - 1], frames[index]],
                "frame_gap": gap,
                "distance_ranks": [ranks[index - 1], ranks[index]],
                "rank_delta": delta,
                "speed": round(speed, 6),
                "direction": "farther" if delta > 0 else "closer",
            }
        )
    return events


def _rare_duration_classes(rare_duration_stats: dict | None) -> dict[str, dict]:
    if not isinstance(rare_duration_stats, dict):
        return {}
    classes = rare_duration_stats.get("classes", rare_duration_stats)
    if not isinstance(classes, dict):
        return {}
    return {str(label): value for label, value in classes.items() if isinstance(value, dict)}


def _candidate_rare_duration_matches(candidate: dict, rare_duration_stats: dict | None) -> list[tuple[str, dict]]:
    matches: list[tuple[str, dict]] = []
    for label, stats in _rare_duration_classes(rare_duration_stats).items():
        field = str(stats.get("field", "")).strip()
        value = str(stats.get("value", "")).strip()
        if not field or _field_value(candidate, field) != value:
            continue
        matches.append((label, stats))
    return matches


def _next_distinct_duration_sec(step_frames: list[int], index: int, fps: float) -> float | None:
    if fps <= 0 or index < 0 or index >= len(step_frames):
        return None
    frame = int(step_frames[index])
    for later_frame in step_frames[index + 1 :]:
        if int(later_frame) > frame:
            return float(int(later_frame) - frame) / float(fps)
    return None


def _duration_bound(stats: dict, key: str) -> float | None:
    value = stats.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rare_duration_penalty_units(
    duration_sec: float | None,
    stats: dict,
    *,
    fps: float,
    lower_key: str,
    upper_key: str,
    missing_penalty_units: int,
) -> int:
    if duration_sec is None:
        return max(0, int(missing_penalty_units))
    lower = _duration_bound(stats, lower_key)
    upper = _duration_bound(stats, upper_key)
    under = 0.0 if lower is None else max(0.0, float(lower) - float(duration_sec))
    over = 0.0 if upper is None else max(0.0, float(duration_sec) - float(upper))
    violation_sec = max(under, over)
    if violation_sec <= 1e-9:
        return 0
    return max(1, int(round(violation_sec * float(fps))))


def _rare_duration_events(
    sequence: list[dict],
    rare_duration_stats: dict | None,
    *,
    fps: float,
    lower_key: str,
    upper_key: str,
    missing_penalty_units: int,
) -> list[dict]:
    events: list[dict] = []
    step_frames = [int(row.get("source_frame", row.get("frame", 0))) for row in sequence]
    for index, row in enumerate(sequence):
        matches = _candidate_rare_duration_matches(row, rare_duration_stats)
        if not matches:
            continue
        duration_sec = _next_distinct_duration_sec(step_frames, index, fps)
        for label, stats in matches:
            penalty_units = _rare_duration_penalty_units(
                duration_sec,
                stats,
                fps=fps,
                lower_key=lower_key,
                upper_key=upper_key,
                missing_penalty_units=missing_penalty_units,
            )
            if penalty_units <= 0:
                continue
            events.append(
                {
                    "index": index,
                    "frame": step_frames[index],
                    "class": label,
                    "duration_sec": duration_sec,
                    "lower_sec": _duration_bound(stats, lower_key),
                    "upper_sec": _duration_bound(stats, upper_key),
                    "penalty_units": int(penalty_units),
                }
        )
    return events


def _bodypart_duration_classes(bodypart_duration_stats: dict | None) -> dict[str, dict]:
    if not isinstance(bodypart_duration_stats, dict):
        return {}
    classes = bodypart_duration_stats.get("classes", bodypart_duration_stats)
    if not isinstance(classes, dict):
        return {}
    return {str(label): value for label, value in classes.items() if isinstance(value, dict)}


def _bodypart_duration_bound(stats: dict, upper_key: str, *, fallback_key: str | None = "max_sec") -> float | None:
    value = stats.get(upper_key)
    if value is None and fallback_key:
        value = stats.get(fallback_key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_combo_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {part.strip() for part in value.split(",") if part.strip()}
    try:
        return {str(part).strip() for part in value if str(part).strip()}
    except TypeError:
        return {str(value).strip()} if str(value).strip() else set()


def _normalize_int_set(value: Any) -> set[int]:
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


def _distance_hold_key(row: dict, mode: str) -> Any:
    normalized = str(mode or "level").strip().lower()
    if normalized in {"distance", "raw_distance"}:
        return str(_field_value(row, "distance"))
    if normalized in {"distance_fov", "pair"}:
        return (str(_field_value(row, "distance")), str(_field_value(row, "fov")))
    if normalized in {"group", "distance_group", "level_group"}:
        return distance_fov_group_label(row)
    return distance_fov_level_label(row)


def _distance_or_fov_changed(candidate: dict, reference: dict) -> bool:
    return (
        str(_field_value(candidate, "distance")) != str(_field_value(reference, "distance"))
        or str(_field_value(candidate, "fov")) != str(_field_value(reference, "fov"))
    )


def _bodypart_duration_upper_bound(
    combo: str,
    stats: dict,
    upper_key: str,
    *,
    upper_blend_combos: Any = None,
    upper_blend_alpha: float = 0.0,
    upper_blend_from_key: str = "p95_sec",
    upper_blend_to_key: str = "max_sec",
) -> float | None:
    blend_combos = _normalize_combo_set(upper_blend_combos)
    if str(combo) in blend_combos:
        lower = _bodypart_duration_bound(stats, str(upper_blend_from_key))
        upper = _bodypart_duration_bound(stats, str(upper_blend_to_key))
        if lower is not None and upper is not None:
            alpha = min(1.0, max(0.0, float(upper_blend_alpha)))
            return float(lower + (upper - lower) * alpha)
    return _bodypart_duration_bound(stats, str(upper_key))


def _bodypart_combo(candidate: dict, proxy_mode: str) -> str:
    return bodypart_combo_label(candidate, proxy_mode=proxy_mode)


def _bodypart_duration_penalty_units(
    duration_sec: float,
    upper_sec: float | None,
    *,
    penalty_sec_per_unit: float,
) -> int:
    if upper_sec is None:
        return 0
    over = max(0.0, float(duration_sec) - float(upper_sec))
    if over <= 1e-9:
        return 0
    return max(1, int(math.ceil(over / max(1e-6, float(penalty_sec_per_unit)))))


def _bodypart_duration_short_penalty_units(
    duration_sec: float,
    lower_sec: float | None,
    *,
    penalty_sec_per_unit: float,
) -> int:
    if lower_sec is None:
        return 0
    under = max(0.0, float(lower_sec) - float(duration_sec))
    if under <= 1e-9:
        return 0
    return max(1, int(math.ceil(under / max(1e-6, float(penalty_sec_per_unit)))))


def _bodypart_run_touches_cut_pair(sequence: list[dict], start: int, end: int) -> bool:
    first_transition = max(1, int(start))
    last_transition = min(len(sequence) - 1, int(end) + 1)
    for index in range(first_transition, last_transition + 1):
        if _is_cut_pair_transition(sequence[index - 1], sequence[index]):
            return True
    return False


def _window_touches_cut_pair_groups(candidates_by_t: list[list[dict]], start: int, end: int) -> bool:
    first_transition = max(1, int(start))
    last_transition = min(len(candidates_by_t) - 1, int(end) + 1)
    for index in range(first_transition, last_transition + 1):
        if _is_cut_pair_transition_groups(candidates_by_t[index - 1], candidates_by_t[index]):
            return True
    return False


def _bodypart_duration_events(
    sequence: list[dict],
    bodypart_duration_stats: dict | None,
    *,
    fps: float,
    proxy_mode: str,
    upper_key: str,
    penalty_sec_per_unit: float,
    upper_blend_combos: Any = None,
    upper_blend_alpha: float = 0.0,
    upper_blend_from_key: str = "p95_sec",
    upper_blend_to_key: str = "max_sec",
    short_enabled: bool = False,
    lower_key: str = "p25_sec",
    short_penalty_sec_per_unit: float = 0.25,
    short_exclude_cut_pairs: bool = True,
) -> list[dict]:
    classes = _bodypart_duration_classes(bodypart_duration_stats)
    if not classes or len(sequence) < 2 or float(fps) <= 0:
        return []
    frames = [int(row.get("source_frame", row.get("frame", 0))) for row in sequence]
    combos = [_bodypart_combo(row, proxy_mode) for row in sequence]
    events: list[dict] = []
    start = 0
    for index in range(1, len(sequence) + 1):
        changed = index == len(sequence) or combos[index] != combos[start]
        if not changed:
            continue
        end_index = index - 1
        if index < len(sequence):
            end_frame = int(frames[index])
        else:
            end_frame = int(frames[end_index])
        duration_sec = max(0.0, float(end_frame - int(frames[start])) / float(fps))
        stats = classes.get(combos[start])
        upper_sec = (
            _bodypart_duration_upper_bound(
                combos[start],
                stats,
                upper_key,
                upper_blend_combos=upper_blend_combos,
                upper_blend_alpha=float(upper_blend_alpha),
                upper_blend_from_key=str(upper_blend_from_key),
                upper_blend_to_key=str(upper_blend_to_key),
            )
            if stats
            else None
        )
        penalty_units = _bodypart_duration_penalty_units(
            duration_sec,
            upper_sec,
            penalty_sec_per_unit=float(penalty_sec_per_unit),
        )
        if penalty_units > 0:
            events.append(
                {
                    "type": "overlong",
                    "start_index": start,
                    "end_index": end_index,
                    "frames": [frames[start], frames[end_index]],
                    "end_frame_exclusive": end_frame,
                    "combo": combos[start],
                    "duration_sec": duration_sec,
                    "upper_sec": upper_sec,
                    "upper_key": str(upper_key),
                    "penalty_units": int(penalty_units),
                }
            )
        if bool(short_enabled) and index < len(sequence):
            lower_sec = _bodypart_duration_bound(stats, lower_key, fallback_key=None) if stats else None
            short_units = _bodypart_duration_short_penalty_units(
                duration_sec,
                lower_sec,
                penalty_sec_per_unit=float(short_penalty_sec_per_unit),
            )
            if (
                short_units > 0
                and (not bool(short_exclude_cut_pairs) or not _bodypart_run_touches_cut_pair(sequence, start, end_index))
            ):
                events.append(
                    {
                        "type": "too_short",
                        "start_index": start,
                        "end_index": end_index,
                        "frames": [frames[start], frames[end_index]],
                        "end_frame_exclusive": end_frame,
                        "combo": combos[start],
                        "duration_sec": duration_sec,
                        "lower_sec": lower_sec,
                        "lower_key": str(lower_key),
                        "penalty_units": int(short_units),
                    }
                )
        start = index
    return events


def _distance_level_duration_classes(distance_level_duration_stats: dict | None) -> dict[str, dict]:
    if not isinstance(distance_level_duration_stats, dict):
        return {}
    classes = distance_level_duration_stats.get("classes", distance_level_duration_stats)
    if not isinstance(classes, dict):
        return {}
    return {str(label): value for label, value in classes.items() if isinstance(value, dict)}


def _distance_hold_duration_classes(distance_level_duration_stats: dict | None, hold_mode: str) -> dict[str, dict]:
    if not isinstance(distance_level_duration_stats, dict):
        return {}
    if str(hold_mode or "").strip().lower() == "group":
        classes = distance_level_duration_stats.get("group_classes", {})
        if isinstance(classes, dict):
            return {str(label): value for label, value in classes.items() if isinstance(value, dict)}
    return _distance_level_duration_classes(distance_level_duration_stats)


def _distance_level_upper_bound(
    level: str,
    stats: dict,
    upper_key: str,
    *,
    override_levels: Any = None,
    override_key: str = "max_sec",
) -> tuple[float | None, str]:
    override_set = _normalize_combo_set(override_levels)
    key = str(override_key) if str(level) in override_set else str(upper_key)
    upper_sec = _duration_bound(stats, key)
    if upper_sec is None and key != "max_sec":
        upper_sec = _duration_bound(stats, "max_sec")
    return upper_sec, key


def _run_duration_end_frame(step_frames: list[int], end_index: int, duration_end_mode: str) -> float:
    mode = str(duration_end_mode or "next_change").strip().lower()
    if end_index + 1 < len(step_frames):
        if mode == "midpoint":
            return (float(step_frames[end_index]) + float(step_frames[end_index + 1])) * 0.5
        return float(step_frames[end_index + 1])
    return float(step_frames[end_index])


def _run_duration_sec(step_frames: list[int], start: int, end_index: int, fps: float, duration_end_mode: str) -> float:
    if float(fps) <= 0:
        return 0.0
    end_frame = _run_duration_end_frame(step_frames, end_index, duration_end_mode)
    return max(0.0, float(end_frame - float(step_frames[start])) / float(fps))


def _distance_level_duration_events(
    sequence: list[dict],
    distance_level_duration_stats: dict | None,
    *,
    fps: float,
    include_levels: Any = None,
    duration_end_mode: str = "next_change",
    upper_key: str,
    upper_override_levels: Any = None,
    upper_override_key: str = "max_sec",
    penalty_sec_per_unit: float,
    short_enabled: bool = False,
    lower_key: str = "p25_sec",
    min_sec: float = 0.0,
    short_penalty_sec_per_unit: float = 0.25,
    short_exclude_cut_pairs: bool = True,
    short_exclude_levels: Any = None,
) -> list[dict]:
    classes = _distance_level_duration_classes(distance_level_duration_stats)
    if not classes or len(sequence) < 2 or float(fps) <= 0:
        return []
    frames = [int(row.get("source_frame", row.get("frame", 0))) for row in sequence]
    levels = [distance_fov_level_label(row) for row in sequence]
    include_level_set = _normalize_combo_set(include_levels)
    events: list[dict] = []
    start = 0
    for index in range(1, len(sequence) + 1):
        changed = index == len(sequence) or levels[index] != levels[start]
        if not changed:
            continue
        end_index = index - 1
        level = levels[start]
        if include_level_set and level not in include_level_set:
            start = index
            continue
        end_frame = _run_duration_end_frame(frames, end_index, str(duration_end_mode))
        duration_sec = _run_duration_sec(frames, start, end_index, float(fps), str(duration_end_mode))
        stats = classes.get(level)
        if stats:
            upper_sec, resolved_upper_key = _distance_level_upper_bound(
                level,
                stats,
                upper_key,
                override_levels=upper_override_levels,
                override_key=upper_override_key,
            )
        else:
            upper_sec, resolved_upper_key = None, str(upper_key)
        over_units = _bodypart_duration_penalty_units(
            duration_sec,
            upper_sec,
            penalty_sec_per_unit=float(penalty_sec_per_unit),
        )
        if over_units > 0:
            events.append(
                {
                    "type": "overlong",
                    "start_index": start,
                    "end_index": end_index,
                    "frames": [frames[start], frames[end_index]],
                    "end_frame_exclusive": end_frame,
                    "level": level,
                    "duration_sec": duration_sec,
                    "upper_sec": upper_sec,
                    "upper_key": str(resolved_upper_key),
                    "penalty_units": int(over_units),
                }
            )
        short_exclude_level_set = _normalize_combo_set(short_exclude_levels)
        if bool(short_enabled) and index < len(sequence) and level not in short_exclude_level_set:
            lower_sec = float(min_sec) if float(min_sec) > 0 else (_duration_bound(stats, lower_key) if stats else None)
            short_units = _bodypart_duration_short_penalty_units(
                duration_sec,
                lower_sec,
                penalty_sec_per_unit=float(short_penalty_sec_per_unit),
            )
            if (
                short_units > 0
                and (not bool(short_exclude_cut_pairs) or not _bodypart_run_touches_cut_pair(sequence, start, end_index))
            ):
                events.append(
                    {
                        "type": "too_short",
                        "start_index": start,
                        "end_index": end_index,
                        "frames": [frames[start], frames[end_index]],
                        "end_frame_exclusive": end_frame,
                        "level": level,
                        "duration_sec": duration_sec,
                        "lower_sec": lower_sec,
                        "lower_key": str(lower_key),
                        "penalty_units": int(short_units),
                    }
                )
        start = index
    return events


def optimize_shot_sequence_cp_sat(
    candidates: list[dict],
    gt_sequence: list[dict],
    fields: list[str] | None = None,
    weights: dict | None = None,
    max_time_sec: float = 30.0,
    local_cost_scale: int = 1000,
    distance_bounce_weight: int | None = None,
    distance_bounce_min_rank_delta: int = 80,
    distance_bounce_max_frame_gap: int = 10,
    cut_identity_weight: int | None = None,
    cut_contrast_weight: int | None = None,
    cut_contrast_min_score: int = 3,
    cut_contrast_medium_rank_delta: int = 100,
    cut_contrast_strong_rank_delta: int = 220,
    post_cut_distance_jump_weight: int | None = None,
    post_cut_distance_jump_min_rank_delta: int = 250,
    post_cut_distance_jump_max_frame_gap: int = 10,
    cut_context_speed_weight: int | None = None,
    cut_context_speed_min_speed: float = 250.0,
    cut_context_speed_max_frame_gap: int = 10,
    cut_context_speed_fps: float = 5.0,
    repeated_distance_bounce_weight: int | None = None,
    repeated_distance_bounce_min_rank_delta: int = 180,
    repeated_distance_bounce_max_frame_gap: int = 15,
    composition_bounce_weight: int | None = None,
    composition_bounce_min_rank_delta: int = 80,
    composition_bounce_max_frame_gap: int = 10,
    vertical_bounce_weight: int | None = None,
    vertical_bounce_min_rank_delta: int = 35,
    vertical_bounce_max_frame_gap: int = 10,
    top_pitch_bounce_weight: int | None = None,
    top_pitch_bounce_max_frame_gap: int = 30,
    top_pitch_bounce_neighbor_pitches: Any = None,
    top_pitch_bounce_return_pitches: Any = None,
    turn_yaw_jitter_weight: int | None = None,
    turn_yaw_jitter_heading_frames: list[int] | None = None,
    turn_yaw_jitter_heading_values: list[float] | None = None,
    turn_yaw_jitter_fps: float = 30.0,
    turn_yaw_jitter_min_frame_gap: int = 1,
    turn_yaw_jitter_max_frame_gap: int = 90,
    turn_yaw_jitter_min_heading_delta: float = 25.0,
    turn_yaw_jitter_min_heading_speed: float = 9999.0,
    turn_yaw_jitter_min_relative_yaw_delta: float = 45.0,
    turn_yaw_jitter_min_world_yaw_delta: float = 60.0,
    turn_yaw_jitter_penalty_degrees_per_unit: float = 15.0,
    turn_yaw_jitter_require_same_direction: bool = True,
    turn_yaw_jitter_exclude_cut_pairs: bool = True,
    reference_sequence: list[dict] | None = None,
    reference_change_weight: int = 0,
    reference_change_fields: list[str] | None = None,
    bodypart_duration_weight: int | None = None,
    bodypart_duration_stats: dict | None = None,
    bodypart_duration_fps: float = 30.0,
    bodypart_duration_proxy_mode: str = "calibrated",
    bodypart_duration_upper_key: str = "max_sec",
    bodypart_duration_penalty_sec_per_unit: float = 0.5,
    bodypart_duration_upper_blend_combos: Any = None,
    bodypart_duration_upper_blend_alpha: float = 0.0,
    bodypart_duration_upper_blend_from_key: str = "p95_sec",
    bodypart_duration_upper_blend_to_key: str = "max_sec",
    bodypart_duration_short_weight: int | None = None,
    bodypart_duration_lower_key: str = "p25_sec",
    bodypart_duration_short_penalty_sec_per_unit: float = 0.25,
    bodypart_duration_short_exclude_cut_pairs: bool = True,
    distance_level_duration_weight: int | None = None,
    distance_level_duration_stats: dict | None = None,
    distance_level_duration_fps: float = 30.0,
    distance_level_duration_include_levels: Any = None,
    distance_level_duration_end_mode: str = "next_change",
    distance_level_duration_upper_key: str = "p95_sec",
    distance_level_duration_upper_override_levels: Any = None,
    distance_level_duration_upper_override_key: str = "max_sec",
    distance_level_duration_penalty_sec_per_unit: float = 0.5,
    distance_level_duration_short_weight: int | None = None,
    distance_level_duration_lower_key: str = "p25_sec",
    distance_level_duration_min_sec: float = 0.7,
    distance_level_duration_short_penalty_sec_per_unit: float = 0.25,
    distance_level_duration_short_exclude_cut_pairs: bool = True,
    distance_level_duration_short_exclude_levels: Any = None,
    distance_level_direction_weight: int | None = None,
    distance_level_direction_close_levels: Any = None,
    distance_level_direction_medium_level: str = "medium",
    distance_level_direction_far_levels: Any = None,
    distance_level_direction_hard: bool = False,
    distance_level_protect_source_levels: Any = None,
    distance_level_protect_min_level: str = "medium_far",
    distance_change_allowed_frames: Any = None,
    distance_level_hold_after_change: bool = False,
    distance_level_hold_after_change_mode: str = "level",
    distance_level_hold_after_change_min_frame_gap: int = 1,
    distance_level_hold_after_change_max_frame_gap: int = 60,
    distance_level_hold_after_change_min_keyframes: int = 2,
    distance_level_hold_after_change_min_sec: float = 0.0,
    distance_level_hold_after_change_min_sec_key: str = "",
    distance_level_hold_after_change_max_sec_key: str = "",
    distance_level_hold_after_change_disallow_cut_pairs: bool = False,
    rare_duration_weight: int | None = None,
    rare_duration_stats: dict | None = None,
    rare_duration_fps: float = 5.0,
    rare_duration_lower_key: str = "p25_sec",
    rare_duration_upper_key: str = "p75_sec",
    rare_duration_missing_penalty_units: int = 0,
    verbose: bool = False,
) -> dict:
    """Select one candidate per frame using CP-SAT sequence-stat matching.

    The solver minimizes weighted deviations from rescaled GT marginal,
    transition, and change-count statistics, with optional local candidate cost.
    """
    optimized_fields = [field for field in list(fields or FIELDS) if field != "transition"]
    resolved_weights = _resolve_weights(weights, optimized_fields)
    candidates_by_t = group_candidates_by_frame(candidates)
    frames = [group[0]["frame"] for group in candidates_by_t]
    T = len(candidates_by_t)
    empty_breakdown = {
        "marginal": 0,
        "transition": 0,
        "change": 0,
        "local": 0,
        "distance_bounce": 0,
        "cut_identity": 0,
        "cut_contrast": 0,
        "post_cut_distance_jump": 0,
        "cut_context_speed": 0,
        "repeated_distance_bounce": 0,
        "composition_bounce": 0,
        "vertical_bounce": 0,
        "top_pitch_bounce": 0,
        "turn_yaw_jitter": 0,
        "reference_change": 0,
        "bodypart_duration": 0,
        "bodypart_duration_short": 0,
        "distance_level_duration": 0,
        "distance_level_duration_short": 0,
        "distance_level_direction": 0,
        "rare_duration": 0,
    }

    gt_raw = compute_gt_stats(gt_sequence, optimized_fields)
    candidate_value_sets = _frame_values(candidates_by_t, optimized_fields)
    gt_value_sets = _gt_values(gt_raw, optimized_fields)
    marginal_values = _merge_value_sets(candidate_value_sets, gt_value_sets)

    candidate_transition_sets = _candidate_transition_values(candidates_by_t, optimized_fields)
    gt_transition_sets = _gt_transition_values(gt_raw, optimized_fields)
    transition_values = _merge_value_sets(candidate_transition_sets, gt_transition_sets)

    target_stats = _rescale_gt_stats(
        gt_stats=gt_raw,
        gt_length=len(gt_sequence),
        candidate_length=T,
        fields=optimized_fields,
        marginal_values=marginal_values,
        transition_values=transition_values,
    )

    if T == 0:
        return {
            "status": "INFEASIBLE",
            "objective": None,
            "selected_sequence": [],
            "selected_indices": [],
            "frames": [],
            "stats": {
                "selected": {"marginal": {}, "transition": {}, "change": {}},
                "target": target_stats,
                "gt_raw": gt_raw,
            },
            "objective_breakdown": empty_breakdown,
        }

    model = cp_model.CpModel()

    x: list[list[cp_model.IntVar]] = []
    for t, candidates_at_t in enumerate(candidates_by_t):
        if not candidates_at_t:
            return {
                "status": "INFEASIBLE",
                "objective": None,
                "selected_sequence": [],
                "selected_indices": [],
                "frames": frames,
                "stats": {
                    "selected": {"marginal": {}, "transition": {}, "change": {}},
                    "target": target_stats,
                    "gt_raw": gt_raw,
                },
                "objective_breakdown": empty_breakdown,
            }
        row_vars = [
            model.NewBoolVar(f"x_t{t}_k{k}")
            for k, _candidate in enumerate(candidates_at_t)
        ]
        model.Add(sum(row_vars) == 1)
        x.append(row_vars)

    # This creates O(sum_t K_{t-1} K_t) adjacent transition variables.
    z: dict[tuple[int, int, int], cp_model.IntVar] = {}
    for t in range(1, T):
        for prev_k in range(len(candidates_by_t[t - 1])):
            for current_k in range(len(candidates_by_t[t])):
                var = model.NewBoolVar(f"z_t{t}_j{prev_k}_k{current_k}")
                model.Add(var <= x[t - 1][prev_k])
                model.Add(var <= x[t][current_k])
                model.Add(var >= x[t - 1][prev_k] + x[t][current_k] - 1)
                z[(t, prev_k, current_k)] = var

    objective_terms: list[Any] = []
    marginal_terms: list[tuple[int, cp_model.IntVar]] = []
    transition_terms: list[tuple[int, cp_model.IntVar]] = []
    change_terms: list[tuple[int, cp_model.IntVar]] = []
    local_terms: list[tuple[int, cp_model.IntVar]] = []
    distance_bounce_terms: list[tuple[int, cp_model.IntVar]] = []
    cut_identity_terms: list[tuple[int, cp_model.IntVar]] = []
    cut_contrast_terms: list[tuple[int, cp_model.IntVar]] = []
    post_cut_distance_jump_terms: list[tuple[int, cp_model.IntVar]] = []
    cut_context_speed_terms: list[tuple[int, cp_model.IntVar]] = []
    repeated_distance_bounce_terms: list[tuple[int, cp_model.IntVar]] = []
    composition_bounce_terms: list[tuple[int, cp_model.IntVar]] = []
    vertical_bounce_terms: list[tuple[int, cp_model.IntVar]] = []
    top_pitch_bounce_terms: list[tuple[int, cp_model.IntVar]] = []
    turn_yaw_jitter_terms: list[tuple[int, cp_model.IntVar]] = []
    reference_change_terms: list[tuple[int, cp_model.IntVar]] = []
    bodypart_duration_terms: list[tuple[int, cp_model.IntVar]] = []
    bodypart_duration_short_terms: list[tuple[int, cp_model.IntVar]] = []
    distance_level_duration_terms: list[tuple[int, cp_model.IntVar]] = []
    distance_level_duration_short_terms: list[tuple[int, cp_model.IntVar]] = []
    distance_level_direction_terms: list[tuple[int, cp_model.IntVar]] = []
    rare_duration_terms: list[tuple[int, cp_model.IntVar]] = []

    for field in optimized_fields:
        weight = int(resolved_weights["marginal"].get(field, 0))
        if weight == 0:
            continue
        for value in marginal_values.get(field, set()):
            selected_terms = [
                x[t][k]
                for t, candidates_at_t in enumerate(candidates_by_t)
                for k, candidate in enumerate(candidates_at_t)
                if _field_value(candidate, field) == value
            ]
            selected_count = sum(selected_terms) if selected_terms else 0
            target_count = int(target_stats["marginal"].get(field, {}).get(value, 0))
            dev = _add_abs_deviation(
                model,
                name=f"dev_marginal_{field}_{repr(value)}",
                expression=selected_count,
                target=target_count,
                upper_bound=T,
            )
            objective_terms.append(weight * dev)
            marginal_terms.append((weight, dev))

    transition_horizon = max(1, T - 1)
    for field in optimized_fields:
        weight = int(resolved_weights["transition"].get(field, 0))
        if weight == 0:
            continue
        for source, target in transition_values.get(field, set()):
            selected_terms = [
                z[(t, prev_k, current_k)]
                for t in range(1, T)
                for prev_k, prev_candidate in enumerate(candidates_by_t[t - 1])
                for current_k, current_candidate in enumerate(candidates_by_t[t])
                if _field_value(prev_candidate, field) == source
                and _field_value(current_candidate, field) == target
            ]
            selected_count = sum(selected_terms) if selected_terms else 0
            target_count = int(target_stats["transition"].get(field, {}).get(source, {}).get(target, 0))
            dev = _add_abs_deviation(
                model,
                name=f"dev_transition_{field}_{repr(source)}_to_{repr(target)}",
                expression=selected_count,
                target=target_count,
                upper_bound=transition_horizon,
            )
            objective_terms.append(weight * dev)
            transition_terms.append((weight, dev))

    for field in optimized_fields:
        weight = int(resolved_weights["change"].get(field, 0))
        if weight == 0:
            continue
        selected_terms = [
            z[(t, prev_k, current_k)]
            for t in range(1, T)
            for prev_k, prev_candidate in enumerate(candidates_by_t[t - 1])
            for current_k, current_candidate in enumerate(candidates_by_t[t])
            if _field_value(prev_candidate, field) != _field_value(current_candidate, field)
        ]
        selected_count = sum(selected_terms) if selected_terms else 0
        target_count = int(target_stats["change"].get(field, 0))
        dev = _add_abs_deviation(
            model,
            name=f"dev_change_{field}",
            expression=selected_count,
            target=target_count,
            upper_bound=transition_horizon,
        )
        objective_terms.append(weight * dev)
        change_terms.append((weight, dev))

    local_weight = int(resolved_weights.get("local", 0))
    if local_weight != 0:
        for t, candidates_at_t in enumerate(candidates_by_t):
            for k, candidate in enumerate(candidates_at_t):
                if "local_cost" not in candidate:
                    continue
                int_local_cost = int(round(float(candidate.get("local_cost", 0.0)) * int(local_cost_scale)))
                coefficient = local_weight * int_local_cost
                if coefficient == 0:
                    continue
                objective_terms.append(coefficient * x[t][k])
                local_terms.append((coefficient, x[t][k]))

    if int(reference_change_weight) > 0 and reference_sequence:
        reference_fields = list(reference_change_fields or optimized_fields)
        reference_by_t = list(reference_sequence)
        for t, candidates_at_t in enumerate(candidates_by_t):
            if t >= len(reference_by_t):
                continue
            reference = reference_by_t[t]
            for k, candidate in enumerate(candidates_at_t):
                diff_count = sum(
                    1
                    for field in reference_fields
                    if str(_field_value(candidate, field)) != str(_field_value(reference, field))
                )
                if diff_count <= 0:
                    continue
                coefficient = int(reference_change_weight) * int(diff_count)
                objective_terms.append(coefficient * x[t][k])
                reference_change_terms.append((coefficient, x[t][k]))

    resolved_distance_level_direction_weight = (
        int(distance_level_direction_weight)
        if distance_level_direction_weight is not None
        else int(resolved_weights.get("distance_level_direction", 0))
    )
    direction_close_levels = _normalize_combo_set(
        distance_level_direction_close_levels or "tight_close,close,medium_close"
    )
    direction_far_levels = _normalize_combo_set(distance_level_direction_far_levels or "medium_far,far,wide")
    direction_medium_index = distance_fov_level_index(str(distance_level_direction_medium_level or "medium"))
    if resolved_distance_level_direction_weight > 0 and reference_sequence and (
        direction_close_levels or direction_far_levels
    ):
        reference_by_t = list(reference_sequence)
        for t, candidates_at_t in enumerate(candidates_by_t):
            if t >= len(reference_by_t):
                continue
            reference_level = distance_fov_level_label(reference_by_t[t])
            if reference_level not in direction_close_levels and reference_level not in direction_far_levels:
                continue
            reference_distance = str(_field_value(reference_by_t[t], "distance"))
            reference_fov = str(_field_value(reference_by_t[t], "fov"))
            for k, candidate in enumerate(candidates_at_t):
                if (
                    str(_field_value(candidate, "distance")) == reference_distance
                    and str(_field_value(candidate, "fov")) == reference_fov
                ):
                    continue
                candidate_level_index = distance_fov_level_index(distance_fov_level_label(candidate))
                penalty_distance = 0
                if reference_level in direction_close_levels and candidate_level_index < direction_medium_index:
                    penalty_distance = direction_medium_index - candidate_level_index
                elif reference_level in direction_far_levels and candidate_level_index > direction_medium_index:
                    penalty_distance = candidate_level_index - direction_medium_index
                if penalty_distance <= 0:
                    continue
                if bool(distance_level_direction_hard):
                    model.Add(x[t][k] == 0)
                    continue
                coefficient = resolved_distance_level_direction_weight * int(penalty_distance)
                objective_terms.append(coefficient * x[t][k])
                distance_level_direction_terms.append((coefficient, x[t][k]))

    distance_level_protect_levels = _normalize_combo_set(distance_level_protect_source_levels)
    if reference_sequence and distance_level_protect_levels:
        reference_by_t = list(reference_sequence)
        protect_min_index = distance_fov_level_index(str(distance_level_protect_min_level or "medium_far"))
        for t, candidates_at_t in enumerate(candidates_by_t):
            if t >= len(reference_by_t):
                continue
            reference = reference_by_t[t]
            reference_level = distance_fov_level_label(reference)
            if reference_level not in distance_level_protect_levels:
                continue
            for k, candidate in enumerate(candidates_at_t):
                if not _distance_or_fov_changed(candidate, reference):
                    continue
                if distance_fov_level_index(distance_fov_level_label(candidate)) < protect_min_index:
                    model.Add(x[t][k] == 0)

    if reference_sequence and distance_change_allowed_frames is not None:
        reference_by_t = list(reference_sequence)
        allowed_distance_change_frames = _normalize_int_set(distance_change_allowed_frames)
        for t, candidates_at_t in enumerate(candidates_by_t):
            if t >= len(reference_by_t):
                continue
            source_frame = _step_source_frame(candidates_at_t)
            if source_frame in allowed_distance_change_frames:
                continue
            allow_cut_pair_tail = (
                bool(distance_level_hold_after_change)
                and bool(distance_level_hold_after_change_disallow_cut_pairs)
                and t + 1 < T
                and _is_cut_pair_transition_groups(candidates_by_t[t], candidates_by_t[t + 1])
            )
            if allow_cut_pair_tail:
                continue
            reference = reference_by_t[t]
            for k, candidate in enumerate(candidates_at_t):
                if _distance_or_fov_changed(candidate, reference):
                    model.Add(x[t][k] == 0)

    if bool(distance_level_hold_after_change) and reference_sequence and T >= 2:
        reference_by_t = list(reference_sequence)
        hold_min_gap = max(1, int(distance_level_hold_after_change_min_frame_gap))
        hold_max_gap = int(distance_level_hold_after_change_max_frame_gap)
        hold_min_keyframes = max(2, int(distance_level_hold_after_change_min_keyframes))
        hold_min_sec = max(0.0, float(distance_level_hold_after_change_min_sec))
        hold_min_sec_key = str(distance_level_hold_after_change_min_sec_key or "").strip()
        hold_max_sec_key = str(distance_level_hold_after_change_max_sec_key or "").strip()
        hold_mode = str(distance_level_hold_after_change_mode or "level")
        step_frames_for_hold = [_step_source_frame(candidates_at_t) for candidates_at_t in candidates_by_t]
        allowed_hold_frames = (
            _normalize_int_set(distance_change_allowed_frames)
            if distance_change_allowed_frames is not None
            else None
        )
        distance_hold_classes = _distance_hold_duration_classes(distance_level_duration_stats, hold_mode)

        def _can_hold_transition(left_index: int, right_index: int) -> bool:
            if left_index < 0 or right_index >= T:
                return False
            if bool(distance_level_hold_after_change_disallow_cut_pairs) and _is_cut_pair_transition_groups(
                candidates_by_t[left_index],
                candidates_by_t[right_index],
            ):
                return False
            gap_value = int(step_frames_for_hold[right_index]) - int(step_frames_for_hold[left_index])
            if gap_value < hold_min_gap:
                return False
            if hold_max_gap > 0 and gap_value > hold_max_gap:
                return False
            return True

        def _valid_hold_window(start_index: int, end_index: int) -> bool:
            if start_index < 0 or end_index >= T:
                return False
            for transition_index in range(start_index, end_index):
                if not _can_hold_transition(transition_index, transition_index + 1):
                    return False
            return True

        def _hold_span_bounds(center_index: int) -> tuple[int, int]:
            left = int(center_index)
            while _can_hold_transition(left - 1, left):
                left -= 1
            right = int(center_index)
            while _can_hold_transition(right, right + 1):
                right += 1
            return left, right

        def _hold_window_min_sec_for_key(hold_key: Any) -> float:
            if hold_min_sec_key:
                stats = distance_hold_classes.get(str(hold_key))
                if stats:
                    resolved = _duration_bound(stats, hold_min_sec_key)
                    if resolved is not None:
                        return max(0.0, float(resolved))
            return hold_min_sec

        def _hold_window_max_sec_for_key(hold_key: Any) -> float | None:
            if not hold_max_sec_key:
                return None
            stats = distance_hold_classes.get(str(hold_key))
            if not stats:
                return None
            return _duration_bound(stats, hold_max_sec_key)

        def _candidate_hold_windows(center_index: int, hold_key: Any) -> list[tuple[int, int]]:
            window_min_sec = _hold_window_min_sec_for_key(hold_key)
            window_max_sec = _hold_window_max_sec_for_key(hold_key)
            if window_min_sec <= 0.0:
                windows: list[tuple[int, int]] = []
                for window_start in range(center_index - hold_min_keyframes + 1, center_index + 1):
                    window_end = window_start + hold_min_keyframes - 1
                    if _valid_hold_window(window_start, window_end):
                        if window_max_sec is not None:
                            duration_sec = _run_duration_sec(
                                step_frames_for_hold,
                                window_start,
                                window_end,
                                float(distance_level_duration_fps),
                                str(distance_level_duration_end_mode),
                            )
                            if duration_sec > float(window_max_sec) + 1e-9:
                                continue
                        windows.append((window_start, window_end))
                return windows

            left_bound, right_bound = _hold_span_bounds(center_index)
            windows = []
            for window_start in range(left_bound, center_index + 1):
                min_window_end = max(center_index, window_start + hold_min_keyframes - 1)
                for window_end in range(min_window_end, right_bound + 1):
                    duration_sec = _run_duration_sec(
                        step_frames_for_hold,
                        window_start,
                        window_end,
                        float(distance_level_duration_fps),
                        str(distance_level_duration_end_mode),
                    )
                    if duration_sec + 1e-9 < window_min_sec:
                        continue
                    if window_max_sec is not None and duration_sec > float(window_max_sec) + 1e-9:
                        continue
                    windows.append((window_start, window_end))
                    break
            return windows

        for t in range(0, T):
            if t >= len(reference_by_t):
                continue
            source_frame = _step_source_frame(candidates_by_t[t])
            is_cut_pair_first = (
                bool(distance_level_hold_after_change_disallow_cut_pairs)
                and t + 1 < T
                and _is_cut_pair_transition_groups(candidates_by_t[t], candidates_by_t[t + 1])
            )
            if allowed_hold_frames is not None and source_frame not in allowed_hold_frames and not is_cut_pair_first:
                continue
            reference = reference_by_t[t]
            for k, candidate in enumerate(candidates_by_t[t]):
                if not _distance_or_fov_changed(candidate, reference):
                    continue
                hold_key = _distance_hold_key(candidate, hold_mode)
                hold_run_terms: list[Any] = []
                for window_start, window_end in _candidate_hold_windows(t, hold_key):
                    same_key_terms: list[Any] = []
                    valid_window = True
                    for window_t in range(window_start, window_end + 1):
                        if window_t == t:
                            same_key_terms.append(x[t][k])
                            continue
                        matching_terms = [
                            x[window_t][candidate_index]
                            for candidate_index, window_candidate in enumerate(candidates_by_t[window_t])
                            if _distance_hold_key(window_candidate, hold_mode) == hold_key
                        ]
                        if not matching_terms:
                            valid_window = False
                            break
                        same_key_terms.append(sum(matching_terms))
                    if not valid_window:
                        continue
                    hold_run = model.NewBoolVar(
                        f"distance_hold_run_t{t}_k{k}_s{window_start}_n{hold_min_keyframes}"
                    )
                    for term in same_key_terms:
                        model.Add(hold_run <= term)
                    model.Add(hold_run >= sum(same_key_terms) - (len(same_key_terms) - 1))
                    hold_run_terms.append(hold_run)
                if hold_run_terms:
                    model.Add(x[t][k] <= sum(hold_run_terms))
                else:
                    model.Add(x[t][k] == 0)

    resolved_distance_bounce_weight = (
        int(distance_bounce_weight)
        if distance_bounce_weight is not None
        else int(resolved_weights.get("distance_bounce", 0))
    )
    if resolved_distance_bounce_weight > 0 and T >= 3:
        step_frames = [_step_source_frame(candidates_at_t) for candidates_at_t in candidates_by_t]
        rank_table = [
            [_distance_rank(candidate) for candidate in candidates_at_t]
            for candidates_at_t in candidates_by_t
        ]
        pair_up: dict[int, cp_model.IntVar] = {}
        pair_down: dict[int, cp_model.IntVar] = {}
        for t in range(1, T):
            gap = step_frames[t] - step_frames[t - 1]
            if gap <= 0 or gap > int(distance_bounce_max_frame_gap):
                continue
            up_terms = []
            down_terms = []
            for prev_k, prev_rank in enumerate(rank_table[t - 1]):
                for current_k, current_rank in enumerate(rank_table[t]):
                    delta = current_rank - prev_rank
                    if delta >= int(distance_bounce_min_rank_delta):
                        up_terms.append(z[(t, prev_k, current_k)])
                    elif delta <= -int(distance_bounce_min_rank_delta):
                        down_terms.append(z[(t, prev_k, current_k)])
            up_var = model.NewBoolVar(f"distance_rank_up_t{t}")
            down_var = model.NewBoolVar(f"distance_rank_down_t{t}")
            model.Add(up_var == (sum(up_terms) if up_terms else 0))
            model.Add(down_var == (sum(down_terms) if down_terms else 0))
            pair_up[t] = up_var
            pair_down[t] = down_var

        for center in range(1, T - 1):
            if center not in pair_up or center + 1 not in pair_up:
                continue
            close_far_close = model.NewBoolVar(f"distance_bounce_close_far_close_t{center}")
            model.Add(close_far_close <= pair_up[center])
            model.Add(close_far_close <= pair_down[center + 1])
            model.Add(close_far_close >= pair_up[center] + pair_down[center + 1] - 1)

            far_close_far = model.NewBoolVar(f"distance_bounce_far_close_far_t{center}")
            model.Add(far_close_far <= pair_down[center])
            model.Add(far_close_far <= pair_up[center + 1])
            model.Add(far_close_far >= pair_down[center] + pair_up[center + 1] - 1)

            objective_terms.append(resolved_distance_bounce_weight * close_far_close)
            objective_terms.append(resolved_distance_bounce_weight * far_close_far)
            distance_bounce_terms.append((resolved_distance_bounce_weight, close_far_close))
            distance_bounce_terms.append((resolved_distance_bounce_weight, far_close_far))

    resolved_repeated_distance_bounce_weight = (
        int(repeated_distance_bounce_weight)
        if repeated_distance_bounce_weight is not None
        else int(resolved_weights.get("repeated_distance_bounce", 0))
    )
    if resolved_repeated_distance_bounce_weight > 0 and T >= 5:
        step_frames = [_step_source_frame(candidates_at_t) for candidates_at_t in candidates_by_t]
        rank_table = [
            [_distance_rank(candidate) for candidate in candidates_at_t]
            for candidates_at_t in candidates_by_t
        ]
        motif_pair_up: dict[int, cp_model.IntVar] = {}
        motif_pair_down: dict[int, cp_model.IntVar] = {}
        for t in range(1, T):
            gap = step_frames[t] - step_frames[t - 1]
            # Same-frame transitions are included here on purpose: repeated
            # cut-slot rank flips are exactly the motif we want to discourage.
            if gap < 0 or gap > int(repeated_distance_bounce_max_frame_gap):
                continue
            up_terms = []
            down_terms = []
            for prev_k, prev_rank in enumerate(rank_table[t - 1]):
                for current_k, current_rank in enumerate(rank_table[t]):
                    delta = current_rank - prev_rank
                    if delta >= int(repeated_distance_bounce_min_rank_delta):
                        up_terms.append(z[(t, prev_k, current_k)])
                    elif delta <= -int(repeated_distance_bounce_min_rank_delta):
                        down_terms.append(z[(t, prev_k, current_k)])
            up_var = model.NewBoolVar(f"repeated_distance_rank_up_t{t}")
            down_var = model.NewBoolVar(f"repeated_distance_rank_down_t{t}")
            model.Add(up_var == (sum(up_terms) if up_terms else 0))
            model.Add(down_var == (sum(down_terms) if down_terms else 0))
            motif_pair_up[t] = up_var
            motif_pair_down[t] = down_var

        for start_t in range(1, T - 3):
            if any(t not in motif_pair_up for t in range(start_t, start_t + 4)):
                continue

            close_far_close_far_close = model.NewBoolVar(
                f"repeated_bounce_close_far_close_far_close_t{start_t}"
            )
            sequence_terms = [
                motif_pair_up[start_t],
                motif_pair_down[start_t + 1],
                motif_pair_up[start_t + 2],
                motif_pair_down[start_t + 3],
            ]
            for term in sequence_terms:
                model.Add(close_far_close_far_close <= term)
            model.Add(close_far_close_far_close >= sum(sequence_terms) - 3)

            far_close_far_close_far = model.NewBoolVar(
                f"repeated_bounce_far_close_far_close_far_t{start_t}"
            )
            sequence_terms = [
                motif_pair_down[start_t],
                motif_pair_up[start_t + 1],
                motif_pair_down[start_t + 2],
                motif_pair_up[start_t + 3],
            ]
            for term in sequence_terms:
                model.Add(far_close_far_close_far <= term)
            model.Add(far_close_far_close_far >= sum(sequence_terms) - 3)

            objective_terms.append(resolved_repeated_distance_bounce_weight * close_far_close_far_close)
            objective_terms.append(resolved_repeated_distance_bounce_weight * far_close_far_close_far)
            repeated_distance_bounce_terms.append(
                (resolved_repeated_distance_bounce_weight, close_far_close_far_close)
            )
            repeated_distance_bounce_terms.append(
                (resolved_repeated_distance_bounce_weight, far_close_far_close_far)
            )

    resolved_composition_bounce_weight = (
        int(composition_bounce_weight)
        if composition_bounce_weight is not None
        else int(resolved_weights.get("composition_bounce", 0))
    )
    if resolved_composition_bounce_weight > 0 and T >= 3:
        step_frames = [_step_source_frame(candidates_at_t) for candidates_at_t in candidates_by_t]
        rank_table = [
            [_composition_rank(candidate) for candidate in candidates_at_t]
            for candidates_at_t in candidates_by_t
        ]
        pair_right: dict[int, cp_model.IntVar] = {}
        pair_left: dict[int, cp_model.IntVar] = {}
        for t in range(1, T):
            gap = step_frames[t] - step_frames[t - 1]
            if gap <= 0 or gap > int(composition_bounce_max_frame_gap):
                continue
            right_terms = []
            left_terms = []
            for prev_k, prev_rank in enumerate(rank_table[t - 1]):
                for current_k, current_rank in enumerate(rank_table[t]):
                    delta = current_rank - prev_rank
                    if delta >= int(composition_bounce_min_rank_delta):
                        right_terms.append(z[(t, prev_k, current_k)])
                    elif delta <= -int(composition_bounce_min_rank_delta):
                        left_terms.append(z[(t, prev_k, current_k)])
            right_var = model.NewBoolVar(f"composition_rank_right_t{t}")
            left_var = model.NewBoolVar(f"composition_rank_left_t{t}")
            model.Add(right_var == (sum(right_terms) if right_terms else 0))
            model.Add(left_var == (sum(left_terms) if left_terms else 0))
            pair_right[t] = right_var
            pair_left[t] = left_var

        for center in range(1, T - 1):
            if center not in pair_right or center + 1 not in pair_right:
                continue
            right_left = model.NewBoolVar(f"composition_bounce_right_left_t{center}")
            model.Add(right_left <= pair_right[center])
            model.Add(right_left <= pair_left[center + 1])
            model.Add(right_left >= pair_right[center] + pair_left[center + 1] - 1)

            left_right = model.NewBoolVar(f"composition_bounce_left_right_t{center}")
            model.Add(left_right <= pair_left[center])
            model.Add(left_right <= pair_right[center + 1])
            model.Add(left_right >= pair_left[center] + pair_right[center + 1] - 1)

            objective_terms.append(resolved_composition_bounce_weight * right_left)
            objective_terms.append(resolved_composition_bounce_weight * left_right)
            composition_bounce_terms.append((resolved_composition_bounce_weight, right_left))
            composition_bounce_terms.append((resolved_composition_bounce_weight, left_right))

    resolved_vertical_bounce_weight = (
        int(vertical_bounce_weight)
        if vertical_bounce_weight is not None
        else int(resolved_weights.get("vertical_bounce", 0))
    )
    if resolved_vertical_bounce_weight > 0 and T >= 3:
        step_frames = [_step_source_frame(candidates_at_t) for candidates_at_t in candidates_by_t]
        rank_table = [
            [_vertical_rank(candidate) for candidate in candidates_at_t]
            for candidates_at_t in candidates_by_t
        ]
        stage_by_t = [
            str(candidates_at_t[0].get("stage", "")).strip().lower() if candidates_at_t else ""
            for candidates_at_t in candidates_by_t
        ]
        pair_up: dict[int, cp_model.IntVar] = {}
        pair_down: dict[int, cp_model.IntVar] = {}
        for t in range(1, T):
            gap = step_frames[t] - step_frames[t - 1]
            if gap <= 0 or gap > int(vertical_bounce_max_frame_gap):
                continue
            up_terms = []
            down_terms = []
            for prev_k, prev_rank in enumerate(rank_table[t - 1]):
                for current_k, current_rank in enumerate(rank_table[t]):
                    prev_candidate = candidates_by_t[t - 1][prev_k]
                    current_candidate = candidates_by_t[t][current_k]
                    if str(prev_candidate.get("distance", "")) != str(current_candidate.get("distance", "")):
                        continue
                    delta = current_rank - prev_rank
                    if delta >= int(vertical_bounce_min_rank_delta):
                        up_terms.append(z[(t, prev_k, current_k)])
                    elif delta <= -int(vertical_bounce_min_rank_delta):
                        down_terms.append(z[(t, prev_k, current_k)])
            up_var = model.NewBoolVar(f"vertical_rank_up_t{t}")
            down_var = model.NewBoolVar(f"vertical_rank_down_t{t}")
            model.Add(up_var == (sum(up_terms) if up_terms else 0))
            model.Add(down_var == (sum(down_terms) if down_terms else 0))
            pair_up[t] = up_var
            pair_down[t] = down_var

        for center in range(1, T - 1):
            if center not in pair_up or center + 1 not in pair_up:
                continue
            if stage_by_t[center] != "cut":
                continue
            if step_frames[center] - step_frames[center - 1] != 1:
                continue
            up_down = model.NewBoolVar(f"vertical_bounce_up_down_t{center}")
            model.Add(up_down <= pair_up[center])
            model.Add(up_down <= pair_down[center + 1])
            model.Add(up_down >= pair_up[center] + pair_down[center + 1] - 1)

            objective_terms.append(resolved_vertical_bounce_weight * up_down)
            vertical_bounce_terms.append((resolved_vertical_bounce_weight, up_down))

    resolved_top_pitch_bounce_weight = (
        int(top_pitch_bounce_weight)
        if top_pitch_bounce_weight is not None
        else int(resolved_weights.get("top_pitch_bounce", 0))
    )
    if resolved_top_pitch_bounce_weight > 0 and T >= 3:
        step_frames = [_step_source_frame(candidates_at_t) for candidates_at_t in candidates_by_t]
        entry_pitches = _normalize_combo_set(top_pitch_bounce_neighbor_pitches or "low,level")
        exit_pitches = _normalize_combo_set(top_pitch_bounce_return_pitches or "low,level,high")
        to_top: dict[int, cp_model.IntVar] = {}
        from_top: dict[int, cp_model.IntVar] = {}
        for t in range(1, T):
            gap = step_frames[t] - step_frames[t - 1]
            if gap < 0 or gap > int(top_pitch_bounce_max_frame_gap):
                continue
            to_top_terms = []
            from_top_terms = []
            for prev_k, prev_candidate in enumerate(candidates_by_t[t - 1]):
                prev_pitch = str(prev_candidate.get("pitch", "level")).strip()
                for current_k, current_candidate in enumerate(candidates_by_t[t]):
                    current_pitch = str(current_candidate.get("pitch", "level")).strip()
                    transition = z[(t, prev_k, current_k)]
                    if prev_pitch in entry_pitches and current_pitch == "top":
                        to_top_terms.append(transition)
                    if prev_pitch == "top" and current_pitch in exit_pitches:
                        from_top_terms.append(transition)
            to_top_var = model.NewBoolVar(f"top_pitch_to_top_t{t}")
            from_top_var = model.NewBoolVar(f"top_pitch_from_top_t{t}")
            model.Add(to_top_var == (sum(to_top_terms) if to_top_terms else 0))
            model.Add(from_top_var == (sum(from_top_terms) if from_top_terms else 0))
            to_top[t] = to_top_var
            from_top[t] = from_top_var

        for center in range(1, T - 1):
            if center not in to_top or center + 1 not in from_top:
                continue
            top_spike = model.NewBoolVar(f"top_pitch_bounce_t{center}")
            model.Add(top_spike <= to_top[center])
            model.Add(top_spike <= from_top[center + 1])
            model.Add(top_spike >= to_top[center] + from_top[center + 1] - 1)
            objective_terms.append(resolved_top_pitch_bounce_weight * top_spike)
            top_pitch_bounce_terms.append((resolved_top_pitch_bounce_weight, top_spike))

    resolved_turn_yaw_jitter_weight = (
        int(turn_yaw_jitter_weight)
        if turn_yaw_jitter_weight is not None
        else int(resolved_weights.get("turn_yaw_jitter", 0))
    )
    if (
        resolved_turn_yaw_jitter_weight > 0
        and T >= 2
        and turn_yaw_jitter_heading_frames
        and turn_yaw_jitter_heading_values
        and float(turn_yaw_jitter_fps) > 0
    ):
        step_frames_for_yaw = [_step_source_frame(candidates_at_t) for candidates_at_t in candidates_by_t]
        heading_by_t = [
            _interpolate_track(
                list(turn_yaw_jitter_heading_frames),
                list(turn_yaw_jitter_heading_values),
                frame,
            )
            for frame in step_frames_for_yaw
        ]
        for t in range(1, T):
            gap = step_frames_for_yaw[t] - step_frames_for_yaw[t - 1]
            if gap < int(turn_yaw_jitter_min_frame_gap) or gap > int(turn_yaw_jitter_max_frame_gap):
                continue
            if bool(turn_yaw_jitter_exclude_cut_pairs) and _is_cut_pair_transition_groups(
                candidates_by_t[t - 1],
                candidates_by_t[t],
            ):
                continue
            for prev_k, prev_candidate in enumerate(candidates_by_t[t - 1]):
                for current_k, current_candidate in enumerate(candidates_by_t[t]):
                    info = _turn_yaw_jitter_pair_info(
                        prev_candidate,
                        current_candidate,
                        left_heading=heading_by_t[t - 1],
                        right_heading=heading_by_t[t],
                        fps=float(turn_yaw_jitter_fps),
                        min_heading_delta=float(turn_yaw_jitter_min_heading_delta),
                        min_heading_speed=float(turn_yaw_jitter_min_heading_speed),
                        min_relative_yaw_delta=float(turn_yaw_jitter_min_relative_yaw_delta),
                        min_world_yaw_delta=float(turn_yaw_jitter_min_world_yaw_delta),
                        require_same_direction=bool(turn_yaw_jitter_require_same_direction),
                    )
                    if info is None:
                        continue
                    excess_deg = abs(float(info["world_yaw_delta_deg"])) - float(turn_yaw_jitter_min_world_yaw_delta)
                    degrees_per_unit = max(1e-6, float(turn_yaw_jitter_penalty_degrees_per_unit))
                    penalty_units = max(1, int(math.ceil(excess_deg / degrees_per_unit)) + 1)
                    coefficient = resolved_turn_yaw_jitter_weight * int(penalty_units)
                    var = z[(t, prev_k, current_k)]
                    objective_terms.append(coefficient * var)
                    turn_yaw_jitter_terms.append((coefficient, var))

    step_frames = [_step_source_frame(candidates_at_t) for candidates_at_t in candidates_by_t]

    resolved_cut_identity_weight = (
        int(cut_identity_weight)
        if cut_identity_weight is not None
        else int(resolved_weights.get("cut_identity", 0))
    )
    if resolved_cut_identity_weight > 0 and T >= 2:
        for t in range(1, T):
            if not _is_cut_pair_transition_groups(candidates_by_t[t - 1], candidates_by_t[t]):
                continue
            for prev_k, prev_candidate in enumerate(candidates_by_t[t - 1]):
                for current_k, current_candidate in enumerate(candidates_by_t[t]):
                    if not _same_shot(prev_candidate, current_candidate, optimized_fields):
                        continue
                    var = z[(t, prev_k, current_k)]
                    objective_terms.append(resolved_cut_identity_weight * var)
                    cut_identity_terms.append((resolved_cut_identity_weight, var))

    resolved_cut_contrast_weight = (
        int(cut_contrast_weight)
        if cut_contrast_weight is not None
        else int(resolved_weights.get("cut_contrast", 0))
    )
    if resolved_cut_contrast_weight > 0 and T >= 2:
        for t in range(1, T):
            if not _is_cut_pair_transition_groups(candidates_by_t[t - 1], candidates_by_t[t]):
                continue
            for prev_k, prev_candidate in enumerate(candidates_by_t[t - 1]):
                for current_k, current_candidate in enumerate(candidates_by_t[t]):
                    score_info = _cut_contrast_score(
                        prev_candidate,
                        current_candidate,
                        fields=optimized_fields,
                        medium_rank_delta=int(cut_contrast_medium_rank_delta),
                        strong_rank_delta=int(cut_contrast_strong_rank_delta),
                    )
                    if score_info["has_major_change"] and int(score_info["score"]) >= int(cut_contrast_min_score):
                        continue
                    penalty_units = max(1, int(cut_contrast_min_score) - int(score_info["score"]))
                    coefficient = resolved_cut_contrast_weight * penalty_units
                    var = z[(t, prev_k, current_k)]
                    objective_terms.append(coefficient * var)
                    cut_contrast_terms.append((coefficient, var))

    resolved_post_cut_distance_jump_weight = (
        int(post_cut_distance_jump_weight)
        if post_cut_distance_jump_weight is not None
        else int(resolved_weights.get("post_cut_distance_jump", 0))
    )
    if resolved_post_cut_distance_jump_weight > 0 and T >= 3:
        rank_table = [
            [_distance_rank(candidate) for candidate in candidates_at_t]
            for candidates_at_t in candidates_by_t
        ]
        for t in range(2, T):
            if not _is_cut_pair_transition_groups(candidates_by_t[t - 2], candidates_by_t[t - 1]):
                continue
            gap = step_frames[t] - step_frames[t - 1]
            if gap <= 0 or gap > int(post_cut_distance_jump_max_frame_gap):
                continue
            for prev_k, prev_rank in enumerate(rank_table[t - 1]):
                for current_k, current_rank in enumerate(rank_table[t]):
                    if abs(current_rank - prev_rank) < int(post_cut_distance_jump_min_rank_delta):
                        continue
                    var = z[(t, prev_k, current_k)]
                    objective_terms.append(resolved_post_cut_distance_jump_weight * var)
                    post_cut_distance_jump_terms.append((resolved_post_cut_distance_jump_weight, var))

    resolved_cut_context_speed_weight = (
        int(cut_context_speed_weight)
        if cut_context_speed_weight is not None
        else int(resolved_weights.get("cut_context_speed", 0))
    )
    if resolved_cut_context_speed_weight > 0 and T >= 3 and float(cut_context_speed_fps) > 0:
        rank_table = [
            [_distance_rank(candidate) for candidate in candidates_at_t]
            for candidates_at_t in candidates_by_t
        ]
        for t in range(1, T):
            gap = step_frames[t] - step_frames[t - 1]
            if gap <= 0 or gap > int(cut_context_speed_max_frame_gap):
                continue

            # Penalize only moves adjacent to a cut pair. The cut pair itself is
            # intentionally exempt.
            before_cut = t + 1 < T and _is_cut_pair_transition_groups(candidates_by_t[t], candidates_by_t[t + 1])
            after_cut = t >= 2 and _is_cut_pair_transition_groups(candidates_by_t[t - 2], candidates_by_t[t - 1])
            if not before_cut and not after_cut:
                continue

            for prev_k, prev_rank in enumerate(rank_table[t - 1]):
                for current_k, current_rank in enumerate(rank_table[t]):
                    delta = current_rank - prev_rank
                    speed = abs(delta) * float(cut_context_speed_fps) / float(gap)
                    if speed < float(cut_context_speed_min_speed):
                        continue
                    penalty_units = max(1, int(round(speed - float(cut_context_speed_min_speed))) + 1)
                    coefficient = resolved_cut_context_speed_weight * penalty_units
                    var = z[(t, prev_k, current_k)]
                    objective_terms.append(coefficient * var)
                    cut_context_speed_terms.append((coefficient, var))

    resolved_bodypart_duration_weight = (
        int(bodypart_duration_weight)
        if bodypart_duration_weight is not None
        else int(resolved_weights.get("bodypart_duration", 0))
    )
    resolved_bodypart_duration_short_weight = (
        int(bodypart_duration_short_weight)
        if bodypart_duration_short_weight is not None
        else int(resolved_weights.get("bodypart_duration_short", 0))
    )
    bodypart_classes = _bodypart_duration_classes(bodypart_duration_stats)
    if (
        (resolved_bodypart_duration_weight > 0 or resolved_bodypart_duration_short_weight > 0)
        and bodypart_classes
        and T >= 3
        and float(bodypart_duration_fps) > 0
    ):
        combo_table = [
            [_bodypart_combo(candidate, str(bodypart_duration_proxy_mode)) for candidate in candidates_at_t]
            for candidates_at_t in candidates_by_t
        ]
        combo_values = sorted(set(bodypart_classes) & {combo for row in combo_table for combo in row})
        for combo in combo_values:
            stats = bodypart_classes.get(combo, {})
            if resolved_bodypart_duration_weight > 0:
                upper_sec = _bodypart_duration_upper_bound(
                    combo,
                    stats,
                    str(bodypart_duration_upper_key),
                    upper_blend_combos=bodypart_duration_upper_blend_combos,
                    upper_blend_alpha=float(bodypart_duration_upper_blend_alpha),
                    upper_blend_from_key=str(bodypart_duration_upper_blend_from_key),
                    upper_blend_to_key=str(bodypart_duration_upper_blend_to_key),
                )
                if upper_sec is not None:
                    for start in range(0, T - 1):
                        for end in range(start + 1, T - 1):
                            duration_sec = float(step_frames[end + 1] - step_frames[start]) / float(bodypart_duration_fps)
                            penalty_units = _bodypart_duration_penalty_units(
                                duration_sec,
                                upper_sec,
                                penalty_sec_per_unit=float(bodypart_duration_penalty_sec_per_unit),
                            )
                            if penalty_units <= 0:
                                continue
                            selected_combo_terms = [
                                sum(
                                    x[t][k]
                                    for k, candidate_combo in enumerate(combo_table[t])
                                    if candidate_combo == combo
                                )
                                for t in range(start, end + 1)
                            ]
                            if not selected_combo_terms:
                                continue
                            all_same = model.NewBoolVar(f"bodypart_duration_{combo}_s{start}_e{end}")
                            for term in selected_combo_terms:
                                model.Add(all_same <= term)
                            model.Add(all_same >= sum(selected_combo_terms) - (end - start))
                            coefficient = resolved_bodypart_duration_weight * int(penalty_units)
                            objective_terms.append(coefficient * all_same)
                            bodypart_duration_terms.append((coefficient, all_same))

            if resolved_bodypart_duration_short_weight > 0:
                lower_sec = _bodypart_duration_bound(stats, str(bodypart_duration_lower_key), fallback_key=None)
                if lower_sec is None:
                    continue
                for start in range(0, T - 1):
                    for end in range(start, T - 1):
                        if bool(bodypart_duration_short_exclude_cut_pairs) and _window_touches_cut_pair_groups(
                            candidates_by_t,
                            start,
                            end,
                        ):
                            continue
                        duration_sec = float(step_frames[end + 1] - step_frames[start]) / float(bodypart_duration_fps)
                        penalty_units = _bodypart_duration_short_penalty_units(
                            duration_sec,
                            lower_sec,
                            penalty_sec_per_unit=float(bodypart_duration_short_penalty_sec_per_unit),
                        )
                        if penalty_units <= 0:
                            continue
                        selected_combo_terms = [
                            sum(
                                x[t][k]
                                for k, candidate_combo in enumerate(combo_table[t])
                                if candidate_combo == combo
                            )
                            for t in range(start, end + 1)
                        ]
                        all_same = model.NewBoolVar(f"bodypart_duration_short_same_{combo}_s{start}_e{end}")
                        for term in selected_combo_terms:
                            model.Add(all_same <= term)
                        model.Add(all_same >= sum(selected_combo_terms) - (end - start))

                        exact_terms: list[Any] = [all_same]
                        if start > 0:
                            left_same = sum(
                                x[start - 1][k]
                                for k, candidate_combo in enumerate(combo_table[start - 1])
                                if candidate_combo == combo
                            )
                            left_not_same = model.NewBoolVar(f"bodypart_duration_short_left_{combo}_s{start}_e{end}")
                            model.Add(left_not_same + left_same == 1)
                            exact_terms.append(left_not_same)
                        if end + 1 < T:
                            right_same = sum(
                                x[end + 1][k]
                                for k, candidate_combo in enumerate(combo_table[end + 1])
                                if candidate_combo == combo
                            )
                            right_not_same = model.NewBoolVar(f"bodypart_duration_short_right_{combo}_s{start}_e{end}")
                            model.Add(right_not_same + right_same == 1)
                            exact_terms.append(right_not_same)

                        exact_run = model.NewBoolVar(f"bodypart_duration_short_{combo}_s{start}_e{end}")
                        for term in exact_terms:
                            model.Add(exact_run <= term)
                        model.Add(exact_run >= sum(exact_terms) - (len(exact_terms) - 1))
                        coefficient = resolved_bodypart_duration_short_weight * int(penalty_units)
                        objective_terms.append(coefficient * exact_run)
                        bodypart_duration_short_terms.append((coefficient, exact_run))

    resolved_distance_level_duration_weight = (
        int(distance_level_duration_weight)
        if distance_level_duration_weight is not None
        else int(resolved_weights.get("distance_level_duration", 0))
    )
    resolved_distance_level_duration_short_weight = (
        int(distance_level_duration_short_weight)
        if distance_level_duration_short_weight is not None
        else int(resolved_weights.get("distance_level_duration_short", 0))
    )
    distance_level_classes = _distance_level_duration_classes(distance_level_duration_stats)
    if (
        (resolved_distance_level_duration_weight > 0 or resolved_distance_level_duration_short_weight > 0)
        and distance_level_classes
        and T >= 3
        and float(distance_level_duration_fps) > 0
    ):
        level_table = [
            [distance_fov_level_label(candidate) for candidate in candidates_at_t]
            for candidates_at_t in candidates_by_t
        ]
        level_values = sorted(set(distance_level_classes) & {level for row in level_table for level in row})
        distance_level_include_set = _normalize_combo_set(distance_level_duration_include_levels)
        if distance_level_include_set:
            level_values = [level for level in level_values if level in distance_level_include_set]
        distance_level_short_exclude_set = _normalize_combo_set(distance_level_duration_short_exclude_levels)
        for level in level_values:
            stats = distance_level_classes.get(level, {})
            if resolved_distance_level_duration_weight > 0:
                upper_sec, _resolved_upper_key = _distance_level_upper_bound(
                    level,
                    stats,
                    str(distance_level_duration_upper_key),
                    override_levels=distance_level_duration_upper_override_levels,
                    override_key=str(distance_level_duration_upper_override_key),
                )
                if upper_sec is not None:
                    for start in range(0, T - 1):
                        for end in range(start, T - 1):
                            duration_sec = _run_duration_sec(
                                step_frames,
                                start,
                                end,
                                float(distance_level_duration_fps),
                                str(distance_level_duration_end_mode),
                            )
                            penalty_units = _bodypart_duration_penalty_units(
                                duration_sec,
                                upper_sec,
                                penalty_sec_per_unit=float(distance_level_duration_penalty_sec_per_unit),
                            )
                            if penalty_units <= 0:
                                continue
                            selected_level_terms = [
                                sum(
                                    x[t][k]
                                    for k, candidate_level in enumerate(level_table[t])
                                    if candidate_level == level
                                )
                                for t in range(start, end + 1)
                            ]
                            all_same = model.NewBoolVar(f"distance_level_duration_{level}_s{start}_e{end}")
                            for term in selected_level_terms:
                                model.Add(all_same <= term)
                            model.Add(all_same >= sum(selected_level_terms) - (end - start))
                            coefficient = resolved_distance_level_duration_weight * int(penalty_units)
                            objective_terms.append(coefficient * all_same)
                            distance_level_duration_terms.append((coefficient, all_same))

            if resolved_distance_level_duration_short_weight > 0 and level not in distance_level_short_exclude_set:
                lower_sec = (
                    float(distance_level_duration_min_sec)
                    if float(distance_level_duration_min_sec) > 0
                    else _duration_bound(stats, str(distance_level_duration_lower_key))
                )
                if lower_sec is None:
                    continue
                for start in range(0, T - 1):
                    for end in range(start, T - 1):
                        if bool(distance_level_duration_short_exclude_cut_pairs) and _window_touches_cut_pair_groups(
                            candidates_by_t,
                            start,
                            end,
                        ):
                            continue
                        duration_sec = _run_duration_sec(
                            step_frames,
                            start,
                            end,
                            float(distance_level_duration_fps),
                            str(distance_level_duration_end_mode),
                        )
                        penalty_units = _bodypart_duration_short_penalty_units(
                            duration_sec,
                            lower_sec,
                            penalty_sec_per_unit=float(distance_level_duration_short_penalty_sec_per_unit),
                        )
                        if penalty_units <= 0:
                            continue
                        selected_level_terms = [
                            sum(
                                x[t][k]
                                for k, candidate_level in enumerate(level_table[t])
                                if candidate_level == level
                            )
                            for t in range(start, end + 1)
                        ]
                        all_same = model.NewBoolVar(f"distance_level_duration_short_same_{level}_s{start}_e{end}")
                        for term in selected_level_terms:
                            model.Add(all_same <= term)
                        model.Add(all_same >= sum(selected_level_terms) - (end - start))

                        exact_terms: list[Any] = [all_same]
                        if start > 0:
                            left_same = sum(
                                x[start - 1][k]
                                for k, candidate_level in enumerate(level_table[start - 1])
                                if candidate_level == level
                            )
                            left_not_same = model.NewBoolVar(f"distance_level_duration_short_left_{level}_s{start}_e{end}")
                            model.Add(left_not_same + left_same == 1)
                            exact_terms.append(left_not_same)
                        if end + 1 < T:
                            right_same = sum(
                                x[end + 1][k]
                                for k, candidate_level in enumerate(level_table[end + 1])
                                if candidate_level == level
                            )
                            right_not_same = model.NewBoolVar(
                                f"distance_level_duration_short_right_{level}_s{start}_e{end}"
                            )
                            model.Add(right_not_same + right_same == 1)
                            exact_terms.append(right_not_same)

                        exact_run = model.NewBoolVar(f"distance_level_duration_short_{level}_s{start}_e{end}")
                        for term in exact_terms:
                            model.Add(exact_run <= term)
                        model.Add(exact_run >= sum(exact_terms) - (len(exact_terms) - 1))
                        coefficient = resolved_distance_level_duration_short_weight * int(penalty_units)
                        objective_terms.append(coefficient * exact_run)
                        distance_level_duration_short_terms.append((coefficient, exact_run))

    resolved_rare_duration_weight = (
        int(rare_duration_weight)
        if rare_duration_weight is not None
        else int(resolved_weights.get("rare_duration", 0))
    )
    if resolved_rare_duration_weight > 0 and rare_duration_stats and T >= 2 and float(rare_duration_fps) > 0:
        for t, candidates_at_t in enumerate(candidates_by_t):
            duration_sec = _next_distinct_duration_sec(
                step_frames,
                t,
                float(rare_duration_fps),
            )
            for k, candidate in enumerate(candidates_at_t):
                penalty_units = 0
                for _label, stats in _candidate_rare_duration_matches(candidate, rare_duration_stats):
                    penalty_units += _rare_duration_penalty_units(
                        duration_sec,
                        stats,
                        fps=float(rare_duration_fps),
                        lower_key=str(rare_duration_lower_key),
                        upper_key=str(rare_duration_upper_key),
                        missing_penalty_units=int(rare_duration_missing_penalty_units),
                    )
                if penalty_units <= 0:
                    continue
                coefficient = resolved_rare_duration_weight * int(penalty_units)
                objective_terms.append(coefficient * x[t][k])
                rare_duration_terms.append((coefficient, x[t][k]))

    model.Minimize(sum(objective_terms) if objective_terms else 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(max_time_sec)
    status_code = solver.Solve(model)
    status = _status_name(status_code)
    has_solution = status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    objective_breakdown = dict(empty_breakdown)
    if has_solution:
        objective_breakdown = {
            "marginal": sum(weight * solver.Value(var) for weight, var in marginal_terms),
            "transition": sum(weight * solver.Value(var) for weight, var in transition_terms),
            "change": sum(weight * solver.Value(var) for weight, var in change_terms),
            "local": sum(coefficient * solver.Value(var) for coefficient, var in local_terms),
            "distance_bounce": sum(weight * solver.Value(var) for weight, var in distance_bounce_terms),
            "cut_identity": sum(weight * solver.Value(var) for weight, var in cut_identity_terms),
            "cut_contrast": sum(coefficient * solver.Value(var) for coefficient, var in cut_contrast_terms),
            "post_cut_distance_jump": sum(weight * solver.Value(var) for weight, var in post_cut_distance_jump_terms),
            "cut_context_speed": sum(coefficient * solver.Value(var) for coefficient, var in cut_context_speed_terms),
            "repeated_distance_bounce": sum(
                weight * solver.Value(var) for weight, var in repeated_distance_bounce_terms
            ),
            "composition_bounce": sum(weight * solver.Value(var) for weight, var in composition_bounce_terms),
            "vertical_bounce": sum(weight * solver.Value(var) for weight, var in vertical_bounce_terms),
            "top_pitch_bounce": sum(weight * solver.Value(var) for weight, var in top_pitch_bounce_terms),
            "turn_yaw_jitter": sum(coefficient * solver.Value(var) for coefficient, var in turn_yaw_jitter_terms),
            "reference_change": sum(coefficient * solver.Value(var) for coefficient, var in reference_change_terms),
            "bodypart_duration": sum(coefficient * solver.Value(var) for coefficient, var in bodypart_duration_terms),
            "bodypart_duration_short": sum(
                coefficient * solver.Value(var) for coefficient, var in bodypart_duration_short_terms
            ),
            "distance_level_duration": sum(
                coefficient * solver.Value(var) for coefficient, var in distance_level_duration_terms
            ),
            "distance_level_duration_short": sum(
                coefficient * solver.Value(var) for coefficient, var in distance_level_duration_short_terms
            ),
            "distance_level_direction": sum(
                coefficient * solver.Value(var) for coefficient, var in distance_level_direction_terms
            ),
            "rare_duration": sum(coefficient * solver.Value(var) for coefficient, var in rare_duration_terms),
        }
        selected_indices: list[int] = []
        selected_sequence: list[dict] = []
        for t, candidates_at_t in enumerate(candidates_by_t):
            selected_k = next(k for k, var in enumerate(x[t]) if solver.Value(var) == 1)
            selected_indices.append(selected_k)
            selected_sequence.append(dict(candidates_at_t[selected_k]))
        selected_stats = {
            "marginal": compute_marginal_counts(selected_sequence, optimized_fields),
            "transition": compute_transition_counts(selected_sequence, optimized_fields),
            "change": compute_change_counts(selected_sequence, optimized_fields),
            "distance_bounce": {
                "count": len(
                    _distance_bounce_events(
                        selected_sequence,
                        min_rank_delta=int(distance_bounce_min_rank_delta),
                        max_frame_gap=int(distance_bounce_max_frame_gap),
                    )
                ),
                "events": _distance_bounce_events(
                    selected_sequence,
                    min_rank_delta=int(distance_bounce_min_rank_delta),
                    max_frame_gap=int(distance_bounce_max_frame_gap),
                ),
            },
            "cut_identity": {
                "count": len(_cut_identity_events(selected_sequence, optimized_fields)),
                "events": _cut_identity_events(selected_sequence, optimized_fields),
            },
            "cut_contrast": {
                "count": len(
                    _cut_contrast_events(
                        selected_sequence,
                        fields=optimized_fields,
                        min_score=int(cut_contrast_min_score),
                        medium_rank_delta=int(cut_contrast_medium_rank_delta),
                        strong_rank_delta=int(cut_contrast_strong_rank_delta),
                    )
                ),
                "events": _cut_contrast_events(
                    selected_sequence,
                    fields=optimized_fields,
                    min_score=int(cut_contrast_min_score),
                    medium_rank_delta=int(cut_contrast_medium_rank_delta),
                    strong_rank_delta=int(cut_contrast_strong_rank_delta),
                ),
            },
            "post_cut_distance_jump": {
                "count": len(
                    _post_cut_distance_jump_events(
                        selected_sequence,
                        min_rank_delta=int(post_cut_distance_jump_min_rank_delta),
                        max_frame_gap=int(post_cut_distance_jump_max_frame_gap),
                    )
                ),
                "events": _post_cut_distance_jump_events(
                    selected_sequence,
                    min_rank_delta=int(post_cut_distance_jump_min_rank_delta),
                    max_frame_gap=int(post_cut_distance_jump_max_frame_gap),
                ),
            },
            "cut_context_speed": {
                "count": len(
                    _cut_context_speed_events(
                        selected_sequence,
                        min_speed=float(cut_context_speed_min_speed),
                        max_frame_gap=int(cut_context_speed_max_frame_gap),
                        fps=float(cut_context_speed_fps),
                    )
                ),
                "events": _cut_context_speed_events(
                    selected_sequence,
                    min_speed=float(cut_context_speed_min_speed),
                    max_frame_gap=int(cut_context_speed_max_frame_gap),
                    fps=float(cut_context_speed_fps),
                ),
            },
            "repeated_distance_bounce": {
                "count": len(
                    _repeated_distance_bounce_events(
                        selected_sequence,
                        min_rank_delta=int(repeated_distance_bounce_min_rank_delta),
                        max_frame_gap=int(repeated_distance_bounce_max_frame_gap),
                    )
                ),
                "events": _repeated_distance_bounce_events(
                    selected_sequence,
                    min_rank_delta=int(repeated_distance_bounce_min_rank_delta),
                    max_frame_gap=int(repeated_distance_bounce_max_frame_gap),
                ),
            },
            "composition_bounce": {
                "count": len(
                    _composition_bounce_events(
                        selected_sequence,
                        min_rank_delta=int(composition_bounce_min_rank_delta),
                        max_frame_gap=int(composition_bounce_max_frame_gap),
                    )
                ),
                "events": _composition_bounce_events(
                    selected_sequence,
                    min_rank_delta=int(composition_bounce_min_rank_delta),
                    max_frame_gap=int(composition_bounce_max_frame_gap),
                ),
            },
            "vertical_bounce": {
                "count": len(
                    _vertical_bounce_events(
                        selected_sequence,
                        min_rank_delta=int(vertical_bounce_min_rank_delta),
                        max_frame_gap=int(vertical_bounce_max_frame_gap),
                    )
                ),
                "events": _vertical_bounce_events(
                    selected_sequence,
                    min_rank_delta=int(vertical_bounce_min_rank_delta),
                    max_frame_gap=int(vertical_bounce_max_frame_gap),
                ),
            },
            "top_pitch_bounce": {
                "count": len(
                    _top_pitch_bounce_events(
                        selected_sequence,
                        neighbor_pitches=top_pitch_bounce_neighbor_pitches or "low,level",
                        return_pitches=top_pitch_bounce_return_pitches or "low,level,high",
                        max_frame_gap=int(top_pitch_bounce_max_frame_gap),
                    )
                ),
                "events": _top_pitch_bounce_events(
                    selected_sequence,
                    neighbor_pitches=top_pitch_bounce_neighbor_pitches or "low,level",
                    return_pitches=top_pitch_bounce_return_pitches or "low,level,high",
                    max_frame_gap=int(top_pitch_bounce_max_frame_gap),
                ),
            },
            "turn_yaw_jitter": {
                "count": len(
                    _turn_yaw_jitter_events(
                        selected_sequence,
                        heading_frames=turn_yaw_jitter_heading_frames,
                        heading_values=turn_yaw_jitter_heading_values,
                        fps=float(turn_yaw_jitter_fps),
                        min_frame_gap=int(turn_yaw_jitter_min_frame_gap),
                        max_frame_gap=int(turn_yaw_jitter_max_frame_gap),
                        min_heading_delta=float(turn_yaw_jitter_min_heading_delta),
                        min_heading_speed=float(turn_yaw_jitter_min_heading_speed),
                        min_relative_yaw_delta=float(turn_yaw_jitter_min_relative_yaw_delta),
                        min_world_yaw_delta=float(turn_yaw_jitter_min_world_yaw_delta),
                        require_same_direction=bool(turn_yaw_jitter_require_same_direction),
                        exclude_cut_pairs=bool(turn_yaw_jitter_exclude_cut_pairs),
                    )
                ),
                "events": _turn_yaw_jitter_events(
                    selected_sequence,
                    heading_frames=turn_yaw_jitter_heading_frames,
                    heading_values=turn_yaw_jitter_heading_values,
                    fps=float(turn_yaw_jitter_fps),
                    min_frame_gap=int(turn_yaw_jitter_min_frame_gap),
                    max_frame_gap=int(turn_yaw_jitter_max_frame_gap),
                    min_heading_delta=float(turn_yaw_jitter_min_heading_delta),
                    min_heading_speed=float(turn_yaw_jitter_min_heading_speed),
                    min_relative_yaw_delta=float(turn_yaw_jitter_min_relative_yaw_delta),
                    min_world_yaw_delta=float(turn_yaw_jitter_min_world_yaw_delta),
                    require_same_direction=bool(turn_yaw_jitter_require_same_direction),
                    exclude_cut_pairs=bool(turn_yaw_jitter_exclude_cut_pairs),
                ),
            },
            "bodypart_duration": {
                "count": len(
                    _bodypart_duration_events(
                        selected_sequence,
                        bodypart_duration_stats,
                        fps=float(bodypart_duration_fps),
                        proxy_mode=str(bodypart_duration_proxy_mode),
                        upper_key=str(bodypart_duration_upper_key),
                        penalty_sec_per_unit=float(bodypart_duration_penalty_sec_per_unit),
                        upper_blend_combos=bodypart_duration_upper_blend_combos,
                        upper_blend_alpha=float(bodypart_duration_upper_blend_alpha),
                        upper_blend_from_key=str(bodypart_duration_upper_blend_from_key),
                        upper_blend_to_key=str(bodypart_duration_upper_blend_to_key),
                        short_enabled=resolved_bodypart_duration_short_weight > 0,
                        lower_key=str(bodypart_duration_lower_key),
                        short_penalty_sec_per_unit=float(bodypart_duration_short_penalty_sec_per_unit),
                        short_exclude_cut_pairs=bool(bodypart_duration_short_exclude_cut_pairs),
                    )
                ),
                "events": _bodypart_duration_events(
                    selected_sequence,
                    bodypart_duration_stats,
                    fps=float(bodypart_duration_fps),
                    proxy_mode=str(bodypart_duration_proxy_mode),
                    upper_key=str(bodypart_duration_upper_key),
                    penalty_sec_per_unit=float(bodypart_duration_penalty_sec_per_unit),
                    upper_blend_combos=bodypart_duration_upper_blend_combos,
                    upper_blend_alpha=float(bodypart_duration_upper_blend_alpha),
                    upper_blend_from_key=str(bodypart_duration_upper_blend_from_key),
                    upper_blend_to_key=str(bodypart_duration_upper_blend_to_key),
                    short_enabled=resolved_bodypart_duration_short_weight > 0,
                    lower_key=str(bodypart_duration_lower_key),
                    short_penalty_sec_per_unit=float(bodypart_duration_short_penalty_sec_per_unit),
                    short_exclude_cut_pairs=bool(bodypart_duration_short_exclude_cut_pairs),
                ),
            },
            "distance_level_duration": {
                "count": len(
                    _distance_level_duration_events(
                        selected_sequence,
                        distance_level_duration_stats,
                        fps=float(distance_level_duration_fps),
                        include_levels=distance_level_duration_include_levels,
                        duration_end_mode=str(distance_level_duration_end_mode),
                        upper_key=str(distance_level_duration_upper_key),
                        upper_override_levels=distance_level_duration_upper_override_levels,
                        upper_override_key=str(distance_level_duration_upper_override_key),
                        penalty_sec_per_unit=float(distance_level_duration_penalty_sec_per_unit),
                        short_enabled=resolved_distance_level_duration_short_weight > 0,
                        lower_key=str(distance_level_duration_lower_key),
                        min_sec=float(distance_level_duration_min_sec),
                        short_penalty_sec_per_unit=float(distance_level_duration_short_penalty_sec_per_unit),
                        short_exclude_cut_pairs=bool(distance_level_duration_short_exclude_cut_pairs),
                        short_exclude_levels=distance_level_duration_short_exclude_levels,
                    )
                ),
                "events": _distance_level_duration_events(
                    selected_sequence,
                    distance_level_duration_stats,
                    fps=float(distance_level_duration_fps),
                    include_levels=distance_level_duration_include_levels,
                    duration_end_mode=str(distance_level_duration_end_mode),
                    upper_key=str(distance_level_duration_upper_key),
                    upper_override_levels=distance_level_duration_upper_override_levels,
                    upper_override_key=str(distance_level_duration_upper_override_key),
                    penalty_sec_per_unit=float(distance_level_duration_penalty_sec_per_unit),
                    short_enabled=resolved_distance_level_duration_short_weight > 0,
                    lower_key=str(distance_level_duration_lower_key),
                    min_sec=float(distance_level_duration_min_sec),
                    short_penalty_sec_per_unit=float(distance_level_duration_short_penalty_sec_per_unit),
                    short_exclude_cut_pairs=bool(distance_level_duration_short_exclude_cut_pairs),
                    short_exclude_levels=distance_level_duration_short_exclude_levels,
                ),
            },
            "rare_duration": {
                "count": len(
                    _rare_duration_events(
                        selected_sequence,
                        rare_duration_stats,
                        fps=float(rare_duration_fps),
                        lower_key=str(rare_duration_lower_key),
                        upper_key=str(rare_duration_upper_key),
                        missing_penalty_units=int(rare_duration_missing_penalty_units),
                    )
                ),
                "events": _rare_duration_events(
                    selected_sequence,
                    rare_duration_stats,
                    fps=float(rare_duration_fps),
                    lower_key=str(rare_duration_lower_key),
                    upper_key=str(rare_duration_upper_key),
                    missing_penalty_units=int(rare_duration_missing_penalty_units),
                ),
            },
        }
        objective_value: float | int | None = solver.ObjectiveValue()
    else:
        selected_indices = []
        selected_sequence = []
        selected_stats = {"marginal": {}, "transition": {}, "change": {}}
        objective_value = None

    result = {
        "status": status,
        "objective": objective_value,
        "selected_sequence": selected_sequence,
        "selected_indices": selected_indices,
        "frames": frames,
        "stats": {
            "selected": selected_stats,
            "target": target_stats,
            "gt_raw": gt_raw,
        },
        "objective_breakdown": objective_breakdown,
    }
    if verbose:
        print(f"status: {status}")
        print(f"objective: {objective_value}")
        print(f"objective_breakdown: {objective_breakdown}")
    return result


if __name__ == "__main__":
    toy_candidates = [
        {
            "frame": 0,
            "target_body": "full_body",
            "shot_size": "far",
            "angle": "front",
            "composition": "center",
            "roll": "none",
            "transition": "linear",
        },
        {
            "frame": 0,
            "target_body": "knee_up",
            "shot_size": "medium",
            "angle": "left_45",
            "composition": "center",
            "roll": "none",
            "transition": "linear",
            "local_cost": 0.1,
        },
        {
            "frame": 1,
            "target_body": "knee_up",
            "shot_size": "medium",
            "angle": "left_45",
            "composition": "center",
            "roll": "none",
            "transition": "linear",
        },
        {
            "frame": 1,
            "target_body": "face",
            "shot_size": "close",
            "angle": "right_45",
            "composition": "left_third",
            "roll": "none",
            "transition": "linear",
        },
        {
            "frame": 2,
            "target_body": "knee_up",
            "shot_size": "medium",
            "angle": "left_45",
            "composition": "center",
            "roll": "none",
            "transition": "linear",
        },
        {
            "frame": 2,
            "target_body": "bust_up",
            "shot_size": "medium",
            "angle": "front",
            "composition": "right_third",
            "roll": "slight_left",
            "transition": "linear",
        },
        {
            "frame": 3,
            "target_body": "knee_up",
            "shot_size": "medium",
            "angle": "left_45",
            "composition": "center",
            "roll": "none",
            "transition": "linear",
        },
        {
            "frame": 3,
            "target_body": "full_body",
            "shot_size": "far",
            "angle": "front",
            "composition": "center",
            "roll": "none",
            "transition": "linear",
        },
        {
            "frame": 4,
            "target_body": "bust_up",
            "shot_size": "medium",
            "angle": "left_45",
            "composition": "center",
            "roll": "none",
            "transition": "linear",
        },
        {
            "frame": 4,
            "target_body": "face",
            "shot_size": "close",
            "angle": "front",
            "composition": "center",
            "roll": "slight_right",
            "transition": "linear",
        },
    ]
    toy_gt = [
        {
            "target_body": "knee_up",
            "shot_size": "medium",
            "angle": "left_45",
            "composition": "center",
            "roll": "none",
            "transition": "linear",
        },
        {
            "target_body": "knee_up",
            "shot_size": "medium",
            "angle": "left_45",
            "composition": "center",
            "roll": "none",
            "transition": "linear",
        },
        {
            "target_body": "knee_up",
            "shot_size": "medium",
            "angle": "left_45",
            "composition": "center",
            "roll": "none",
            "transition": "linear",
        },
        {
            "target_body": "bust_up",
            "shot_size": "medium",
            "angle": "left_45",
            "composition": "center",
            "roll": "none",
            "transition": "linear",
        },
        {
            "target_body": "bust_up",
            "shot_size": "medium",
            "angle": "front",
            "composition": "center",
            "roll": "none",
            "transition": "linear",
        },
    ]

    toy_result = optimize_shot_sequence_cp_sat(
        toy_candidates,
        toy_gt,
        verbose=True,
        max_time_sec=10.0,
    )
    assert toy_result["status"] in {"OPTIMAL", "FEASIBLE"}
    assert len(toy_result["selected_sequence"]) == len(group_candidates_by_frame(toy_candidates))
    assert len(toy_result["selected_indices"]) == len(toy_result["frames"])
    for selected_index, group in zip(toy_result["selected_indices"], group_candidates_by_frame(toy_candidates)):
        assert 0 <= selected_index < len(group)
    assert "transition" not in toy_result["stats"]["selected"]["marginal"]
    mismatch_result = optimize_shot_sequence_cp_sat(
        toy_candidates,
        toy_gt[:3],
        verbose=False,
        max_time_sec=10.0,
    )
    assert mismatch_result["status"] in {"OPTIMAL", "FEASIBLE"}
    assert len(mismatch_result["stats"]["target"]["marginal"]["shot_size"]) > 0

    print("selected_sequence:")
    pprint(toy_result["selected_sequence"])
    print("selected_stats:")
    pprint(toy_result["stats"]["selected"])
    print("target_stats:")
    pprint(toy_result["stats"]["target"])
    print("objective_breakdown:")
    pprint(toy_result["objective_breakdown"])
