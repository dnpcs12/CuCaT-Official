from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from camera_autoshot.geometry import Vec3, add, direction_from_yaw_pitch, dot, flatten_xz, normalize, scale, sub
from camera_autoshot.providers import FrameTargetContext, JsonFrameTargetProvider
from camera_autoshot.spec import CompositionPreset, RollPreset, TargetBodyPreset, TransitionPreset
from camera_autoshot.vmd import CameraVmdKey, classify_camera_transition, read_camera_vmd

from .solver_v3 import (
    COMPOSITION_TARGETS,
    DISTANCE_HEIGHT_MULTIPLIERS,
    FOV_DEGREES,
    PITCH_OFFSETS_DEG,
    WORLD_FORWARD,
    YAW_OFFSETS_DEG,
    _anchor_bounds,
    _anchor_point,
    _eye_from_mmd_camera,
    _project_bounds,
)
from .spec_v3 import (
    DistancePreset,
    FovPreset,
    InterpolationPreset,
    PitchPreset,
    ShotKeyframeRequestV3,
    TargetAnchorPreset,
    YawPreset,
)

ZERO_DISTANCE_EPS = 1e-4
ZERO_DISTANCE_LOOK_AT_DEPTH = 30.0
FOV_PHYSICAL_CHANGE_THRESHOLD_DEG = 5.0
FOV_SCREEN_FALLBACK_SIZE_SCORE = 0.22
FOV_SCREEN_FALLBACK_MIN_IMPROVEMENT = 0.08


@dataclass(frozen=True)
class InferredPlanKeyV3:
    frame: int
    target_anchor: str
    distance: str
    yaw: str
    pitch: str
    composition: str
    roll: str
    fov: str
    transition: str
    interpolation: str
    camera_reason: str


@dataclass(frozen=True)
class DistanceFovCandidate:
    distance: DistancePreset
    fov: FovPreset
    score: float
    size_score: float
    distance_ratio: float
    fov_deg: float
    physical_distance_ratio: float
    physical_fov_deg: float
    debug: dict[str, object]


@dataclass
class _PreparedPlanRowV3:
    key: CameraVmdKey
    transition: TransitionPreset
    context: FrameTargetContext
    eye: Vec3
    look_at_for_projection: Vec3
    zero_distance_rotation_look_at: bool
    target_anchor: TargetAnchorPreset
    top_anchor_debug: dict[str, float]
    yaw: YawPreset
    pitch: PitchPreset
    candidates: list[DistanceFovCandidate]
    distance_ratio: float
    roll: RollPreset
    composition: CompositionPreset
    projected: dict[str, float]
    anchor_scores: dict[str, float]


def _same_pose(a: CameraVmdKey, b: CameraVmdKey, tolerance: float = 1e-4) -> bool:
    if a.view_angle_deg != b.view_angle_deg or a.perspective != b.perspective:
        return False
    if abs(a.distance - b.distance) > tolerance:
        return False
    for av, bv in zip(a.interest, b.interest):
        if abs(av - bv) > tolerance:
            return False
    for av, bv in zip(a.rotate, b.rotate):
        if abs(av - bv) > tolerance:
            return False
    return True


def _collapse_cut_patterns(keys: list[CameraVmdKey]) -> list[tuple[CameraVmdKey, TransitionPreset]]:
    collapsed: list[tuple[CameraVmdKey, TransitionPreset]] = []
    index = 0
    while index < len(keys):
        key = keys[index]
        if index + 2 < len(keys):
            hold = keys[index + 1]
            next_key = keys[index + 2]
            if _same_pose(key, hold) and hold.frame + 1 == next_key.frame:
                collapsed.append((key, TransitionPreset.CUT))
                index += 2
                continue
        collapsed.append((key, classify_camera_transition(key.interpolation)))
        index += 1
    return collapsed


def _camera_interpolation_channels(interpolation: bytes) -> dict[str, tuple[int, int, int, int]]:
    padded = bytes(interpolation) + bytes(max(0, 24 - len(interpolation)))
    return {
        "interest_x": tuple(int(value) for value in padded[0:4]),
        "interest_y": tuple(int(value) for value in padded[4:8]),
        "interest_z": tuple(int(value) for value in padded[8:12]),
        "rotation": tuple(int(value) for value in padded[12:16]),
        "distance": tuple(int(value) for value in padded[16:20]),
        "fov": tuple(int(value) for value in padded[20:24]),
    }


def _classify_interpolation_cp(cp: tuple[int, int, int, int]) -> InterpolationPreset:
    # The project stores VMD control points in x1, x2, y1, y2 order.
    # Linear is therefore close to (20, 107, 20, 107), not x1, y1, x2, y2.
    x1, x2, y1, y2 = (float(value) for value in cp)
    if abs(y1 - x1) + abs(y2 - x2) <= 16.0:
        return InterpolationPreset.LINEAR
    start_delta = y1 - x1
    end_delta = y2 - x2
    if start_delta < 0.0 and end_delta < 0.0:
        return InterpolationPreset.EASE_IN
    if start_delta > 0.0 and end_delta > 0.0:
        return InterpolationPreset.EASE_OUT
    return InterpolationPreset.EASE_IN_OUT


def _dominant_interpolation_channel(key: CameraVmdKey, next_key: CameraVmdKey | None) -> str:
    if next_key is None:
        return "rotation"

    deltas = {
        "interest_x": abs(float(next_key.interest[0]) - float(key.interest[0])),
        "interest_y": abs(float(next_key.interest[1]) - float(key.interest[1])),
        "interest_z": abs(float(next_key.interest[2]) - float(key.interest[2])),
        "rotation": sum(abs(float(b) - float(a)) for a, b in zip(key.rotate, next_key.rotate)),
        "distance": abs(float(next_key.distance) - float(key.distance)),
        "fov": abs(float(next_key.view_angle_deg) - float(key.view_angle_deg)),
    }
    return max(deltas, key=deltas.get)


def _infer_interpolation(
    key: CameraVmdKey,
    next_key: CameraVmdKey | None,
    transition: TransitionPreset,
) -> tuple[InterpolationPreset, dict[str, object]]:
    channels = _camera_interpolation_channels(key.interpolation)
    dominant_channel = _dominant_interpolation_channel(key, next_key)
    interpolation = _classify_interpolation_cp(channels[dominant_channel])
    if transition is TransitionPreset.CUT:
        interpolation = InterpolationPreset.LINEAR
    return interpolation, {
        "dominant_channel": dominant_channel,
        "dominant_cp": list(channels[dominant_channel]),
        "channels": {name: list(values) for name, values in channels.items()},
        "raw_hex": key.interpolation.hex(),
        "transition": transition.value,
    }


def _look_at_for_zero_distance_reverse(key: CameraVmdKey, eye: Vec3) -> tuple[Vec3, bool]:
    """Synthesize a projection look-at for direct-camera distance=0 keys.

    Some source VMDs encode the camera as a direct eye position with distance 0.
    In that representation key.interest is effectively the eye, so projecting
    bounds with eye==interest degenerates and V3 previously fell back to
    extreme_close.  Match the pre-V3 reverse behavior by using the MMD rotation
    to create a virtual look-at in front of the camera for inference only.
    """

    if abs(float(key.distance)) > ZERO_DISTANCE_EPS:
        return key.interest, False

    synthetic_eye = _eye_from_mmd_camera(
        key.interest,
        key.rotate,
        -ZERO_DISTANCE_LOOK_AT_DEPTH,
    )
    forward = (
        float(key.interest[0]) - float(synthetic_eye[0]),
        float(key.interest[1]) - float(synthetic_eye[1]),
        float(key.interest[2]) - float(synthetic_eye[2]),
    )
    length = math.sqrt(forward[0] * forward[0] + forward[1] * forward[1] + forward[2] * forward[2])
    if length <= 1e-8:
        return key.interest, False
    forward = (forward[0] / length, forward[1] / length, forward[2] / length)
    return (
        float(eye[0]) + forward[0] * ZERO_DISTANCE_LOOK_AT_DEPTH,
        float(eye[1]) + forward[1] * ZERO_DISTANCE_LOOK_AT_DEPTH,
        float(eye[2]) + forward[2] * ZERO_DISTANCE_LOOK_AT_DEPTH,
    ), True


# Anchors that the V3.1 labels use. The coarse LEGS/TORSO anchors exist in the schema only for the
# "coarse target anchor" representation ablation, which maps fine labels to coarse ones afterwards;
# reverse labeling never selects them.
REVERSE_TARGET_ANCHORS = tuple(
    anchor for anchor in TargetAnchorPreset if anchor not in (TargetAnchorPreset.LEGS, TargetAnchorPreset.TORSO)
)


def _infer_target_anchor(
    context: FrameTargetContext,
    interest: Vec3,
    eye: Vec3 | None = None,
    fov_deg: float | None = None,
    aspect_ratio: float = 16.0 / 9.0,
) -> tuple[TargetAnchorPreset, dict[str, float]]:
    best = TargetAnchorPreset.CHEST
    best_score = float("inf")
    scores: dict[str, float] = {}
    full = context.bounds_for(TargetBodyPreset.FULL_BODY)
    height = max(full.size()[1], 1e-4)
    for anchor in REVERSE_TARGET_ANCHORS:
        point = _anchor_point(context, anchor)
        score = abs(float(interest[1]) - point[1]) / height
        if eye is not None and fov_deg is not None:
            full_projected = _project_bounds(full, eye, interest, float(fov_deg), aspect_ratio)
            full_half_h = float(full_projected.get("height_frac", 1e9))
            projected = _project_bounds(_anchor_bounds(context, anchor), eye, interest, float(fov_deg), aspect_ratio)
            center_y = float(projected.get("center_y", 0.0))
            half_h = float(projected.get("height_frac", 1e9))
            if half_h >= 1e8:
                projection_penalty = 2.5
            else:
                top = center_y + half_h
                bottom = center_y - half_h
                visible_y = max(0.0, min(top, 1.0) - max(bottom, -1.0))
                visible_ratio = min(1.0, visible_y / max(half_h * 2.0, 1e-6))
                offscreen_center = max(0.0, abs(center_y) - 0.95)
                projection_penalty = (1.0 - visible_ratio) * 0.25 + offscreen_center * 0.55
            if 0.0 < full_half_h <= 1.15 and anchor in (
                TargetAnchorPreset.FEET,
                TargetAnchorPreset.LOWER_LEG,
            ):
                # When the GT camera already shows nearly the whole body, a
                # low look-at point produces the "feet-only" crop after solve.
                projection_penalty += 0.75
            if 0.0 < full_half_h <= 1.0 and anchor is TargetAnchorPreset.THIGH:
                projection_penalty += 0.30
            full_center_y = float(full_projected.get("center_y", 0.0))
            if full_half_h > 2.0 and full_center_y < -0.6:
                # These GT close-ups keep the upper body in frame even when the
                # raw interest point sits near the waist.  Prefer a slightly
                # higher semantic anchor so the solver does not crop the face.
                if anchor in (TargetAnchorPreset.FEET, TargetAnchorPreset.LOWER_LEG):
                    projection_penalty += 0.70
                elif anchor is TargetAnchorPreset.THIGH:
                    projection_penalty += 0.45
                elif anchor is TargetAnchorPreset.HIP:
                    projection_penalty += 0.28
                elif anchor is TargetAnchorPreset.WAIST:
                    projection_penalty += 0.18
            if full_half_h > 2.0 and full_center_y < -1.2 and anchor is TargetAnchorPreset.CHEST:
                projection_penalty += 0.08
            if full_half_h > 1.8 and full_center_y > 0.35:
                # A high full-body center with a large projected body is often
                # a whole-body crop with the feet near the edge, not a feet shot.
                if anchor in (TargetAnchorPreset.FEET, TargetAnchorPreset.LOWER_LEG):
                    projection_penalty += 0.45
                elif anchor is TargetAnchorPreset.THIGH:
                    projection_penalty += 0.12
            if fov_deg >= 35.0 and 1.2 < full_half_h < 1.9 and -0.85 < full_center_y < -0.35:
                # Wide-FOV close shots can keep the upper body dominant even
                # when the raw look-at is near the hip.  Without this, the
                # reconstruction drifts down during upward camera moves.
                if anchor in (TargetAnchorPreset.FEET, TargetAnchorPreset.LOWER_LEG):
                    projection_penalty += 0.75
                elif anchor is TargetAnchorPreset.THIGH:
                    projection_penalty += 0.55
                elif anchor is TargetAnchorPreset.HIP:
                    projection_penalty += 0.45
                elif anchor is TargetAnchorPreset.WAIST:
                    projection_penalty += 0.18
                elif anchor is TargetAnchorPreset.FACE:
                    projection_penalty += 0.06
            if full_half_h > 2.0 and anchor is TargetAnchorPreset.THIGH:
                projection_penalty += 0.10
            score += projection_penalty
        scores[anchor.value] = score
        if score < best_score:
            best_score = score
            best = anchor
    return best, scores


def _infer_distance(context: FrameTargetContext, eye: Vec3, interest: Vec3) -> tuple[DistancePreset, float]:
    full = context.bounds_for(TargetBodyPreset.FULL_BODY)
    body_height = max(full.size()[1], 1e-4)
    ratio = math.sqrt(sum((eye[i] - interest[i]) ** 2 for i in range(3))) / body_height
    best = min(DistancePreset, key=lambda item: abs(DISTANCE_HEIGHT_MULTIPLIERS[item] - ratio))
    return best, ratio


def _infer_fov(view_angle_deg: int) -> FovPreset:
    value = float(view_angle_deg)
    return min(FovPreset, key=lambda item: abs(FOV_DEGREES[item] - value))


def _safe_log_ratio(candidate: float, target: float) -> float:
    candidate = max(float(candidate), 1e-6)
    target = max(float(target), 1e-6)
    return abs(math.log(candidate / target))


def _build_distance_fov_screen_candidates(
    context: FrameTargetContext,
    anchor: TargetAnchorPreset,
    yaw: YawPreset,
    pitch: PitchPreset,
    eye: Vec3,
    interest: Vec3,
    view_angle_deg: int,
    aspect_ratio: float,
    yaw_mode: str = "actor",
) -> tuple[list[DistanceFovCandidate], float]:
    """Enumerate discrete distance/FOV candidates with screen-size scores."""
    full = context.bounds_for(TargetBodyPreset.FULL_BODY)
    body_height = max(full.size()[1], 1e-4)
    physical_ratio = math.sqrt(sum((eye[i] - interest[i]) ** 2 for i in range(3))) / body_height
    physical_fov = float(view_angle_deg)
    # Distance/FOV should describe the final apparent zoom of the whole
    # character when the full body can be projected.  For heavy close-ups or
    # steep views the full-body box may cross behind the camera, so fall back
    # to the selected anchor instead of reverting to raw physical distance.
    anchor_bounds = _anchor_bounds(context, anchor)
    fit_bounds = full
    target_projected = _project_bounds(fit_bounds, eye, interest, physical_fov, aspect_ratio)
    target_height = float(target_projected.get("height_frac", 0.0))
    target_width = float(target_projected.get("width_frac", 0.0))

    if target_height >= 1e8 or target_width >= 1e8 or target_height <= 1e-6 or target_width <= 1e-6:
        fit_bounds = anchor_bounds
        target_projected = _project_bounds(fit_bounds, eye, interest, physical_fov, aspect_ratio)
        target_height = float(target_projected.get("height_frac", 0.0))
        target_width = float(target_projected.get("width_frac", 0.0))

    if target_height >= 1e8 or target_width >= 1e8 or target_height <= 1e-6 or target_width <= 1e-6:
        distance, ratio = _infer_distance(context, eye, interest)
        fov = _infer_fov(view_angle_deg)
        return [
            DistanceFovCandidate(
                distance=distance,
                fov=fov,
                score=0.0,
                size_score=0.0,
                distance_ratio=DISTANCE_HEIGHT_MULTIPLIERS[distance],
                fov_deg=FOV_DEGREES[fov],
                physical_distance_ratio=ratio,
                physical_fov_deg=physical_fov,
                debug={
                    "mode": "fallback_physical_after_invalid_projection",
                    "target_height_frac": target_height,
                    "target_width_frac": target_width,
                },
            )
        ], ratio

    base_forward = WORLD_FORWARD if yaw_mode == "world" or pitch is PitchPreset.TOP else context.actor_forward
    eye_direction = direction_from_yaw_pitch(base_forward, YAW_OFFSETS_DEG[yaw], PITCH_OFFSETS_DEG[pitch])
    candidate_interest = _anchor_point(context, anchor)

    def evaluate_candidate(distance: DistancePreset, fov: FovPreset) -> tuple[float, dict[str, object]] | None:
        candidate_distance = body_height * DISTANCE_HEIGHT_MULTIPLIERS[distance]
        candidate_eye = add(candidate_interest, scale(eye_direction, candidate_distance))
        candidate_fov = FOV_DEGREES[fov]
        projected = _project_bounds(fit_bounds, candidate_eye, candidate_interest, candidate_fov, aspect_ratio)
        candidate_height = float(projected.get("height_frac", 0.0))
        candidate_width = float(projected.get("width_frac", 0.0))
        if candidate_height >= 1e8 or candidate_width >= 1e8:
            return None

        # Primary objective: apparent bbox size.  Tiny secondary terms keep ties
        # stable without letting raw physical distance/FOV dominate framing.
        size_score = _safe_log_ratio(candidate_height, target_height) + 0.35 * _safe_log_ratio(
            candidate_width, target_width
        )
        distance_tie = abs(DISTANCE_HEIGHT_MULTIPLIERS[distance] - physical_ratio) * 0.01
        fov_tie = abs(candidate_fov - physical_fov) / 90.0 * 0.01
        score = size_score + distance_tie + fov_tie
        return score, {
            "fit_bounds": "anchor" if fit_bounds is anchor_bounds else "full_body",
            "score": score,
            "size_score": size_score,
            "target_height_frac": target_height,
            "target_width_frac": target_width,
            "candidate_height_frac": candidate_height,
            "candidate_width_frac": candidate_width,
            "physical_distance_ratio_body_height": physical_ratio,
            "candidate_distance_ratio_body_height": DISTANCE_HEIGHT_MULTIPLIERS[distance],
            "physical_fov_deg": physical_fov,
            "candidate_fov_deg": candidate_fov,
        }

    scored: list[tuple[float, DistancePreset, FovPreset, dict[str, object]]] = []
    for distance in DistancePreset:
        for fov in FovPreset:
            evaluated = evaluate_candidate(distance, fov)
            if evaluated is None:
                continue
            score, debug = evaluated
            scored.append((score, distance, fov, debug))

    if not scored:
        distance, ratio = _infer_distance(context, eye, interest)
        fov = _infer_fov(view_angle_deg)
        return [
            DistanceFovCandidate(
                distance=distance,
                fov=fov,
                score=0.0,
                size_score=0.0,
                distance_ratio=DISTANCE_HEIGHT_MULTIPLIERS[distance],
                fov_deg=FOV_DEGREES[fov],
                physical_distance_ratio=ratio,
                physical_fov_deg=physical_fov,
                debug={
                    "mode": "fallback_physical_after_no_valid_screen_candidate",
                    "target_height_frac": target_height,
                    "target_width_frac": target_width,
                },
            )
        ], ratio

    candidates: list[DistanceFovCandidate] = []
    for score, distance, fov, debug in scored:
        candidates.append(
            DistanceFovCandidate(
                distance=distance,
                fov=fov,
                score=score,
                size_score=float(debug.get("size_score", score)),
                distance_ratio=DISTANCE_HEIGHT_MULTIPLIERS[distance],
                fov_deg=FOV_DEGREES[fov],
                physical_distance_ratio=physical_ratio,
                physical_fov_deg=physical_fov,
                debug=debug,
            )
        )
    candidates.sort(key=lambda item: item.score)
    return candidates, physical_ratio


def _infer_distance_and_fov_by_screen_size(
    context: FrameTargetContext,
    anchor: TargetAnchorPreset,
    yaw: YawPreset,
    pitch: PitchPreset,
    eye: Vec3,
    interest: Vec3,
    view_angle_deg: int,
    aspect_ratio: float,
    fov_policy: str = "joint",
    yaw_mode: str = "actor",
) -> tuple[DistancePreset, FovPreset, float, dict[str, object]]:
    """Choose distance/FOV by bbox score or an explicit FOV policy."""

    candidates, physical_ratio = _build_distance_fov_screen_candidates(
        context,
        anchor,
        yaw,
        pitch,
        eye,
        interest,
        view_angle_deg,
        aspect_ratio,
        yaw_mode=yaw_mode,
    )

    best_joint = min(candidates, key=lambda item: item.score)
    best_joint_score = best_joint.score
    best_joint_distance = best_joint.distance
    best_joint_fov = best_joint.fov
    best_joint_debug = best_joint.debug
    physical_fov = float(view_angle_deg)
    physical_fov_preset = _infer_fov(view_angle_deg)
    physical_fov_delta_from_normal = abs(physical_fov - FOV_DEGREES[FovPreset.NORMAL])
    physical_fov_changed = physical_fov_delta_from_normal >= FOV_PHYSICAL_CHANGE_THRESHOLD_DEG
    if fov_policy == "joint":
        best_joint_debug = dict(best_joint_debug)
        best_joint_debug.update(
            {
                "mode": "screen_size_joint",
                "fov_policy": "joint_legacy",
                "preferred_fov": "",
                "physical_fov_preset": physical_fov_preset.value,
                "physical_fov_changed": physical_fov_changed,
                "physical_fov_delta_from_normal_deg": physical_fov_delta_from_normal,
                "best_joint_score": best_joint_score,
                "best_joint_fov": best_joint_fov.value,
                "used_joint_fallback": False,
            }
        )
        return best_joint_distance, best_joint_fov, physical_ratio, best_joint_debug

    if fov_policy == "temporal_delta":
        # Temporal selection is handled at the sequence level in infer_plan_keys_v3.
        return best_joint_distance, best_joint_fov, physical_ratio, best_joint_debug

    if fov_policy != "physical_normal_first":
        raise ValueError(f"unsupported fov_policy: {fov_policy}")

    preferred_fov = physical_fov_preset if physical_fov_changed else FovPreset.NORMAL
    preferred_candidate = min(
        [item for item in candidates if item.fov is preferred_fov] or candidates,
        key=lambda item: item.score,
    )
    preferred_score = preferred_candidate.score
    preferred_distance = preferred_candidate.distance
    selected_fov = preferred_candidate.fov
    selected_debug = preferred_candidate.debug

    # FOV policy:
    # - If raw VMD FOV is near MMD's normal value, keep NORMAL and solve zoom via distance.
    # - If raw VMD FOV is clearly narrow/wide, keep that physical FOV bin.
    # - Only fall back to fully joint bbox matching when the preferred FOV leaves
    #   a visibly large bbox mismatch and the joint candidate is meaningfully better.
    joint_improvement = preferred_score - best_joint_score
    use_joint_fallback = (
        best_joint_fov is not selected_fov
        and float(selected_debug.get("size_score", 0.0)) > FOV_SCREEN_FALLBACK_SIZE_SCORE
        and joint_improvement > FOV_SCREEN_FALLBACK_MIN_IMPROVEMENT
    )
    if use_joint_fallback:
        selected_distance = best_joint_distance
        selected_fov = best_joint_fov
        selected_debug = best_joint_debug
        mode = "screen_size_joint_fov_fallback"
    else:
        selected_distance = preferred_distance
        mode = "screen_size_preferred_fov"

    selected_debug = dict(selected_debug)
    selected_debug.update(
        {
            "mode": mode,
            "preferred_fov": preferred_fov.value,
            "physical_fov_preset": physical_fov_preset.value,
            "physical_fov_changed": physical_fov_changed,
            "physical_fov_delta_from_normal_deg": physical_fov_delta_from_normal,
            "preferred_score": preferred_score,
            "best_joint_score": best_joint_score,
            "best_joint_fov": best_joint_fov.value,
            "joint_improvement": joint_improvement,
            "used_joint_fallback": use_joint_fallback,
            "fov_policy": "physical_bin" if physical_fov_changed else "normal_default",
        }
    )
    return selected_distance, selected_fov, physical_ratio, selected_debug


def _stabilize_top_view_anchor(
    context: FrameTargetContext,
    anchor: TargetAnchorPreset,
    pitch: PitchPreset,
    eye: Vec3,
    interest: Vec3,
    fov_deg: float,
    aspect_ratio: float,
) -> tuple[TargetAnchorPreset, dict[str, float]]:
    """Avoid top-view crops that over-center the lower legs.

    MMD top cameras can encode a whole-body overhead shot with an interest
    point that numerically sits near the legs.  If we keep that as a semantic
    lower-leg anchor, the solver reproduces a too-close leg-centered shot and
    clips the head.  Only apply this to top views whose full-body projection is
    already roughly centered, so explicit feet/leg detail shots remain intact.
    """

    debug = {"applied": 0.0}
    if pitch is not PitchPreset.TOP or anchor not in (
        TargetAnchorPreset.FEET,
        TargetAnchorPreset.LOWER_LEG,
        TargetAnchorPreset.THIGH,
    ):
        return anchor, debug

    projected = _project_bounds(context.bounds_for(TargetBodyPreset.FULL_BODY), eye, interest, fov_deg, aspect_ratio)
    center_y = float(projected.get("center_y", 0.0))
    height = float(projected.get("height_frac", 1e9))
    debug.update({"center_y": center_y, "height_frac": height})

    if 0.75 <= height <= 1.45 and -0.55 <= center_y <= 0.15:
        debug["applied"] = 1.0
        debug["from_anchor"] = anchor.value
        debug["to_anchor"] = TargetAnchorPreset.HIP.value
        return TargetAnchorPreset.HIP, debug

    return anchor, debug


def _stabilize_top_view_distance_fov(
    distance: DistancePreset,
    fov: FovPreset,
    pitch: PitchPreset,
    physical_distance_ratio: float,
    top_anchor_debug: dict[str, float],
    original_view_angle_deg: int,
) -> tuple[DistancePreset, FovPreset, dict[str, str | float]]:
    """Keep stabilized overhead whole-body shots from becoming close-ups."""

    debug: dict[str, str | float] = {"applied": 0.0}
    top_anchor_applied = float(top_anchor_debug.get("applied", 0.0)) >= 0.5
    far_physical_top = (
        pitch is PitchPreset.TOP
        and physical_distance_ratio >= 8.0
        and float(original_view_angle_deg) <= 25.0
    )
    if not top_anchor_applied and not far_physical_top:
        return distance, fov, debug

    debug.update(
        {
            "applied": 1.0,
            "top_anchor_applied": 1.0 if top_anchor_applied else 0.0,
            "far_physical_top": 1.0 if far_physical_top else 0.0,
            "from_distance": distance.value,
            "from_fov": fov.value,
        }
    )
    if far_physical_top:
        if distance in (
            DistancePreset.EXTREME_CLOSE,
            DistancePreset.CLOSE,
            DistancePreset.MEDIUM_CLOSE,
        ):
            distance = DistancePreset.MEDIUM
    elif distance in (DistancePreset.EXTREME_CLOSE, DistancePreset.CLOSE, DistancePreset.MEDIUM_CLOSE):
        distance = DistancePreset.MEDIUM
    if far_physical_top or top_anchor_applied:
        fov = FovPreset.NORMAL
    else:
        fov = _infer_fov(original_view_angle_deg)
    debug.update({"to_distance": distance.value, "to_fov": fov.value})
    return distance, fov, debug


def _infer_pitch(direction: Vec3) -> PitchPreset:
    vertical_deg = math.degrees(math.asin(max(-1.0, min(1.0, direction[1]))))
    if vertical_deg >= 35.0:
        return PitchPreset.TOP
    if vertical_deg >= 8.0:
        return PitchPreset.HIGH
    if vertical_deg <= -7.0:
        return PitchPreset.LOW
    return PitchPreset.LEVEL


def _infer_yaw(context: FrameTargetContext, direction: Vec3, pitch: PitchPreset, yaw_mode: str = "actor") -> YawPreset:
    flat_direction = flatten_xz(direction)
    base_forward = WORLD_FORWARD if yaw_mode == "world" or pitch is PitchPreset.TOP else context.actor_forward
    best = YawPreset.FRONT
    best_dot = -1e9
    for preset, yaw_deg in YAW_OFFSETS_DEG.items():
        candidate = flatten_xz(direction_from_yaw_pitch(base_forward, yaw_deg, 0.0))
        score = dot(flat_direction, candidate)
        if score > best_dot:
            best_dot = score
            best = preset
    return best


def _normalize_degrees(degrees: float) -> float:
    return (degrees + 180.0) % 360.0 - 180.0


def _roll_degrees_from_rotate(rotate) -> float:
    return _normalize_degrees(math.degrees(float(rotate[2])))


def _infer_roll(rotate) -> RollPreset:
    roll_deg = _roll_degrees_from_rotate(rotate)
    abs_roll = abs(roll_deg)
    if abs_roll < 2.5:
        return RollPreset.NONE
    if abs_roll < 12.0:
        magnitude = "slight"
    elif abs_roll < 30.0:
        magnitude = "normal"
    else:
        magnitude = "strong"
    if roll_deg >= 0.0:
        if magnitude == "slight":
            return RollPreset.SLIGHT_LEFT
        if magnitude == "normal":
            return RollPreset.LEFT
        return RollPreset.STRONG_LEFT
    if magnitude == "slight":
        return RollPreset.SLIGHT_RIGHT
    if magnitude == "normal":
        return RollPreset.RIGHT
    return RollPreset.STRONG_RIGHT


def _infer_composition(
    context: FrameTargetContext,
    anchor: TargetAnchorPreset,
    eye: Vec3,
    interest: Vec3,
    fov_deg: float,
    aspect_ratio: float,
) -> tuple[CompositionPreset, dict[str, float]]:
    projected = _project_bounds(context.bounds_for(TargetBodyPreset.FULL_BODY), eye, interest, fov_deg, aspect_ratio)
    if float(projected.get("height_frac", 1e9)) >= 1e8:
        projected = _project_bounds(_anchor_bounds(context, anchor), eye, interest, fov_deg, aspect_ratio)
    center_x = float(projected["center_x"])
    best = min(CompositionPreset, key=lambda item: abs(center_x - COMPOSITION_TARGETS[item]))
    return best, projected


def _build_reason(
    target_anchor: TargetAnchorPreset,
    distance: DistancePreset,
    yaw: YawPreset,
    pitch: PitchPreset,
    composition: CompositionPreset,
    roll: RollPreset,
    fov: FovPreset,
) -> str:
    roll_text = "" if roll is RollPreset.NONE else f", {roll.value} roll"
    return (
        f"A {distance.value} shot anchored at {target_anchor.value}, "
        f"yaw={yaw.value}, pitch={pitch.value}, fov={fov.value}, "
        f"composition={composition.value}{roll_text}."
    )


def _prepare_plan_rows_v3(
    provider: JsonFrameTargetProvider,
    collapsed: list[tuple[CameraVmdKey, TransitionPreset]],
    aspect_ratio: float,
    yaw_mode: str = "actor",
) -> list[_PreparedPlanRowV3]:
    rows: list[_PreparedPlanRowV3] = []
    for key, transition in collapsed:
        context = provider.get_context(key.frame)
        eye = _eye_from_mmd_camera(key.interest, key.rotate, key.distance)
        look_at_for_projection, zero_distance_rotation_look_at = _look_at_for_zero_distance_reverse(key, eye)
        direction = normalize(sub(eye, look_at_for_projection))
        pitch = _infer_pitch(direction)
        yaw = _infer_yaw(context, direction, pitch, yaw_mode=yaw_mode)
        target_anchor, anchor_scores = _infer_target_anchor(
            context,
            look_at_for_projection,
            eye=eye,
            fov_deg=float(key.view_angle_deg),
            aspect_ratio=aspect_ratio,
        )
        target_anchor, top_anchor_debug = _stabilize_top_view_anchor(
            context,
            target_anchor,
            pitch,
            eye,
            look_at_for_projection,
            float(key.view_angle_deg),
            aspect_ratio,
        )
        candidates, distance_ratio = _build_distance_fov_screen_candidates(
            context,
            target_anchor,
            yaw,
            pitch,
            eye,
            look_at_for_projection,
            key.view_angle_deg,
            aspect_ratio,
            yaw_mode=yaw_mode,
        )
        roll = _infer_roll(key.rotate)
        composition, projected = _infer_composition(
            context,
            target_anchor,
            eye,
            look_at_for_projection,
            float(key.view_angle_deg),
            aspect_ratio,
        )
        rows.append(
            _PreparedPlanRowV3(
                key=key,
                transition=transition,
                context=context,
                eye=eye,
                look_at_for_projection=look_at_for_projection,
                zero_distance_rotation_look_at=zero_distance_rotation_look_at,
                target_anchor=target_anchor,
                top_anchor_debug=top_anchor_debug,
                yaw=yaw,
                pitch=pitch,
                candidates=candidates,
                distance_ratio=distance_ratio,
                roll=roll,
                composition=composition,
                projected=projected,
                anchor_scores=anchor_scores,
            )
        )
    return rows


def _temporal_distance_fov_local_score(candidate: DistanceFovCandidate) -> float:
    # Local bbox fit remains important, but FOV should not be selected purely
    # because an absolute source FOV bin is narrow/wide.  Prefer normal FOV
    # unless temporal FOV deltas make a non-normal bin useful.
    fov_prior = 0.0 if candidate.fov is FovPreset.NORMAL else 0.04
    opposing_zoom_prior = 0.0
    if candidate.fov is FovPreset.NARROW and candidate.distance in (DistancePreset.FAR, DistancePreset.WIDE):
        opposing_zoom_prior += 0.025
    if candidate.fov is FovPreset.WIDE and candidate.distance in (
        DistancePreset.EXTREME_CLOSE,
        DistancePreset.CLOSE,
    ):
        opposing_zoom_prior += 0.025
    return candidate.size_score + fov_prior + opposing_zoom_prior


def _temporal_distance_fov_transition_score(
    prev_candidate: DistanceFovCandidate,
    candidate: DistanceFovCandidate,
) -> float:
    physical_distance_delta = candidate.physical_distance_ratio - prev_candidate.physical_distance_ratio
    selected_distance_delta = candidate.distance_ratio - prev_candidate.distance_ratio
    physical_fov_delta = candidate.physical_fov_deg - prev_candidate.physical_fov_deg
    selected_fov_delta = candidate.fov_deg - prev_candidate.fov_deg

    distance_delta_error = abs(selected_distance_delta - physical_distance_delta) * 0.12
    fov_delta_error = abs(selected_fov_delta - physical_fov_delta) / 10.0 * 0.16

    # If the original VMD barely changes FOV/distance, avoid inventing a
    # discrete jump just to improve one frame's bbox fit.
    invented_fov_jump = 0.08 if abs(physical_fov_delta) < 2.0 and abs(selected_fov_delta) >= 8.0 else 0.0
    invented_distance_jump = (
        0.08 if abs(physical_distance_delta) < 0.25 and abs(selected_distance_delta) >= 1.2 else 0.0
    )
    return distance_delta_error + fov_delta_error + invented_fov_jump + invented_distance_jump


def _select_distance_fov_temporal_delta(
    rows: list[_PreparedPlanRowV3],
) -> tuple[list[DistanceFovCandidate], list[dict[str, object]]]:
    if not rows:
        return [], []

    costs: list[list[float]] = []
    back: list[list[int]] = []
    first_costs = [_temporal_distance_fov_local_score(candidate) for candidate in rows[0].candidates]
    costs.append(first_costs)
    back.append([-1 for _ in rows[0].candidates])

    for index in range(1, len(rows)):
        current_costs: list[float] = []
        current_back: list[int] = []
        prev_candidates = rows[index - 1].candidates
        candidates = rows[index].candidates
        for candidate in candidates:
            local = _temporal_distance_fov_local_score(candidate)
            best_prev_index = 0
            best_cost = float("inf")
            for prev_index, prev_candidate in enumerate(prev_candidates):
                transition_cost = _temporal_distance_fov_transition_score(prev_candidate, candidate)
                total = costs[index - 1][prev_index] + transition_cost + local
                if total < best_cost:
                    best_cost = total
                    best_prev_index = prev_index
            current_costs.append(best_cost)
            current_back.append(best_prev_index)
        costs.append(current_costs)
        back.append(current_back)

    selected_indices = [0 for _ in rows]
    selected_indices[-1] = min(range(len(costs[-1])), key=lambda item: costs[-1][item])
    for index in range(len(rows) - 1, 0, -1):
        selected_indices[index - 1] = back[index][selected_indices[index]]

    selected = [row.candidates[candidate_index] for row, candidate_index in zip(rows, selected_indices)]
    debug_extras: list[dict[str, object]] = []
    for index, (row, candidate, candidate_index) in enumerate(zip(rows, selected, selected_indices)):
        extra: dict[str, object] = {
            "mode": "screen_size_temporal_delta",
            "fov_policy": "temporal_delta",
            "temporal_candidate_index": candidate_index,
            "temporal_local_score": _temporal_distance_fov_local_score(candidate),
            "temporal_total_cost": costs[index][candidate_index],
        }
        if index > 0:
            prev = selected[index - 1]
            extra.update(
                {
                    "physical_distance_delta": candidate.physical_distance_ratio - prev.physical_distance_ratio,
                    "selected_distance_delta": candidate.distance_ratio - prev.distance_ratio,
                    "physical_fov_delta_deg": candidate.physical_fov_deg - prev.physical_fov_deg,
                    "selected_fov_delta_deg": candidate.fov_deg - prev.fov_deg,
                    "temporal_transition_score": _temporal_distance_fov_transition_score(prev, candidate),
                }
            )
        debug_extras.append(extra)
    return selected, debug_extras


def infer_plan_keys_v3(
    vmd_path: Path,
    targets_path: Path,
    sample_fps: float = 0.0,
    timeline_fps: float = 30.0,
    aspect_ratio: float = 16.0 / 9.0,
    fov_policy: str = "joint",
    yaw_mode: str = "actor",
) -> tuple[list[InferredPlanKeyV3], list[dict]]:
    provider = JsonFrameTargetProvider(targets_path)
    camera_keys = read_camera_vmd(vmd_path)
    collapsed = _collapse_cut_patterns(camera_keys)
    prepared_rows = _prepare_plan_rows_v3(provider, collapsed, aspect_ratio, yaw_mode=yaw_mode)
    if fov_policy == "temporal_delta":
        temporal_candidates, temporal_debugs = _select_distance_fov_temporal_delta(prepared_rows)
    else:
        temporal_candidates = []
        temporal_debugs = [{} for _ in prepared_rows]

    frame_scale = sample_fps / timeline_fps if sample_fps > 0.0 and timeline_fps > 0.0 else 1.0
    plan_keys: list[InferredPlanKeyV3] = []
    debug_rows: list[dict] = []

    for row_index, row in enumerate(prepared_rows):
        key = row.key
        next_key = prepared_rows[row_index + 1].key if row_index + 1 < len(prepared_rows) else None
        transition = row.transition
        interpolation, interpolation_debug = _infer_interpolation(key, next_key, transition)
        context = row.context
        eye = row.eye
        look_at_for_projection = row.look_at_for_projection
        pitch = row.pitch
        yaw = row.yaw
        target_anchor = row.target_anchor
        top_anchor_debug = row.top_anchor_debug
        if fov_policy == "temporal_delta":
            selected_candidate = temporal_candidates[row_index]
            distance = selected_candidate.distance
            fov = selected_candidate.fov
            distance_ratio = row.distance_ratio
            distance_fov_debug = dict(selected_candidate.debug)
            distance_fov_debug.update(temporal_debugs[row_index])
        else:
            distance, fov, distance_ratio, distance_fov_debug = _infer_distance_and_fov_by_screen_size(
                context,
                target_anchor,
                yaw,
                pitch,
                eye,
                look_at_for_projection,
                key.view_angle_deg,
                aspect_ratio,
                fov_policy=fov_policy,
                yaw_mode=yaw_mode,
            )
        distance, fov, top_distance_fov_debug = _stabilize_top_view_distance_fov(
            distance,
            fov,
            pitch,
            distance_ratio,
            top_anchor_debug,
            key.view_angle_deg,
        )
        roll = row.roll
        composition = row.composition
        projected = row.projected
        output_frame = int(round(key.frame * frame_scale)) if frame_scale != 1.0 else int(key.frame)
        plan_keys.append(
            InferredPlanKeyV3(
                frame=output_frame,
                target_anchor=target_anchor.value,
                distance=distance.value,
                yaw=yaw.value,
                pitch=pitch.value,
                composition=composition.value,
                roll=roll.value,
                fov=fov.value,
                transition=transition.value,
                interpolation=interpolation.value,
                camera_reason=_build_reason(
                    target_anchor,
                    distance,
                    yaw,
                    pitch,
                    composition,
                    roll,
                    fov,
                ),
            )
        )
        debug_rows.append(
            {
                "timeline_frame": key.frame,
                "output_frame": output_frame,
                "transition": transition.value,
                "interpolation": interpolation.value,
                "interpolation_raw": interpolation_debug,
                "target_anchor": target_anchor.value,
                "distance": distance.value,
                "distance_ratio_body_height": distance_ratio,
                "yaw": yaw.value,
                "pitch": pitch.value,
                "composition": composition.value,
                "roll": roll.value,
                "roll_degrees_original": _roll_degrees_from_rotate(key.rotate),
                "fov": fov.value,
                "distance_fov_joint": distance_fov_debug,
                "zero_distance_rotation_look_at": row.zero_distance_rotation_look_at,
                "look_at_for_projection": look_at_for_projection,
                "top_anchor_stabilization": top_anchor_debug,
                "top_distance_fov_stabilization": top_distance_fov_debug,
                "anchor_scores": row.anchor_scores,
                "projected": projected,
                "camera_key": {
                    "frame": key.frame,
                    "distance": key.distance,
                    "interest": key.interest,
                    "rotate": key.rotate,
                    "view_angle_deg": key.view_angle_deg,
                    "perspective": key.perspective,
                },
                "eye": eye,
                "context_frame": context.frame,
                "yaw_reference": yaw_mode,
            }
        )
    return plan_keys, debug_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer a Reverse V3 shot-plan JSON from a camera VMD.")
    parser.add_argument("--camera-vmd", required=True, help="Input camera-only VMD path.")
    parser.add_argument("--targets", required=True, help="Target dump JSON path.")
    parser.add_argument("--output", required=True, help="Output V3 plan JSON path.")
    parser.add_argument("--debug-json", default="", help="Optional debug JSON output path.")
    parser.add_argument("--sample-fps", type=float, default=0.0)
    parser.add_argument("--timeline-fps", type=float, default=30.0)
    parser.add_argument("--aspect-ratio", type=float, default=16.0 / 9.0)
    parser.add_argument(
        "--fov-policy",
        choices=("joint", "physical_normal_first", "temporal_delta"),
        default="joint",
        help="Distance/FOV reverse policy. joint preserves the legacy V3 behavior.",
    )
    parser.add_argument(
        "--yaw-mode",
        choices=("actor", "world"),
        default="actor",
        help="Yaw reference frame. actor preserves V3/V3.1; world is used by V3.2.",
    )
    parser.add_argument("--schema-name", default="camera_autoshot.reverse_v3")
    parser.add_argument("--omit-camera-reason", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plan_keys, debug_rows = infer_plan_keys_v3(
        Path(args.camera_vmd),
        Path(args.targets),
        sample_fps=args.sample_fps,
        timeline_fps=args.timeline_fps,
        aspect_ratio=args.aspect_ratio,
        fov_policy=args.fov_policy,
        yaw_mode=args.yaw_mode,
    )
    payload: dict[str, object] = {
        "schema": args.schema_name,
        "yaw_reference": args.yaw_mode,
        "keyframes": [],
    }
    if args.sample_fps > 0.0:
        payload["sample_fps"] = float(args.sample_fps)
    for key in plan_keys:
        row = asdict(key)
        if args.omit_camera_reason:
            row.pop("camera_reason", None)
        payload["keyframes"].append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote V3 inferred plan: {output_path}")
    print(f"keyframe_count={len(plan_keys)}")

    if args.debug_json:
        debug_path = Path(args.debug_json)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(json.dumps(debug_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote V3 debug JSON: {debug_path}")


if __name__ == "__main__":
    main()
