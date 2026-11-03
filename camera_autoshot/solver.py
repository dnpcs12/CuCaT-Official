from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import Vec3, add, cross, direction_from_yaw_pitch, dot, length, normalize, scale, sub
from .providers import Bounds3D, FrameTargetProvider
from .spec import (
    AnglePreset,
    CompositionPreset,
    RollPreset,
    ShotKeyframeRequest,
    ShotSizePreset,
    TargetBodyPreset,
    TransitionPreset,
)


WORLD_UP: Vec3 = (0.0, 1.0, 0.0)
WORLD_FORWARD: Vec3 = (0.0, 0.0, 1.0)


ANGLE_OFFSETS = {
    AnglePreset.FRONT: (0.0, 0.0),
    AnglePreset.BACK: (180.0, 0.0),
    AnglePreset.TOP: (0.0, 72.0),
    AnglePreset.LEFT_45: (35.0, 0.0),
    AnglePreset.RIGHT_45: (-35.0, 0.0),
    AnglePreset.PROFILE_LEFT: (80.0, 0.0),
    AnglePreset.PROFILE_RIGHT: (-80.0, 0.0),
    AnglePreset.HIGH_FRONT: (0.0, 18.0),
    AnglePreset.LOW_FRONT: (0.0, -12.0),
}

SHOT_HEIGHT_FRACTIONS = {
    ShotSizePreset.FAR: 0.46,
    ShotSizePreset.MEDIUM: 0.66,
    ShotSizePreset.CLOSE: 0.84,
}


SHOT_HEIGHT_REQUEST_OVERRIDES = {
    (TargetBodyPreset.ABDOMEN, ShotSizePreset.FAR): 0.98,
    (TargetBodyPreset.ABDOMEN, ShotSizePreset.MEDIUM): 0.74,
    (TargetBodyPreset.ABDOMEN, ShotSizePreset.CLOSE): 0.84,
}

SHOT_FOV_DEGREES = {
    ShotSizePreset.FAR: 34.0,
    ShotSizePreset.MEDIUM: 30.0,
    ShotSizePreset.CLOSE: 24.0,
}

COMPOSITION_TARGETS = {
    CompositionPreset.CENTER: (0.0, 0.0),
    CompositionPreset.LEFT_THIRD: (-0.35, 0.0),
    CompositionPreset.RIGHT_THIRD: (0.35, 0.0),
}


DISTANCE_SCALES = {
    (TargetBodyPreset.FACE, ShotSizePreset.FAR): 1.32,
    (TargetBodyPreset.FACE, ShotSizePreset.CLOSE): 1.12,
    (TargetBodyPreset.FACE, ShotSizePreset.MEDIUM): 1.13,
    (TargetBodyPreset.BUST_UP, ShotSizePreset.FAR): 0.96,
    (TargetBodyPreset.BUST_UP, ShotSizePreset.MEDIUM): 0.90,
    (TargetBodyPreset.BUST_UP, ShotSizePreset.CLOSE): 0.96,
    (TargetBodyPreset.WAIST_UP, ShotSizePreset.FAR): 1.07,
    (TargetBodyPreset.WAIST_UP, ShotSizePreset.MEDIUM): 1.07,
    (TargetBodyPreset.WAIST_UP, ShotSizePreset.CLOSE): 1.07,
    (TargetBodyPreset.KNEE_UP, ShotSizePreset.FAR): 0.95,
    (TargetBodyPreset.KNEE_UP, ShotSizePreset.MEDIUM): 0.95,
    (TargetBodyPreset.KNEE_UP, ShotSizePreset.CLOSE): 0.95,
    (TargetBodyPreset.ABDOMEN, ShotSizePreset.MEDIUM): 1.25,
    (TargetBodyPreset.ABDOMEN, ShotSizePreset.CLOSE): 0.90,
    (TargetBodyPreset.FEET, ShotSizePreset.FAR): 1.06,
    (TargetBodyPreset.FEET, ShotSizePreset.MEDIUM): 1.02,
    (TargetBodyPreset.FEET, ShotSizePreset.CLOSE): 0.98,
}


INTEREST_Y_SCALES = {
    TargetBodyPreset.FEET: -0.04,
    TargetBodyPreset.KNEE_UP: 0.12,
    TargetBodyPreset.ABDOMEN: 0.02,
    TargetBodyPreset.WAIST_UP: 0.04,
    TargetBodyPreset.BUST_UP: 0.06,
}


INTEREST_Y_REQUEST_SCALES = {
    (TargetBodyPreset.ABDOMEN, ShotSizePreset.FAR): 0.05,
    (TargetBodyPreset.ABDOMEN, ShotSizePreset.MEDIUM): 0.04,
    (TargetBodyPreset.ABDOMEN, ShotSizePreset.CLOSE): 0.04,
    (TargetBodyPreset.BUST_UP, ShotSizePreset.MEDIUM): 0.11,
    (TargetBodyPreset.BUST_UP, ShotSizePreset.CLOSE): 0.10,
    (TargetBodyPreset.FACE, ShotSizePreset.FAR): 0.05,
    (TargetBodyPreset.FACE, ShotSizePreset.MEDIUM): 0.06,
    (TargetBodyPreset.FACE, ShotSizePreset.CLOSE): 0.07,
}


COMPOSITION_Y_REQUEST_TARGETS = {
    (TargetBodyPreset.ABDOMEN, ShotSizePreset.MEDIUM): 0.0,
    (TargetBodyPreset.ABDOMEN, ShotSizePreset.FAR): 0.14,
}


ROLL_OFFSETS_DEG = {
    RollPreset.NONE: 0.0,
    RollPreset.SLIGHT_LEFT: 5.0,
    RollPreset.LEFT: 18.0,
    RollPreset.STRONG_LEFT: 45.0,
    RollPreset.SLIGHT_RIGHT: -5.0,
    RollPreset.RIGHT: -18.0,
    RollPreset.STRONG_RIGHT: -45.0,
}


@dataclass(frozen=True)
class CameraSolveConfig:
    aspect_ratio: float = 16.0 / 9.0
    min_distance: float = 1.0
    max_distance: float = 500.0
    fit_iterations: int = 40
    perspective: bool = True
    synthesize_start_keyframe: bool = True
    timeline_start_frame: int = 0
    start_camera_interest: Vec3 = (0.0, 10.0, 0.0)
    start_camera_rotate: Vec3 = (0.0, 0.0, 0.0)
    start_camera_distance: float = 50.0
    start_camera_fov_deg: int = 30
    start_camera_transition: TransitionPreset = TransitionPreset.LINEAR
    medium_distance_multiplier: float = 1.0
    close_distance_multiplier: float = 1.0
    full_body_close_distance_multiplier: float = 1.0
    legs_close_distance_multiplier: float = 1.0


@dataclass(frozen=True)
class SolvedCameraKey:
    frame: int
    interest: Vec3
    rotate: Vec3
    distance: float
    view_angle_deg: int
    perspective: bool
    transition_to_next: TransitionPreset
    debug_eye: Vec3
    debug_body_bounds: Bounds3D
    debug_shot_height_fraction: float


def _unwrap_angle(prev: float, current: float) -> float:
    value = float(current)
    while value - prev > math.pi:
        value -= 2.0 * math.pi
    while value - prev < -math.pi:
        value += 2.0 * math.pi
    return value


def _unwrap_rotation_sequence(keys: list[SolvedCameraKey]) -> list[SolvedCameraKey]:
    if not keys:
        return keys

    unwrapped = [keys[0]]
    prev = keys[0].rotate
    for key in keys[1:]:
        rotate = (
            _unwrap_angle(prev[0], key.rotate[0]),
            _unwrap_angle(prev[1], key.rotate[1]),
            _unwrap_angle(prev[2], key.rotate[2]),
        )
        unwrapped_key = SolvedCameraKey(
            frame=key.frame,
            interest=key.interest,
            rotate=rotate,
            distance=key.distance,
            view_angle_deg=key.view_angle_deg,
            perspective=key.perspective,
            transition_to_next=key.transition_to_next,
            debug_eye=key.debug_eye,
            debug_body_bounds=key.debug_body_bounds,
            debug_shot_height_fraction=key.debug_shot_height_fraction,
        )
        unwrapped.append(unwrapped_key)
        prev = rotate
    return unwrapped


def _camera_basis(eye: Vec3, interest: Vec3) -> tuple[Vec3, Vec3, Vec3]:
    forward = normalize(sub(interest, eye))
    right = normalize(cross(forward, WORLD_UP), (1.0, 0.0, 0.0))
    up = normalize(cross(right, forward), WORLD_UP)
    return right, up, forward


def _project_bounds(bounds: Bounds3D, eye: Vec3, interest: Vec3, fov_deg: float, aspect_ratio: float) -> dict[str, float]:
    right, up, forward = _camera_basis(eye, interest)
    tan_half = math.tan(math.radians(fov_deg) * 0.5)

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for point in bounds.corners():
        rel = sub(point, eye)
        cam_x = dot(rel, right)
        cam_y = dot(rel, up)
        cam_z = dot(rel, forward)
        if cam_z <= 1e-5:
            return {
                "min_x": -1e9,
                "max_x": 1e9,
                "min_y": -1e9,
                "max_y": 1e9,
                "center_x": 0.0,
                "center_y": 0.0,
                "width_frac": 1e9,
                "height_frac": 1e9,
                "focus_depth": 1.0,
            }
        xs.append(cam_x / (tan_half * aspect_ratio * cam_z))
        ys.append(cam_y / (tan_half * cam_z))
        zs.append(cam_z)

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    return {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "center_x": (min_x + max_x) * 0.5,
        "center_y": (min_y + max_y) * 0.5,
        "width_frac": (max_x - min_x) * 0.5,
        "height_frac": (max_y - min_y) * 0.5,
        "focus_depth": sum(zs) / len(zs),
    }


def _fit_distance(
    bounds: Bounds3D,
    interest: Vec3,
    eye_direction: Vec3,
    fov_deg: float,
    shot_height_fraction: float,
    config: CameraSolveConfig,
) -> float:
    target_height = shot_height_fraction
    target_width = min(0.92, shot_height_fraction * 0.8)

    lo = config.min_distance
    hi = config.max_distance
    for _ in range(config.fit_iterations):
        mid = (lo + hi) * 0.5
        eye = add(interest, scale(eye_direction, mid))
        projected = _project_bounds(bounds, eye, interest, fov_deg, config.aspect_ratio)
        too_big = (
            projected["height_frac"] > target_height
            or projected["width_frac"] > target_width
        )
        if too_big:
            lo = mid
        else:
            hi = mid
    return hi


def _apply_composition(
    eye: Vec3,
    interest: Vec3,
    bounds: Bounds3D,
    fov_deg: float,
    request: ShotKeyframeRequest,
    aspect_ratio: float,
) -> tuple[Vec3, Vec3]:
    desired_x, desired_y = COMPOSITION_TARGETS[request.composition]
    desired_y = COMPOSITION_Y_REQUEST_TARGETS.get(
        (request.target_body, request.shot_size),
        desired_y,
    )
    if request.target_body is TargetBodyPreset.FULL_BODY:
        if request.angle is AnglePreset.HIGH_FRONT and request.shot_size in (ShotSizePreset.MEDIUM, ShotSizePreset.FAR):
            desired_y = max(desired_y, 0.10)
        elif request.angle is AnglePreset.TOP and request.shot_size in (ShotSizePreset.MEDIUM, ShotSizePreset.FAR):
            desired_y = max(desired_y, 0.18)
    projected = _project_bounds(bounds, eye, interest, fov_deg, aspect_ratio)
    delta_x = desired_x - projected["center_x"]
    delta_y = desired_y - projected["center_y"]

    if abs(delta_x) <= 1e-5 and abs(delta_y) <= 1e-5:
        return eye, interest

    right, up, _forward = _camera_basis(eye, interest)
    focus_depth = max(1e-4, projected["focus_depth"])
    half_h = math.tan(math.radians(fov_deg) * 0.5) * focus_depth
    half_w = half_h * aspect_ratio

    shift = add(scale(right, delta_x * half_w), scale(up, delta_y * half_h))
    return add(eye, shift), add(interest, shift)


def _mmd_rotate_from_eye_interest(eye: Vec3, interest: Vec3) -> Vec3:
    offset = sub(eye, interest)
    horizontal = math.sqrt(offset[0] * offset[0] + offset[2] * offset[2])
    pitch = math.atan2(-offset[1], max(1e-8, horizontal))
    yaw = math.atan2(offset[0], offset[2])
    roll = 0.0
    return (pitch, yaw, roll)


def _roll_rad_for_request(request: ShotKeyframeRequest) -> float:
    return math.radians(ROLL_OFFSETS_DEG[request.roll])


def _distance_scale_for_request(request: ShotKeyframeRequest, config: CameraSolveConfig) -> float:
    scale = DISTANCE_SCALES.get((request.target_body, request.shot_size), 1.0)
    if request.shot_size is ShotSizePreset.MEDIUM:
        scale *= config.medium_distance_multiplier
    elif request.shot_size is ShotSizePreset.CLOSE:
        scale *= config.close_distance_multiplier
        if request.target_body is TargetBodyPreset.FULL_BODY:
            scale *= config.full_body_close_distance_multiplier
        elif request.target_body is TargetBodyPreset.LEGS:
            scale *= config.legs_close_distance_multiplier
    return scale


def _interest_for_request(body_bounds: Bounds3D, request: ShotKeyframeRequest) -> Vec3:
    center = body_bounds.center()
    y_scale = INTEREST_Y_REQUEST_SCALES.get(
        (request.target_body, request.shot_size),
        INTEREST_Y_SCALES.get(request.target_body, 0.0),
    )
    if abs(y_scale) <= 1e-8:
        return center
    size_y = body_bounds.size()[1]
    return (
        center[0],
        center[1] + size_y * y_scale,
        center[2],
    )


def _abdomen_medium_fit_bounds(full_body: Bounds3D) -> Bounds3D:
    min_x, min_y, min_z = full_body.min
    max_x, _max_y, max_z = full_body.max
    size_x, size_y, size_z = full_body.size()
    center_x = (min_x + max_x) * 0.5
    center_z = (min_z + max_z) * 0.5
    y0 = min_y + size_y * 0.10
    y1 = min_y + size_y * 0.76
    half_x = size_x * 0.30
    half_z = size_z * 0.20
    return Bounds3D(
        (center_x - half_x, y0, center_z - half_z),
        (center_x + half_x, y1, center_z + half_z),
    )


def _fit_bounds_for_request(context, request: ShotKeyframeRequest) -> Bounds3D:
    if request.target_body is not TargetBodyPreset.ABDOMEN:
        return context.bounds_for(request.target_body)

    full_body = context.bounds_for(TargetBodyPreset.FULL_BODY)
    if request.shot_size is ShotSizePreset.FAR:
        return full_body
    if request.shot_size is ShotSizePreset.MEDIUM:
        return context.bounds_for(TargetBodyPreset.ABDOMEN)
    return context.bounds_for(TargetBodyPreset.ABDOMEN)


def _interest_bounds_for_request(context, request: ShotKeyframeRequest) -> Bounds3D:
    if request.target_body is TargetBodyPreset.ABDOMEN:
        return context.bounds_for(TargetBodyPreset.ABDOMEN)
    return context.bounds_for(request.target_body)


def _shot_height_fraction_for_request(request: ShotKeyframeRequest) -> float:
    return SHOT_HEIGHT_REQUEST_OVERRIDES.get(
        (request.target_body, request.shot_size),
        SHOT_HEIGHT_FRACTIONS[request.shot_size],
    )


def _eye_from_mmd_camera(interest: Vec3, rotate: Vec3, distance: float) -> Vec3:
    pitch, yaw, _roll = rotate
    abs_distance = abs(distance)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    sin_yaw = math.sin(yaw)
    cos_yaw = math.cos(yaw)
    offset = (
        abs_distance * sin_yaw * cos_pitch,
        -abs_distance * sin_pitch,
        abs_distance * cos_yaw * cos_pitch,
    )
    return add(interest, offset)


def solve_camera_keyframe(
    provider: FrameTargetProvider,
    request: ShotKeyframeRequest,
    config: CameraSolveConfig | None = None,
) -> SolvedCameraKey:
    config = config or CameraSolveConfig()
    context = provider.get_context(request.frame)
    fit_bounds = _fit_bounds_for_request(context, request)
    interest_bounds = _interest_bounds_for_request(context, request)

    view_angle_deg = int(round(SHOT_FOV_DEGREES[request.shot_size]))
    shot_height_fraction = _shot_height_fraction_for_request(request)
    yaw_deg, pitch_deg = ANGLE_OFFSETS[request.angle]

    interest = _interest_for_request(interest_bounds, request)
    # Top shots should stay locked to the stage/world instead of orbiting with
    # the actor as the character turns.
    base_forward = WORLD_FORWARD if request.angle is AnglePreset.TOP else context.actor_forward
    eye_direction = direction_from_yaw_pitch(base_forward, yaw_deg, pitch_deg)
    distance = _fit_distance(fit_bounds, interest, eye_direction, view_angle_deg, shot_height_fraction, config)
    distance *= _distance_scale_for_request(request, config)
    eye = add(interest, scale(eye_direction, distance))
    eye, interest = _apply_composition(eye, interest, fit_bounds, view_angle_deg, request, config.aspect_ratio)

    pitch, yaw, _roll = _mmd_rotate_from_eye_interest(eye, interest)
    rotate = (pitch, yaw, _roll_rad_for_request(request))
    return SolvedCameraKey(
        frame=request.frame,
        interest=interest,
        rotate=rotate,
        distance=length(sub(eye, interest)),
        view_angle_deg=view_angle_deg,
        perspective=config.perspective,
        transition_to_next=request.transition_to_next,
        debug_eye=eye,
        debug_body_bounds=fit_bounds,
        debug_shot_height_fraction=shot_height_fraction,
    )


def solve_camera_timeline(
    provider: FrameTargetProvider,
    requests: list[ShotKeyframeRequest],
    config: CameraSolveConfig | None = None,
) -> list[SolvedCameraKey]:
    config = config or CameraSolveConfig()
    ordered = sorted(requests, key=lambda item: item.frame)
    if (
        ordered
        and config.synthesize_start_keyframe
        and ordered[0].frame > config.timeline_start_frame
    ):
        start_context = provider.get_context(config.timeline_start_frame)
        ordered = [
            ShotKeyframeRequest(
                frame=config.timeline_start_frame,
                target_body=ordered[0].target_body,
                shot_size=ordered[0].shot_size,
                angle=ordered[0].angle,
                composition=ordered[0].composition,
                roll=ordered[0].roll,
                transition_to_next=config.start_camera_transition,
            )
        ] + ordered

        default_key = SolvedCameraKey(
            frame=config.timeline_start_frame,
            interest=config.start_camera_interest,
            rotate=config.start_camera_rotate,
            distance=config.start_camera_distance,
            view_angle_deg=config.start_camera_fov_deg,
            perspective=config.perspective,
            transition_to_next=config.start_camera_transition,
            debug_eye=_eye_from_mmd_camera(
                config.start_camera_interest,
                config.start_camera_rotate,
                config.start_camera_distance,
            ),
            debug_body_bounds=start_context.bounds_for(TargetBodyPreset.FULL_BODY),
            debug_shot_height_fraction=SHOT_HEIGHT_FRACTIONS[ordered[1].shot_size],
        )
        return _unwrap_rotation_sequence(
            [default_key] + [solve_camera_keyframe(provider, request, config) for request in ordered[1:]]
        )

    return _unwrap_rotation_sequence(
        [solve_camera_keyframe(provider, request, config) for request in ordered]
    )
