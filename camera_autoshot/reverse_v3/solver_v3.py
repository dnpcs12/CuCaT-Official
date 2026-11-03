from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from camera_autoshot.geometry import Vec3, add, cross, direction_from_yaw_pitch, dot, length, normalize, scale, sub
from camera_autoshot.providers import Bounds3D, FrameTargetContext, FrameTargetProvider, JsonFrameTargetProvider
from camera_autoshot.spec import CompositionPreset, RollPreset, TargetBodyPreset, TransitionPreset
from camera_autoshot.vmd import write_camera_vmd

from .spec_v3 import (
    DistancePreset,
    FovPreset,
    InterpolationPreset,
    PitchPreset,
    ShotKeyframeRequestV3,
    TargetAnchorPreset,
    YawPreset,
)


WORLD_UP: Vec3 = (0.0, 1.0, 0.0)
WORLD_FORWARD: Vec3 = (0.0, 0.0, 1.0)


YAW_OFFSETS_DEG = {
    YawPreset.FRONT: 0.0,
    YawPreset.LEFT_45: 45.0,
    YawPreset.PROFILE_LEFT: 90.0,
    YawPreset.BACK_LEFT_45: 135.0,
    YawPreset.BACK: 180.0,
    YawPreset.BACK_RIGHT_45: -135.0,
    YawPreset.PROFILE_RIGHT: -90.0,
    YawPreset.RIGHT_45: -45.0,
}


PITCH_OFFSETS_DEG = {
    PitchPreset.LOW: -12.0,
    PitchPreset.LEVEL: 0.0,
    PitchPreset.HIGH: 18.0,
    PitchPreset.TOP: 72.0,
}


FOV_DEGREES = {
    FovPreset.NARROW: 22.0,
    FovPreset.NORMAL: 30.0,
    FovPreset.WIDE: 40.0,
}


# Physical distance is intentionally decoupled from target_anchor.  These are
# multiples of full-body height, so each distance token means roughly the same
# camera offset for every anchor.
DISTANCE_HEIGHT_MULTIPLIERS = {
    DistancePreset.EXTREME_CLOSE: 0.80,
    DistancePreset.CLOSE: 1.30,
    DistancePreset.MEDIUM_CLOSE: 1.90,
    DistancePreset.MEDIUM: 2.70,
    DistancePreset.MEDIUM_FAR: 3.90,
    DistancePreset.FAR: 5.50,
    DistancePreset.WIDE: 7.50,
}


TARGET_ANCHOR_Y_FRACTIONS = {
    TargetAnchorPreset.FEET: 0.05,
    TargetAnchorPreset.LOWER_LEG: 0.22,
    TargetAnchorPreset.THIGH: 0.38,
    TargetAnchorPreset.HIP: 0.51,
    TargetAnchorPreset.WAIST: 0.63,
    TargetAnchorPreset.CHEST: 0.76,
    # Coarse-anchor ablation bins.  Keep the original fine-grained anchors
    # intact; these are only used by labels that explicitly opt into them.
    TargetAnchorPreset.LEGS: 0.30,
    TargetAnchorPreset.TORSO: 0.64,
    TargetAnchorPreset.FACE: 0.90,
}


COMPOSITION_TARGETS = {
    CompositionPreset.LEFT_THIRD: 0.35,
    CompositionPreset.CENTER: 0.0,
    CompositionPreset.RIGHT_THIRD: -0.35,
}


ROLL_OFFSETS_DEG = {
    RollPreset.NONE: 0.0,
    RollPreset.SLIGHT_LEFT: 8.0,
    RollPreset.LEFT: 18.0,
    RollPreset.STRONG_LEFT: 45.0,
    RollPreset.SLIGHT_RIGHT: -8.0,
    RollPreset.RIGHT: -18.0,
    RollPreset.STRONG_RIGHT: -45.0,
}


@dataclass(frozen=True)
class CameraSolveConfigV3:
    aspect_ratio: float = 16.0 / 9.0
    min_distance: float = 1.0
    max_distance: float = 500.0
    perspective: bool = True
    synthesize_start_keyframe: bool = True
    timeline_start_frame: int = 0
    start_camera_interest: Vec3 = (0.0, 10.0, 0.0)
    start_camera_rotate: Vec3 = (0.0, 0.0, 0.0)
    start_camera_distance: float = 50.0
    start_camera_fov_deg: int = 30
    start_camera_transition: TransitionPreset = TransitionPreset.LINEAR
    top_uses_world_forward: bool = True
    yaw_uses_world_forward: bool = False
    yaw_offset_deg: float = 0.0
    vmd_distance_sign: float = -1.0
    composition_y_centering: bool = True
    # Optional calibration hook for experiments that need to nudge a target
    # anchor vertically only for specific semantic shot conditions.  Each tuple
    # is (target_anchor_value, distance_value, full_body_height_fraction_delta).
    anchor_y_calibration_offsets: tuple[tuple[str, str, float], ...] = ()
    # Optional per-anchor distance calibration.  Each tuple is
    # (target_anchor_value, distance_value, body_height_multiplier_delta).
    distance_multiplier_calibration_offsets: tuple[tuple[str, str, float], ...] = ()


@dataclass(frozen=True)
class SolvedCameraKeyV3:
    frame: int
    interest: Vec3
    rotate: Vec3
    distance: float
    view_angle_deg: int
    perspective: bool
    transition_to_next: TransitionPreset
    interpolation_to_next: InterpolationPreset
    debug_eye: Vec3
    debug_body_bounds: Bounds3D
    debug_target_anchor: str
    debug_distance_multiplier: float


def _unwrap_angle(prev: float, current: float) -> float:
    value = float(current)
    while value - prev > math.pi:
        value -= 2.0 * math.pi
    while value - prev < -math.pi:
        value += 2.0 * math.pi
    return value


def _unwrap_rotation_sequence(keys: list[SolvedCameraKeyV3]) -> list[SolvedCameraKeyV3]:
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
        unwrapped.append(
            SolvedCameraKeyV3(
                frame=key.frame,
                interest=key.interest,
                rotate=rotate,
                distance=key.distance,
                view_angle_deg=key.view_angle_deg,
                perspective=key.perspective,
                transition_to_next=key.transition_to_next,
                interpolation_to_next=key.interpolation_to_next,
                debug_eye=key.debug_eye,
                debug_body_bounds=key.debug_body_bounds,
                debug_target_anchor=key.debug_target_anchor,
                debug_distance_multiplier=key.debug_distance_multiplier,
            )
        )
        prev = rotate
    return unwrapped


def _full_body_bounds(context: FrameTargetContext) -> Bounds3D:
    return context.bounds_for(TargetBodyPreset.FULL_BODY)


def _anchor_point(
    context: FrameTargetContext,
    anchor: TargetAnchorPreset,
    y_offset_fraction: float = 0.0,
) -> Vec3:
    full = _full_body_bounds(context)
    size_y = full.size()[1]
    y_offset = size_y * y_offset_fraction
    if anchor is TargetAnchorPreset.FACE:
        face = context.bounds_for(TargetBodyPreset.FACE)
        center = face.center()
        return (center[0], center[1] + y_offset, center[2])

    min_x, min_y, min_z = full.min
    max_x, _max_y, max_z = full.max
    size_y = full.size()[1]
    frac = TARGET_ANCHOR_Y_FRACTIONS[anchor]
    return ((min_x + max_x) * 0.5, min_y + size_y * frac + y_offset, (min_z + max_z) * 0.5)


def _anchor_bounds(
    context: FrameTargetContext,
    anchor: TargetAnchorPreset,
    y_offset_fraction: float = 0.0,
) -> Bounds3D:
    if anchor is TargetAnchorPreset.FACE:
        face = context.bounds_for(TargetBodyPreset.FACE)
        full = _full_body_bounds(context)
        y_offset = full.size()[1] * y_offset_fraction
        return Bounds3D(
            (face.min[0], face.min[1] + y_offset, face.min[2]),
            (face.max[0], face.max[1] + y_offset, face.max[2]),
        )

    full = _full_body_bounds(context)
    center = _anchor_point(context, anchor, y_offset_fraction)
    size_x, size_y, size_z = full.size()
    half_x = max(size_x * 0.18, 0.01)
    half_y = max(size_y * 0.045, 0.01)
    half_z = max(size_z * 0.18, 0.01)
    return Bounds3D(
        (center[0] - half_x, center[1] - half_y, center[2] - half_z),
        (center[0] + half_x, center[1] + half_y, center[2] + half_z),
    )


def _anchor_y_offset_fraction(request: ShotKeyframeRequestV3, config: CameraSolveConfigV3) -> float:
    """Return a vertical anchor calibration delta for this request, if configured."""

    anchor_value = request.target_anchor.value
    distance_value = request.distance.value
    for target_anchor, distance, offset in config.anchor_y_calibration_offsets:
        if anchor_value == str(target_anchor) and distance_value == str(distance):
            return float(offset)
    return 0.0


def _distance_multiplier_offset(request: ShotKeyframeRequestV3, config: CameraSolveConfigV3) -> float:
    """Return a per-anchor distance multiplier delta for this request."""

    anchor_value = request.target_anchor.value
    distance_value = request.distance.value
    total = 0.0
    for target_anchor, distance, offset in config.distance_multiplier_calibration_offsets:
        if anchor_value == str(target_anchor) and distance_value == str(distance):
            total += float(offset)
    return total


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
        "center_x": (min_x + max_x) * 0.5,
        "center_y": (min_y + max_y) * 0.5,
        "width_frac": (max_x - min_x) * 0.5,
        "height_frac": (max_y - min_y) * 0.5,
        "focus_depth": sum(zs) / len(zs),
    }


def _apply_composition(
    eye: Vec3,
    interest: Vec3,
    anchor_bounds: Bounds3D,
    fov_deg: float,
    request: ShotKeyframeRequestV3,
    aspect_ratio: float,
    composition_y_centering: bool = True,
) -> tuple[Vec3, Vec3]:
    desired_x = COMPOSITION_TARGETS[request.composition]
    projected = _project_bounds(anchor_bounds, eye, interest, fov_deg, aspect_ratio)
    # Moving both eye and interest right makes the subject project left, so the
    # world-space dolly offset must use the inverse of the screen-space error.
    delta_x = projected["center_x"] - desired_x
    delta_y = projected["center_y"] if composition_y_centering else 0.0
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
    return (pitch, yaw, 0.0)


def _roll_rad_for_request(request: ShotKeyframeRequestV3) -> float:
    return math.radians(ROLL_OFFSETS_DEG[request.roll])


def _eye_from_mmd_camera(interest: Vec3, rotate: Vec3, distance: float) -> Vec3:
    pitch, yaw, _roll = rotate
    abs_distance = abs(distance)
    cos_pitch = math.cos(pitch)
    offset = (
        abs_distance * math.sin(yaw) * cos_pitch,
        -abs_distance * math.sin(pitch),
        abs_distance * math.cos(yaw) * cos_pitch,
    )
    return add(interest, offset)


def solve_camera_keyframe_v3(
    provider: FrameTargetProvider,
    request: ShotKeyframeRequestV3,
    config: CameraSolveConfigV3 | None = None,
) -> SolvedCameraKeyV3:
    config = config or CameraSolveConfigV3()
    context = provider.get_context(request.frame)
    full = _full_body_bounds(context)
    body_height = max(full.size()[1], 1e-4)
    distance_multiplier = DISTANCE_HEIGHT_MULTIPLIERS[request.distance] + _distance_multiplier_offset(request, config)
    distance_multiplier = max(0.05, distance_multiplier)
    view_angle_deg = int(round(FOV_DEGREES[request.fov]))

    anchor_y_offset = _anchor_y_offset_fraction(request, config)
    interest = _anchor_point(context, request.target_anchor, anchor_y_offset)
    yaw_deg = YAW_OFFSETS_DEG[request.yaw] + float(config.yaw_offset_deg)
    pitch_deg = PITCH_OFFSETS_DEG[request.pitch]
    base_forward = (
        WORLD_FORWARD
        if config.yaw_uses_world_forward or (config.top_uses_world_forward and request.pitch is PitchPreset.TOP)
        else context.actor_forward
    )
    eye_direction = direction_from_yaw_pitch(base_forward, yaw_deg, pitch_deg)
    distance = max(config.min_distance, min(config.max_distance, body_height * distance_multiplier))
    eye = add(interest, scale(eye_direction, distance))

    fit_bounds = _anchor_bounds(context, request.target_anchor, anchor_y_offset)
    eye, interest = _apply_composition(
        eye,
        interest,
        fit_bounds,
        view_angle_deg,
        request,
        config.aspect_ratio,
        config.composition_y_centering,
    )
    pitch, yaw, _roll = _mmd_rotate_from_eye_interest(eye, interest)
    rotate = (pitch, yaw, _roll_rad_for_request(request))
    return SolvedCameraKeyV3(
        frame=request.frame,
        interest=interest,
        rotate=rotate,
        distance=length(sub(eye, interest)) * config.vmd_distance_sign,
        view_angle_deg=view_angle_deg,
        perspective=config.perspective,
        transition_to_next=request.transition_to_next,
        interpolation_to_next=request.interpolation,
        debug_eye=eye,
        debug_body_bounds=fit_bounds,
        debug_target_anchor=request.target_anchor.value,
        debug_distance_multiplier=distance_multiplier,
    )


def solve_camera_timeline_v3(
    provider: FrameTargetProvider,
    requests: list[ShotKeyframeRequestV3],
    config: CameraSolveConfigV3 | None = None,
) -> list[SolvedCameraKeyV3]:
    config = config or CameraSolveConfigV3()
    ordered = sorted(requests, key=lambda item: item.frame)
    if ordered and config.synthesize_start_keyframe and ordered[0].frame > config.timeline_start_frame:
        start_context = provider.get_context(config.timeline_start_frame)
        default_key = SolvedCameraKeyV3(
            frame=config.timeline_start_frame,
            interest=config.start_camera_interest,
            rotate=config.start_camera_rotate,
            distance=config.start_camera_distance * config.vmd_distance_sign,
            view_angle_deg=config.start_camera_fov_deg,
            perspective=config.perspective,
            transition_to_next=config.start_camera_transition,
            interpolation_to_next=InterpolationPreset.LINEAR,
            debug_eye=_eye_from_mmd_camera(
                config.start_camera_interest,
                config.start_camera_rotate,
                config.start_camera_distance,
            ),
            debug_body_bounds=start_context.bounds_for(TargetBodyPreset.FULL_BODY),
            debug_target_anchor=ordered[0].target_anchor.value,
            debug_distance_multiplier=DISTANCE_HEIGHT_MULTIPLIERS[ordered[0].distance],
        )
        solved = [default_key] + [solve_camera_keyframe_v3(provider, request, config) for request in ordered]
        return _unwrap_rotation_sequence(solved)
    return _unwrap_rotation_sequence([solve_camera_keyframe_v3(provider, request, config) for request in ordered])


def _read_requests(path: Path) -> list[ShotKeyframeRequestV3]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("keyframes", [])
    try:
        from .close_dense_schema import (
            apply_close_dense_distance_schema_from_payload,
            internalize_close_dense_row,
            uses_close_dense_distance_schema,
        )

        if uses_close_dense_distance_schema(raw):
            apply_close_dense_distance_schema_from_payload(raw)
            rows = [internalize_close_dense_row(item) if isinstance(item, dict) else item for item in rows]
    except Exception:
        # Keep the legacy solver usable even if the optional V3.1 close-dense
        # helper is unavailable.
        pass
    return [ShotKeyframeRequestV3.from_dict(item) for item in rows]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve a Reverse V3 shot-plan JSON into a camera VMD.")
    parser.add_argument("--plan", required=True, help="Input V3 camera_poses JSON.")
    parser.add_argument("--targets", required=True, help="Target dump JSON with body bounds and actor forward.")
    parser.add_argument("--output-vmd", required=True, help="Output camera VMD path.")
    parser.add_argument("--debug-json", default="", help="Optional solved debug JSON path.")
    parser.add_argument("--no-start-keyframe", action="store_true", help="Disable synthetic frame-0 start key.")
    parser.add_argument(
        "--yaw-mode",
        choices=("actor", "world"),
        default="actor",
        help="Yaw reference frame. actor preserves V3/V3.1; world is used by V3.2.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    provider = JsonFrameTargetProvider(Path(args.targets))
    config = CameraSolveConfigV3(
        synthesize_start_keyframe=not args.no_start_keyframe,
        yaw_uses_world_forward=(args.yaw_mode == "world"),
    )
    requests = _read_requests(Path(args.plan))
    solved = solve_camera_timeline_v3(provider, requests, config)
    write_camera_vmd(solved, Path(args.output_vmd), model_name="Autoshot V3 Camera")

    if args.debug_json:
        debug_path = Path(args.debug_json)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(json.dumps([asdict(key) for key in solved], ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote V3 camera VMD: {args.output_vmd}")
    print(f"keyframe_count={len(solved)}")


if __name__ == "__main__":
    main()
