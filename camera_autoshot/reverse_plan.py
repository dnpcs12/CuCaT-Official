from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .providers import FrameTargetContext, JsonFrameTargetProvider
from .solver import (
    ANGLE_OFFSETS,
    CameraSolveConfig,
    _distance_scale_for_request,
    _eye_from_mmd_camera,
    _fit_bounds_for_request,
    _interest_for_request,
    _interest_bounds_for_request,
    _project_bounds,
    _shot_height_fraction_for_request,
)
from .spec import (
    AnglePreset,
    CompositionPreset,
    ShotKeyframeRequest,
    ShotSizePreset,
    TargetBodyPreset,
    TransitionPreset,
)
from .vmd import CameraVmdKey, classify_camera_transition, read_camera_vmd


@dataclass(frozen=True)
class InferredPlanKey:
    frame: int
    target_body: str
    shot_size: str
    angle: str
    composition: str
    transition: str
    camera_reason: str


# NOTE:
# The current forward autoshot solver applies composition with a screen-x sign
# that is effectively mirrored from the intuitive "left/right of the frame"
# wording. For reverse inference we intentionally infer the label that
# reconstructs the observed screen placement when fed back into the current
# solver, so reconstructed videos match the original camera more closely.
RECONSTRUCTION_COMPOSITION_TARGETS = {
    CompositionPreset.CENTER: 0.0,
    CompositionPreset.LEFT_THIRD: 0.35,
    CompositionPreset.RIGHT_THIRD: -0.35,
}


INFER_BODY_CANDIDATES = [
    TargetBodyPreset.FULL_BODY,
    TargetBodyPreset.LEGS,
    TargetBodyPreset.KNEE_UP,
    TargetBodyPreset.ABDOMEN,
    TargetBodyPreset.WAIST_UP,
    TargetBodyPreset.BUST_UP,
    TargetBodyPreset.FACE,
]


def _request_for_inference(frame: int, body: TargetBodyPreset, size: ShotSizePreset) -> ShotKeyframeRequest:
    return ShotKeyframeRequest(
        frame=frame,
        target_body=body,
        shot_size=size,
        angle=AnglePreset.FRONT,
        composition=CompositionPreset.CENTER,
        transition_to_next=TransitionPreset.LINEAR,
    )


def _expected_height_fraction(request: ShotKeyframeRequest) -> float:
    scale = _distance_scale_for_request(request, CameraSolveConfig())
    return _shot_height_fraction_for_request(request) / max(scale, 1e-8)


def _expected_width_fraction(request: ShotKeyframeRequest) -> float:
    scale = _distance_scale_for_request(request, CameraSolveConfig())
    return min(0.92, _shot_height_fraction_for_request(request) * 0.8) / max(scale, 1e-8)


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
        transition = classify_camera_transition(key.interpolation)
        collapsed.append((key, transition))
        index += 1
    return collapsed


def _infer_angle(context: FrameTargetContext, eye, interest) -> AnglePreset:
    offset = (
        eye[0] - interest[0],
        eye[1] - interest[1],
        eye[2] - interest[2],
    )
    length = math.sqrt(offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2])
    if length <= 1e-8:
        return AnglePreset.FRONT
    direction = (offset[0] / length, offset[1] / length, offset[2] / length)

    best = AnglePreset.FRONT
    best_dot = -1e9
    dot_by_preset: dict[AnglePreset, float] = {}
    for preset, (yaw_deg, pitch_deg) in ANGLE_OFFSETS.items():
        # Reuse the same canonical direction convention as the forward solver.
        from .geometry import direction_from_yaw_pitch

        candidate = direction_from_yaw_pitch(context.actor_forward, yaw_deg, pitch_deg)
        dot = direction[0] * candidate[0] + direction[1] * candidate[1] + direction[2] * candidate[2]
        dot_by_preset[preset] = dot
        if dot > best_dot:
            best_dot = dot
            best = preset

    front_dot = dot_by_preset.get(AnglePreset.FRONT, -1e9)
    high_dot = dot_by_preset.get(AnglePreset.HIGH_FRONT, -1e9)
    low_dot = dot_by_preset.get(AnglePreset.LOW_FRONT, -1e9)
    top_dot = dot_by_preset.get(AnglePreset.TOP, -1e9)

    # Favor vertical-angle presets when the view direction clearly has a
    # vertical component and the front-facing preset only wins by a tiny
    # margin. This helps recover "head-to-legs" high/low shots that otherwise
    # collapse into plain front.
    if direction[1] >= 0.08 and high_dot >= front_dot - 0.02 and high_dot >= best_dot - 0.02:
        best = AnglePreset.HIGH_FRONT
        best_dot = high_dot
    if direction[1] <= -0.08 and low_dot >= front_dot - 0.02 and low_dot >= best_dot - 0.02:
        best = AnglePreset.LOW_FRONT
        best_dot = low_dot
    if direction[1] >= 0.45 and top_dot >= best_dot - 0.08:
        best = AnglePreset.TOP
    return best


def _infer_body_and_size(context: FrameTargetContext, key: CameraVmdKey, eye, aspect_ratio: float) -> tuple[TargetBodyPreset, ShotSizePreset, dict]:
    best_body = TargetBodyPreset.FULL_BODY
    best_size = ShotSizePreset.MEDIUM
    best_projected = None
    best_score = float("inf")
    eye_offset = (
        eye[0] - key.interest[0],
        eye[1] - key.interest[1],
        eye[2] - key.interest[2],
    )
    eye_offset_len = math.sqrt(
        eye_offset[0] * eye_offset[0]
        + eye_offset[1] * eye_offset[1]
        + eye_offset[2] * eye_offset[2]
    )
    vertical_strength = abs(eye_offset[1]) / max(eye_offset_len, 1e-8)
    face_projected = _project_bounds(
        context.bounds_for(TargetBodyPreset.FACE),
        eye,
        key.interest,
        float(key.view_angle_deg),
        aspect_ratio,
    )
    face_visible = (
        face_projected["height_frac"] < 1e8
        and face_projected["height_frac"] >= 0.12
        and face_projected["max_y"] > -0.05
        and face_projected["min_y"] < 0.95
    )

    def center_score_for(bounds, request: ShotKeyframeRequest) -> float:
        center = _interest_for_request(bounds, request)
        size = bounds.size()
        sx = max(abs(size[0]), 1e-4)
        sy = max(abs(size[1]), 1e-4)
        sz = max(abs(size[2]), 1e-4)
        dx = abs(key.interest[0] - center[0]) / sx
        dy = abs(key.interest[1] - center[1]) / sy
        dz = abs(key.interest[2] - center[2]) / sz
        # Match the forward solver's expected look-at point, not the raw bounds
        # center. This is especially important for body regions like bust_up,
        # knee_up, face, and abdomen, where the forward solver intentionally
        # shifts the interest point vertically.
        #
        # Depth (Z) is much less trustworthy than X/Y for reverse classification:
        # angled cameras can change the inferred local depth a lot even when the
        # shot is still clearly targeting the same body region on screen.
        y_weight = 1.25
        if vertical_strength >= 0.35:
            if request.target_body in (
                TargetBodyPreset.BUST_UP,
                TargetBodyPreset.WAIST_UP,
                TargetBodyPreset.KNEE_UP,
                TargetBodyPreset.FULL_BODY,
            ):
                y_weight *= 0.55
            elif request.target_body is TargetBodyPreset.ABDOMEN:
                y_weight *= 0.75
        return dx * 0.35 + dy * y_weight + dz * 0.03

    for body in INFER_BODY_CANDIDATES:
        for size in ShotSizePreset:
            request = _request_for_inference(context.frame, body, size)
            fit_bounds = _fit_bounds_for_request(context, request)
            interest_bounds = _interest_bounds_for_request(context, request)
            projected = _project_bounds(fit_bounds, eye, key.interest, float(key.view_angle_deg), aspect_ratio)
            if projected["height_frac"] >= 1e8 or projected["width_frac"] >= 1e8:
                continue
            center_score = center_score_for(interest_bounds, request)
            expected_height = _expected_height_fraction(request)
            expected_width = _expected_width_fraction(request)
            framing_score = (
                abs(projected["height_frac"] - expected_height)
                + abs(projected["width_frac"] - expected_width) * 0.25
            )
            semantic_penalty = 0.0
            if face_visible and body is TargetBodyPreset.LEGS:
                semantic_penalty += 0.9
            if face_visible and vertical_strength >= 0.35 and body in (
                TargetBodyPreset.BUST_UP,
                TargetBodyPreset.KNEE_UP,
                TargetBodyPreset.FULL_BODY,
            ):
                semantic_penalty -= 0.12
            score = framing_score + center_score + semantic_penalty
            if score < best_score:
                best_score = score
                best_body = body
                best_size = size
                best_projected = {
                    **projected,
                    "center_score": center_score,
                    "framing_score": framing_score,
                    "semantic_penalty": semantic_penalty,
                    "total_score": score,
                }

    if best_projected is None:
        best_projected = {
            "center_x": 0.0,
            "center_y": 0.0,
            "width_frac": 0.0,
            "height_frac": 0.0,
        }
    return best_body, best_size, best_projected


def _infer_composition(projected: dict) -> CompositionPreset:
    center_x = float(projected["center_x"])
    best = CompositionPreset.CENTER
    best_score = float("inf")
    for preset, target_x in RECONSTRUCTION_COMPOSITION_TARGETS.items():
        score = abs(center_x - target_x)
        if score < best_score:
            best_score = score
            best = preset
    return best


def _build_reason(body: TargetBodyPreset, size: ShotSizePreset, angle: AnglePreset, composition: CompositionPreset) -> str:
    labels = {
        TargetBodyPreset.FULL_BODY: "full-body",
        TargetBodyPreset.LEGS: "legs",
        TargetBodyPreset.KNEE_UP: "knee-up",
        TargetBodyPreset.ABDOMEN: "abdomen",
        TargetBodyPreset.WAIST_UP: "waist-up",
        TargetBodyPreset.BUST_UP: "bust-up",
        TargetBodyPreset.FACE: "face",
    }
    return (
        f"A {size.value} {labels[body]} shot from {angle.value}, "
        f"placed at {composition.value} composition."
    )


def infer_plan_keys(
    vmd_path: Path,
    targets_path: Path,
    sample_fps: float = 0.0,
    timeline_fps: float = 30.0,
    aspect_ratio: float = 16.0 / 9.0,
) -> tuple[list[InferredPlanKey], list[dict]]:
    provider = JsonFrameTargetProvider(targets_path)
    camera_keys = read_camera_vmd(vmd_path)
    collapsed = _collapse_cut_patterns(camera_keys)

    frame_scale = 1.0
    if sample_fps > 0.0 and timeline_fps > 0.0:
        frame_scale = sample_fps / timeline_fps

    plan_keys: list[InferredPlanKey] = []
    debug_rows: list[dict] = []
    for key, transition in collapsed:
        context = provider.get_context(key.frame)
        eye = _eye_from_mmd_camera(key.interest, key.rotate, key.distance)
        body, size, projected = _infer_body_and_size(context, key, eye, aspect_ratio)
        composition = _infer_composition(projected)
        angle = _infer_angle(context, eye, key.interest)
        output_frame = int(round(key.frame * frame_scale)) if frame_scale != 1.0 else int(key.frame)

        plan_keys.append(
            InferredPlanKey(
                frame=output_frame,
                target_body=body.value,
                shot_size=size.value,
                angle=angle.value,
                composition=composition.value,
                transition=transition.value,
                camera_reason=_build_reason(body, size, angle, composition),
            )
        )
        debug_rows.append(
            {
                "timeline_frame": key.frame,
                "output_frame": output_frame,
                "transition": transition.value,
                "target_body": body.value,
                "shot_size": size.value,
                "angle": angle.value,
                "composition": composition.value,
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
            }
        )
    return plan_keys, debug_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer a camera_poses-style plan JSON from a camera VMD and target data.")
    parser.add_argument("--camera-vmd", required=True, help="Input camera-only VMD path.")
    parser.add_argument("--targets", required=True, help="JSON file containing per-frame body bounds and actor forward vectors.")
    parser.add_argument("--output", required=True, help="Output plan JSON path.")
    parser.add_argument("--debug-json", default="", help="Optional debug JSON output path.")
    parser.add_argument("--sample-fps", type=float, default=0.0, help="Optional output plan FPS. When set, frames are mapped from timeline FPS back to this sample FPS.")
    parser.add_argument("--timeline-fps", type=float, default=30.0, help="Timeline FPS used by the input VMD.")
    parser.add_argument("--aspect-ratio", type=float, default=16.0 / 9.0)
    parser.add_argument("--omit-camera-reason", action="store_true", help="Do not include camera_reason in the output plan JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plan_keys, debug_rows = infer_plan_keys(
        Path(args.camera_vmd),
        Path(args.targets),
        sample_fps=args.sample_fps,
        timeline_fps=args.timeline_fps,
        aspect_ratio=args.aspect_ratio,
    )

    payload: dict[str, object] = {
        "keyframes": [],
    }
    for key in plan_keys:
        row = asdict(key)
        if args.omit_camera_reason:
            row.pop("camera_reason", None)
        payload["keyframes"].append(row)
    if args.sample_fps > 0.0:
        payload["sample_fps"] = float(args.sample_fps)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote inferred plan: {output_path}")
    print(f"keyframe_count={len(plan_keys)}")

    if args.debug_json:
        debug_path = Path(args.debug_json)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(json.dumps(debug_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote debug JSON: {debug_path}")


if __name__ == "__main__":
    main()
