from __future__ import annotations

import argparse
from pathlib import Path

from .plan_timeline import load_plan_payload
from .providers import JsonFrameTargetProvider
from .solver import CameraSolveConfig, solve_camera_timeline
from .spec import ShotKeyframeRequest
from .vmd import dump_debug_json, read_camera_vmd_stats, write_camera_vmd


def load_requests(
    plan_path: Path,
    plan_sample_fps: float = 0.0,
    timeline_fps: float = 30.0,
) -> list[ShotKeyframeRequest]:
    payload = load_plan_payload(
        Path(plan_path),
        plan_sample_fps=plan_sample_fps,
        timeline_fps=timeline_fps,
    )
    requests: list[ShotKeyframeRequest] = []
    for item in payload.get("keyframes", []):
        normalized = dict(item["original"])
        normalized["frame"] = int(item["timeline_frame"])
        requests.append(ShotKeyframeRequest.from_dict(normalized))
    return requests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate camera-only VMD from natural-language shot presets.")
    parser.add_argument("--targets", required=True, help="JSON file containing per-frame body bounds and actor forward vectors.")
    parser.add_argument("--plan", required=True, help="JSON file containing keyframe shot requests.")
    parser.add_argument(
        "--plan-sample-fps",
        type=float,
        default=0.0,
        help="Optional FPS of the sampled images used to author the plan. Frames are scaled onto the timeline FPS.",
    )
    parser.add_argument(
        "--timeline-fps",
        type=float,
        default=30.0,
        help="Target timeline FPS used by the output VMD.",
    )
    parser.add_argument("--output", required=True, help="Output camera VMD path.")
    parser.add_argument("--debug-json", default="", help="Optional debug JSON output path.")
    parser.add_argument("--aspect-ratio", type=float, default=16.0 / 9.0)
    parser.add_argument("--model-name", default="Autoshot Camera")
    parser.add_argument(
        "--medium-distance-multiplier",
        type=float,
        default=1.0,
        help="Extra distance multiplier applied only to medium shots.",
    )
    parser.add_argument(
        "--close-distance-multiplier",
        type=float,
        default=1.0,
        help="Extra distance multiplier applied only to close shots.",
    )
    parser.add_argument(
        "--full-body-close-distance-multiplier",
        type=float,
        default=1.0,
        help="Additional distance multiplier applied only to full_body close shots.",
    )
    parser.add_argument(
        "--legs-close-distance-multiplier",
        type=float,
        default=1.0,
        help="Additional distance multiplier applied only to legs close shots.",
    )
    parser.add_argument(
        "--linear-translation",
        action="store_true",
        help="Force x/y/z camera translation interpolation to linear while keeping other channels on their requested transition.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    provider = JsonFrameTargetProvider(Path(args.targets))
    requests = load_requests(
        Path(args.plan),
        plan_sample_fps=args.plan_sample_fps,
        timeline_fps=args.timeline_fps,
    )
    solved = solve_camera_timeline(
        provider,
        requests,
        CameraSolveConfig(
            aspect_ratio=args.aspect_ratio,
            medium_distance_multiplier=args.medium_distance_multiplier,
            close_distance_multiplier=args.close_distance_multiplier,
            full_body_close_distance_multiplier=args.full_body_close_distance_multiplier,
            legs_close_distance_multiplier=args.legs_close_distance_multiplier,
        ),
    )

    output_path = write_camera_vmd(
        solved,
        Path(args.output),
        model_name=args.model_name,
        linear_translation=args.linear_translation,
    )
    stats = read_camera_vmd_stats(output_path)

    if args.debug_json:
        dump_debug_json(solved, Path(args.debug_json))

    print(f"Wrote: {output_path}")
    print(f"camera_count={stats['camera_count']} camera_max_frame={stats['camera_max_frame']}")
    for key in solved:
        print(
            f"frame={key.frame} distance={key.distance:.3f} "
            f"view_angle={key.view_angle_deg} interest={key.interest} rotate={key.rotate}"
        )


if __name__ == "__main__":
    main()
