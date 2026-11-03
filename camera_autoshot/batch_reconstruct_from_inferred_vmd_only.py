from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from .cli import load_requests
from .providers import JsonFrameTargetProvider
from .solver import CameraSolveConfig, _eye_from_mmd_camera, solve_camera_timeline
from .vmd import evaluate_camera_vmd, read_camera_vmd, write_camera_vmd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def release_dir() -> Path:
    return repo_root() / "saba_mmd" / "build" / "Release"


def _load_latest_ok_rows(csv_path: Path) -> dict[int, dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: dict[int, dict[str, str]] = {}
        for row in reader:
            try:
                index = int(row["index"])
            except (KeyError, TypeError, ValueError):
                continue
            rows[index] = row
        return rows


def _resolve_camera_path(index: int, camera_path: Path) -> Path:
    if camera_path.exists():
        return camera_path
    build_release = str(release_dir())
    if build_release not in sys.path:
        sys.path.insert(0, build_release)
    import batch_render_gt_range as batch_render  # type: ignore

    folder = batch_render.BASE_DIR / f"amc{index}"
    preferred = camera_path.name if str(camera_path) else ""
    resolved = batch_render.resolve_vmd(folder, preferred, "camera", index)
    if resolved is None:
        return camera_path
    return resolved


def _vec_l2(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _angle_diff_rad(a: float, b: float) -> float:
    diff = (float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi
    return diff


def _rotate_l2_deg(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.degrees(
        math.sqrt(sum(_angle_diff_rad(x, y) ** 2 for x, y in zip(a, b)))
    )


def _write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "index",
        "status",
        "original_camera_vmd",
        "reconstructed_camera_vmd",
        "frame_count",
        "original_key_count",
        "reconstructed_key_count",
        "mean_eye_error",
        "max_eye_error",
        "max_eye_error_frame",
        "mean_interest_error",
        "max_interest_error",
        "max_interest_error_frame",
        "mean_rotate_error_deg",
        "max_rotate_error_deg",
        "max_rotate_error_deg_frame",
        "mean_distance_error",
        "max_distance_error",
        "max_distance_error_frame",
        "mean_fov_error_deg",
        "max_fov_error_deg",
        "max_fov_error_deg_frame",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _compare_vmds(
    original_vmd: Path,
    reconstructed_vmd: Path,
    frame_count: int,
) -> dict[str, object]:
    original_keys = read_camera_vmd(original_vmd)
    reconstructed_keys = read_camera_vmd(reconstructed_vmd)

    eye_errors: list[tuple[int, float]] = []
    interest_errors: list[tuple[int, float]] = []
    rotate_errors: list[tuple[int, float]] = []
    distance_errors: list[tuple[int, float]] = []
    fov_errors: list[tuple[int, float]] = []

    for frame in range(frame_count):
        original = evaluate_camera_vmd(original_keys, frame)
        reconstructed = evaluate_camera_vmd(reconstructed_keys, frame)
        if original is None or reconstructed is None:
            continue

        original_eye = _eye_from_mmd_camera(original.interest, original.rotate, original.distance)
        reconstructed_eye = _eye_from_mmd_camera(reconstructed.interest, reconstructed.rotate, reconstructed.distance)

        eye_errors.append((frame, _vec_l2(original_eye, reconstructed_eye)))
        interest_errors.append((frame, _vec_l2(original.interest, reconstructed.interest)))
        rotate_errors.append((frame, _rotate_l2_deg(original.rotate, reconstructed.rotate)))
        distance_errors.append((frame, abs(original.distance - reconstructed.distance)))
        fov_errors.append((frame, abs(original.view_angle_deg - reconstructed.view_angle_deg)))

    def summarize(series: list[tuple[int, float]], prefix: str) -> dict[str, object]:
        if not series:
            return {
                f"mean_{prefix}": "",
                f"max_{prefix}": "",
                f"max_{prefix}_frame": "",
            }
        values = [value for _frame, value in series]
        frame, value = max(series, key=lambda item: item[1])
        return {
            f"mean_{prefix}": sum(values) / len(values),
            f"max_{prefix}": value,
            f"max_{prefix}_frame": frame,
        }

    summary: dict[str, object] = {
        "frame_count": frame_count,
        "original_key_count": len(original_keys),
        "reconstructed_key_count": len(reconstructed_keys),
    }
    summary.update(summarize(eye_errors, "eye_error"))
    summary.update(summarize(interest_errors, "interest_error"))
    summary.update(summarize(rotate_errors, "rotate_error_deg"))
    summary.update(summarize(distance_errors, "distance_error"))
    summary.update(summarize(fov_errors, "fov_error_deg"))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstruct VMDs from inferred camera_poses JSON and compare against original base camera VMDs without rendering MP4s.")
    parser.add_argument("--input-root", required=True, help="Root directory produced by batch_reverse_plans.py")
    parser.add_argument(
        "--latest-ok-csv",
        default=str(release_dir() / "batch_render_result_latest_ok.csv"),
        help="Path to batch_render_result_latest_ok.csv",
    )
    parser.add_argument(
        "--targets-root",
        default=str(Path("camera_autoshot") / "results"),
        help="Root directory containing cNN/real_targets.json",
    )
    parser.add_argument(
        "--targets-filename",
        default="real_targets.json",
        help="Target data filename inside each cNN folder.",
    )
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--plan-sample-fps", type=float, default=0.0)
    parser.add_argument("--timeline-fps", type=float, default=30.0)
    parser.add_argument("--aspect-ratio", type=float, default=16.0 / 9.0)
    parser.add_argument("--medium-distance-multiplier", type=float, default=1.0)
    parser.add_argument("--close-distance-multiplier", type=float, default=1.0)
    parser.add_argument("--full-body-close-distance-multiplier", type=float, default=1.0)
    parser.add_argument("--legs-close-distance-multiplier", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_root = Path(args.input_root).resolve()
    latest_ok_csv = Path(args.latest_ok_csv).resolve()
    targets_root = Path(args.targets_root).resolve()

    latest_rows = _load_latest_ok_rows(latest_ok_csv)
    summary_rows: list[dict[str, object]] = []

    for index in range(args.start, args.end + 1):
        row: dict[str, object] = {
            "index": index,
            "status": "",
            "original_camera_vmd": "",
            "reconstructed_camera_vmd": "",
        }

        latest = latest_rows.get(index)
        item_dir = input_root / f"c{index}"
        inferred_plan = item_dir / "camera_poses_inferred.json"
        reconstructed_vmd = item_dir / "camera_poses_reconstructed.vmd"
        reconstructed_debug = item_dir / "camera_poses_reconstructed_debug.json"
        compare_json = item_dir / "comparison_vmd_only.json"

        if latest is None:
            row["status"] = "missing-latest-ok"
            summary_rows.append(row)
            continue

        original_camera_vmd = _resolve_camera_path(index, Path(latest.get("camera", "")))
        targets_json = targets_root / f"c{index}" / args.targets_filename

        row["original_camera_vmd"] = str(original_camera_vmd)
        row["reconstructed_camera_vmd"] = str(reconstructed_vmd)

        if not inferred_plan.exists():
            row["status"] = "missing-inferred-plan"
            summary_rows.append(row)
            continue
        if not original_camera_vmd.exists():
            row["status"] = "missing-original-camera-vmd"
            summary_rows.append(row)
            continue
        if not targets_json.exists():
            row["status"] = "missing-targets-json"
            summary_rows.append(row)
            continue

        provider = JsonFrameTargetProvider(targets_json)
        requests = load_requests(
            inferred_plan,
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
        reconstructed_vmd.parent.mkdir(parents=True, exist_ok=True)
        write_camera_vmd(solved, reconstructed_vmd, model_name=f"Reconstructed c{index}")
        reconstructed_debug.write_text(
            json.dumps(
                [
                    {
                        "frame": key.frame,
                        "interest": key.interest,
                        "rotate": key.rotate,
                        "distance": key.distance,
                        "view_angle_deg": key.view_angle_deg,
                        "perspective": key.perspective,
                        "transition_to_next": key.transition_to_next.value,
                    }
                    for key in solved
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        export_frame_count = int(float(latest["export_frame_count"]))
        comparison = _compare_vmds(original_camera_vmd, reconstructed_vmd, export_frame_count)
        compare_json.write_text(
            json.dumps(
                {
                    "index": index,
                    "status": "ok",
                    "original_camera_vmd": str(original_camera_vmd),
                    "reconstructed_camera_vmd": str(reconstructed_vmd),
                    **comparison,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        row["status"] = "ok"
        row.update(comparison)
        summary_rows.append(row)
        print(f"[ok] c{index}")

    summary_json = input_root / "reconstruction_summary_vmd_only.json"
    summary_csv = input_root / "reconstruction_summary_vmd_only.csv"
    summary_json.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(summary_rows, summary_csv)
    print(f"Wrote summary: {summary_json}")
    print(f"Wrote summary: {summary_csv}")


if __name__ == "__main__":
    main()
