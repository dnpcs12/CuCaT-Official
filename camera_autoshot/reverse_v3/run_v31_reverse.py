from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from statistics import median

from camera_autoshot.batch_reconstruct_from_inferred_vmd_only import _compare_vmds, _write_csv
from camera_autoshot.batch_reverse_plans import _load_latest_ok_rows, _resolve_camera_path
from camera_autoshot.providers import JsonFrameTargetProvider
from camera_autoshot.vmd import write_camera_vmd

from camera_autoshot.reverse_v3.reverse_plan_v3 import InferredPlanKeyV3, infer_plan_keys_v3
from camera_autoshot.reverse_v3.solver_v3 import (
    CameraSolveConfigV3,
    DISTANCE_HEIGHT_MULTIPLIERS,
    solve_camera_timeline_v3,
)
from camera_autoshot.reverse_v3.spec_v3 import DistancePreset, ShotKeyframeRequestV3


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "camera_autoshot" / "reverse_v3" / "labels" / "v3_1_full"
DEFAULT_LATEST_OK = REPO_ROOT / "saba_mmd" / "build" / "Release" / "batch_render_result_latest_ok.csv"
DEFAULT_TARGETS_ROOT = REPO_ROOT / "camera_autoshot" / "reverses"
DEFAULT_PRUNED_ROOT = (
    REPO_ROOT / "camera_autoshot" / "analysis" / "minimal_noncut_camera_keyframes_linear_reduction60plus_vmd"
)


# Keep the current Reverse V3 modules untouched.  Internally, reuse the existing
# DistancePreset.WIDE slot as the new closest candidate and remap it on output.
INTERNAL_DISTANCE_MULTIPLIERS = {
    DistancePreset.WIDE: 0.55,  # exported as extreme_close
    DistancePreset.EXTREME_CLOSE: 0.80,  # exported as very_close
    DistancePreset.CLOSE: 1.20,
    DistancePreset.MEDIUM_CLOSE: 1.75,
    DistancePreset.MEDIUM: 2.55,
    DistancePreset.MEDIUM_FAR: 3.70,
    DistancePreset.FAR: 5.80,
}

INTERNAL_TO_EXPORT_DISTANCE = {
    "wide": "extreme_close",
    "extreme_close": "very_close",
    "close": "close",
    "medium_close": "medium_close",
    "medium": "medium",
    "medium_far": "medium_far",
    "far": "far",
}

EXPORT_TO_INTERNAL_DISTANCE = {value: key for key, value in INTERNAL_TO_EXPORT_DISTANCE.items()}


def apply_close_dense_distance_schema() -> None:
    """Patch only process-local distance multipliers for this experiment."""

    for preset, value in INTERNAL_DISTANCE_MULTIPLIERS.items():
        DISTANCE_HEIGHT_MULTIPLIERS[preset] = value


def export_plan_key(key: InferredPlanKeyV3) -> dict[str, object]:
    row = asdict(key)
    row["distance"] = INTERNAL_TO_EXPORT_DISTANCE.get(str(row["distance"]), str(row["distance"]))
    return row


def export_debug_row(row: dict[str, object]) -> dict[str, object]:
    copied = dict(row)
    if "distance" in copied:
        copied["distance"] = INTERNAL_TO_EXPORT_DISTANCE.get(str(copied["distance"]), str(copied["distance"]))
    joint = copied.get("distance_fov_joint")
    if isinstance(joint, dict):
        joint = dict(joint)
        for key in ("best_joint_distance", "from_distance", "to_distance"):
            if key in joint:
                joint[key] = INTERNAL_TO_EXPORT_DISTANCE.get(str(joint[key]), str(joint[key]))
        copied["distance_fov_joint"] = joint
    top = copied.get("top_distance_fov_stabilization")
    if isinstance(top, dict):
        top = dict(top)
        for key in ("from_distance", "to_distance"):
            if key in top:
                top[key] = INTERNAL_TO_EXPORT_DISTANCE.get(str(top[key]), str(top[key]))
        copied["top_distance_fov_stabilization"] = top
    return copied


def write_export_plan(path: Path, plan_keys: list[InferredPlanKeyV3], sample_fps: float = 0.0) -> None:
    payload: dict[str, object] = {
        "schema": "camera_autoshot.reverse_v3.close_dense_no_wide",
        "distance_schema": {
            "distance_values": [
                "extreme_close",
                "very_close",
                "close",
                "medium_close",
                "medium",
                "medium_far",
                "far",
            ],
            "notes": (
                "Experimental close-dense distance bins. Existing Reverse V3 files are not modified; "
                "internal wide is remapped to exported extreme_close, and internal extreme_close is "
                "remapped to exported very_close."
            ),
            "body_height_multipliers": {
                "extreme_close": 0.55,
                "very_close": 0.80,
                "close": 1.20,
                "medium_close": 1.75,
                "medium": 2.55,
                "medium_far": 3.70,
                "far": 5.80,
            },
        },
        "keyframes": [export_plan_key(key) for key in plan_keys],
    }
    if sample_fps > 0.0:
        payload["sample_fps"] = float(sample_fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def internal_requests_from_plan_keys(plan_keys: list[InferredPlanKeyV3]) -> list[ShotKeyframeRequestV3]:
    requests: list[ShotKeyframeRequestV3] = []
    for key in plan_keys:
        requests.append(
            ShotKeyframeRequestV3.from_dict(
                {
                    "frame": key.frame,
                    "target_anchor": key.target_anchor,
                    "distance": key.distance,
                    "yaw": key.yaw,
                    "pitch": key.pitch,
                    "composition": key.composition,
                    "roll": key.roll,
                    "fov": key.fov,
                    "transition": key.transition,
                }
            )
        )
    return requests


def internal_requests_from_export_plan(path: Path) -> list[ShotKeyframeRequestV3]:
    """Read an exported close-dense JSON and convert distance labels back to internal labels."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    requests: list[ShotKeyframeRequestV3] = []
    for raw in payload.get("keyframes", []):
        row = dict(raw)
        row["distance"] = EXPORT_TO_INTERNAL_DISTANCE.get(str(row.get("distance", "")), str(row.get("distance", "")))
        requests.append(ShotKeyframeRequestV3.from_dict(row))
    return requests


def resolve_reverse_input(index: int, original_camera_vmd: Path, pruned_root: Path) -> tuple[Path, str]:
    pruned = pruned_root / f"c{index}" / "camera_minimal_noncut_linear.vmd"
    if pruned.exists():
        return pruned, "minimal_noncut_linear_reduction60plus"
    return original_camera_vmd, "latest_ok_original"


def summarize_metric(rows: list[dict[str, object]], name: str) -> dict[str, float]:
    values: list[float] = []
    weighted_sum = 0.0
    weight_total = 0.0
    for row in rows:
        if row.get("status") != "ok":
            continue
        value = row.get(name, "")
        if value == "":
            continue
        numeric = float(value)
        values.append(numeric)
        weight = float(row.get("frame_count") or 0)
        weighted_sum += numeric * weight
        weight_total += weight
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0, "weighted_mean": 0.0}
    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.9)))
    return {
        "mean": sum(values) / len(values),
        "median": median(values),
        "p90": ordered[p90_index],
        "max": max(values),
        "weighted_mean": weighted_sum / weight_total if weight_total > 0.0 else sum(values) / len(values),
    }


def write_metric_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    metrics = [
        "mean_eye_error",
        "mean_interest_error",
        "mean_rotate_error_deg",
        "mean_distance_error",
        "mean_fov_error_deg",
        "max_eye_error",
        "max_interest_error",
        "max_rotate_error_deg",
        "max_distance_error",
        "max_fov_error_deg",
    ]
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    payload = {
        "experiment": "close_dense_temporal_delta",
        "baseline_reference": str(
            REPO_ROOT / "camera_autoshot" / "reverse_v3" / "final" / "middle_candidate_with_pruned_high_quality"
        ),
        "common_song_count": len(ok_rows),
        "distance_schema": {
            "extreme_close": 0.55,
            "very_close": 0.80,
            "close": 1.20,
            "medium_close": 1.75,
            "medium": 2.55,
            "medium_far": 3.70,
            "far": 5.80,
        },
        "metrics": {metric: summarize_metric(rows, metric) for metric in metrics},
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_manifest_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "index",
        "status",
        "reverse_input_policy",
        "reverse_input_vmd",
        "original_camera_vmd",
        "targets",
        "plan_json",
        "debug_json",
        "reconstructed_camera_vmd",
        "reconstructed_debug_json",
        "keyframe_count",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run close-dense temporal-delta Reverse V3 experiment without touching final V3."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=107)
    parser.add_argument("--latest-ok-csv", type=Path, default=DEFAULT_LATEST_OK)
    parser.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS_ROOT)
    parser.add_argument("--targets-filename", default="real_targets.json")
    parser.add_argument("--pruned-root", type=Path, default=DEFAULT_PRUNED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-fps", type=float, default=0.0)
    parser.add_argument("--timeline-fps", type=float, default=30.0)
    parser.add_argument("--aspect-ratio", type=float, default=16.0 / 9.0)
    parser.add_argument(
        "--fov-policy",
        choices=("joint", "physical_normal_first", "temporal_delta"),
        default="temporal_delta",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    apply_close_dense_distance_schema()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    latest_rows = _load_latest_ok_rows(args.latest_ok_csv.resolve())
    targets_root = args.targets_root.resolve()
    pruned_root = args.pruned_root.resolve()

    manifest_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for index in range(args.start, args.end + 1):
        latest = latest_rows.get(index)
        item_dir = output_root / f"c{index}"
        item_dir.mkdir(parents=True, exist_ok=True)
        plan_path = item_dir / "camera_poses_v3_close_dense.json"
        debug_path = item_dir / "camera_poses_v3_close_dense_debug.json"
        reconstructed_vmd = item_dir / "camera_poses_v3_close_dense_reconstructed.vmd"
        reconstructed_debug = item_dir / "camera_poses_v3_close_dense_reconstructed_debug.json"
        compare_json = item_dir / "comparison_v3_close_dense.json"
        targets_path = targets_root / f"c{index}" / args.targets_filename

        manifest_row: dict[str, object] = {
            "index": index,
            "status": "",
            "reverse_input_policy": "",
            "reverse_input_vmd": "",
            "original_camera_vmd": "",
            "targets": str(targets_path),
            "plan_json": str(plan_path),
            "debug_json": str(debug_path),
            "reconstructed_camera_vmd": str(reconstructed_vmd),
            "reconstructed_debug_json": str(reconstructed_debug),
            "keyframe_count": "",
        }
        summary_row: dict[str, object] = {
            "index": index,
            "status": "",
            "original_camera_vmd": "",
            "reconstructed_camera_vmd": str(reconstructed_vmd),
        }

        if latest is None:
            manifest_row["status"] = summary_row["status"] = "missing-latest-ok"
            manifest_rows.append(manifest_row)
            summary_rows.append(summary_row)
            continue

        original_camera_vmd = _resolve_camera_path(index, Path(latest.get("camera", "")))
        reverse_input_vmd, reverse_input_policy = resolve_reverse_input(index, original_camera_vmd, pruned_root)
        manifest_row["original_camera_vmd"] = summary_row["original_camera_vmd"] = str(original_camera_vmd)
        manifest_row["reverse_input_vmd"] = str(reverse_input_vmd)
        manifest_row["reverse_input_policy"] = reverse_input_policy

        if not original_camera_vmd.exists():
            manifest_row["status"] = summary_row["status"] = "missing-original-camera-vmd"
            manifest_rows.append(manifest_row)
            summary_rows.append(summary_row)
            continue
        if not reverse_input_vmd.exists():
            manifest_row["status"] = summary_row["status"] = "missing-reverse-input-vmd"
            manifest_rows.append(manifest_row)
            summary_rows.append(summary_row)
            continue
        if not targets_path.exists():
            manifest_row["status"] = summary_row["status"] = "missing-targets-json"
            manifest_rows.append(manifest_row)
            summary_rows.append(summary_row)
            continue

        try:
            export_frame_count = int(float(latest["export_frame_count"]))
        except (KeyError, TypeError, ValueError):
            manifest_row["status"] = summary_row["status"] = "invalid-export-frame-count"
            manifest_rows.append(manifest_row)
            summary_rows.append(summary_row)
            continue

        plan_keys, debug_rows = infer_plan_keys_v3(
            reverse_input_vmd,
            targets_path,
            sample_fps=args.sample_fps,
            timeline_fps=args.timeline_fps,
            aspect_ratio=args.aspect_ratio,
            fov_policy=args.fov_policy,
        )
        write_export_plan(plan_path, plan_keys, sample_fps=args.sample_fps)
        debug_path.write_text(
            json.dumps([export_debug_row(row) for row in debug_rows], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        provider = JsonFrameTargetProvider(targets_path)
        solved = solve_camera_timeline_v3(
            provider,
            internal_requests_from_plan_keys(plan_keys),
            CameraSolveConfigV3(aspect_ratio=args.aspect_ratio),
        )
        write_camera_vmd(solved, reconstructed_vmd, model_name=f"Reverse V3 close dense c{index}")
        reconstructed_debug.write_text(
            json.dumps([asdict(key) for key in solved], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        comparison = _compare_vmds(original_camera_vmd, reconstructed_vmd, export_frame_count)
        compare_json.write_text(
            json.dumps(
                {
                    "index": index,
                    "status": "ok",
                    "reverse_input_policy": reverse_input_policy,
                    "reverse_input_vmd": str(reverse_input_vmd),
                    "original_camera_vmd": str(original_camera_vmd),
                    "reconstructed_camera_vmd": str(reconstructed_vmd),
                    **comparison,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        manifest_row["status"] = summary_row["status"] = "ok"
        manifest_row["keyframe_count"] = len(plan_keys)
        summary_row["reverse_input_policy"] = reverse_input_policy
        summary_row.update(comparison)
        manifest_rows.append(manifest_row)
        summary_rows.append(summary_row)
        print(
            f"[ok] c{index}: input={reverse_input_policy} keys={len(plan_keys)} "
            f"eye={float(comparison['mean_eye_error']):.4f} "
            f"rot={float(comparison['mean_rotate_error_deg']):.4f} "
            f"dist={float(comparison['mean_distance_error']):.4f} "
            f"fov={float(comparison['mean_fov_error_deg']):.4f}"
        )

    (output_root / "manifest.json").write_text(json.dumps(manifest_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_manifest_csv(manifest_rows, output_root / "manifest.csv")
    (output_root / "reconstruction_summary_vmd_only.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(summary_rows, output_root / "reconstruction_summary_vmd_only.csv")
    write_metric_summary(summary_rows, output_root / "camera_error_summary.json")
    print(f"Wrote root: {output_root}")
    print(f"Wrote metrics: {output_root / 'camera_error_summary.json'}")


if __name__ == "__main__":
    main()
