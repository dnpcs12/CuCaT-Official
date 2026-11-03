"""Generate cameras for the 42 DCM test clips and evaluate them.

    python -m evaluation.run_dcm_test --output-root evaluation/results/cucat \
        --dca-root <DanceCamAnimator checkout> --dca-python <its python>

Steps: stage-1 timing from the released timing models (``evaluation.dcm_test_timing``; skipped when
``--timing-root`` is given) -> shot plans from the timing (``evaluation.dcm_test_plans``) ->
optionally top-k candidates and CP-SAT (``--cp-sat``) -> interpolation stage and VMD export with the
body targets (``shotplan_transformer.apply_v3_interpolation_stage``) -> DCA metrics
(``evaluation.run_dca_eval``). Without ``--cp-sat`` the shot plans are decoded as for the paper's
quantitative tables.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.dcm_common import (
    DEFAULT_FEATURE_ROOT,
    DEFAULT_INTERPOLATION_RUN_DIR,
    DEFAULT_SHOTPLAN_RUN_DIR,
    DEFAULT_TARGETS_ROOT,
    DEFAULT_TEST_CLIPS,
    DEFAULT_TIMING_FILENAME,
    REPO_ROOT,
    load_json,
    write_json,
)
from evaluation.run_dca_eval import DEFAULT_SPLIT_GT_ROOT, evaluate_vmd_root


DISTANCE_LEVEL_STATS = REPO_ROOT / "camera_autoshot" / "analysis" / "v31_distance_level_duration_stats_midpoint.json"
# CP-SAT settings of the paper (supplementary Table S1).
CP_SAT_ARGS = [
    "--schema", "v3_1",
    "--top-k", "10",
    "--max-time-sec", "30.0",
    "--local-cost-scale", "100",
    "--allow-missing-gt",
    "--manifest-name", "cp_sat_manifest.json",
    "--disable-sequence-stat-objective",
    "--distance-bounce-weight", "400",
    "--distance-bounce-min-rank-delta", "80",
    "--distance-bounce-max-frame-gap", "60",
    "--composition-bounce-weight", "100",
    "--composition-bounce-min-rank-delta", "80",
    "--composition-bounce-max-frame-gap", "60",
    "--top-pitch-bounce-weight", "400",
    "--top-pitch-bounce-max-frame-gap", "30",
    "--cut-identity-weight", "400",
    "--cut-contrast-weight", "100",
    "--cut-contrast-min-score", "3",
    "--cut-contrast-medium-rank-delta", "100",
    "--cut-contrast-strong-rank-delta", "220",
    "--cut-context-speed-weight", "400",
    "--cut-context-speed-min-speed", "250.0",
    "--cut-context-speed-max-frame-gap", "60",
    "--cut-context-speed-fps", "30.0",
    "--repeated-distance-bounce-weight", "1200",
    "--repeated-distance-bounce-min-rank-delta", "180",
    "--repeated-distance-bounce-max-frame-gap", "90",
    "--distance-level-duration-weight", "160",
    "--distance-level-duration-stats", str(DISTANCE_LEVEL_STATS),
    "--distance-level-duration-fps", "30.0",
    "--distance-level-duration-include-levels", "tight_close,close,medium_close",
    "--distance-level-duration-end-mode", "midpoint",
    "--distance-level-duration-upper-key", "p99_sec",
    "--distance-level-duration-penalty-sec-per-unit", "0.5",
    "--disallow-new-top-pitch-candidates",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and evaluate cameras for the DCM test clips.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--timing-root",
        type=Path,
        default=None,
        help="Folder with a stage-1 cut_key_timings_by_song.json to use instead of generating it with evaluation.dcm_test_timing.",
    )
    parser.add_argument("--timing-filename", default=DEFAULT_TIMING_FILENAME)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS_ROOT)
    parser.add_argument("--test-clips", type=Path, default=DEFAULT_TEST_CLIPS)
    parser.add_argument("--shotplan-run-dir", type=Path, default=DEFAULT_SHOTPLAN_RUN_DIR)
    parser.add_argument("--interpolation-run-dir", type=Path, default=DEFAULT_INTERPOLATION_RUN_DIR)
    parser.add_argument("--cp-sat", action="store_true", help="Select among the top-10 candidates with CP-SAT before interpolation.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dca-root", type=Path, default=None, help="DanceCamAnimator-Official checkout; omit with --skip-eval.")
    parser.add_argument("--dca-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--split-gt-root", type=Path, default=DEFAULT_SPLIT_GT_ROOT)
    parser.add_argument("--skip-eval", action="store_true", help="Only generate the cameras.")
    return parser


def run(module: str, *arguments: str) -> None:
    command = [sys.executable, "-m", module, *arguments]
    print("$ " + " ".join(command), flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=True)


def optimized_path(candidate_path: Path) -> Path:
    stem = re.sub(r"_top\d+_candidates$", "_optimized_cp_sat", candidate_path.stem)
    if stem == candidate_path.stem:
        stem = f"{candidate_path.stem}_optimized_cp_sat"
    return candidate_path.with_name(f"{stem}.json")


def candidate_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("c*/*top10_candidates*.json") if "_optimized_cp_sat" not in path.name)


def build_optimized_manifest(plan_root: Path, opt_root: Path) -> None:
    """Point the plan manifest at the CP-SAT-optimized plans."""

    candidates_by_clip: dict[tuple[int, str], Path] = {}
    for path in candidate_paths(opt_root):
        payload = load_json(path)
        index = int(re.search(r"(\d+)", str(payload.get("source_song_id", path.parent.name))).group(1))
        candidates_by_clip[(index, str(payload.get("clip_id", "")))] = path
    rows: list[dict[str, Any]] = []
    for source_row in load_json(plan_root / "manifest.json"):
        if not isinstance(source_row, dict) or not str(source_row.get("status", "")).startswith("ok"):
            continue
        key = (int(source_row["index"]), str(source_row.get("clip_id", "")))
        if key not in candidates_by_clip:
            raise RuntimeError(f"Missing top-k candidates for {key}")
        plan = optimized_path(candidates_by_clip[key])
        if not plan.exists():
            raise FileNotFoundError(f"Missing optimized plan: {plan}")
        row = dict(source_row)
        row.update({"status": "ok", "plan": str(plan), "source_plan": str(source_row.get("plan", ""))})
        rows.append(row)
    write_json(opt_root / "manifest.json", rows)


def main() -> None:
    args = build_parser().parse_args()
    output_root = args.output_root.resolve()
    timing_root = args.timing_root.resolve() if args.timing_root is not None else output_root / "timing"
    if args.timing_root is None:
        run(
            "evaluation.dcm_test_timing",
            "--output-root", str(timing_root),
            "--feature-root", str(args.feature_root.resolve()),
            "--test-clips", str(args.test_clips.resolve()),
            "--device", str(args.device),
        )
    common = [
        "--feature-root", str(args.feature_root.resolve()),
        "--timing-root", str(timing_root),
        "--timing-filename", str(args.timing_filename if args.timing_root is not None else DEFAULT_TIMING_FILENAME),
    ]

    plan_root = output_root / "plans"
    run(
        "evaluation.dcm_test_plans",
        *common,
        "--run-dir", str(args.shotplan_run_dir.resolve()),
        "--test-clips", str(args.test_clips.resolve()),
        "--targets-root", str(args.targets_root.resolve()),
        "--output-root", str(plan_root),
        "--device", str(args.device),
    )

    interp_input = plan_root
    camera_root = output_root / "cameras"
    if args.cp_sat:
        topk_root = output_root / "topk"
        run(
            "evaluation.dcm_test_topk",
            *common,
            "--run-dir", str(args.shotplan_run_dir.resolve()),
            "--plan-root", str(plan_root),
            "--output-root", str(topk_root),
            "--device", str(args.device),
        )
        opt_root = output_root / "cp_sat"
        if opt_root.exists():
            shutil.rmtree(opt_root)
        for source in candidate_paths(topk_root):
            destination = opt_root / source.parent.name / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        run(
            "shotplan_transformer.optimize_topk_candidate_files",
            "--source-root", str(opt_root),
            "--candidate-glob", "c*/*top10_candidates*.json",
            *CP_SAT_ARGS,
        )
        build_optimized_manifest(plan_root, opt_root)
        interp_input = opt_root
        camera_root = output_root / "cameras_cp_sat"

    run(
        "shotplan_transformer.apply_v3_interpolation_stage",
        "--actor-forward-stabilization",
        "--run-dir", str(args.interpolation_run_dir.resolve()),
        "--checkpoint-name", "best.pt",
        "--plan-root", str(interp_input),
        "--output-root", str(camera_root),
        "--targets-root", str(args.targets_root.resolve()),
        "--device", str(args.device),
    )

    if args.skip_eval:
        print(f"cameras: {camera_root}", flush=True)
        return
    if args.dca_root is None:
        raise SystemExit("Pass --dca-root (DanceCamAnimator checkout) or --skip-eval.")
    metrics = evaluate_vmd_root(
        camera_root,
        dca_root=args.dca_root,
        dca_python=args.dca_python,
        split_gt_root=args.split_gt_root,
        test_clips=args.test_clips,
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
