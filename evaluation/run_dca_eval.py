"""Compute the DCM metrics (FID_k, Dist_k, FID_s, Dist_s, DMR, LCD) for a folder of test-clip VMDs.

    python -m evaluation.run_dca_eval --vmd-root <folder> --dca-root <DanceCamAnimator checkout> \
        --dca-python <python of the DanceCamAnimator environment>

``<folder>/manifest.json`` lists one row per DCM test clip (``index``, ``clip_id``,
``clip_start_frame``, ``clip_end_frame``, ``vmd``; VMD paths may be relative to the folder). The
outputs of ``evaluation.run_dcm_test`` use this layout.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.dca_manifest import build_dca_manifest, build_split_lookup
from evaluation.dcm_common import DEFAULT_TEST_CLIPS, REPO_ROOT, load_json, write_json


ADAPTER = Path(__file__).resolve().parent / "dca_evaluate_vmd.py"
DEFAULT_SPLIT_GT_ROOT = REPO_ROOT / "local_assets" / "DCM++" / "Test"
# DanceCamAnimator metric keys -> names used in the paper.
METRIC_NAMES = {
    "fid_k_res": "FID_k",
    "div_k_res": "Dist_k",
    "fid_s_res": "FID_s",
    "div_s_res": "Dist_s",
    "res_dmr": "DMR",
    "lcd": "LCD",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DCM test-clip VMDs with DanceCamAnimator's metrics.")
    parser.add_argument("--vmd-root", type=Path, required=True, help="Folder with manifest.json and the test-clip VMDs.")
    parser.add_argument("--dca-root", type=Path, required=True, help="DanceCamAnimator-Official checkout (patched with evaluation.patch_dca).")
    parser.add_argument("--dca-python", type=Path, default=Path(sys.executable), help="Python of the DanceCamAnimator environment.")
    parser.add_argument("--split-gt-root", type=Path, default=DEFAULT_SPLIT_GT_ROOT, help="DCM++ Test folder.")
    parser.add_argument("--test-clips", type=Path, default=DEFAULT_TEST_CLIPS)
    parser.add_argument("--output-root", type=Path, default=None, help="Defaults to <vmd-root>/dca_eval.")
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep the converted cameras and bone masks.")
    return parser


def clean_intermediates(output_root: Path, vmd_root: Path) -> None:
    for child in output_root.iterdir() if output_root.exists() else []:
        if child.is_dir() and (child.name == "cache" or child.name.startswith("gt_subset_")):
            shutil.rmtree(child)
    result_dir = output_root / f"results_{vmd_root.name}"
    if result_dir.exists():
        for child in result_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)


def evaluate_vmd_root(
    vmd_root: Path,
    *,
    dca_root: Path,
    dca_python: Path,
    split_gt_root: Path,
    test_clips: Path = DEFAULT_TEST_CLIPS,
    output_root: Path | None = None,
    keep_intermediate: bool = False,
) -> dict[str, Any]:
    vmd_root = vmd_root.resolve()
    split_gt_root = split_gt_root.resolve()
    output_root = (output_root or vmd_root / "dca_eval").resolve()
    if not (split_gt_root / "CameraCentric").exists():
        raise FileNotFoundError(f"DCM++ Test ground truth not found: {split_gt_root}")
    manifest_path, summary = build_dca_manifest(vmd_root, build_split_lookup(test_clips), split_gt_root)
    print(f"[dca] {summary['record_count']} clips from {summary['unique_indices']} songs", flush=True)
    subprocess.run(
        [
            str(dca_python),
            str(ADAPTER),
            "--dca-root", str(dca_root.resolve()),
            "--generated-root", str(vmd_root),
            "--manifest-path", str(manifest_path),
            "--split-gt-root", str(split_gt_root),
            "--output-root", str(output_root),
        ],
        cwd=str(dca_root.resolve()),
        check=True,
    )
    raw_metrics = load_json(output_root / f"results_{vmd_root.name}" / "metrics_new.json")
    if not keep_intermediate:
        clean_intermediates(output_root, vmd_root)
    metrics = {name: float(raw_metrics[key]) for key, name in METRIC_NAMES.items()}
    write_json(vmd_root / "metrics.json", {"metrics": metrics, "dca_metrics": raw_metrics, "clip_count": summary["record_count"]})
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    metrics = evaluate_vmd_root(
        args.vmd_root,
        dca_root=args.dca_root,
        dca_python=args.dca_python,
        split_gt_root=args.split_gt_root,
        test_clips=args.test_clips,
        output_root=args.output_root,
        keep_intermediate=bool(args.keep_intermediate),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
