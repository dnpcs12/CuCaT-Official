from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

LOCAL_ASSETS_DIR = Path(__file__).resolve().parents[1] / "local_assets"
# PMX/PMD used to evaluate dance motions into body targets. Not distributed; pass --model-path or place it here.
DEFAULT_BASE_MODEL = LOCAL_ASSETS_DIR / "default_target_model.pmx"
# Per-song model, motion, and frame range of the body targets used for the paper (--paper-targets).
PAPER_TARGET_MAPPING = Path(__file__).resolve().parents[1] / "datasets" / "dcm_target_models.csv"
DEFAULT_PAPER_MODEL_DIR = LOCAL_ASSETS_DIR / "target_models"


MOTION_OVERRIDES_BY_INDEX = {
    # c50 has both 1p and 5p motions under the same raw-data folder. The DCM
    # test clip K_50_1 and Task 1 renders use the 1p TDA motion; dumping targets
    # from any 5p motion makes the solver aim at the wrong dancer.
    50: LOCAL_ASSETS_DIR / "amc_raw_data" / "amc50" / "_WORLD" / "1p ver" / "motion(1p ver.)TDA用.vmd",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_dump_exe() -> Path:
    return repo_root() / "saba_mmd" / "build" / "Release" / "mmd_target_dump.exe"


def default_latest_ok_csv() -> Path:
    return repo_root() / "saba_mmd" / "build" / "Release" / "batch_render_result_latest_ok.csv"


def default_output_for_index(index: int) -> Path:
    return repo_root() / "camera_autoshot" / "examples" / f"c{index}_real_targets.json"


def safe_print(value: object) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    text = str(value).encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(text)


def read_latest_ok_row(csv_path: Path, index: int) -> dict[str, str]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        try:
            if int(row.get("index", "-1")) == index:
                return row
        except ValueError:
            continue
    raise SystemExit(f"Index {index} not found in {csv_path}")


def normalize_model_path(model_path: Path) -> Path:
    if model_path.exists():
        return model_path
    model_text = str(model_path)
    if "MMD" in model_text and "AIChannel" in model_text and "APOSS" in model_text and DEFAULT_BASE_MODEL.exists():
        return DEFAULT_BASE_MODEL
    return model_path


def normalize_motion_path(index: int, motion_path: Path) -> Path:
    override = MOTION_OVERRIDES_BY_INDEX.get(index)
    if override is not None and override.exists():
        return override

    try:
        from .render_autoshot import load_batch_module

        batch_render = load_batch_module()
        if motion_path.exists():
            stats = batch_render.read_vmd_stats(motion_path)
            if stats.get("motion_max_frame", 0) > 0 or stats.get("camera_count", 0) == 0:
                return motion_path

        folder = batch_render.BASE_DIR / f"amc{index}"
        basename = motion_path.name if str(motion_path) else ""
        if basename:
            basename_matches = list(folder.rglob(basename))
            for candidate in basename_matches:
                stats = batch_render.read_vmd_stats(candidate)
                if stats.get("motion_max_frame", 0) > 0 or stats.get("camera_count", 0) == 0:
                    return candidate
        resolved = batch_render.resolve_vmd(folder, basename, "motion", index)
        if resolved is not None and resolved.exists():
            return resolved
    except Exception:
        pass
    return motion_path


def read_paper_target_row(index: int) -> dict[str, str]:
    with PAPER_TARGET_MAPPING.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["index"]) == index:
                return row
    raise SystemExit(f"Index {index} is not a DCM song in {PAPER_TARGET_MAPPING}")


def find_model(model_dir: Path, model_name: str) -> Path:
    matches = sorted(model_dir.rglob(model_name)) if model_dir.exists() else []
    if not matches:
        raise SystemExit(
            f"Model {model_name} not found under {model_dir}. Put the PMX files listed in "
            f"{PAPER_TARGET_MAPPING.name} there (with their folders) or pass --model-dir."
        )
    return matches[0]


def resolve_paths(args: argparse.Namespace) -> tuple[Path, list[Path], Path, int | None]:
    if args.paper_targets:
        if args.index is None:
            raise SystemExit("--paper-targets needs --index.")
        row = read_paper_target_row(args.index)
        model_path = Path(args.model_path) if args.model_path else find_model(Path(args.model_dir), row["model"])
        vmd_paths = [LOCAL_ASSETS_DIR / "amc_raw_data" / row["motion"]]
        output_path = (
            Path(args.output)
            if args.output
            else repo_root() / "camera_autoshot" / "reverses" / f"c{args.index}" / "real_targets.json"
        )
        end_frame = args.end if args.end is not None else int(row["end_frame"])
        return model_path, vmd_paths, output_path, end_frame

    if args.index is not None:
        row = read_latest_ok_row(Path(args.latest_ok_csv), args.index)
        model_path = normalize_model_path(Path(row["model"]))
        vmd_paths = [normalize_motion_path(args.index, Path(row["motion"]))]
        output_path = Path(args.output) if args.output else default_output_for_index(args.index)
        end_frame = args.end
        if end_frame is None:
            try:
                end_frame = int(row["export_frame_count"]) - 1
            except (KeyError, TypeError, ValueError):
                end_frame = None
        return model_path, vmd_paths, output_path, end_frame

    if not args.model_path or not args.motion_path:
        raise SystemExit("Either --index or both --model-path and --motion-path are required.")

    model_path = normalize_model_path(Path(args.model_path))
    vmd_paths = [Path(args.motion_path)]
    output_path = (
        Path(args.output)
        if args.output
        else repo_root() / "camera_autoshot" / "results" / "autoshot" / "target_data.json"
    )
    return model_path, vmd_paths, output_path, args.end


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dump per-frame target data JSON for camera_autoshot using the standalone mmd_target_dump executable."
    )
    parser.add_argument("--index", type=int, default=None, help="Render index to resolve from batch_render_result_latest_ok.csv")
    parser.add_argument("--model-path", default="", help="Explicit model path when not using --index")
    parser.add_argument("--motion-path", default="", help="Explicit motion VMD path when not using --index")
    parser.add_argument("--output", default="", help="Output JSON path")
    parser.add_argument("--start", type=int, default=0, help="Start frame")
    parser.add_argument("--end", type=int, default=None, help="End frame. Defaults to export_frame_count-1 for --index")
    parser.add_argument("--step", type=int, default=1, help="Sampling step in frames")
    parser.add_argument(
        "--legacy-targets-v1",
        action="store_true",
        help="Reproduce the April 2026 target-region schema used by the paper evaluation.",
    )
    parser.add_argument(
        "--paper-targets",
        action="store_true",
        help=(
            "Regenerate the paper's body targets for --index: model, motion, and frame range from "
            "datasets/dcm_target_models.csv, with --legacy-targets-v1. Output defaults to "
            "camera_autoshot/reverses/c<index>/real_targets.json."
        ),
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_PAPER_MODEL_DIR),
        help="Folder searched for the PMX files named in datasets/dcm_target_models.csv (--paper-targets).",
    )
    parser.add_argument(
        "--tool",
        default=str(default_dump_exe()),
        help="Path to mmd_target_dump.exe",
    )
    parser.add_argument(
        "--latest-ok-csv",
        default=str(default_latest_ok_csv()),
        help="Path to batch_render_result_latest_ok.csv used by --index",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.paper_targets:
        args.legacy_targets_v1 = True
    tool_path = Path(args.tool)
    if not tool_path.exists():
        raise SystemExit(f"mmd_target_dump.exe not found: {tool_path}")

    model_path, vmd_paths, output_path, end_frame = resolve_paths(args)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    for vmd_path in vmd_paths:
        if not vmd_path.exists():
            raise SystemExit(f"VMD not found: {vmd_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [str(tool_path), str(model_path)]
    for vmd_path in vmd_paths:
        command.extend(["-vmd", str(vmd_path)])
    command.extend(["-o", str(output_path), "-start", str(args.start), "-step", str(max(1, args.step))])
    if args.legacy_targets_v1:
        command.append("-legacy-targets-v1")
    if end_frame is not None:
        command.extend(["-end", str(end_frame)])

    subprocess.run(command, check=True)
    safe_print(f"model={model_path}")
    safe_print(f"motion={vmd_paths[0]}")
    safe_print(f"output={output_path}")
    if end_frame is not None:
        safe_print(f"range={args.start}-{end_frame}")
    else:
        safe_print(f"range={args.start}-auto")


if __name__ == "__main__":
    main()
