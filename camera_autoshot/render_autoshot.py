from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_BASE_MODEL = Path(__file__).resolve().parents[1] / "local_assets" / "default_render_model.pmx"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def release_dir() -> Path:
    return repo_root() / "saba_mmd" / "build" / "Release"


def load_batch_module():
    build_release = str(release_dir())
    if build_release not in sys.path:
        sys.path.insert(0, build_release)
    import batch_render_gt_range as batch_render  # type: ignore

    return batch_render


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


def looks_like_motion_vmd(batch_render, vmd_path: Path) -> bool:
    stats = batch_render.read_vmd_stats(vmd_path)
    return stats.get("motion_max_frame", 0) > 0 or stats.get("camera_count", 0) == 0


def normalize_motion_path(batch_render, index: int, motion_path: Path) -> Path:
    if motion_path.exists() and looks_like_motion_vmd(batch_render, motion_path):
        return motion_path
    folder = batch_render.BASE_DIR / f"amc{index}"
    preferred = motion_path.name if str(motion_path) else ""
    resolved = batch_render.resolve_vmd(folder, preferred, "motion", index)
    if resolved is None:
        return motion_path
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render an autoshot camera VMD onto the latest-ok assets for an index.")
    parser.add_argument("--index", type=int, required=True, help="Render index to resolve from batch_render_result_latest_ok.csv")
    parser.add_argument("--camera", required=True, help="Camera VMD path")
    parser.add_argument("--output", required=True, help="Final MP4 output path")
    parser.add_argument(
        "--latest-ok-csv",
        default=str(release_dir() / "batch_render_result_latest_ok.csv"),
        help="Path to batch_render_result_latest_ok.csv",
    )
    parser.add_argument("--model-path", default="", help="Optional explicit model override")
    parser.add_argument("--model-alias", default="", help="Optional model alias override")
    parser.add_argument(
        "--fps",
        type=int,
        default=0,
        help="Optional output FPS for faster preview renders. Default keeps the normal render FPS.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Optional output width for faster preview renders. Default keeps the normal render width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Optional output height for faster preview renders. Default keeps the normal render height.",
    )
    parser.add_argument(
        "--no-msaa",
        action="store_true",
        help="Disable MSAA for faster preview renders.",
    )
    parser.add_argument(
        "--msaa-count",
        type=int,
        default=8,
        help="MSAA sample count when MSAA is enabled. Default matches the normal renderer.",
    )
    parser.add_argument(
        "--worker-id",
        default="auto",
        help="Saba worker id to lease. Use auto to pick a free worker from the pool.",
    )
    parser.add_argument(
        "--worker-dir",
        default="",
        help="Optional explicit Saba worker directory. Overrides --worker-id when set.",
    )
    parser.add_argument(
        "--worker-lock-wait-sec",
        type=float,
        default=600.0,
        help="Seconds to wait for a free Saba worker before failing.",
    )
    parser.add_argument(
        "--worker-lock-stale-sec",
        type=float,
        default=12 * 60 * 60,
        help="Seconds after which an ownerless worker lock can be reclaimed.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    batch_render = load_batch_module()

    latest_ok_csv = Path(args.latest_ok_csv)
    row = read_latest_ok_row(latest_ok_csv, args.index)

    camera_path = Path(args.camera).resolve()
    if not camera_path.exists():
        raise SystemExit(f"Camera VMD not found: {camera_path}")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    motion_path = normalize_motion_path(batch_render, args.index, Path(row["motion"]))
    wav_path = Path(row["wav"])
    model_path = normalize_model_path(Path(row["model"]))
    if args.model_path or args.model_alias:
        model_path = batch_render.resolve_model_choice(args.model_path, args.model_alias, model_path)

    if not motion_path.exists():
        raise SystemExit(f"Motion VMD not found: {motion_path}")
    if not wav_path.exists():
        raise SystemExit(f"WAV not found: {wav_path}")
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    try:
        export_frame_count = int(row["export_frame_count"])
        fps = int(row.get("fps", str(batch_render.FPS)) or batch_render.FPS)
        duration_sec = export_frame_count / fps
    except (KeyError, TypeError, ValueError):
        fps = batch_render.FPS
        duration_sec = float(row["export_duration_sec"])

    batch_render.MODEL_PATH = model_path

    worker_dir_arg = Path(args.worker_dir).resolve() if args.worker_dir else None
    with batch_render.acquire_saba_worker(
        args.worker_id,
        worker_dir=worker_dir_arg,
        wait_sec=args.worker_lock_wait_sec,
        stale_sec=args.worker_lock_stale_sec,
        lease_info={
            "script": "camera_autoshot.render_autoshot",
            "index": args.index,
            "output": str(output_path),
        },
    ) as worker:
        temp_output_path = worker.worker_dir / (
            f"__autoshot_tmp_{args.index}_{os.getpid()}_{int(time.time() * 1000)}_{output_path.name}"
        )
        batch_render.write_init_lua(
            motion_path,
            camera_path,
            wav_path,
            temp_output_path.name,
            duration_sec,
            fps=args.fps or None,
            width=args.width or None,
            height=args.height or None,
            msaa_enable=not args.no_msaa,
            msaa_count=args.msaa_count,
            work_dir=worker.worker_dir,
        )
        proc = None
        try:
            proc = subprocess.Popen([str(batch_render.VIEWER_EXE)], cwd=str(worker.worker_dir))
            worker.update(viewer_pid=proc.pid)
            ok = batch_render.wait_until_output_stable(proc, temp_output_path)
            batch_render.terminate_viewer_process(proc)
            batch_render.cleanup_export_frame_dirs(worker.worker_dir)

            if not ok or not temp_output_path.exists() or temp_output_path.stat().st_size <= 0:
                raise SystemExit(f"Render failed for index {args.index}")

            if output_path.exists():
                output_path.unlink()
            shutil.move(str(temp_output_path), str(output_path))
        finally:
            if proc is not None:
                batch_render.terminate_viewer_process(proc)
            batch_render.cleanup_export_frame_dirs(worker.worker_dir)
            worker.update(viewer_pid=0)
            if temp_output_path.exists():
                try:
                    temp_output_path.unlink()
                except PermissionError:
                    time.sleep(1.0)
                    if temp_output_path.exists():
                        temp_output_path.unlink()

    safe_print(f"index={args.index}")
    safe_print(f"worker={worker.worker_id}")
    safe_print(f"model={model_path}")
    safe_print(f"motion={motion_path}")
    safe_print(f"camera={camera_path}")
    safe_print(f"wav={wav_path}")
    safe_print(f"output={output_path}")
    safe_print(f"duration_sec={duration_sec:.6f}")
    if args.fps or args.width or args.height or args.no_msaa or args.msaa_count != 8:
        safe_print(
            "preview_render="
            f"fps={args.fps or batch_render.FPS}, "
            f"width={args.width or batch_render.WIDTH}, "
            f"height={args.height or batch_render.HEIGHT}, "
            f"msaa={'off' if args.no_msaa else args.msaa_count}"
        )


if __name__ == "__main__":
    main()
