from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .reverse_plan import infer_plan_keys


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
    build_release = str((Path(__file__).resolve().parents[1] / "saba_mmd" / "build" / "Release"))
    if build_release not in sys.path:
        sys.path.insert(0, build_release)
    import batch_render_gt_range as batch_render  # type: ignore

    folder = batch_render.BASE_DIR / f"amc{index}"
    preferred = camera_path.name if str(camera_path) else ""
    resolved = batch_render.resolve_vmd(folder, preferred, "camera", index)
    if resolved is None:
        return camera_path
    return resolved


def _write_manifest_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "index",
        "status",
        "camera",
        "targets",
        "output",
        "debug_json",
        "keyframe_count",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-infer camera_poses-style JSON plans from base camera VMDs in batch_render_result_latest_ok.csv.")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument(
        "--latest-ok-csv",
        default=str(Path("saba_mmd") / "build" / "Release" / "batch_render_result_latest_ok.csv"),
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
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory to write inferred plans into. Per-index subfolders will be created inside.",
    )
    parser.add_argument("--sample-fps", type=float, default=0.0)
    parser.add_argument("--timeline-fps", type=float, default=30.0)
    parser.add_argument("--aspect-ratio", type=float, default=16.0 / 9.0)
    parser.add_argument("--omit-camera-reason", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    latest_ok_csv = Path(args.latest_ok_csv).resolve()
    targets_root = Path(args.targets_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    latest_rows = _load_latest_ok_rows(latest_ok_csv)
    manifest_rows: list[dict[str, object]] = []

    for index in range(args.start, args.end + 1):
        source = latest_rows.get(index)
        item_dir = output_root / f"c{index}"
        item_dir.mkdir(parents=True, exist_ok=True)

        plan_path = item_dir / "camera_poses_inferred.json"
        debug_path = item_dir / "camera_poses_inferred_debug.json"
        targets_path = targets_root / f"c{index}" / args.targets_filename

        row: dict[str, object] = {
            "index": index,
            "status": "",
            "camera": "",
            "targets": str(targets_path),
            "output": str(plan_path),
            "debug_json": str(debug_path),
            "keyframe_count": "",
        }

        if source is None:
            row["status"] = "missing-latest-ok"
            manifest_rows.append(row)
            continue

        camera_path = _resolve_camera_path(index, Path(source.get("camera", "")))
        row["camera"] = str(camera_path)

        if not camera_path.exists():
            row["status"] = "missing-camera-vmd"
            manifest_rows.append(row)
            continue

        if not targets_path.exists():
            row["status"] = "missing-targets-json"
            manifest_rows.append(row)
            continue

        inferred_keys, debug_rows = infer_plan_keys(
            camera_path,
            targets_path,
            sample_fps=args.sample_fps,
            timeline_fps=args.timeline_fps,
            aspect_ratio=args.aspect_ratio,
        )

        payload: dict[str, object] = {"keyframes": []}
        if args.sample_fps > 0.0:
            payload["sample_fps"] = float(args.sample_fps)

        for key in inferred_keys:
            key_row = asdict(key)
            if args.omit_camera_reason:
                key_row.pop("camera_reason", None)
            payload["keyframes"].append(key_row)

        plan_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        debug_path.write_text(
            json.dumps(
                {
                    "index": index,
                    "camera_vmd": str(camera_path),
                    "targets": str(targets_path),
                    "keyframes": debug_rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        row["status"] = "ok"
        row["keyframe_count"] = len(inferred_keys)
        manifest_rows.append(row)
        print(f"[ok] c{index}: {len(inferred_keys)} keys")

    manifest_json_path = output_root / "manifest.json"
    manifest_csv_path = output_root / "manifest.csv"
    manifest_json_path.write_text(json.dumps(manifest_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_manifest_csv(manifest_rows, manifest_csv_path)
    print(f"Wrote manifest: {manifest_json_path}")
    print(f"Wrote manifest: {manifest_csv_path}")


if __name__ == "__main__":
    main()
