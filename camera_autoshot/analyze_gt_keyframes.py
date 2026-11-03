from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from .vmd import read_camera_vmd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def release_dir() -> Path:
    return repo_root() / "saba_mmd" / "build" / "Release"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract GT camera keyframe timings from camera VMD files."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=107)
    parser.add_argument(
        "--latest-ok-csv",
        default=str(release_dir() / "batch_render_result_latest_ok.csv"),
        help="Path to batch_render_result_latest_ok.csv",
    )
    parser.add_argument(
        "--output-root",
        default=str(Path("camera_autoshot") / "analysis" / "gt_keyframes"),
        help="Directory for GT keyframe timing outputs.",
    )
    return parser


def _safe_float(text: str, default: float = 0.0) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _safe_int(text: str, default: int = -1) -> int:
    try:
        return int(text)
    except (TypeError, ValueError):
        return default


def _mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def _median_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _stringify_ints(values: list[int]) -> str:
    return ";".join(str(value) for value in values)


def _stringify_floats(values: list[float]) -> str:
    return ";".join(f"{value:.6f}" for value in values)


def _intervals(values: list[int]) -> list[int]:
    if len(values) < 2:
        return []
    return [values[index + 1] - values[index] for index in range(len(values) - 1)]


def main() -> None:
    args = build_parser().parse_args()
    latest_ok_csv = Path(args.latest_ok_csv).resolve()
    output_root = (repo_root() / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(latest_ok_csv.open("r", encoding="utf-8-sig", newline="")))
    latest_ok_by_index: dict[int, dict[str, str]] = {}
    for row in rows:
        index = _safe_int(row.get("index", ""))
        if index < 0:
            continue
        latest_ok_by_index[index] = row

    per_song_rows: list[dict[str, object]] = []
    missing_indices: list[int] = []

    total_duration_sec = 0.0
    total_keyframe_count = 0
    analyzed_indices: list[int] = []

    key_counts: list[float] = []
    keyframes_per_minute_values: list[float] = []
    interval_frame_values: list[float] = []
    interval_second_values: list[float] = []

    for index in range(args.start, args.end + 1):
        row = latest_ok_by_index.get(index)
        if row is None:
            missing_indices.append(index)
            continue

        camera_path = Path(row.get("camera", ""))
        if not camera_path.exists():
            missing_indices.append(index)
            continue

        keys = read_camera_vmd(camera_path)
        frame_list = [int(key.frame) for key in keys]
        duration_sec = _safe_float(row.get("export_duration_sec", "0"))
        duration_min = duration_sec / 60.0 if duration_sec > 0 else 0.0
        time_list_sec = [round(frame / 30.0, 6) for frame in frame_list]
        interval_frames = _intervals(frame_list)
        interval_seconds = [round(value / 30.0, 6) for value in interval_frames]
        keyframe_count = len(frame_list)
        keyframes_per_minute = (keyframe_count / duration_min) if duration_min > 0 else 0.0

        analyzed_indices.append(index)
        total_duration_sec += duration_sec
        total_keyframe_count += keyframe_count
        key_counts.append(float(keyframe_count))
        keyframes_per_minute_values.append(float(keyframes_per_minute))
        interval_frame_values.extend(float(value) for value in interval_frames)
        interval_second_values.extend(float(value) for value in interval_seconds)

        per_song_rows.append(
            {
                "index": index,
                "camera_vmd": str(camera_path),
                "duration_sec": round(duration_sec, 6),
                "duration_min": round(duration_min, 6),
                "keyframe_count": keyframe_count,
                "keyframes_per_minute": round(keyframes_per_minute, 6),
                "first_key_frame": frame_list[0] if frame_list else "",
                "last_key_frame": frame_list[-1] if frame_list else "",
                "min_interval_frames": min(interval_frames) if interval_frames else "",
                "max_interval_frames": max(interval_frames) if interval_frames else "",
                "median_interval_frames": round(_median_or_zero([float(v) for v in interval_frames]), 6),
                "mean_interval_frames": round(_mean_or_zero([float(v) for v in interval_frames]), 6),
                "median_interval_sec": round(_median_or_zero([float(v) for v in interval_seconds]), 6),
                "mean_interval_sec": round(_mean_or_zero([float(v) for v in interval_seconds]), 6),
                "keyframe_frames": frame_list,
                "keyframe_times_sec": time_list_sec,
                "keyframe_frames_text": _stringify_ints(frame_list),
                "keyframe_times_sec_text": _stringify_floats(time_list_sec),
            }
        )

    per_song_rows.sort(key=lambda row: int(row["index"]))
    by_key_count_desc = sorted(per_song_rows, key=lambda row: (-int(row["keyframe_count"]), int(row["index"])))
    by_kpm_desc = sorted(
        per_song_rows,
        key=lambda row: (-float(row["keyframes_per_minute"]), int(row["index"])),
    )

    summary = {
        "definition": {
            "keyframe": "Every camera key stored in the GT camera VMD.",
            "keyframe_time_sec": "Camera key frame divided by 30 fps.",
            "interval": "Difference between consecutive key frames.",
        },
        "range": {
            "start": args.start,
            "end": args.end,
        },
        "latest_ok_csv": str(latest_ok_csv),
        "output_root": str(output_root),
        "analyzed_song_count": len(per_song_rows),
        "analyzed_indices": analyzed_indices,
        "missing_indices": missing_indices,
        "total_keyframe_count": total_keyframe_count,
        "total_duration_sec": round(total_duration_sec, 6),
        "total_duration_min": round(total_duration_sec / 60.0, 6) if total_duration_sec > 0 else 0.0,
        "overall_keyframes_per_minute": round((total_keyframe_count * 60.0 / total_duration_sec), 6)
        if total_duration_sec > 0
        else 0.0,
        "mean_keyframe_count_per_song": round(_mean_or_zero(key_counts), 6),
        "median_keyframe_count_per_song": round(_median_or_zero(key_counts), 6),
        "mean_keyframes_per_minute": round(_mean_or_zero(keyframes_per_minute_values), 6),
        "median_keyframes_per_minute": round(_median_or_zero(keyframes_per_minute_values), 6),
        "mean_interval_frames_global": round(_mean_or_zero(interval_frame_values), 6),
        "median_interval_frames_global": round(_median_or_zero(interval_frame_values), 6),
        "mean_interval_sec_global": round(_mean_or_zero(interval_second_values), 6),
        "median_interval_sec_global": round(_median_or_zero(interval_second_values), 6),
        "top_by_keyframe_count": [
            {
                "index": int(row["index"]),
                "keyframe_count": int(row["keyframe_count"]),
                "keyframes_per_minute": float(row["keyframes_per_minute"]),
            }
            for row in by_key_count_desc[:15]
        ],
        "top_by_keyframes_per_minute": [
            {
                "index": int(row["index"]),
                "keyframe_count": int(row["keyframe_count"]),
                "keyframes_per_minute": float(row["keyframes_per_minute"]),
            }
            for row in by_kpm_desc[:15]
        ],
    }

    summary_json = output_root / "gt_keyframe_summary.json"
    per_song_json = output_root / "gt_keyframes_per_song.json"
    per_song_csv = output_root / "gt_keyframes_per_song.csv"

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    per_song_json.write_text(json.dumps(per_song_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_fieldnames = [
        "index",
        "camera_vmd",
        "duration_sec",
        "duration_min",
        "keyframe_count",
        "keyframes_per_minute",
        "first_key_frame",
        "last_key_frame",
        "min_interval_frames",
        "max_interval_frames",
        "median_interval_frames",
        "mean_interval_frames",
        "median_interval_sec",
        "mean_interval_sec",
        "keyframe_frames_text",
        "keyframe_times_sec_text",
    ]
    with per_song_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames)
        writer.writeheader()
        for row in per_song_rows:
            writer.writerow({field: row.get(field, "") for field in csv_fieldnames})

    print(f"Wrote: {summary_json}")
    print(f"Wrote: {per_song_json}")
    print(f"Wrote: {per_song_csv}")
    print(f"Analyzed songs: {len(per_song_rows)}")
    print(f"Total keyframes: {total_keyframe_count}")
    if total_duration_sec > 0:
        print(f"Overall keyframes/min: {total_keyframe_count * 60.0 / total_duration_sec:.6f}")


if __name__ == "__main__":
    main()
