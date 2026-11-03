from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from .reverse_plan import _same_pose
from .vmd import read_camera_vmd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def release_dir() -> Path:
    return repo_root() / "saba_mmd" / "build" / "Release"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze cut-like adjacent camera jumps in GT camera VMD files."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=107)
    parser.add_argument(
        "--latest-ok-csv",
        default=str(release_dir() / "batch_render_result_latest_ok.csv"),
        help="Path to batch_render_result_latest_ok.csv",
    )
    parser.add_argument(
        "--stats-output-root",
        default=str(Path("camera_autoshot") / "analysis" / "gt_cut_stats"),
        help="Directory for aggregate cut statistics outputs.",
    )
    parser.add_argument(
        "--frames-output-root",
        default=str(Path("camera_autoshot") / "analysis" / "gt_cut_frames"),
        help="Directory for per-song cut-frame list outputs.",
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


def _median_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    ordered = sorted(values)

    def at(prob: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * prob))))
        return float(ordered[index])

    return {
        "p50": at(0.50),
        "p75": at(0.75),
        "p90": at(0.90),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": float(ordered[-1]),
    }


def _stringify_frames(frames: list[int]) -> str:
    return ";".join(str(frame) for frame in frames)


def _stringify_boundaries(boundaries: list[list[int]]) -> str:
    return ";".join(f"{start}->{end}" for start, end in boundaries)


def main() -> None:
    args = build_parser().parse_args()
    latest_ok_csv = Path(args.latest_ok_csv).resolve()
    stats_output_root = (repo_root() / args.stats_output_root).resolve()
    frames_output_root = (repo_root() / args.frames_output_root).resolve()
    stats_output_root.mkdir(parents=True, exist_ok=True)
    frames_output_root.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(latest_ok_csv.open("r", encoding="utf-8-sig", newline="")))
    latest_ok_by_index: dict[int, dict[str, str]] = {}
    for row in rows:
        index = _safe_int(row.get("index", ""))
        if index < 0:
            continue
        latest_ok_by_index[index] = row

    per_song_rows: list[dict[str, object]] = []
    per_song_frame_rows: list[dict[str, object]] = []
    missing_indices: list[int] = []

    all_interest_moves: list[float] = []
    all_rotation_degs: list[float] = []
    all_distance_rels: list[float] = []
    all_fov_deltas: list[float] = []

    total_cut_count = 0
    total_duration_sec = 0.0
    analyzed_indices: list[int] = []

    for index in range(args.start, args.end + 1):
        row = latest_ok_by_index.get(index)
        if row is None:
            missing_indices.append(index)
            continue

        camera_path = Path(row.get("camera", ""))
        if not camera_path.exists():
            missing_indices.append(index)
            continue

        analyzed_indices.append(index)
        duration_sec = _safe_float(row.get("export_duration_sec", "0"))
        total_duration_sec += duration_sec

        keys = read_camera_vmd(camera_path)
        adjacent_pairs = 0
        ignored_same_pose_pairs = 0
        cut_count = 0

        interest_moves: list[float] = []
        rotation_degs: list[float] = []
        distance_rels: list[float] = []
        fov_deltas: list[float] = []
        cut_frames: list[int] = []
        cut_boundaries: list[list[int]] = []

        for current_key, next_key in zip(keys, keys[1:]):
            if next_key.frame - current_key.frame != 1:
                continue

            adjacent_pairs += 1
            if _same_pose(current_key, next_key):
                ignored_same_pose_pairs += 1
                continue

            cut_count += 1
            cut_frames.append(int(next_key.frame))
            cut_boundaries.append([int(current_key.frame), int(next_key.frame)])

            interest_move = math.dist(current_key.interest, next_key.interest)
            rotation_deg = math.degrees(
                math.sqrt(sum((a - b) * (a - b) for a, b in zip(current_key.rotate, next_key.rotate)))
            )
            average_distance = max(
                (abs(current_key.distance) + abs(next_key.distance)) * 0.5,
                1e-8,
            )
            distance_rel = abs(next_key.distance - current_key.distance) / average_distance
            fov_delta = abs(next_key.view_angle_deg - current_key.view_angle_deg)

            interest_moves.append(interest_move)
            rotation_degs.append(rotation_deg)
            distance_rels.append(distance_rel)
            fov_deltas.append(float(fov_delta))

            all_interest_moves.append(interest_move)
            all_rotation_degs.append(rotation_deg)
            all_distance_rels.append(distance_rel)
            all_fov_deltas.append(float(fov_delta))

        total_cut_count += cut_count
        duration_min = duration_sec / 60.0 if duration_sec > 0 else 0.0
        cuts_per_minute = cut_count / duration_min if duration_min > 0 else 0.0

        per_song_rows.append(
            {
                "index": index,
                "camera_vmd": str(camera_path),
                "duration_sec": round(duration_sec, 6),
                "duration_min": round(duration_min, 6),
                "camera_key_count": len(keys),
                "adjacent_key_pairs": adjacent_pairs,
                "ignored_same_pose_pairs": ignored_same_pose_pairs,
                "cut_count": cut_count,
                "cuts_per_minute": round(cuts_per_minute, 6),
                "median_interest_move": round(_median_or_zero(interest_moves), 6),
                "median_rotation_deg": round(_median_or_zero(rotation_degs), 6),
                "median_distance_rel": round(_median_or_zero(distance_rels), 6),
                "median_fov_delta": round(_median_or_zero(fov_deltas), 6),
                "mean_interest_move": round(_mean_or_zero(interest_moves), 6),
                "mean_rotation_deg": round(_mean_or_zero(rotation_degs), 6),
                "mean_distance_rel": round(_mean_or_zero(distance_rels), 6),
                "mean_fov_delta": round(_mean_or_zero(fov_deltas), 6),
            }
        )

        per_song_frame_rows.append(
            {
                "index": index,
                "camera_vmd": str(camera_path),
                "cut_count": cut_count,
                "cut_frames": cut_frames,
                "cut_boundaries": cut_boundaries,
                "cut_frames_text": _stringify_frames(cut_frames),
                "cut_boundaries_text": _stringify_boundaries(cut_boundaries),
            }
        )

    per_song_rows.sort(key=lambda row: int(row["index"]))
    per_song_frame_rows.sort(key=lambda row: int(row["index"]))
    by_cut_count_desc = sorted(per_song_rows, key=lambda row: (-int(row["cut_count"]), int(row["index"])))
    by_cpm_desc = sorted(per_song_rows, key=lambda row: (-float(row["cuts_per_minute"]), int(row["index"])))

    summary = {
        "definition": {
            "cut_event": "A pair of consecutive GT camera keys where frame_gap == 1 and the two poses are not identical.",
            "cut_frame": "For each cut event, the stored cut frame is the later key frame in the pair.",
            "cut_boundary": "A stored [start_frame, end_frame] pair where end_frame = start_frame + 1.",
            "ignored_case": "Adjacent keys with effectively the same pose are excluded.",
        },
        "range": {
            "start": args.start,
            "end": args.end,
        },
        "latest_ok_csv": str(latest_ok_csv),
        "stats_output_root": str(stats_output_root),
        "frames_output_root": str(frames_output_root),
        "analyzed_song_count": len(per_song_rows),
        "analyzed_indices": analyzed_indices,
        "missing_indices": missing_indices,
        "total_cut_count": total_cut_count,
        "total_duration_sec": round(total_duration_sec, 6),
        "total_duration_min": round(total_duration_sec / 60.0, 6) if total_duration_sec > 0 else 0.0,
        "overall_cuts_per_minute": round((total_cut_count * 60.0 / total_duration_sec), 6)
        if total_duration_sec > 0
        else 0.0,
        "mean_cut_count_per_song": round(_mean_or_zero([float(row["cut_count"]) for row in per_song_rows]), 6),
        "median_cut_count_per_song": round(_median_or_zero([float(row["cut_count"]) for row in per_song_rows]), 6),
        "mean_cuts_per_minute": round(_mean_or_zero([float(row["cuts_per_minute"]) for row in per_song_rows]), 6),
        "median_cuts_per_minute": round(_median_or_zero([float(row["cuts_per_minute"]) for row in per_song_rows]), 6),
        "global_jump_distribution": {
            "interest_move": _quantiles(all_interest_moves),
            "rotation_deg": _quantiles(all_rotation_degs),
            "distance_rel": _quantiles(all_distance_rels),
            "fov_delta": _quantiles(all_fov_deltas),
        },
        "top_by_cut_count": [
            {
                "index": int(row["index"]),
                "cut_count": int(row["cut_count"]),
                "cuts_per_minute": float(row["cuts_per_minute"]),
            }
            for row in by_cut_count_desc[:15]
        ],
        "top_by_cuts_per_minute": [
            {
                "index": int(row["index"]),
                "cut_count": int(row["cut_count"]),
                "cuts_per_minute": float(row["cuts_per_minute"]),
            }
            for row in by_cpm_desc[:15]
        ],
    }

    summary_json = stats_output_root / "gt_cut_summary.json"
    per_song_json = stats_output_root / "gt_cut_per_song.json"
    per_song_csv = stats_output_root / "gt_cut_per_song.csv"
    cut_frames_json = frames_output_root / "gt_cut_frames_per_song.json"
    cut_frames_csv = frames_output_root / "gt_cut_frames_per_song.csv"

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    per_song_json.write_text(json.dumps(per_song_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    cut_frames_json.write_text(json.dumps(per_song_frame_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    stats_fieldnames = [
        "index",
        "camera_vmd",
        "duration_sec",
        "duration_min",
        "camera_key_count",
        "adjacent_key_pairs",
        "ignored_same_pose_pairs",
        "cut_count",
        "cuts_per_minute",
        "median_interest_move",
        "median_rotation_deg",
        "median_distance_rel",
        "median_fov_delta",
        "mean_interest_move",
        "mean_rotation_deg",
        "mean_distance_rel",
        "mean_fov_delta",
    ]
    with per_song_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=stats_fieldnames)
        writer.writeheader()
        writer.writerows(per_song_rows)

    cut_frame_fieldnames = [
        "index",
        "camera_vmd",
        "cut_count",
        "cut_frames_text",
        "cut_boundaries_text",
    ]
    with cut_frames_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cut_frame_fieldnames)
        writer.writeheader()
        for row in per_song_frame_rows:
            writer.writerow({field: row.get(field, "") for field in cut_frame_fieldnames})

    print(f"Wrote: {summary_json}")
    print(f"Wrote: {per_song_json}")
    print(f"Wrote: {per_song_csv}")
    print(f"Wrote: {cut_frames_json}")
    print(f"Wrote: {cut_frames_csv}")
    print(f"Analyzed songs: {len(per_song_rows)}")
    print(f"Total cuts: {total_cut_count}")
    if total_duration_sec > 0:
        print(f"Overall cuts/min: {total_cut_count * 60.0 / total_duration_sec:.6f}")


if __name__ == "__main__":
    main()
