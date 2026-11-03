from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from music_processing.dance_motion_features import BODY_PART_SPECS, DEFAULT_MOTION_ROOT
from music_processing.motion_body_activity import build_body_part_activity_result


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "motion_body_frame_labels"
DEFAULT_GROUP_THRESHOLD_SCALES = {
    "head": 1.27,
    "arm": 0.76,
    "leg": 0.72,
    "hand": 0.67,
    "full_body": 1.0,
}
DEFAULT_GROUP_RAW_MINS = {
    "head": 0.041,
    "arm": 0.107,
    "leg": 0.029,
    "hand": 0.156,
    "full_body": 0.005,
}


def parse_key_values(items: list[str] | None, defaults: dict[str, float]) -> dict[str, float]:
    values = dict(defaults)
    if not items:
        return values
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Expected key=value setting, got: {item}")
        key, raw_value = item.split("=", 1)
        values[key.strip()] = float(raw_value)
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-frame binary body-part activity labels from motion body activity windows."
    )
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive).")
    parser.add_argument("--end", type=int, default=107, help="End index (inclusive).")
    parser.add_argument("--indexes", type=int, nargs="*", help="Explicit index list. Overrides --start/--end.")
    parser.add_argument("--motion-root", type=Path, default=DEFAULT_MOTION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--window-frames", type=int, default=30)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--reducer", choices=["mean", "max", "p75", "p90"], default="p75")
    parser.add_argument("--score-mode", choices=["relative_to_global", "absolute"], default="relative_to_global")
    parser.add_argument("--activity-quantile", type=float, default=75.0)
    parser.add_argument("--min-active-score", type=float, default=0.0)
    parser.add_argument(
        "--threshold-scale",
        action="append",
        metavar="GROUP=VALUE",
        help="Override group threshold multiplier. Example: --threshold-scale arm=0.76",
    )
    parser.add_argument(
        "--raw-min",
        action="append",
        metavar="GROUP=VALUE",
        help="Override group raw-score minimum. Example: --raw-min hand=0.156",
    )
    return parser.parse_args()


def json_ready_float(value: float) -> float:
    return round(float(value), 6)


def resolve_indexes(args: argparse.Namespace) -> list[int]:
    if args.indexes:
        return sorted(set(int(index) for index in args.indexes))
    return list(range(int(args.start), int(args.end) + 1))


def build_frame_labels(
    activity: dict,
    *,
    part_keys: list[str],
    threshold_scales: dict[str, float],
    raw_mins: dict[str, float],
) -> tuple[np.ndarray, dict[str, int]]:
    num_frames = int(activity["num_frames"])
    first_frame = int(activity["frame_range"]["start"])
    labels = np.zeros((num_frames, len(part_keys)), dtype=np.uint8)

    for window in activity["windows"]:
        start_row = int(window["start_frame"]) - first_frame
        end_row = int(window["end_frame"]) - first_frame + 1
        start_row = max(0, start_row)
        end_row = min(num_frames, end_row)
        if end_row <= start_row:
            continue

        for part_index, part_key in enumerate(part_keys):
            part_meta = activity["body_parts"][part_key]
            state = window["body_parts"][part_key]
            group = str(part_meta.get("threshold_group", part_key))
            threshold = float(state["threshold"]) * float(threshold_scales.get(group, 1.0))
            raw_min = float(raw_mins.get(group, 0.0))
            active = float(state["score"]) >= threshold and float(state.get("raw_score", 0.0)) >= raw_min
            if active:
                labels[start_row:end_row, part_index] = 1

    active_frame_counts = {
        part_key: int(labels[:, part_index].sum()) for part_index, part_key in enumerate(part_keys)
    }
    return labels, active_frame_counts


def write_csv(path: Path, frame_numbers: np.ndarray, labels: np.ndarray, part_keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", *part_keys])
        for frame, row in zip(frame_numbers.tolist(), labels.tolist()):
            writer.writerow([int(frame), *[int(value) for value in row]])


def write_npz(path: Path, frame_numbers: np.ndarray, labels: np.ndarray, part_keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frames=frame_numbers.astype(np.int32),
        labels=labels.astype(np.uint8),
        part_keys=np.asarray(part_keys),
    )


def main() -> None:
    args = parse_args()
    threshold_scales = parse_key_values(args.threshold_scale, DEFAULT_GROUP_THRESHOLD_SCALES)
    raw_mins = parse_key_values(args.raw_min, DEFAULT_GROUP_RAW_MINS)
    part_keys = [spec.key for spec in BODY_PART_SPECS]

    songs_dir = Path(args.output_root) / "songs"
    npz_dir = songs_dir / "npz"
    csv_dir = songs_dir / "csv"
    skipped: list[dict] = []
    songs: list[dict] = []

    for index in resolve_indexes(args):
        song_id = f"c{int(index)}"
        motion_path = Path(args.motion_root) / f"m{int(index)}_GlobalTransform.json"
        if not motion_path.exists():
            skipped.append({"id": song_id, "reason": f"Motion file not found: {motion_path}"})
            continue

        activity = build_body_part_activity_result(
            index=int(index),
            motion_path=motion_path,
            motion_root=Path(args.motion_root),
            fps=float(args.fps),
            window_frames=int(args.window_frames),
            stride=int(args.stride),
            reducer=str(args.reducer),
            activity_quantile=float(args.activity_quantile),
            min_active_score=float(args.min_active_score),
            score_mode=str(args.score_mode),
        )
        frame_numbers = np.arange(
            int(activity["frame_range"]["start"]),
            int(activity["frame_range"]["end"]) + 1,
            dtype=np.int32,
        )
        labels, active_frame_counts = build_frame_labels(
            activity,
            part_keys=part_keys,
            threshold_scales=threshold_scales,
            raw_mins=raw_mins,
        )

        npz_path = npz_dir / f"{song_id}.npz"
        csv_path = csv_dir / f"{song_id}.csv"
        write_npz(npz_path, frame_numbers, labels, part_keys)
        write_csv(csv_path, frame_numbers, labels, part_keys)

        songs.append(
            {
                "id": song_id,
                "index": int(index),
                "motion_path": str(motion_path.resolve()),
                "num_frames": int(labels.shape[0]),
                "frame_range": activity["frame_range"],
                "npz_path": str(npz_path),
                "csv_path": str(csv_path),
                "active_frame_counts": active_frame_counts,
            }
        )

    manifest = {
        "label_type": "motion_body_frame_binary",
        "description": "Per-frame binary labels expanded from overlapping motion-body activity windows. A frame is 1 for a part if any covering window is active for that part.",
        "fps": float(args.fps),
        "window_frames": int(args.window_frames),
        "window_duration_sec": json_ready_float(float(args.window_frames) / float(args.fps)),
        "stride_frames": int(args.stride),
        "stride_sec": json_ready_float(float(args.stride) / float(args.fps)),
        "reducer": str(args.reducer),
        "score_mode": str(args.score_mode),
        "activity_quantile": float(args.activity_quantile),
        "threshold_scales": {key: float(value) for key, value in threshold_scales.items()},
        "raw_mins": {key: float(value) for key, value in raw_mins.items()},
        "part_keys": part_keys,
        "song_count": int(len(songs)),
        "songs": songs,
        "skipped": skipped,
    }

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote: {output_root / 'manifest.json'}")
    print(f"Songs: {len(songs)}")
    print(f"Skipped: {len(skipped)}")


if __name__ == "__main__":
    main()
