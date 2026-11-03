from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BODY_ROOT = Path(__file__).resolve().parent / "results" / "motion_body_frame_labels"
DEFAULT_GT_CUT_PATH = WORKSPACE_ROOT / "camera_autoshot" / "analysis" / "gt_cut_frames" / "gt_cut_frames_per_song.json"
DEFAULT_SPLIT_PATH = WORKSPACE_ROOT / "datasets" / "music_dance_song_split.json"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "cut_transition_dataset_motion_body_labels"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cut-transition dataset using only per-frame motion body-part activation labels."
    )
    parser.add_argument("--body-root", type=Path, default=DEFAULT_BODY_ROOT)
    parser.add_argument("--gt-cut-path", type=Path, default=DEFAULT_GT_CUT_PATH)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--soft-radius",
        type=int,
        default=3,
        help="Triangular soft-label radius in frames around each GT cut.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(path.resolve())


def _song_id_from_index(index: int | str) -> str:
    text = str(index)
    return text if text.startswith("c") else f"c{int(text)}"


def _load_cut_map(path: Path) -> dict[str, list[int]]:
    rows = _load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in GT cut file: {path}")
    cut_map: dict[str, list[int]] = {}
    for row in rows:
        if not isinstance(row, dict) or "index" not in row:
            continue
        song_id = _song_id_from_index(row["index"])
        cut_map[song_id] = [int(frame) for frame in row.get("cut_frames", [])]
    return cut_map


def _soft_labels(cut_frames: np.ndarray, num_frames: int, radius: int) -> np.ndarray:
    soft = np.zeros(num_frames, dtype=np.float32)
    if num_frames <= 0:
        return soft
    radius = max(0, int(radius))
    for frame in cut_frames:
        if frame < 0 or frame >= num_frames:
            continue
        if radius <= 0:
            soft[int(frame)] = 1.0
            continue
        start = max(0, int(frame) - radius)
        end = min(num_frames - 1, int(frame) + radius)
        for idx in range(start, end + 1):
            value = 1.0 - (abs(idx - int(frame)) / float(radius + 1))
            if value > soft[idx]:
                soft[idx] = float(value)
    return soft


def _cut_labels(cut_frames: np.ndarray, num_frames: int) -> np.ndarray:
    labels = np.zeros(num_frames, dtype=np.float32)
    valid = cut_frames[(cut_frames >= 0) & (cut_frames < num_frames)]
    if len(valid) > 0:
        labels[valid.astype(np.int32)] = 1.0
    return labels


def _load_split(path: Path, available_ids: set[str]) -> tuple[dict[str, list[str]], list[str]]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected split JSON object: {path}")
    source = payload["split"] if isinstance(payload.get("split"), dict) else payload
    split = {
        name: [str(song_id) for song_id in source.get(name, []) if str(song_id) in available_ids]
        for name in ("train", "val", "test")
    }
    excluded = [str(song_id) for song_id in source.get("excluded", []) if str(song_id) in available_ids]
    return split, excluded


def main() -> None:
    args = parse_args()
    body_manifest_path = args.body_root / "manifest.json"
    if not body_manifest_path.exists():
        raise SystemExit(f"Motion-body manifest not found: {body_manifest_path}")
    if not args.gt_cut_path.exists():
        raise SystemExit(f"GT cut file not found: {args.gt_cut_path}")
    if args.output_root.exists() and args.force:
        shutil.rmtree(args.output_root)

    body_manifest = _load_json(body_manifest_path)
    if not isinstance(body_manifest, dict):
        raise SystemExit(f"Invalid motion-body manifest: {body_manifest_path}")
    body_songs = body_manifest.get("songs", [])
    if not body_songs:
        raise SystemExit(f"No songs found in motion-body manifest: {body_manifest_path}")

    cut_map = _load_cut_map(args.gt_cut_path)
    available_ids = {str(entry["id"]) for entry in body_songs}
    split, excluded = _load_split(args.split_path, available_ids) if args.split_path.exists() else (
        {"train": sorted(available_ids), "val": [], "test": []},
        [],
    )

    songs_dir = args.output_root / "songs"
    songs_dir.mkdir(parents=True, exist_ok=True)
    part_keys = [str(part) for part in body_manifest.get("part_keys", [])]
    feature_names = [f"body_active_{part}" for part in part_keys]
    converted_songs: list[dict] = []
    skipped: list[dict] = []

    for entry in sorted(body_songs, key=lambda item: int(item.get("index", 0))):
        song_id = str(entry["id"])
        npz_path = Path(str(entry.get("npz_path", "")))
        if not npz_path.exists():
            skipped.append({"id": song_id, "reason": f"missing body npz: {npz_path}"})
            continue
        if song_id not in cut_map:
            skipped.append({"id": song_id, "reason": "missing GT cut frames"})
            continue

        with np.load(npz_path, allow_pickle=False) as data:
            features = np.asarray(data["labels"], dtype=np.float32)
            frames = np.asarray(data["frames"], dtype=np.int32)
            if not feature_names:
                feature_names = [f"body_active_{part}" for part in np.asarray(data["part_keys"]).astype(str).tolist()]

        num_frames = int(features.shape[0])
        cut_frames = np.asarray(cut_map[song_id], dtype=np.int32)
        cut_frames = np.unique(cut_frames[(cut_frames >= 0) & (cut_frames < num_frames)])
        labels = _cut_labels(cut_frames, num_frames)
        soft_labels = _soft_labels(cut_frames, num_frames, int(args.soft_radius))

        output_npz_path = songs_dir / f"{song_id}.npz"
        np.savez_compressed(
            output_npz_path,
            features=features,
            labels=labels,
            soft_labels=soft_labels,
            cut_frames=cut_frames,
            frames=frames,
            body_part_keys=np.asarray(part_keys, dtype=str),
        )

        converted_songs.append(
            {
                "id": song_id,
                "index": int(entry.get("index", song_id.lstrip("c"))),
                "fps": float(body_manifest.get("fps", 30.0)),
                "num_frames": num_frames,
                "duration_sec": round(num_frames / float(body_manifest.get("fps", 30.0)), 6),
                "cut_count": int(len(cut_frames)),
                "positive_frame_count": int(len(cut_frames)),
                "positive_frame_ratio": float(len(cut_frames) / max(1, num_frames)),
                "feature_path": _repo_relative(output_npz_path),
                "feature_dim": int(features.shape[1]),
                "source_body_npz_path": str(npz_path),
            }
        )

    converted_ids = {song["id"] for song in converted_songs}
    split = {name: [song_id for song_id in split[name] if song_id in converted_ids] for name in ("train", "val", "test")}
    excluded = [song_id for song_id in excluded if song_id in converted_ids]

    output_manifest = {
        "dataset_type": "motion_body_labels_cut_transition",
        "version": 1,
        "description": "Frame-level cut-transition dataset using only 8D motion body-part binary activation features.",
        "source_body_manifest": str(body_manifest_path),
        "source_gt_cut_frames": str(args.gt_cut_path),
        "source_split": str(args.split_path) if args.split_path.exists() else None,
        "fps": float(body_manifest.get("fps", 30.0)),
        "soft_radius": int(args.soft_radius),
        "feature_names": feature_names,
        "feature_dim": len(feature_names),
        "song_count": len(converted_songs),
        "total_frames": int(sum(song["num_frames"] for song in converted_songs)),
        "total_cuts": int(sum(song["cut_count"] for song in converted_songs)),
        "positive_frame_count": int(sum(song["positive_frame_count"] for song in converted_songs)),
        "positive_frame_ratio": float(
            sum(song["positive_frame_count"] for song in converted_songs)
            / max(1, sum(song["num_frames"] for song in converted_songs))
        ),
        "split": {**split, "excluded": excluded},
        "songs": converted_songs,
        "skipped": skipped,
    }
    _write_json(args.output_root / "manifest.json", output_manifest)

    print(f"Wrote dataset: {args.output_root}")
    print(
        "Songs train/val/test/excluded: "
        f"{len(split['train'])}/{len(split['val'])}/{len(split['test'])}/{len(excluded)}"
    )
    print(f"Feature dim: {len(feature_names)}")
    print(f"Total frames/cuts: {output_manifest['total_frames']}/{output_manifest['total_cuts']}")
    if skipped:
        print(f"Skipped: {len(skipped)}")


if __name__ == "__main__":
    main()
