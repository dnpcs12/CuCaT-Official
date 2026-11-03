from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from shutil import copy2
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent / "results" / "cut_keyframe_multimodal_dataset_dcm"
DEFAULT_KEYFRAME_PATH = WORKSPACE_ROOT / "camera_autoshot" / "analysis" / "gt_keyframes" / "gt_keyframes_per_song.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh embedded keyframe labels in multimodal NPZ datasets from gt_keyframes_per_song.json."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--keyframe-labels", type=Path, default=DEFAULT_KEYFRAME_PATH)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--soft-radius", type=int, default=None)
    parser.add_argument("--no-backup", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_local_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (WORKSPACE_ROOT / path).resolve()


def _song_id_from_index(index: int) -> str:
    return f"c{int(index)}"


def _entry_song_id(entry: dict) -> str | None:
    if "source_song_id" in entry:
        return str(entry["source_song_id"])
    if "index" in entry:
        return _song_id_from_index(int(entry["index"]))
    text = str(entry.get("id", ""))
    if text.startswith("c") and text[1:].isdigit():
        return text
    return None


def _event_frames_to_labels(event_frames: np.ndarray, *, num_frames: int) -> np.ndarray:
    labels = np.zeros(num_frames, dtype=np.float32)
    frames = np.asarray(event_frames, dtype=np.int32)
    frames = frames[(frames >= 0) & (frames < num_frames)]
    if len(frames) > 0:
        labels[np.unique(frames)] = 1.0
    return labels


def _event_frames_to_soft_labels(event_frames: np.ndarray, *, num_frames: int, radius: int) -> np.ndarray:
    soft = np.zeros(num_frames, dtype=np.float32)
    frames = np.asarray(event_frames, dtype=np.int32)
    frames = frames[(frames >= 0) & (frames < num_frames)]
    if len(frames) == 0:
        return soft
    for frame in np.unique(frames):
        start = max(0, int(frame) - int(radius))
        end = min(num_frames, int(frame) + int(radius) + 1)
        distances = np.abs(np.arange(start, end, dtype=np.int32) - int(frame))
        values = 1.0 - distances.astype(np.float32) / float(max(1, radius) + 1)
        soft[start:end] = np.maximum(soft[start:end], values)
    return soft


def _load_keyframes_by_song(path: Path) -> dict[str, np.ndarray]:
    rows = _load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in {path}")
    result: dict[str, np.ndarray] = {}
    for row in rows:
        if not isinstance(row, dict) or "index" not in row:
            continue
        result[_song_id_from_index(int(row["index"]))] = np.asarray(row.get("keyframe_frames", []), dtype=np.int32)
    return result


def _refresh_entry(
    entry: dict,
    *,
    keyframes_by_song: dict[str, np.ndarray],
    selected_indices: set[int] | None,
    soft_radius: int,
) -> dict | None:
    song_id = _entry_song_id(entry)
    if song_id is None or song_id not in keyframes_by_song:
        return None
    if selected_indices is not None and int(song_id[1:]) not in selected_indices:
        return None

    feature_path = _resolve_local_path(str(entry["feature_path"]))
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    with np.load(feature_path) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}

    num_frames = int(arrays["audio_features"].shape[0])
    start_frame = int(entry.get("source_start_frame", 0) or 0)
    end_frame = int(entry.get("source_end_frame", start_frame + num_frames - 1) or (start_frame + num_frames - 1))
    full_frames = keyframes_by_song[song_id]
    local_frames = full_frames[(full_frames >= start_frame) & (full_frames <= end_frame)] - start_frame
    local_frames = np.asarray(local_frames, dtype=np.int32)

    old_count = int(len(arrays.get("keyframe_frames", [])))
    arrays["keyframe_frames"] = local_frames
    arrays["keyframe_labels"] = _event_frames_to_labels(local_frames, num_frames=num_frames)
    arrays["keyframe_soft_labels"] = _event_frames_to_soft_labels(
        local_frames,
        num_frames=num_frames,
        radius=int(soft_radius),
    )
    arrays["keyframe_label_mask"] = np.ones(num_frames, dtype=np.float32)
    np.savez_compressed(feature_path, **arrays)

    entry["keyframe_count"] = int(len(local_frames))
    entry["keyframe_label_available"] = True
    return {
        "id": str(entry.get("id")),
        "song_id": song_id,
        "feature_path": str(feature_path),
        "old_count": old_count,
        "new_count": int(len(local_frames)),
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Dataset manifest not found: {manifest_path}")

    manifest = _load_json(manifest_path)
    songs = manifest.get("songs", [])
    if not isinstance(songs, list):
        raise SystemExit(f"Expected manifest.songs list in {manifest_path}")

    keyframes_by_song = _load_keyframes_by_song(args.keyframe_labels.resolve())
    selected_indices = set(int(index) for index in args.indices) if args.indices else None
    soft_radius = int(args.soft_radius if args.soft_radius is not None else manifest.get("soft_radius_frames", 3))

    if not args.no_backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        copy2(manifest_path, manifest_path.with_name(f"{manifest_path.name}.bak_{timestamp}"))

    updates: list[dict] = []
    for entry in songs:
        updated = _refresh_entry(
            entry,
            keyframes_by_song=keyframes_by_song,
            selected_indices=selected_indices,
            soft_radius=soft_radius,
        )
        if updated is not None:
            updates.append(updated)

    manifest["keyframe_labels"] = str(args.keyframe_labels.resolve())
    manifest["keyframe_refresh"] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "indices": sorted(selected_indices) if selected_indices is not None else "all",
        "soft_radius_frames": soft_radius,
        "updated_entry_count": len(updates),
    }
    _write_json(manifest_path, manifest)
    print(json.dumps(updates, ensure_ascii=False, indent=2))
    print(f"Updated entries: {len(updates)}")
    print(f"Wrote: {manifest_path}")


if __name__ == "__main__":
    main()
