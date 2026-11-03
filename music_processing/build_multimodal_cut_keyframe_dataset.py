from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from music_processing.dance_motion_features import (
    align_motion_positions,
    align_motion_rotations,
    build_motion_features,
    load_motion_positions,
    load_motion_transforms,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DATASET_ROOT = Path(__file__).resolve().parent / "results" / "cut_transition_dataset"
DEFAULT_MOTION_ROOT = Path(__file__).resolve().parents[1] / "local_assets" / "Simplified_MotionGlobalTransform"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "cut_keyframe_multimodal_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a frame-level multimodal dataset by aligning audio features with dance motion transforms."
    )
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive).")
    parser.add_argument("--end", type=int, default=107, help="End index (inclusive).")
    parser.add_argument(
        "--indexes",
        type=int,
        nargs="*",
        help="Optional explicit list of indexes. When set, --start/--end are ignored.",
    )
    parser.add_argument(
        "--source-dataset-root",
        type=Path,
        default=DEFAULT_SOURCE_DATASET_ROOT,
        help="Root of the existing frame-level cut-transition dataset.",
    )
    parser.add_argument(
        "--motion-root",
        type=Path,
        default=DEFAULT_MOTION_ROOT,
        help="Directory containing mN_GlobalTransform.json files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root where merged multimodal .npz files will be written.",
    )
    parser.add_argument(
        "--keyframe-labels",
        type=Path,
        default=None,
        help="Optional JSON file containing future keyframe labels. Missing labels stay masked out.",
    )
    parser.add_argument(
        "--soft-radius",
        type=int,
        default=3,
        help="Triangular soft-label radius used for optional keyframe labels.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip songs that already have a saved multimodal .npz file.",
    )
    parser.add_argument(
        "--rotation-feature-set",
        choices=["none", "root_head_world_rot6d", "root_head_wrist_rel_rot6d_vectors"],
        default="none",
        help=(
            "Optional dance rotation/vector features to append. "
            "root_head_wrist_rel_rot6d_vectors includes root/head world rot6d, "
            "wrist relative rot6d, and foot/root/head vectors."
        ),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (WORKSPACE_ROOT / path).resolve()


def _resolve_indexes(args: argparse.Namespace) -> list[int]:
    if args.indexes:
        return sorted(set(int(index) for index in args.indexes))
    return list(range(int(args.start), int(args.end) + 1))


def _normalize_song_id(value: str | int) -> str:
    if isinstance(value, int):
        return f"c{value}"
    text = str(value).strip()
    if not text:
        raise ValueError("Empty song id")
    if text[0] in {"c", "m"} and text[1:].isdigit():
        return f"c{int(text[1:])}"
    if text.isdigit():
        return f"c{int(text)}"
    return text


def _extract_frames_field(payload: dict) -> list[int] | None:
    for key in ("keyframe_frames", "frames", "keyframes", "frame_indices", "labels"):
        value = payload.get(key)
        if isinstance(value, list):
            return [int(item) for item in value]
    return None


def _load_keyframe_label_map(path: Path | None) -> dict[str, list[int]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Keyframe label file not found: {path}")
    payload = _load_json(path)
    label_map: dict[str, list[int]] = {}

    def add_entry(song_id: str | int, frames: list[int]) -> None:
        normalized_id = _normalize_song_id(song_id)
        label_map[normalized_id] = sorted(set(int(frame) for frame in frames))

    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            frames = _extract_frames_field(row)
            if frames is None:
                continue
            song_id = row.get("id", row.get("song_id", row.get("index", row.get("clip_id"))))
            if song_id is None:
                continue
            add_entry(song_id, frames)
        return label_map

    if isinstance(payload, dict):
        if isinstance(payload.get("songs"), list):
            for row in payload["songs"]:
                if not isinstance(row, dict):
                    continue
                frames = _extract_frames_field(row)
                if frames is None:
                    continue
                song_id = row.get("id", row.get("song_id", row.get("index", row.get("clip_id"))))
                if song_id is None:
                    continue
                add_entry(song_id, frames)
            return label_map
        for key, value in payload.items():
            if isinstance(value, list):
                add_entry(key, [int(item) for item in value])
            elif isinstance(value, dict):
                frames = _extract_frames_field(value)
                if frames is not None:
                    add_entry(value.get("id", value.get("song_id", key)), frames)
        return label_map

    raise ValueError(f"Unsupported keyframe label format: {path}")


def _event_frames_to_labels(event_frames: list[int] | np.ndarray, *, num_frames: int) -> np.ndarray:
    labels = np.zeros(num_frames, dtype=np.float32)
    frames = np.asarray(event_frames, dtype=np.int32)
    frames = frames[(frames >= 0) & (frames < num_frames)]
    if len(frames) > 0:
        labels[np.unique(frames)] = 1.0
    return labels


def _event_frames_to_soft_labels(event_frames: list[int] | np.ndarray, *, num_frames: int, radius: int) -> np.ndarray:
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


def main() -> None:
    args = parse_args()
    indexes = set(_resolve_indexes(args))
    source_manifest_path = args.source_dataset_root / "manifest.json"
    if not source_manifest_path.exists():
        raise SystemExit(f"Source dataset manifest not found: {source_manifest_path}")
    source_manifest = _load_json(source_manifest_path)
    if not isinstance(source_manifest, dict):
        raise SystemExit(f"Unexpected source manifest format: {source_manifest_path}")

    source_songs = source_manifest.get("songs", [])
    if not isinstance(source_songs, list) or not source_songs:
        raise SystemExit(f"No songs found in source manifest: {source_manifest_path}")

    keyframe_label_map = _load_keyframe_label_map(args.keyframe_labels)
    songs_out_dir = args.output_root / "songs"
    songs_out_dir.mkdir(parents=True, exist_ok=True)

    output_songs: list[dict] = []
    skipped_songs: list[dict] = []
    audio_feature_names = list(source_manifest.get("feature_names", []))
    dance_feature_names: list[str] | None = None

    for entry in source_songs:
        index = int(entry["index"])
        if index not in indexes:
            continue

        song_id = str(entry["id"])
        output_path = songs_out_dir / f"{song_id}.npz"
        if args.skip_existing and output_path.exists():
            output_songs.append(
                {
                    "id": song_id,
                    "index": index,
                    "feature_path": str(output_path),
                    "skipped_existing": True,
                }
            )
            continue

        source_feature_path = _resolve_path(entry["feature_path"])
        motion_path = args.motion_root / f"m{index}_GlobalTransform.json"

        if not source_feature_path.exists():
            skipped_songs.append({"id": song_id, "index": index, "reason": f"missing audio dataset file: {source_feature_path}"})
            continue
        if not motion_path.exists():
            skipped_songs.append({"id": song_id, "index": index, "reason": f"missing motion file: {motion_path}"})
            continue

        source_data = np.load(source_feature_path)
        audio_features = np.asarray(source_data["features"], dtype=np.float32)
        cut_labels = np.asarray(source_data["labels"], dtype=np.float32)
        cut_soft_labels = np.asarray(source_data["soft_labels"], dtype=np.float32)
        frame_times_sec = np.asarray(source_data["frame_times_sec"], dtype=np.float32)
        cut_frames = np.asarray(source_data["cut_frames"], dtype=np.int32)
        num_frames = int(audio_features.shape[0])

        rotation_feature_set = str(args.rotation_feature_set)
        include_root_head_world_rot6d = rotation_feature_set in {
            "root_head_world_rot6d",
            "root_head_wrist_rel_rot6d_vectors",
        }
        include_wrist_relative_rot6d = rotation_feature_set == "root_head_wrist_rel_rot6d_vectors"
        include_foot_root_head_vectors = rotation_feature_set == "root_head_wrist_rel_rot6d_vectors"
        if include_root_head_world_rot6d or include_wrist_relative_rot6d or include_foot_root_head_vectors:
            _, raw_positions, raw_rotations, _ = load_motion_transforms(motion_path)
            aligned_positions, alignment_note = align_motion_positions(raw_positions, target_frames=num_frames)
            aligned_rotations, rotation_alignment_note = align_motion_rotations(raw_rotations, target_frames=num_frames)
            if rotation_alignment_note != alignment_note:
                alignment_note = f"positions_{alignment_note};rotations_{rotation_alignment_note}"
            dance_features, current_dance_feature_names = build_motion_features(
                aligned_positions,
                rotations=aligned_rotations,
                include_root_head_world_rot6d=include_root_head_world_rot6d,
                include_wrist_relative_rot6d=include_wrist_relative_rot6d,
                include_foot_root_head_vectors=include_foot_root_head_vectors,
            )
        else:
            _, raw_positions, _ = load_motion_positions(motion_path)
            aligned_positions, alignment_note = align_motion_positions(raw_positions, target_frames=num_frames)
            dance_features, current_dance_feature_names = build_motion_features(aligned_positions)

        if dance_feature_names is None:
            dance_feature_names = current_dance_feature_names
        elif dance_feature_names != current_dance_feature_names:
            raise SystemExit(f"Dance feature-name mismatch detected at {motion_path}")

        keyframe_frames = np.asarray(keyframe_label_map.get(song_id, []), dtype=np.int32)
        keyframe_labels = _event_frames_to_labels(keyframe_frames, num_frames=num_frames)
        keyframe_soft_labels = _event_frames_to_soft_labels(
            keyframe_frames,
            num_frames=num_frames,
            radius=int(args.soft_radius),
        )
        keyframe_label_mask = (
            np.ones(num_frames, dtype=np.float32) if song_id in keyframe_label_map else np.zeros(num_frames, dtype=np.float32)
        )

        np.savez_compressed(
            output_path,
            audio_features=audio_features,
            dance_features=dance_features,
            cut_labels=cut_labels,
            cut_soft_labels=cut_soft_labels,
            keyframe_labels=keyframe_labels,
            keyframe_soft_labels=keyframe_soft_labels,
            keyframe_label_mask=keyframe_label_mask,
            frame_times_sec=frame_times_sec,
            cut_frames=cut_frames,
            keyframe_frames=keyframe_frames,
        )

        output_songs.append(
            {
                "id": song_id,
                "index": index,
                "fps": float(entry["fps"]),
                "num_frames": num_frames,
                "duration_sec": float(entry["duration_sec"]),
                "audio_feature_dim": int(audio_features.shape[1]),
                "dance_feature_dim": int(dance_features.shape[1]),
                "joint_count": int(aligned_positions.shape[1]),
                "motion_source_frames": int(raw_positions.shape[0]),
                "motion_alignment": alignment_note,
                "cut_count": int(len(cut_frames)),
                "keyframe_label_available": bool(song_id in keyframe_label_map),
                "keyframe_count": int(len(keyframe_frames)),
                "feature_path": str(output_path),
                "source_feature_path": str(source_feature_path),
                "motion_path": str(motion_path),
            }
        )

    manifest = {
        "dataset_type": "multimodal_cut_keyframe",
        "version": 1,
        "source_dataset_manifest": str(source_manifest_path),
        "motion_root": str(args.motion_root),
        "keyframe_labels": str(args.keyframe_labels) if args.keyframe_labels is not None else None,
        "soft_radius_frames": int(args.soft_radius),
        "dance_rotation_feature_set": str(args.rotation_feature_set),
        "song_count": len(output_songs),
        "skipped_count": len(skipped_songs),
        "audio_feature_names": audio_feature_names,
        "dance_feature_names": dance_feature_names or [],
        "songs": output_songs,
        "skipped": skipped_songs,
    }
    manifest_path = args.output_root / "manifest.json"
    _write_json(manifest_path, manifest)
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved songs: {len(output_songs)}")
    if skipped_songs:
        print(f"Skipped songs: {len(skipped_songs)}")


if __name__ == "__main__":
    main()
