from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DATASET_ROOT = Path(__file__).resolve().parent / "results" / "cut_keyframe_multimodal_dataset"
DEFAULT_DCM_SPLIT_ROOT = WORKSPACE_ROOT / "local_assets" / "split"
DEFAULT_DCM_OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "cut_keyframe_multimodal_dataset_dcm"
DEFAULT_SONG_SPLIT_PATH = DEFAULT_SOURCE_DATASET_ROOT / "music_dance_song_split.json"
DEFAULT_DCM_SUMMARY_PATH = DEFAULT_DCM_OUTPUT_ROOT / "music_dance_dcm_split_summary.json"

FIXED_VAL_IDS = ["c9", "c10", "c36", "c51", "c70", "c71", "c79", "c86", "c88", "c93"]
FIXED_TEST_IDS = ["c8", "c17", "c34", "c38", "c39", "c40", "c46", "c61", "c68", "c81"]
FIXED_EXCLUDED_IDS = ["c6", "c23", "c26", "c41", "c85"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build split metadata for the music+dance multimodal dataset and "
            "materialize the official DCM short-clip dataset variant."
        )
    )
    parser.add_argument("--source-dataset-root", type=Path, default=DEFAULT_SOURCE_DATASET_ROOT)
    parser.add_argument("--dcm-split-root", type=Path, default=DEFAULT_DCM_SPLIT_ROOT)
    parser.add_argument("--dcm-output-root", type=Path, default=DEFAULT_DCM_OUTPUT_ROOT)
    parser.add_argument("--song-split-path", type=Path, default=DEFAULT_SONG_SPLIT_PATH)
    parser.add_argument("--dcm-summary-path", type=Path, default=DEFAULT_DCM_SUMMARY_PATH)
    parser.add_argument("--force", action="store_true", help="Overwrite existing JSON/NPZ outputs.")
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite without --force: {path}")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_npz(path: Path, *, force: bool, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite without --force: {path}")
    np.savez_compressed(path, **arrays)


def _resolve_feature_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (WORKSPACE_ROOT / path).resolve()


def _sort_song_ids(song_ids: list[str]) -> list[str]:
    def sort_key(song_id: str) -> tuple[int, str]:
        if song_id.startswith("c") and song_id[1:].isdigit():
            return (int(song_id[1:]), song_id)
        return (10**9, song_id)

    return sorted(song_ids, key=sort_key)


def _load_official_split_lists(split_root: Path) -> dict[str, Any]:
    if not split_root.exists():
        raise FileNotFoundError(f"Official DCM split directory not found: {split_root}")
    payload = {
        "train": list(_load_json(split_root / "train.json")),
        "val": list(_load_json(split_root / "validation.json")),
        "test": list(_load_json(split_root / "test.json")),
        "long2short": _load_json(split_root / "long2short.json"),
    }
    for split_name in ("train", "val", "test"):
        if not isinstance(payload[split_name], list):
            raise ValueError(f"Unexpected split format for {split_name}.json")
    if not isinstance(payload["long2short"], dict):
        raise ValueError("Unexpected long2short.json format")
    return payload


def _parse_official_short_id(short_id: str) -> tuple[str, int, int | None]:
    parts = str(short_id).split("_")
    if len(parts) == 2:
        prefix, index_text = parts
        return prefix, int(index_text), None
    if len(parts) == 3:
        prefix, index_text, segment_text = parts
        return prefix, int(index_text), int(segment_text)
    raise ValueError(f"Unsupported official short id format: {short_id}")


def _normalize_long2short(raw: dict[str, list[list[int]]]) -> dict[int, list[tuple[int, int]]]:
    normalized: dict[int, list[tuple[int, int]]] = {}
    for index_text, ranges in raw.items():
        normalized[int(index_text)] = [(int(start), int(end)) for start, end in ranges]
    return normalized


def _slice_event_frames(event_frames: np.ndarray, *, start_frame: int, end_frame: int) -> np.ndarray:
    local = event_frames[(event_frames >= start_frame) & (event_frames <= end_frame)] - int(start_frame)
    return np.asarray(local, dtype=np.int32)


def _build_song_level_split(
    source_manifest: dict,
    *,
    source_manifest_path: Path,
    song_split_path: Path,
    force: bool,
) -> dict:
    available_ids = _sort_song_ids([str(song["id"]) for song in source_manifest["songs"]])
    available_set = set(available_ids)
    val_ids = [song_id for song_id in FIXED_VAL_IDS if song_id in available_set]
    test_ids = [song_id for song_id in FIXED_TEST_IDS if song_id in available_set]
    excluded_present = [song_id for song_id in FIXED_EXCLUDED_IDS if song_id in available_set]
    train_ids = [
        song_id
        for song_id in available_ids
        if song_id not in set(FIXED_VAL_IDS) | set(FIXED_TEST_IDS) | set(FIXED_EXCLUDED_IDS)
    ]

    payload = {
        "dataset_type": "multimodal_cut_keyframe_split",
        "split_variant": "song_level_fixed",
        "dataset_root": str(source_manifest_path.parent),
        "dataset_manifest": str(source_manifest_path),
        "description": "Fixed clip-level split that keeps whole songs together for the music+dance multimodal dataset.",
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "excluded": list(FIXED_EXCLUDED_IDS),
        "available_song_ids": available_ids,
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "available": len(available_ids),
            "excluded_present_in_dataset": len(excluded_present),
        },
        "missing_requested_ids": {
            "val": [song_id for song_id in FIXED_VAL_IDS if song_id not in available_set],
            "test": [song_id for song_id in FIXED_TEST_IDS if song_id not in available_set],
            "excluded": [song_id for song_id in FIXED_EXCLUDED_IDS if song_id not in available_set],
        },
    }
    _write_json(song_split_path, payload, force=force)
    return payload


def _build_dcm_segment_dataset(
    *,
    source_manifest: dict,
    source_manifest_path: Path,
    split_root: Path,
    output_root: Path,
    summary_path: Path,
    force: bool,
) -> dict:
    official = _load_official_split_lists(split_root)
    long2short = _normalize_long2short(official["long2short"])
    entries_by_song_id = {str(song["id"]): song for song in source_manifest["songs"]}
    built_entries_by_id: dict[str, dict] = {}
    skipped: list[dict] = []
    built_ids: set[str] = set()

    for split_name in ("train", "val", "test"):
        for short_id in official[split_name]:
            prefix, index, segment_idx = _parse_official_short_id(short_id)
            source_song_id = f"c{index}"
            entry = entries_by_song_id.get(source_song_id)
            if entry is None:
                skipped.append(
                    {
                        "id": short_id,
                        "split": split_name,
                        "reason": f"source song missing from multimodal dataset: {source_song_id}",
                    }
                )
                continue

            source_data = np.load(_resolve_feature_path(entry["feature_path"]))
            total_frames = int(source_data["audio_features"].shape[0])
            ranges = long2short.get(index, [])

            if segment_idx is None:
                start_frame = 0
                end_frame = total_frames - 1
            else:
                if segment_idx < 0 or segment_idx >= len(ranges):
                    skipped.append(
                        {
                            "id": short_id,
                            "split": split_name,
                            "reason": f"segment index {segment_idx} out of range for c{index}",
                        }
                    )
                    continue
                start_frame, end_frame = ranges[segment_idx]
                start_frame = max(0, int(start_frame))
                end_frame = min(total_frames - 1, int(end_frame))

            if end_frame < start_frame:
                skipped.append(
                    {
                        "id": short_id,
                        "split": split_name,
                        "reason": f"empty frame range [{start_frame}, {end_frame}]",
                    }
                )
                continue

            segment_slice = slice(start_frame, end_frame + 1)
            frame_times_sec = np.asarray(source_data["frame_times_sec"][segment_slice], dtype=np.float32)
            if len(frame_times_sec) <= 0:
                skipped.append({"id": short_id, "split": split_name, "reason": "segment has zero frames"})
                continue
            frame_times_sec = frame_times_sec - frame_times_sec[0]

            cut_frames = _slice_event_frames(
                np.asarray(source_data["cut_frames"], dtype=np.int32),
                start_frame=start_frame,
                end_frame=end_frame,
            )
            keyframe_frames = _slice_event_frames(
                np.asarray(source_data["keyframe_frames"], dtype=np.int32),
                start_frame=start_frame,
                end_frame=end_frame,
            )

            segment_output_path = output_root / "songs" / f"{short_id}.npz"
            _write_npz(
                segment_output_path,
                force=force,
                audio_features=np.asarray(source_data["audio_features"][segment_slice], dtype=np.float32),
                dance_features=np.asarray(source_data["dance_features"][segment_slice], dtype=np.float32),
                cut_labels=np.asarray(source_data["cut_labels"][segment_slice], dtype=np.float32),
                cut_soft_labels=np.asarray(source_data["cut_soft_labels"][segment_slice], dtype=np.float32),
                keyframe_labels=np.asarray(source_data["keyframe_labels"][segment_slice], dtype=np.float32),
                keyframe_soft_labels=np.asarray(source_data["keyframe_soft_labels"][segment_slice], dtype=np.float32),
                keyframe_label_mask=np.asarray(source_data["keyframe_label_mask"][segment_slice], dtype=np.float32),
                frame_times_sec=frame_times_sec,
                cut_frames=cut_frames,
                keyframe_frames=keyframe_frames,
            )

            built_entries_by_id[short_id] = {
                "id": short_id,
                "source_song_id": source_song_id,
                "index": index,
                "prefix": prefix,
                "segment_index": segment_idx,
                "official_split": split_name,
                "fps": float(entry["fps"]),
                "num_frames": int(end_frame - start_frame + 1),
                "duration_sec": round(float(frame_times_sec[-1]) if len(frame_times_sec) > 0 else 0.0, 6),
                "cut_count": int(len(cut_frames)),
                "keyframe_count": int(len(keyframe_frames)),
                "source_start_frame": int(start_frame),
                "source_end_frame": int(end_frame),
                "source_num_frames": int(total_frames),
                "feature_path": str(segment_output_path),
                "source_feature_path": str(_resolve_feature_path(entry["feature_path"])),
            }
            built_ids.add(short_id)

    split_ids = {
        split_name: [short_id for short_id in official[split_name] if short_id in built_ids]
        for split_name in ("train", "val", "test")
    }
    songs = [built_entries_by_id[short_id] for split_name in ("train", "val", "test") for short_id in split_ids[split_name]]

    output_manifest = {
        "dataset_type": "multimodal_cut_keyframe_dcm_short",
        "version": 1,
        "split_variant": "official_dcm_short",
        "source_dataset_manifest": str(source_manifest_path),
        "official_split_root": str(split_root),
        "audio_feature_names": source_manifest.get("audio_feature_names", []),
        "dance_feature_names": source_manifest.get("dance_feature_names", []),
        "song_count": len(songs),
        "clip_count": len(songs),
        "skipped_count": len(skipped),
        "split": split_ids,
        "songs": songs,
        "skipped": skipped,
    }
    _write_json(output_root / "manifest.json", output_manifest, force=force)

    summary = {
        "dataset_type": "multimodal_cut_keyframe_dcm_short_summary",
        "dataset_root": str(output_root),
        "dataset_manifest": str(output_root / "manifest.json"),
        "official_split_root": str(split_root),
        "official_counts": {split_name: len(official[split_name]) for split_name in ("train", "val", "test")},
        "built_counts": {split_name: len(split_ids[split_name]) for split_name in ("train", "val", "test")},
        "missing_counts": {
            split_name: len(official[split_name]) - len(split_ids[split_name])
            for split_name in ("train", "val", "test")
        },
        "built_split_ids": split_ids,
        "skipped": skipped,
    }
    _write_json(summary_path, summary, force=force)
    return summary


def main() -> None:
    args = parse_args()
    source_manifest_path = args.source_dataset_root / "manifest.json"
    if not source_manifest_path.exists():
        raise SystemExit(f"Source multimodal manifest not found: {source_manifest_path}")
    source_manifest = _load_json(source_manifest_path)
    if not isinstance(source_manifest, dict) or not isinstance(source_manifest.get("songs"), list):
        raise SystemExit(f"Unexpected source multimodal manifest format: {source_manifest_path}")

    song_split = _build_song_level_split(
        source_manifest,
        source_manifest_path=source_manifest_path,
        song_split_path=args.song_split_path,
        force=bool(args.force),
    )
    dcm_summary = _build_dcm_segment_dataset(
        source_manifest=source_manifest,
        source_manifest_path=source_manifest_path,
        split_root=args.dcm_split_root,
        output_root=args.dcm_output_root,
        summary_path=args.dcm_summary_path,
        force=bool(args.force),
    )

    print(f"Saved song-level split: {args.song_split_path}")
    print(
        "Song-level counts: "
        f"train={song_split['counts']['train']} "
        f"val={song_split['counts']['val']} "
        f"test={song_split['counts']['test']}"
    )
    print(f"Saved DCM short-clip dataset manifest: {args.dcm_output_root / 'manifest.json'}")
    print(
        "DCM built counts: "
        f"train={dcm_summary['built_counts']['train']} "
        f"val={dcm_summary['built_counts']['val']} "
        f"test={dcm_summary['built_counts']['test']}"
    )
    if dcm_summary["skipped"]:
        print(f"Skipped DCM short clips: {len(dcm_summary['skipped'])}")


if __name__ == "__main__":
    main()
