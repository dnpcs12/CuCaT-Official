from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from music_processing.build_audio_only_cut_dataset_from_multimodal import (  # noqa: E402
    DEMUX_EVENTS_NO_RMS_AUDIO_FEATURES,
    _columns_for_names,
    _compact_dance_payload,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = Path(__file__).resolve().parent / "results"
DEFAULT_SOURCE_ROOT = RESULTS_ROOT / "cut_keyframe_multimodal_dataset_dcm"
DEFAULT_BODY_DATASET_ROOT = RESULTS_ROOT / "01_datasets" / "cut_transition_dataset_motion_body_labels"
DEFAULT_MULTIMODAL_OUTPUT_ROOT = (
    RESULTS_ROOT / "01_datasets" / "cut_keyframe_multimodal_dataset_dcm_demucs_no_rms_compact_dance"
)
DEFAULT_KEYFRAME_OUTPUT_ROOT = (
    RESULTS_ROOT / "01_datasets" / "keyframe_timing_dataset_demucs_no_rms_compact_dance_dcm"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build keyframe datasets that use the same input features as the cut model: "
            "Demucs no-RMS meter audio + compact dance."
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--body-dataset-root", type=Path, default=DEFAULT_BODY_DATASET_ROOT)
    parser.add_argument("--multimodal-output-root", type=Path, default=DEFAULT_MULTIMODAL_OUTPUT_ROOT)
    parser.add_argument("--keyframe-output-root", type=Path, default=DEFAULT_KEYFRAME_OUTPUT_ROOT)
    parser.add_argument(
        "--include-body-activity",
        action="store_true",
        help="Append 8D full-song body activity labels to the compact dance stream.",
    )
    parser.add_argument("--exclude-song-ids", nargs="*", default=["c47", "c105"])
    parser.add_argument(
        "--exclude-scope",
        choices=["train", "train_val", "all", "none"],
        default="train_val",
        help="Where exclusions apply for the Stage-B keyframe timing dataset. The multimodal output keeps all entries.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _normalize_song_id(song_id: str | int) -> str:
    text = str(song_id).strip()
    if text.lower().startswith("c"):
        return "c" + str(int(text[1:]))
    return "c" + str(int(text))


def _resolve_repo_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (WORKSPACE_ROOT / path).resolve()
    if path.exists():
        return path

    text = str(path)
    replacements = [
        (
            str(RESULTS_ROOT / "cut_keyframe_multimodal_dataset_dcm"),
            str(RESULTS_ROOT / "01_datasets" / "cut_keyframe_multimodal_dataset_dcm"),
        ),
        (
            str(RESULTS_ROOT / "cut_keyframe_multimodal_dataset"),
            str(RESULTS_ROOT / "01_datasets" / "cut_keyframe_multimodal_dataset"),
        ),
        (
            str(RESULTS_ROOT / "cut_transition_dataset_motion_body_labels"),
            str(RESULTS_ROOT / "01_datasets" / "cut_transition_dataset_motion_body_labels"),
        ),
    ]
    for old, new in replacements:
        if old in text:
            candidate = Path(text.replace(old, new))
            if candidate.exists():
                return candidate
    return path


def _align_rows(values: np.ndarray, target_frames: int) -> np.ndarray:
    target_frames = int(target_frames)
    if len(values) == target_frames:
        return values.astype(np.float32, copy=False)
    if len(values) > target_frames:
        return values[:target_frames].astype(np.float32, copy=False)
    if len(values) == 0:
        return np.zeros((target_frames, values.shape[1]), dtype=np.float32)
    pad = np.repeat(values[-1:, :], target_frames - len(values), axis=0)
    return np.concatenate([values, pad], axis=0).astype(np.float32, copy=False)


def _exclude_applies(entry: dict, excluded_ids: set[str], scope: str) -> bool:
    if scope == "none":
        return False
    song_id = _normalize_song_id(entry.get("source_song_id", entry.get("id", "")))
    if song_id not in excluded_ids:
        return False
    split = str(entry.get("official_split", "")).lower()
    if scope == "all":
        return True
    if scope == "train":
        return split == "train"
    return split in {"train", "val"}


def _split_for_entry(entry: dict) -> str:
    split = str(entry.get("official_split", "train")).lower()
    if split == "validation":
        return "val"
    if split not in {"train", "val", "test"}:
        return "train"
    return split


def _feature_payload(
    data: np.lib.npyio.NpzFile,
    *,
    audio_names: list[str],
    dance_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict]:
    audio = np.asarray(data["audio_features"], dtype=np.float32)
    dance = np.asarray(data["dance_features"], dtype=np.float32)
    audio_cols = _columns_for_names(audio_names, DEMUX_EVENTS_NO_RMS_AUDIO_FEATURES)
    compact_dance, compact_names, compact_metadata = _compact_dance_payload(dance, dance_names)
    return (
        audio[:, audio_cols].astype(np.float32),
        compact_dance.astype(np.float32),
        list(DEMUX_EVENTS_NO_RMS_AUDIO_FEATURES),
        compact_names,
        compact_metadata,
    )


def _entry_payload(entry: dict, *, out_path: Path, feature_dim: int, keyframe_count: int | None = None) -> dict:
    positive_count = int(keyframe_count if keyframe_count is not None else entry.get("keyframe_count", 0))
    num_frames = int(entry["num_frames"])
    return {
        "id": str(entry["id"]),
        "source_song_id": _normalize_song_id(entry.get("source_song_id", entry["id"])),
        "index": int(entry.get("index", 0)),
        "prefix": entry.get("prefix"),
        "segment_index": entry.get("segment_index"),
        "official_split": entry.get("official_split"),
        "fps": float(entry.get("fps", 30.0)),
        "num_frames": num_frames,
        "duration_sec": float(entry.get("duration_sec", num_frames / 30.0)),
        "cut_count": int(entry.get("cut_count", 0)),
        "keyframe_count": positive_count,
        "positive_frame_count": positive_count,
        "positive_frame_ratio": float(positive_count / max(1, num_frames)),
        "source_start_frame": int(entry.get("source_start_frame", 0)),
        "source_end_frame": int(entry.get("source_end_frame", num_frames - 1)),
        "source_num_frames": int(entry.get("source_num_frames", num_frames)),
        "feature_path": _relative_to_repo(out_path),
        "feature_dim": int(feature_dim),
    }


def main() -> None:
    args = parse_args()
    source_manifest_path = args.source_root / "manifest.json"
    if not source_manifest_path.exists():
        raise SystemExit(f"Source manifest not found: {source_manifest_path}")

    source_manifest = _load_json(source_manifest_path)
    source_entries = list(source_manifest.get("songs", []))
    if not source_entries:
        raise SystemExit(f"No source songs found: {source_manifest_path}")

    audio_names = list(source_manifest.get("audio_feature_names", []))
    dance_names = list(source_manifest.get("dance_feature_names", []))
    excluded_ids = {_normalize_song_id(song_id) for song_id in args.exclude_song_ids}
    body_manifest = None
    body_entries_by_id: dict[str, dict] = {}
    body_feature_names: list[str] = []
    body_cache: dict[str, np.ndarray] = {}
    if args.include_body_activity:
        body_manifest_path = args.body_dataset_root / "manifest.json"
        if not body_manifest_path.exists():
            raise SystemExit(f"Body activity manifest not found: {body_manifest_path}")
        body_manifest = _load_json(body_manifest_path)
        body_entries_by_id = {str(entry["id"]): entry for entry in body_manifest.get("songs", [])}
        body_feature_names = list(body_manifest.get("feature_names", []))

    multimodal_root = args.multimodal_output_root.resolve()
    keyframe_root = args.keyframe_output_root.resolve()
    multimodal_songs_dir = multimodal_root / "songs"
    keyframe_songs_dir = keyframe_root / "songs"
    multimodal_songs_dir.mkdir(parents=True, exist_ok=True)
    keyframe_songs_dir.mkdir(parents=True, exist_ok=True)

    multimodal_entries: list[dict] = []
    keyframe_entries: list[dict] = []
    skipped_by_exclusion: list[dict] = []
    split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    selected_audio_names: list[str] | None = None
    compact_dance_names: list[str] | None = None
    output_dance_names: list[str] | None = None
    compact_metadata: dict | None = None

    for entry in source_entries:
        source_path = _resolve_repo_path(entry["feature_path"])
        if not source_path.exists():
            raise SystemExit(f"Missing source feature file: {source_path}")

        data = np.load(source_path)
        audio_features, dance_features, selected_audio_names, compact_dance_names, compact_metadata = _feature_payload(
            data,
            audio_names=audio_names,
            dance_names=dance_names,
        )
        output_dance_names = list(compact_dance_names)

        if args.include_body_activity:
            source_song_id = _normalize_song_id(entry.get("source_song_id", entry["id"]))
            body_entry = body_entries_by_id.get(source_song_id)
            if body_entry is None:
                raise SystemExit(f"Missing body activity entry for {entry['id']} source song {source_song_id}")
            if source_song_id not in body_cache:
                body_path = _resolve_repo_path(body_entry["feature_path"])
                if not body_path.exists():
                    raise SystemExit(f"Missing body activity feature file: {body_path}")
                with np.load(body_path, allow_pickle=False) as body_data:
                    body_cache[source_song_id] = np.asarray(body_data["features"], dtype=np.float32)
            clip_start = int(entry.get("source_start_frame", 0))
            clip_frames = int(audio_features.shape[0])
            body_features = _align_rows(body_cache[source_song_id][clip_start : clip_start + clip_frames], clip_frames)
            dance_features = np.concatenate([dance_features, body_features], axis=1).astype(np.float32)
            output_dance_names = list(compact_dance_names) + body_feature_names

        multimodal_path = multimodal_songs_dir / f"{entry['id']}.npz"
        if not multimodal_path.exists() or args.force:
            np.savez_compressed(
                multimodal_path,
                audio_features=audio_features,
                dance_features=dance_features,
                cut_labels=np.asarray(data["cut_labels"], dtype=np.float32),
                cut_soft_labels=np.asarray(data["cut_soft_labels"], dtype=np.float32),
                keyframe_labels=np.asarray(data["keyframe_labels"], dtype=np.float32),
                keyframe_soft_labels=np.asarray(data["keyframe_soft_labels"], dtype=np.float32),
                keyframe_label_mask=np.asarray(data["keyframe_label_mask"], dtype=np.float32),
                frame_times_sec=np.asarray(data["frame_times_sec"], dtype=np.float32),
                cut_frames=np.asarray(data["cut_frames"], dtype=np.int32),
                keyframe_frames=np.asarray(data["keyframe_frames"], dtype=np.int32),
            )
        multimodal_entries.append(
            {
                **_entry_payload(
                    entry,
                    out_path=multimodal_path,
                    feature_dim=int(audio_features.shape[1] + dance_features.shape[1]),
                    keyframe_count=int(np.asarray(data["keyframe_labels"], dtype=np.float32).sum()),
                ),
                "audio_feature_dim": int(audio_features.shape[1]),
                "dance_feature_dim": int(dance_features.shape[1]),
                "body_feature_dim": int(len(body_feature_names)) if args.include_body_activity else 0,
                "source_feature_path": _relative_to_repo(source_path),
            }
        )

        if _exclude_applies(entry, excluded_ids, str(args.exclude_scope)):
            skipped_by_exclusion.append(
                {
                    "id": str(entry["id"]),
                    "source_song_id": _normalize_song_id(entry.get("source_song_id", entry["id"])),
                    "official_split": entry.get("official_split"),
                }
            )
            continue

        features = np.concatenate([audio_features, dance_features], axis=1).astype(np.float32)
        keyframe_path = keyframe_songs_dir / f"{entry['id']}.npz"
        keyframe_labels = np.asarray(data["keyframe_labels"], dtype=np.float32)
        if not keyframe_path.exists() or args.force:
            np.savez_compressed(
                keyframe_path,
                features=features,
                labels=keyframe_labels,
                soft_labels=np.asarray(data["keyframe_soft_labels"], dtype=np.float32),
                keyframe_frames=np.asarray(data["keyframe_frames"], dtype=np.int32),
                cut_frames=np.asarray(data["keyframe_frames"], dtype=np.int32),
                gt_cut_frames=np.asarray(data["cut_frames"], dtype=np.int32),
                frame_times_sec=np.asarray(data["frame_times_sec"], dtype=np.float32),
            )

        keyframe_entry = _entry_payload(
            entry,
            out_path=keyframe_path,
            feature_dim=int(features.shape[1]),
            keyframe_count=int(keyframe_labels.sum()),
        )
        keyframe_entries.append(keyframe_entry)
        split_name = _split_for_entry(entry)
        split[split_name].append(str(entry["id"]))

    if selected_audio_names is None or output_dance_names is None or compact_dance_names is None:
        raise SystemExit("No entries converted.")

    feature_set = "audio_demucs_events_no_rms_compact_dance"
    if args.include_body_activity:
        feature_set += "_body_activity"
    multimodal_manifest = {
        "dataset_type": "multimodal_cut_keyframe_dcm_short_cut_aligned_features",
        "version": 1,
        "feature_set": feature_set,
        "source_dataset_manifest": str(source_manifest_path.resolve()),
        "source_body_activity_manifest": str((args.body_dataset_root / "manifest.json").resolve())
        if args.include_body_activity
        else None,
        "source_dataset_type": source_manifest.get("dataset_type"),
        "audio_feature_names": selected_audio_names,
        "dance_feature_names": output_dance_names,
        "audio_feature_dim": len(selected_audio_names),
        "dance_feature_dim": len(output_dance_names),
        "body_feature_names": body_feature_names,
        "body_feature_dim": len(body_feature_names),
        "compact_dance_metadata": compact_metadata,
        "song_count": len(multimodal_entries),
        "clip_count": len(multimodal_entries),
        "split": source_manifest.get("split"),
        "songs": multimodal_entries,
    }
    _write_json(multimodal_root / "manifest.json", multimodal_manifest)

    feature_names = selected_audio_names + output_dance_names
    keyframe_manifest = {
        "dataset_type": "frame_level_keyframe_timing_cnn",
        "version": 1,
        "feature_set": feature_set,
        "source_dataset_manifest": str(source_manifest_path.resolve()),
        "source_body_activity_manifest": str((args.body_dataset_root / "manifest.json").resolve())
        if args.include_body_activity
        else None,
        "target": "keyframe_labels",
        "target_note": (
            "cut_frames is intentionally aliased to keyframe_frames for generic compatibility; "
            "gt_cut_frames stores the actual GT cut frames."
        ),
        "excluded_song_ids": sorted(excluded_ids),
        "exclude_scope": str(args.exclude_scope),
        "feature_groups": {
            "audio_demucs_events_no_rms": selected_audio_names,
            "compact_dance": compact_metadata,
            "body_activity": body_feature_names,
        },
        "feature_names": feature_names,
        "song_count": len(keyframe_entries),
        "total_frames": int(sum(int(row["num_frames"]) for row in keyframe_entries)),
        "total_keyframes": int(sum(int(row["keyframe_count"]) for row in keyframe_entries)),
        "positive_frame_count": int(sum(int(row["positive_frame_count"]) for row in keyframe_entries)),
        "positive_frame_ratio": float(
            sum(int(row["positive_frame_count"]) for row in keyframe_entries)
            / max(1, sum(int(row["num_frames"]) for row in keyframe_entries))
        ),
        "split": split,
        "songs": keyframe_entries,
        "skipped_by_exclusion": skipped_by_exclusion,
    }
    _write_json(keyframe_root / "manifest.json", keyframe_manifest)

    print(f"Source root: {args.source_root}")
    print(f"Multimodal output root: {multimodal_root}")
    print(f"Keyframe output root: {keyframe_root}")
    print(f"Multimodal entries: {len(multimodal_entries)}")
    print(f"Keyframe entries: {len(keyframe_entries)}")
    print(f"Skipped by exclusion: {len(skipped_by_exclusion)}")
    print(f"Feature dims: audio={len(selected_audio_names)} dance={len(output_dance_names)} total={len(feature_names)}")


if __name__ == "__main__":
    main()
