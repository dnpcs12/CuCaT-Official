from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_SOURCE_ROOT = Path(__file__).resolve().parent / "results" / "cut_keyframe_multimodal_dataset_dcm"
DEFAULT_OUTPUT_ROOTS = {
    "audio": Path(__file__).resolve().parent / "results" / "cut_transition_dataset_from_multimodal_dcm",
    "audio_demucs_events_no_rms": Path(__file__).resolve().parent
    / "results"
    / "cut_transition_dataset_audio_demucs_events_no_rms_from_multimodal_dcm",
    "dance": Path(__file__).resolve().parent / "results" / "cut_transition_dataset_dance_from_multimodal_dcm",
    "audio_dance": Path(__file__).resolve().parent / "results" / "cut_transition_dataset_audio_dance_from_multimodal_dcm",
    "compact_dance": Path(__file__).resolve().parent / "results" / "cut_transition_dataset_compact_dance_from_multimodal_dcm",
    "audio_compact_dance": Path(__file__).resolve().parent
    / "results"
    / "cut_transition_dataset_audio_compact_dance_from_multimodal_dcm",
    "audio_demucs_events_no_rms_compact_dance": Path(__file__).resolve().parent
    / "results"
    / "cut_transition_dataset_audio_demucs_events_no_rms_compact_dance_from_multimodal_dcm",
}

DEMUX_EVENTS_NO_RMS_AUDIO_FEATURES = [
    "vocals_abs_mean",
    "vocals_peak",
    "bass_abs_mean",
    "bass_peak",
    "drums_abs_mean",
    "drums_peak",
    "other_abs_mean",
    "other_peak",
    "beat_pulse",
    "beat_since",
    "beat_until",
    "downbeat_pulse",
    "downbeat_since",
    "downbeat_until",
    "bar_pulse",
    "bar_since",
    "bar_until",
]

COMPACT_ROOT_JOINT_INDEX = 0
COMPACT_DANCE_JOINT_INDICES = [
    0,  # 下半身
    1,  # 上半身
    2,  # 上半身2
    4,  # 頭
    9,  # 左ひざ
    10,  # 左足首
    11,  # 左つま先
    13,  # 右ひざ
    14,  # 右足首
    15,  # 右つま先
    19,  # 左ひじ
    20,  # 左手首
    25,  # 左親指２
    28,  # 左人指３
    31,  # 左中指３
    34,  # 左薬指３
    37,  # 左小指３
    41,  # 右ひじ
    42,  # 右手首
    47,  # 右親指２
    50,  # 右人指３
    53,  # 右中指３
    56,  # 右薬指３
    59,  # 右小指３
]
COMPACT_DANCE_JOINT_NAMES = {
    0: "lower_body",
    1: "upper_body",
    2: "upper_body2",
    4: "head",
    9: "left_knee",
    10: "left_ankle",
    11: "left_toe",
    13: "right_knee",
    14: "right_ankle",
    15: "right_toe",
    19: "left_elbow",
    20: "left_wrist",
    25: "left_thumb_tip",
    28: "left_index_tip",
    31: "left_middle_tip",
    34: "left_ring_tip",
    37: "left_pinky_tip",
    41: "right_elbow",
    42: "right_wrist",
    47: "right_thumb_tip",
    50: "right_index_tip",
    53: "right_middle_tip",
    56: "right_ring_tip",
    59: "right_pinky_tip",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a multimodal cut/keyframe dataset into a frame-CNN cut-transition dataset."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--feature-set",
        choices=[
            "audio",
            "audio_demucs_events_no_rms",
            "dance",
            "audio_dance",
            "compact_dance",
            "audio_compact_dance",
            "audio_demucs_events_no_rms_compact_dance",
        ],
        default="audio",
        help="Which multimodal features to expose as the single frame-CNN input tensor.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root. Defaults to a feature-set-specific folder under music_processing/results.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_repo_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parents[1] / path).resolve()


def _relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path(__file__).resolve().parents[1]))
    except ValueError:
        return str(path.resolve())


def _columns_for_names(all_names: list[str], wanted_names: list[str]) -> list[int]:
    index_by_name = {name: idx for idx, name in enumerate(all_names)}
    missing = [name for name in wanted_names if name not in index_by_name]
    if missing:
        raise ValueError(f"Missing expected dance feature names: {missing[:8]}")
    return [index_by_name[name] for name in wanted_names]


def _compact_dance_payload(dance_features: np.ndarray, dance_names: list[str]) -> tuple[np.ndarray, list[str], dict]:
    non_root_indices = [idx for idx in COMPACT_DANCE_JOINT_INDICES if idx != COMPACT_ROOT_JOINT_INDEX]
    relpos_names = [
        f"dance_relpos_j{joint_idx:02d}_{axis}"
        for joint_idx in non_root_indices
        for axis in ("x", "y", "z")
    ]
    relvel_names = [
        f"dance_relvel_j{joint_idx:02d}_{axis}"
        for joint_idx in non_root_indices
        for axis in ("x", "y", "z")
    ]
    root_pos_names = [f"dance_root_pos_{axis}" for axis in ("x", "y", "z")]
    root_vel_names = [f"dance_root_vel_{axis}" for axis in ("x", "y", "z")]
    speed_names = [f"dance_joint_speed_j{joint_idx:02d}" for joint_idx in non_root_indices]
    selected_names = relpos_names + relvel_names + root_pos_names + root_vel_names + speed_names
    selected_columns = _columns_for_names(dance_names, selected_names)
    speed_columns = _columns_for_names(dance_names, speed_names)

    compact_features = np.concatenate(
        [
            dance_features[:, selected_columns],
            np.mean(dance_features[:, speed_columns], axis=1, keepdims=True),
        ],
        axis=1,
    ).astype(np.float32)
    compact_names = selected_names + ["dance_compact_motion_energy"]
    metadata = {
        "compact_root_joint_index": COMPACT_ROOT_JOINT_INDEX,
        "compact_joint_indices": COMPACT_DANCE_JOINT_INDICES,
        "compact_non_root_joint_indices": non_root_indices,
        "compact_joint_names": {str(idx): COMPACT_DANCE_JOINT_NAMES[idx] for idx in COMPACT_DANCE_JOINT_INDICES},
        "compact_dance_feature_dim": len(compact_names),
        "compact_motion_energy": "mean speed over selected non-root joints",
    }
    return compact_features, compact_names, metadata


def _feature_payload(data: np.lib.npyio.NpzFile, source_manifest: dict, feature_set: str) -> tuple[np.ndarray, list[str]]:
    audio_features = np.asarray(data["audio_features"], dtype=np.float32)
    dance_features = np.asarray(data["dance_features"], dtype=np.float32)
    audio_names = list(source_manifest.get("audio_feature_names", []))
    dance_names = list(source_manifest.get("dance_feature_names", []))
    compact_dance, compact_dance_names, _ = _compact_dance_payload(dance_features, dance_names)
    no_rms_audio_columns = _columns_for_names(audio_names, DEMUX_EVENTS_NO_RMS_AUDIO_FEATURES)
    no_rms_audio = audio_features[:, no_rms_audio_columns]

    if feature_set == "audio":
        return audio_features, audio_names
    if feature_set == "audio_demucs_events_no_rms":
        return no_rms_audio, DEMUX_EVENTS_NO_RMS_AUDIO_FEATURES
    if feature_set == "dance":
        return dance_features, dance_names
    if feature_set == "compact_dance":
        return compact_dance, compact_dance_names
    if feature_set == "audio_compact_dance":
        return np.concatenate([audio_features, compact_dance], axis=1), audio_names + compact_dance_names
    if feature_set == "audio_demucs_events_no_rms_compact_dance":
        return (
            np.concatenate([no_rms_audio, compact_dance], axis=1),
            DEMUX_EVENTS_NO_RMS_AUDIO_FEATURES + compact_dance_names,
        )
    return np.concatenate([audio_features, dance_features], axis=1), audio_names + dance_names


def main() -> None:
    args = parse_args()
    source_manifest_path = args.source_root / "manifest.json"
    if not source_manifest_path.exists():
        raise SystemExit(f"Source manifest not found: {source_manifest_path}")

    source_manifest = _load_json(source_manifest_path)
    source_songs = source_manifest.get("songs", [])
    if not source_songs:
        raise SystemExit(f"No songs found in source manifest: {source_manifest_path}")

    output_root = (args.output_root or DEFAULT_OUTPUT_ROOTS[str(args.feature_set)]).resolve()
    songs_output_dir = output_root / "songs"
    songs_output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = list(source_manifest.get("audio_feature_names", []))
    if args.feature_set != "audio":
        first_feature_path = _resolve_repo_path(source_songs[0]["feature_path"])
        first_data = np.load(first_feature_path)
        _, feature_names = _feature_payload(first_data, source_manifest, str(args.feature_set))

    converted_songs: list[dict] = []
    for entry in source_songs:
        source_feature_path = _resolve_repo_path(entry["feature_path"])
        if not source_feature_path.exists():
            raise SystemExit(f"Missing feature file: {source_feature_path}")
        output_feature_path = songs_output_dir / f"{entry['id']}.npz"
        if not output_feature_path.exists() or args.force:
            data = np.load(source_feature_path)
            features, feature_names = _feature_payload(data, source_manifest, str(args.feature_set))
            np.savez_compressed(
                output_feature_path,
                features=features,
                labels=np.asarray(data["cut_labels"], dtype=np.float32),
                soft_labels=np.asarray(data["cut_soft_labels"], dtype=np.float32),
                cut_frames=np.asarray(data["cut_frames"], dtype=np.int32),
            )

        converted_entry = {
            "id": str(entry["id"]),
            "index": int(entry.get("index", len(converted_songs))),
            "fps": float(entry.get("fps", 30.0)),
            "num_frames": int(entry["num_frames"]),
            "duration_sec": float(entry["duration_sec"]),
            "cut_count": int(entry.get("cut_count", 0)),
            "positive_frame_count": int(entry.get("cut_count", 0)),
            "positive_frame_ratio": float(entry.get("cut_count", 0) / max(1, int(entry["num_frames"]))),
            "feature_path": _relative_to_repo(output_feature_path),
            "feature_dim": int(len(feature_names)),
            "source_song_id": str(entry.get("source_song_id", entry["id"])),
            "source_start_frame": int(entry.get("source_start_frame", 0)),
            "source_end_frame": int(entry.get("source_end_frame", entry["num_frames"] - 1)),
            "source_num_frames": int(entry.get("source_num_frames", entry["num_frames"])),
            "official_split": entry.get("official_split"),
        }
        converted_songs.append(converted_entry)

    output_manifest = {
        "dataset_type": "frame_level_cut_transition_from_multimodal",
        "version": 1,
        "feature_set": str(args.feature_set),
        "source_dataset_manifest": str(source_manifest_path),
        "source_dataset_type": source_manifest.get("dataset_type"),
        "feature_names": feature_names,
        "song_count": len(converted_songs),
        "total_frames": int(sum(int(song["num_frames"]) for song in converted_songs)),
        "total_cuts": int(sum(int(song["cut_count"]) for song in converted_songs)),
        "positive_frame_count": int(sum(int(song["positive_frame_count"]) for song in converted_songs)),
        "positive_frame_ratio": float(
            sum(int(song["positive_frame_count"]) for song in converted_songs)
            / max(1, sum(int(song["num_frames"]) for song in converted_songs))
        ),
        "compact_dance_metadata": _compact_dance_payload(
            np.zeros((1, len(source_manifest.get("dance_feature_names", []))), dtype=np.float32),
            list(source_manifest.get("dance_feature_names", [])),
        )[2]
        if str(args.feature_set) in {"compact_dance", "audio_compact_dance"}
        else None,
        "split": source_manifest.get("split"),
        "songs": converted_songs,
    }
    _write_json(output_root / "manifest.json", output_manifest)

    print(f"Source root: {args.source_root}")
    print(f"Output root: {output_root}")
    print(f"Converted entries: {len(converted_songs)}")
    print(f"Saved manifest: {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
