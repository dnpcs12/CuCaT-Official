from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from music_processing.build_audio_only_cut_dataset_from_multimodal import DEMUX_EVENTS_NO_RMS_AUDIO_FEATURES


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DATASET_ROOT = (
    Path(__file__).resolve().parent / "results" / "01_datasets" / "cut_keyframe_multimodal_dataset_dcm_demucs_no_rms_compact_dance"
)
DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parent / "results" / "01_datasets" / "cut_conditioned_keyframe_dataset_dcm_audio17_compact_dance"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a cut-conditioned keyframe dataset by slicing multimodal songs/clips "
            "into GT-cut-bounded segments, then sub-splitting long segments to <= 10 seconds."
        )
    )
    parser.add_argument("--source-dataset-root", type=Path, default=DEFAULT_SOURCE_DATASET_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-segment-seconds", type=float, default=10.0)
    parser.add_argument("--soft-radius", type=int, default=3)
    parser.add_argument(
        "--audio-feature-set",
        choices=["source", "demucs_events_no_rms"],
        default="source",
        help=(
            "Which audio columns to keep for Stage-B. `source` preserves the source dataset; "
            "`demucs_events_no_rms` keeps the 17-dim audio feature set used by cut/Stage-A."
        ),
    )
    parser.add_argument("--force", action="store_true")
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


def _columns_for_names(all_names: list[str], wanted_names: list[str]) -> list[int]:
    index_by_name = {str(name): idx for idx, name in enumerate(all_names)}
    missing = [name for name in wanted_names if name not in index_by_name]
    if missing:
        raise ValueError(f"Missing requested audio feature columns: {missing}")
    return [index_by_name[name] for name in wanted_names]


def _resolve_local_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (WORKSPACE_ROOT / path).resolve()


def _resolve_source_feature_path(path_str: str, source_dataset_root: Path) -> Path:
    path = _resolve_local_path(path_str)
    if path.exists():
        return path

    # Older manifests can contain absolute paths from before results were grouped
    # under 01_datasets. Prefer the explicit source root when the basename matches.
    source_root_candidate = (source_dataset_root / "songs" / path.name).resolve()
    if source_root_candidate.exists():
        return source_root_candidate

    return path


def _event_frames_to_soft_labels(event_frames: np.ndarray, *, num_frames: int, radius: int) -> np.ndarray:
    """Create distance-weighted soft labels around each event.

    With the default radius 3, frames at distance 0/1/2/3 receive
    weights 1.0/0.75/0.5/0.25. Frames outside the radius stay 0.
    """

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


def _load_explicit_split(path: Path) -> dict[str, list[str]]:
    payload = _load_json(path)
    source = payload["split"] if isinstance(payload, dict) and isinstance(payload.get("split"), dict) else payload
    if not isinstance(source, dict):
        raise ValueError(f"Unexpected split manifest format: {path}")
    return {
        "train": [str(item) for item in source.get("train", [])],
        "val": [str(item) for item in source.get("val", [])],
        "test": [str(item) for item in source.get("test", [])],
    }


def _resolve_split_map(manifest: dict, split_manifest_path: Path | None) -> tuple[dict[str, str], dict[str, list[str]] | None, str | None]:
    split: dict[str, list[str]] | None = None
    split_source: str | None = None
    if split_manifest_path is not None:
        split = _load_explicit_split(split_manifest_path)
        split_source = str(split_manifest_path)
    elif isinstance(manifest.get("split"), dict):
        source = manifest["split"]
        split = {
            "train": [str(item) for item in source.get("train", [])],
            "val": [str(item) for item in source.get("val", [])],
            "test": [str(item) for item in source.get("test", [])],
        }
        split_source = "source_dataset_manifest"

    split_map: dict[str, str] = {}
    if split is not None:
        for split_name in ("train", "val", "test"):
            for item in split[split_name]:
                split_map[str(item)] = split_name
    return split_map, split, split_source


def _group_consecutive_frames(frames: np.ndarray, *, num_frames: int) -> list[tuple[int, int]]:
    clipped = sorted(
        {
            int(frame)
            for frame in np.asarray(frames, dtype=np.int32).tolist()
            if 0 <= int(frame) < int(num_frames)
        }
    )
    if not clipped:
        return []

    ranges: list[tuple[int, int]] = []
    start = clipped[0]
    prev = clipped[0]
    for frame in clipped[1:]:
        if frame == prev + 1:
            prev = frame
            continue
        ranges.append((start, prev))
        start = frame
        prev = frame
    ranges.append((start, prev))
    return ranges


def _clip_unique_frames(frames: np.ndarray, *, num_frames: int) -> list[int]:
    return sorted(
        {
            int(frame)
            for frame in np.asarray(frames, dtype=np.int32).tolist()
            if 0 <= int(frame) < int(num_frames)
        }
    )


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ordered = sorted((int(start), int(end)) for start, end in ranges)
    merged: list[tuple[int, int]] = []
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end + 1:
            current_end = max(current_end, end)
            continue
        merged.append((current_start, current_end))
        current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def _cut_frame_to_keyframe_range(cut_frame: int, *, keyframe_set: set[int], num_frames: int) -> tuple[int, int]:
    """Expand one GT cut frame to the adjacent camera-keyframe pair that encodes it.

    In the camera data, a visual cut at frame 16 is commonly represented by
    two consecutive camera keyframes `(15, 16)`.  Stage-2 keyframe prediction
    must exclude both frames, otherwise the model simply learns the cut-pair
    keyframe instead of a motion keyframe between cuts.
    """

    cut = int(cut_frame)

    def has(frame: int) -> bool:
        return 0 <= int(frame) < int(num_frames) and int(frame) in keyframe_set

    if has(cut - 1) and has(cut):
        return (cut - 1, cut)
    if has(cut) and has(cut + 1):
        return (cut, cut + 1)
    if has(cut - 1):
        return (max(0, cut - 1), cut)
    if has(cut + 1):
        return (cut, min(int(num_frames) - 1, cut + 1))
    return (cut, cut)


def _build_cut_keyframe_ranges(
    *,
    num_frames: int,
    cut_frames: np.ndarray,
    keyframe_frames: np.ndarray,
) -> list[dict[str, int]]:
    clipped_cut_frames = _clip_unique_frames(cut_frames, num_frames=num_frames)
    keyframe_set = set(_clip_unique_frames(keyframe_frames, num_frames=num_frames))
    raw_ranges: list[tuple[int, int]] = [
        _cut_frame_to_keyframe_range(cut_frame, keyframe_set=keyframe_set, num_frames=num_frames)
        for cut_frame in clipped_cut_frames
    ]
    merged_ranges = _merge_ranges(raw_ranges)

    rows: list[dict[str, int]] = []
    for start, end in merged_ranges:
        member_cut_frames = [
            int(cut_frame)
            for cut_frame in clipped_cut_frames
            if int(start) <= int(cut_frame) <= int(end)
        ]
        representative = member_cut_frames[-1] if member_cut_frames else int(end)
        rows.append(
            {
                "cut_frame": int(representative),
                "cut_start_frame": int(start),
                "cut_end_frame": int(end),
            }
        )
    return rows


def _build_prediction_intervals(
    num_frames: int,
    cut_frames: np.ndarray,
    keyframe_frames: np.ndarray,
) -> list[dict[str, int | str | None]]:
    """Build non-cut regions around and between cut-keyframe pairs.

    A GT cut frame can be encoded in the camera keyframes as an adjacent pair:
    if frame `16` is a cut, the data may contain keyframes `(15, 16)` or
    `(16, 17)`. Stage-2 keyframe prediction excludes that whole pair.
    For cut pairs `(0, 1)` and `(15, 16)`, the middle interval is `2..14`.
    The leading region before the first cut pair and the trailing region after
    the last cut pair are also kept when they exist.
    """

    cut_ranges = _build_cut_keyframe_ranges(
        num_frames=num_frames,
        cut_frames=cut_frames,
        keyframe_frames=keyframe_frames,
    )
    intervals: list[dict[str, int | str | None]] = []

    def add_interval(
        *,
        start: int,
        end: int,
        left_cut: dict[str, int] | None,
        right_cut: dict[str, int] | None,
        interval_position: str,
    ) -> None:
        if int(start) > int(end):
            return

        def value(cut: dict[str, int] | None, key: str) -> int | None:
            if cut is None:
                return None
            return int(cut[key])

        intervals.append(
            {
                "interval_start_frame": int(start),
                "interval_end_frame": int(end),
                "interval_position": interval_position,
                "left_cut_frame": value(left_cut, "cut_frame"),
                "left_cut_start_frame": value(left_cut, "cut_start_frame"),
                "left_cut_end_frame": value(left_cut, "cut_end_frame"),
                "right_cut_frame": value(right_cut, "cut_frame"),
                "right_cut_start_frame": value(right_cut, "cut_start_frame"),
                "right_cut_end_frame": value(right_cut, "cut_end_frame"),
            }
        )

    if not cut_ranges:
        add_interval(
            start=0,
            end=int(num_frames) - 1,
            left_cut=None,
            right_cut=None,
            interval_position="full",
        )
        return intervals

    first_cut = cut_ranges[0]
    add_interval(
        start=0,
        end=int(first_cut["cut_start_frame"]) - 1,
        left_cut=None,
        right_cut=first_cut,
        interval_position="head",
    )

    for left_cut, right_cut in zip(cut_ranges[:-1], cut_ranges[1:]):
        add_interval(
            start=int(left_cut["cut_end_frame"]) + 1,
            end=int(right_cut["cut_start_frame"]) - 1,
            left_cut=left_cut,
            right_cut=right_cut,
            interval_position="middle",
        )

    last_cut = cut_ranges[-1]
    add_interval(
        start=int(last_cut["cut_end_frame"]) + 1,
        end=int(num_frames) - 1,
        left_cut=last_cut,
        right_cut=None,
        interval_position="tail",
    )
    return intervals


def _split_long_interval(start_frame: int, end_frame: int, *, max_frames: int) -> list[tuple[int, int]]:
    interval_len = int(end_frame - start_frame + 1)
    if interval_len <= max_frames:
        return [(int(start_frame), int(end_frame))]
    chunk_count = int(math.ceil(interval_len / float(max_frames)))
    chunk_size = int(math.ceil(interval_len / float(chunk_count)))
    chunks: list[tuple[int, int]] = []
    current = int(start_frame)
    while current <= end_frame:
        chunk_end = min(int(end_frame), current + chunk_size - 1)
        chunks.append((current, chunk_end))
        current = chunk_end + 1
    return chunks


def _build_condition_features(
    *,
    chunk_start: int,
    chunk_end: int,
    interval_start: int,
    interval_end: int,
    has_left_cut: bool,
    has_right_cut: bool,
) -> np.ndarray:
    frame_indices = np.arange(chunk_start, chunk_end + 1, dtype=np.int32)
    interval_len = max(1, int(interval_end - interval_start))
    chunk_len = max(1, int(chunk_end - chunk_start))
    pos_in_interval = (frame_indices - int(interval_start)).astype(np.float32) / float(interval_len)
    dist_from_left_norm = (frame_indices - int(interval_start)).astype(np.float32) / float(interval_len)
    dist_to_right_norm = (int(interval_end) - frame_indices).astype(np.float32) / float(interval_len)
    pos_in_chunk = (frame_indices - int(chunk_start)).astype(np.float32) / float(chunk_len)
    left_cut_visible = np.zeros(len(frame_indices), dtype=np.float32)
    right_cut_visible = np.zeros(len(frame_indices), dtype=np.float32)
    if has_left_cut and chunk_start == interval_start:
        left_cut_visible[0] = 1.0
    if has_right_cut and chunk_end == interval_end:
        right_cut_visible[-1] = 1.0
    return np.stack(
        [
            left_cut_visible,
            right_cut_visible,
            pos_in_interval,
            dist_from_left_norm,
            dist_to_right_norm,
            pos_in_chunk,
        ],
        axis=1,
    ).astype(np.float32)


def _soft_label_weight_table(radius: int) -> dict[str, float]:
    clipped_radius = max(0, int(radius))
    return {
        str(distance): round(1.0 - float(distance) / float(max(1, clipped_radius) + 1), 6)
        for distance in range(clipped_radius + 1)
    }


def main() -> None:
    args = parse_args()
    source_manifest_path = args.source_dataset_root / "manifest.json"
    if not source_manifest_path.exists():
        raise SystemExit(f"Source dataset manifest not found: {source_manifest_path}")
    source_manifest = _load_json(source_manifest_path)
    songs = source_manifest.get("songs", [])
    if not isinstance(songs, list) or not songs:
        raise SystemExit(f"No songs found in source manifest: {source_manifest_path}")

    source_audio_feature_names = [str(name) for name in source_manifest.get("audio_feature_names", [])]
    if args.audio_feature_set == "demucs_events_no_rms":
        selected_audio_feature_names = list(DEMUX_EVENTS_NO_RMS_AUDIO_FEATURES)
        selected_audio_columns = _columns_for_names(source_audio_feature_names, selected_audio_feature_names)
    else:
        selected_audio_feature_names = list(source_audio_feature_names)
        selected_audio_columns = None

    split_map, source_split, split_source = _resolve_split_map(source_manifest, args.split_manifest)
    entries_out: list[dict] = []
    skipped: list[dict] = []
    segments_by_split = {"train": [], "val": [], "test": []}
    max_segment_frames_overall = 0

    for source_entry in songs:
        source_id = str(source_entry["id"])
        source_path = _resolve_source_feature_path(source_entry["feature_path"], args.source_dataset_root)
        if not source_path.exists():
            skipped.append({"source_id": source_id, "reason": f"missing source npz: {source_path}"})
            continue

        payload = np.load(source_path)
        audio_features = np.asarray(payload["audio_features"], dtype=np.float32)
        if selected_audio_columns is not None:
            audio_features = audio_features[:, selected_audio_columns]
        dance_features = np.asarray(payload["dance_features"], dtype=np.float32)
        keyframe_labels = np.asarray(payload["keyframe_labels"], dtype=np.float32)
        keyframe_mask = np.asarray(payload.get("keyframe_label_mask", np.ones(len(audio_features), dtype=np.float32)), dtype=np.float32)
        keyframe_frames = np.asarray(payload["keyframe_frames"], dtype=np.int32)
        cut_frames = np.asarray(payload["cut_frames"], dtype=np.int32)
        num_frames = int(audio_features.shape[0])
        fps = float(source_entry.get("fps", 30.0))
        max_segment_frames = max(1, int(round(float(args.max_segment_seconds) * fps)))
        max_segment_frames_overall = max(max_segment_frames_overall, max_segment_frames)

        if float(np.max(keyframe_mask)) <= 0.0:
            skipped.append({"source_id": source_id, "reason": "no keyframe labels available"})
            continue

        prediction_intervals = _build_prediction_intervals(num_frames, cut_frames, keyframe_frames)
        segment_serial = 0
        for interval in prediction_intervals:
            interval_start = int(interval["interval_start_frame"])
            interval_end = int(interval["interval_end_frame"])
            if interval_end < interval_start:
                continue
            chunks = _split_long_interval(interval_start, interval_end, max_frames=max_segment_frames)
            for chunk_index, (chunk_start, chunk_end) in enumerate(chunks):
                local_slice = slice(chunk_start, chunk_end + 1)
                local_len = int(chunk_end - chunk_start + 1)
                local_keyframe_frames = keyframe_frames[
                    (keyframe_frames >= chunk_start) & (keyframe_frames <= chunk_end)
                ] - int(chunk_start)
                local_keyframe_frames = np.asarray(local_keyframe_frames, dtype=np.int32)
                local_keyframe_labels = np.zeros(local_len, dtype=np.float32)
                if len(local_keyframe_frames) > 0:
                    local_keyframe_labels[np.unique(local_keyframe_frames)] = 1.0
                local_keyframe_soft = _event_frames_to_soft_labels(
                    local_keyframe_frames,
                    num_frames=local_len,
                    radius=int(args.soft_radius),
                )
                local_keyframe_mask = np.asarray(keyframe_mask[local_slice], dtype=np.float32)
                condition_features = _build_condition_features(
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    interval_start=interval_start,
                    interval_end=interval_end,
                    has_left_cut=interval["left_cut_frame"] is not None,
                    has_right_cut=interval["right_cut_frame"] is not None,
                )

                segment_id = f"{source_id}__seg_{segment_serial:04d}"
                segment_serial += 1
                output_path = args.output_root / "segments" / f"{segment_id}.npz"
                _write_npz(
                    output_path,
                    force=bool(args.force),
                    audio_features=np.asarray(audio_features[local_slice], dtype=np.float32),
                    dance_features=np.asarray(dance_features[local_slice], dtype=np.float32),
                    condition_features=condition_features,
                    keyframe_labels=local_keyframe_labels,
                    keyframe_soft_labels=local_keyframe_soft,
                    keyframe_label_mask=local_keyframe_mask,
                    keyframe_frames=local_keyframe_frames,
                )

                split_name = split_map.get(source_id)
                entry = {
                    "id": segment_id,
                    "source_id": source_id,
                    "source_index": int(source_entry.get("index", -1)),
                    "fps": fps,
                    "num_frames": local_len,
                    "duration_sec": round(local_len / fps, 6),
                    "interval_start_frame": int(interval_start),
                    "interval_end_frame": int(interval_end),
                    "interval_position": interval["interval_position"],
                    "left_cut_frame": interval["left_cut_frame"],
                    "left_cut_start_frame": interval["left_cut_start_frame"],
                    "left_cut_end_frame": interval["left_cut_end_frame"],
                    "right_cut_frame": interval["right_cut_frame"],
                    "right_cut_start_frame": interval["right_cut_start_frame"],
                    "right_cut_end_frame": interval["right_cut_end_frame"],
                    "chunk_start_frame": int(chunk_start),
                    "chunk_end_frame": int(chunk_end),
                    "chunk_index": int(chunk_index),
                    "chunk_count": int(len(chunks)),
                    "keyframe_count": int(len(local_keyframe_frames)),
                    "feature_path": str(output_path),
                    "split": split_name,
                }
                entries_out.append(entry)
                if split_name in segments_by_split:
                    segments_by_split[split_name].append(segment_id)

    manifest = {
        "dataset_type": "cut_conditioned_keyframe",
        "version": 4,
        "source_dataset_manifest": str(source_manifest_path),
        "source_split_source": split_source,
        "interval_definition": (
            "Each GT cut frame is expanded to its adjacent camera-keyframe pair, "
            "for example cut frame 16 with keyframes (15, 16), and that whole pair "
            "is excluded from input/target frames. Prediction intervals include "
            "the leading region before the first cut pair, all middle regions "
            "between cut pairs, and the trailing region after the last cut pair."
        ),
        "max_segment_seconds": float(args.max_segment_seconds),
        "max_segment_frames": int(max_segment_frames_overall),
        "soft_radius_frames": int(args.soft_radius),
        "soft_label_weighting": {
            "description": (
                "Distance-weighted soft targets for keyframes. Distance 0 is the exact GT frame; "
                "distances up to soft_radius_frames are treated as partial positives in training."
            ),
            "distance_frame_to_target_weight": _soft_label_weight_table(int(args.soft_radius)),
        },
        "evaluation_default_tolerance_frames": 3,
        "audio_feature_set": str(args.audio_feature_set),
        "audio_feature_names": selected_audio_feature_names,
        "dance_feature_names": source_manifest.get("dance_feature_names", []),
        "condition_feature_names": [
            "left_cut_visible",
            "right_cut_visible",
            "pos_in_interval",
            "dist_from_left_norm",
            "dist_to_right_norm",
            "pos_in_chunk",
        ],
        "segment_count": len(entries_out),
        "split": segments_by_split if source_split is not None else None,
        "segments": entries_out,
        "skipped": skipped,
    }
    _write_json(args.output_root / "manifest.json", manifest, force=bool(args.force))
    print(f"Saved manifest: {args.output_root / 'manifest.json'}")
    print(f"Saved segments: {len(entries_out)}")
    if source_split is not None:
        print(
            "Split counts: "
            f"train={len(segments_by_split['train'])} "
            f"val={len(segments_by_split['val'])} "
            f"test={len(segments_by_split['test'])}"
        )
    if skipped:
        print(f"Skipped: {len(skipped)}")


if __name__ == "__main__":
    main()
