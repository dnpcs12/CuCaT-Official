from __future__ import annotations

import argparse
import json
import sys
from bisect import bisect_right
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DCM_MANIFEST = (
    WORKSPACE_ROOT / "music_processing" / "results" / "cut_keyframe_multimodal_dataset_dcm" / "manifest.json"
)
DEFAULT_FULL_LABEL_ROOT = (
    WORKSPACE_ROOT
    / "camera_autoshot"
    / "reverse_v3"
    / "labels"
    / "v3_1_full"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "camera_autoshot"
    / "reverse_v3"
    / "dcm_clip_labels"
    / "v3_1_close_dense_temporal_delta"
)
DEFAULT_SPLIT_MANIFEST = (
    WORKSPACE_ROOT
    / "shotplan_transformer"
    / "results"
    / "dcm_clip_split_manifest_v3_1_close_dense_temporal_delta.json"
)
V3_LABEL_FIELDS = (
    "target_anchor",
    "distance",
    "yaw",
    "pitch",
    "composition",
    "roll",
    "fov",
    "transition",
    "interpolation",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_full_keyframes(path: Path) -> list[dict[str, object]]:
    payload = _load_json(path)
    rows = payload.get("keyframes", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected V3 label payload: {path}")
    keyframes = [row for row in rows if isinstance(row, dict) and "frame" in row]
    return sorted(keyframes, key=lambda row: int(row["frame"]))


def _default_label_value(field: str) -> str:
    if field == "fov":
        # Representation ablations may intentionally remove FOV from the
        # source label.  Keep clip-label building robust by falling back to
        # the solver's default FOV token.
        return "normal"
    if field == "transition":
        return "linear"
    if field == "interpolation":
        return "linear"
    raise KeyError(field)


def _copy_label_at_frame(row: dict[str, object], *, local_frame: int, source_frame: int) -> dict[str, object]:
    copied = {"frame": int(local_frame), "source_frame": int(source_frame)}
    for field in V3_LABEL_FIELDS:
        copied[field] = str(row[field]) if field in row else _default_label_value(field)
    return copied


def _active_row_at(frames: list[int], rows: list[dict[str, object]], source_frame: int) -> dict[str, object]:
    index = bisect_right(frames, int(source_frame)) - 1
    if index < 0:
        index = 0
    return rows[index]


def _dedupe_by_frame_and_label(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[object, ...]] = set()
    output: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: (int(item["frame"]), int(item.get("source_frame", item["frame"])))):
        signature = tuple([int(row["frame"])] + [row.get(field, "") for field in V3_LABEL_FIELDS])
        if signature in seen:
            continue
        seen.add(signature)
        output.append(row)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clip full-song Reverse V3 labels into official DCM split clips.")
    parser.add_argument("--dcm-manifest", type=Path, default=DEFAULT_DCM_MANIFEST)
    parser.add_argument("--full-label-root", type=Path, default=DEFAULT_FULL_LABEL_ROOT)
    parser.add_argument("--full-label-name", default="camera_poses_v3_close_dense.json")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-label-name", default="camera_poses_v3.json")
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dcm_manifest_path = args.dcm_manifest.resolve()
    full_label_root = args.full_label_root.resolve()
    output_root = args.output_root.resolve()
    split_manifest_path = args.split_manifest.resolve()

    dcm_manifest = _load_json(dcm_manifest_path)
    clips = [clip for clip in dcm_manifest.get("songs", []) if isinstance(clip, dict)]
    full_label_cache: dict[str, list[dict[str, object]]] = {}
    split_rows = {"train": [], "val": [], "test": []}
    clip_metadata: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    for clip in clips:
        clip_id = str(clip["id"])
        source_song_id = str(clip["source_song_id"])
        official_split = str(clip.get("official_split", "train"))
        if official_split not in split_rows:
            skipped.append({"id": clip_id, "reason": f"unknown split {official_split}"})
            continue

        full_label_path = full_label_root / source_song_id / str(args.full_label_name)
        if source_song_id not in full_label_cache:
            if not full_label_path.exists():
                skipped.append({"id": clip_id, "reason": "missing full-song V3 label", "path": str(full_label_path)})
                continue
            full_label_cache[source_song_id] = _load_full_keyframes(full_label_path)
        full_rows = full_label_cache[source_song_id]
        if not full_rows:
            skipped.append({"id": clip_id, "reason": "empty full-song V3 label", "path": str(full_label_path)})
            continue

        full_frames = [int(row["frame"]) for row in full_rows]
        source_start = int(clip["source_start_frame"])
        source_end = int(clip["source_end_frame"])
        num_frames = int(clip["num_frames"])
        if num_frames <= 0 or source_end < source_start:
            skipped.append({"id": clip_id, "reason": "invalid clip frame range"})
            continue

        local_rows: list[dict[str, object]] = []
        start_row = _active_row_at(full_frames, full_rows, source_start)
        local_rows.append(_copy_label_at_frame(start_row, local_frame=0, source_frame=source_start))

        for row in full_rows:
            source_frame = int(row["frame"])
            if source_start < source_frame <= source_end:
                local_frame = source_frame - source_start
                if 0 <= local_frame < num_frames:
                    local_rows.append(_copy_label_at_frame(row, local_frame=local_frame, source_frame=source_frame))

        end_row = _active_row_at(full_frames, full_rows, source_end)
        local_rows.append(_copy_label_at_frame(end_row, local_frame=num_frames - 1, source_frame=source_end))
        local_rows = _dedupe_by_frame_and_label(local_rows)

        payload = {
            "schema": "camera_autoshot.reverse_v3.dcm_clip_labels",
            "source_schema": "camera_autoshot.reverse_v3",
            "clip_id": clip_id,
            "source_song_id": source_song_id,
            "source_start_frame": source_start,
            "source_end_frame": source_end,
            "num_frames": num_frames,
            "fps": float(clip.get("fps", 30.0)),
            "official_split": official_split,
            "full_label_path": str(full_label_path),
            "keyframes": local_rows,
        }
        output_path = output_root / clip_id / str(args.output_label_name)
        _write_json(output_path, payload)

        split_rows[official_split].append(clip_id)
        clip_metadata.append(
            {
                "clip_id": clip_id,
                "source_song_id": source_song_id,
                "official_split": official_split,
                "source_start_frame": source_start,
                "source_end_frame": source_end,
                "num_frames": num_frames,
                "keyframe_count": len(local_rows),
                "label_path": str(output_path),
                "feature_id": clip_id,
            }
        )

    split_manifest = {
        "dataset_type": "dcm_clip_split_for_v3_shotplan_transformer",
        "version": 1,
        "source_dcm_manifest": str(dcm_manifest_path),
        "label_root": str(output_root),
        "label_filename": str(args.output_label_name),
        "available_song_ids": [row["clip_id"] for row in clip_metadata],
        "train": split_rows["train"],
        "val": split_rows["val"],
        "test": split_rows["test"],
        "clips": clip_metadata,
        "skipped": skipped,
    }
    _write_json(split_manifest_path, split_manifest)

    print(f"label root: {output_root}")
    print(f"split manifest: {split_manifest_path}")
    print(f"clips: {len(clip_metadata)} train={len(split_rows['train'])} val={len(split_rows['val'])} test={len(split_rows['test'])}")
    print(f"skipped: {len(skipped)}")


if __name__ == "__main__":
    main()
