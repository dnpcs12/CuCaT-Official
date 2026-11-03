from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = (
    Path(__file__).resolve().parent / "results" / "cut_keyframe_multimodal_dataset_dcm_dca_aist_full_dance"
)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "keyframe_density_stagea_dataset_dcm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Stage-A keyframe-density dataset. Each sample is a fixed-length "
            "window whose target is keyframe-frame occupancy inside the window."
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--window-frames", type=int, default=90, help="3 seconds at 30fps by default.")
    parser.add_argument("--stride-frames", type=int, default=15, help="0.5 seconds at 30fps by default.")
    parser.add_argument(
        "--include-tail-window",
        action="store_true",
        help="Also include the final full window when stride does not land exactly on it.",
    )
    parser.add_argument(
        "--target-label-source",
        choices=["hard", "soft"],
        default="hard",
        help="Use binary keyframe labels or soft keyframe labels to compute density.",
    )
    parser.add_argument(
        "--keep-cut-keyframe-pairs",
        action="store_true",
        help="Do not remove adjacent camera-keyframe pairs that encode GT cuts from the keyframe target.",
    )
    parser.add_argument(
        "--exclude-song-ids",
        default="",
        help="Comma-separated source song ids to exclude, e.g. c47,c105.",
    )
    parser.add_argument(
        "--exclude-scope",
        choices=["all", "train", "train_val"],
        default="all",
        help="Which split scope --exclude-song-ids applies to.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite without --force: {path}")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict], *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite without --force: {path}")
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_repo_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (WORKSPACE_ROOT / path).resolve()


def _relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(path.resolve())


def _clip_unique_frames(frames: np.ndarray, *, num_frames: int) -> list[int]:
    return sorted(
        {
            int(frame)
            for frame in np.asarray(frames, dtype=np.int32).tolist()
            if 0 <= int(frame) < int(num_frames)
        }
    )


def _cut_frame_to_keyframe_range(cut_frame: int, *, keyframe_set: set[int], num_frames: int) -> tuple[int, int]:
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


def _cut_keyframe_exclusion_mask(*, num_frames: int, cut_frames: np.ndarray, keyframe_frames: np.ndarray) -> np.ndarray:
    mask = np.zeros(num_frames, dtype=bool)
    clipped_cut_frames = _clip_unique_frames(cut_frames, num_frames=num_frames)
    keyframe_set = set(_clip_unique_frames(keyframe_frames, num_frames=num_frames))
    for cut_frame in clipped_cut_frames:
        start, end = _cut_frame_to_keyframe_range(cut_frame, keyframe_set=keyframe_set, num_frames=num_frames)
        mask[int(start) : int(end) + 1] = True
    return mask


def _build_window_starts(length: int, *, window_frames: int, stride_frames: int, include_tail: bool) -> list[int]:
    if length < window_frames:
        return []
    starts = list(range(0, length - window_frames + 1, max(1, stride_frames)))
    last_start = length - window_frames
    if include_tail and starts and starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _split_for_entry(entry: dict, manifest_split: dict | None) -> str:
    entry_id = str(entry["id"])
    if isinstance(manifest_split, dict):
        for split_name in ("train", "val", "test"):
            if entry_id in {str(item) for item in manifest_split.get(split_name, [])}:
                return split_name
    official = str(entry.get("official_split", "")).lower()
    if official in {"train", "val", "test"}:
        return official
    if official == "validation":
        return "val"
    return "train"


def _previous_cut_frames(cut_frames: np.ndarray, start_frame: int) -> tuple[int | None, int | None]:
    previous = [int(frame) for frame in np.asarray(cut_frames, dtype=np.int32).tolist() if int(frame) < int(start_frame)]
    if not previous:
        return None, None
    last_cut = previous[-1]
    previous_cut = previous[-2] if len(previous) >= 2 else None
    return last_cut, previous_cut


def _parse_id_set(text: str) -> set[str]:
    return {item.strip() for item in str(text).split(",") if item.strip()}


def _excluded_by_scope(*, split_name: str, scope: str) -> bool:
    if scope == "all":
        return True
    if scope == "train":
        return split_name == "train"
    if scope == "train_val":
        return split_name in {"train", "val"}
    return False


def main() -> None:
    args = parse_args()
    manifest_path = args.source_root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Source manifest not found: {manifest_path}")
    source_manifest = _load_json(manifest_path)
    source_songs = source_manifest.get("songs", [])
    if not source_songs:
        raise SystemExit(f"No songs found in source manifest: {manifest_path}")

    output_root = args.output_root.resolve()
    rows: list[dict] = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_positive_counts = {"train": 0, "val": 0, "test": 0}
    total_keyframe_count = 0
    excluded_cut_keyframe_count = 0
    excluded_song_ids = _parse_id_set(args.exclude_song_ids)
    excluded_entry_count = 0
    manifest_split = source_manifest.get("split") if isinstance(source_manifest.get("split"), dict) else None

    for entry in source_songs:
        source_song_id = str(entry.get("source_song_id", entry["id"]))
        split_name = _split_for_entry(entry, manifest_split)
        if (
            (source_song_id in excluded_song_ids or str(entry["id"]) in excluded_song_ids)
            and _excluded_by_scope(split_name=split_name, scope=str(args.exclude_scope))
        ):
            excluded_entry_count += 1
            continue
        feature_path = _resolve_repo_path(entry["feature_path"])
        if not feature_path.exists():
            raise SystemExit(f"Missing feature file: {feature_path}")

        data = np.load(feature_path)
        audio_features = np.asarray(data["audio_features"], dtype=np.float32)
        num_frames = int(audio_features.shape[0])
        if num_frames < int(args.window_frames):
            continue

        target_key = "keyframe_soft_labels" if str(args.target_label_source) == "soft" else "keyframe_labels"
        key_labels = np.asarray(data[target_key], dtype=np.float32).copy()
        keyframe_frames = np.asarray(data["keyframe_frames"], dtype=np.int32)
        cut_frames = np.asarray(data["cut_frames"], dtype=np.int32)
        total_keyframe_count += int(np.asarray(data["keyframe_labels"], dtype=np.float32).sum())

        if not args.keep_cut_keyframe_pairs:
            exclusion_mask = _cut_keyframe_exclusion_mask(
                num_frames=num_frames,
                cut_frames=cut_frames,
                keyframe_frames=keyframe_frames,
            )
            excluded_cut_keyframe_count += int(np.asarray(data["keyframe_labels"], dtype=np.float32)[exclusion_mask].sum())
            key_labels[exclusion_mask] = 0.0

        starts = _build_window_starts(
            num_frames,
            window_frames=int(args.window_frames),
            stride_frames=int(args.stride_frames),
            include_tail=bool(args.include_tail_window),
        )
        for start in starts:
            end = int(start) + int(args.window_frames)
            window_target = key_labels[start:end]
            density = float(np.sum(window_target) / float(max(1, int(args.window_frames))))
            hard_count = int(np.sum(np.asarray(data["keyframe_labels"], dtype=np.float32)[start:end] > 0))
            effective_count = float(np.sum(window_target))
            last_cut, previous_cut = _previous_cut_frames(cut_frames, int(start))
            row = {
                "id": f"{entry['id']}__w{int(start):06d}",
                "entry_id": str(entry["id"]),
                "source_song_id": source_song_id,
                "split": split_name,
                "feature_path": _relative_to_repo(feature_path),
                "start_frame": int(start),
                "end_frame_exclusive": int(end),
                "window_frames": int(args.window_frames),
                "fps": float(entry.get("fps", 30.0)),
                "target_density": round(density, 8),
                "target_keyframe_count": round(effective_count, 6),
                "hard_keyframe_count": hard_count,
                "last_cut_frame": last_cut,
                "previous_cut_frame": previous_cut,
            }
            rows.append(row)
            split_counts[split_name] = split_counts.get(split_name, 0) + 1
            if density > 0:
                split_positive_counts[split_name] = split_positive_counts.get(split_name, 0) + 1

    rows.sort(key=lambda row: (row["split"], row["entry_id"], row["start_frame"]))
    windows_path = output_root / "windows.jsonl"
    _write_jsonl(windows_path, rows, force=bool(args.force))

    output_manifest = {
        "dataset_type": "keyframe_density_stagea",
        "version": 1,
        "source_dataset_manifest": str(manifest_path),
        "source_dataset_type": source_manifest.get("dataset_type"),
        "window_frames": int(args.window_frames),
        "stride_frames": int(args.stride_frames),
        "target_label_source": str(args.target_label_source),
        "cut_keyframe_pairs_excluded": not bool(args.keep_cut_keyframe_pairs),
        "audio_feature_names": list(source_manifest.get("audio_feature_names", [])),
        "dance_feature_names": list(source_manifest.get("dance_feature_names", [])),
        "audio_feature_dim": len(source_manifest.get("audio_feature_names", [])),
        "dance_feature_dim": len(source_manifest.get("dance_feature_names", [])),
        "prev_cut_feature_names": [
            "prev_cut_exists",
            "since_prev_cut_clipped",
            "prev_cut_gap_clipped",
            "prev_segment_progress_clipped",
        ],
        "window_index_path": _relative_to_repo(windows_path),
        "window_count": len(rows),
        "split_counts": split_counts,
        "positive_window_counts": split_positive_counts,
        "total_source_keyframes": int(total_keyframe_count),
        "excluded_cut_pair_keyframes": int(excluded_cut_keyframe_count),
        "excluded_song_ids": sorted(excluded_song_ids),
        "exclude_scope": str(args.exclude_scope),
        "excluded_entry_count": int(excluded_entry_count),
        "split": {
            "train": [row["id"] for row in rows if row["split"] == "train"],
            "val": [row["id"] for row in rows if row["split"] == "val"],
            "test": [row["id"] for row in rows if row["split"] == "test"],
        },
    }
    _write_json(output_root / "manifest.json", output_manifest, force=bool(args.force))

    print(f"Source root: {args.source_root}")
    print(f"Output root: {output_root}")
    print(f"Windows: {len(rows)}")
    print(f"Split counts: {split_counts}")
    print(f"Positive windows: {split_positive_counts}")
    print(f"Excluded cut-pair keyframes: {excluded_cut_keyframe_count}/{total_keyframe_count}")
    print(f"Excluded source songs: {sorted(excluded_song_ids)} scope={args.exclude_scope} ({excluded_entry_count} entries)")
    print(f"Saved manifest: {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
