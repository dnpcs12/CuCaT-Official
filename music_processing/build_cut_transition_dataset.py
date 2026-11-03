from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import librosa
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from music_processing.analyze_audio_segments import DEFAULT_OUTPUT_ROOT as DEFAULT_ANALYSIS_ROOT
from music_processing.analyze_audio_segments import run_analysis
from music_processing.audio_peaks import DEFAULT_RESULTS_ROOT, DEFAULT_VIDEO_ROOT, resolve_audio_path, resolve_video_path
from music_processing.madmom_downbeats import save_madmom_downbeats


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CUT_LABELS_PATH = (
    WORKSPACE_ROOT / "camera_autoshot" / "analysis" / "gt_cut_frames" / "gt_cut_frames_per_song.json"
)
DEFAULT_DEMUCS_ROOT = DEFAULT_RESULTS_ROOT / "demucs"
DEFAULT_OUTPUT_ROOT = DEFAULT_RESULTS_ROOT / "cut_transition_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build frame-level cut-transition datasets from waveform, Demucs stems, and beat/bar events."
    )
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=107, help="End index (inclusive)")
    parser.add_argument(
        "--indexes",
        type=int,
        nargs="*",
        help="Optional explicit list of indexes. When set, --start/--end are ignored.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root folder where per-song frame-level dataset files will be written.",
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=DEFAULT_ANALYSIS_ROOT,
        help="Root folder containing heuristic audio summaries.",
    )
    parser.add_argument(
        "--demucs-root",
        type=Path,
        default=DEFAULT_DEMUCS_ROOT,
        help="Root folder containing Demucs separation outputs.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=DEFAULT_VIDEO_ROOT,
        help="Folder that contains cN.mp4 files.",
    )
    parser.add_argument(
        "--cut-labels",
        type=Path,
        default=DEFAULT_CUT_LABELS_PATH,
        help="Path to gt_cut_frames_per_song.json.",
    )
    parser.add_argument(
        "--soft-radius",
        type=int,
        default=3,
        help="Triangular soft-label radius in frames around each GT cut.",
    )
    parser.add_argument(
        "--distance-clip-frames",
        type=int,
        default=90,
        help="Clip distance-to-event features at this many frames before normalization.",
    )
    parser.add_argument(
        "--refresh-analysis",
        action="store_true",
        help="Recompute heuristic analysis if summary.json is missing or stale.",
    )
    parser.add_argument(
        "--refresh-madmom",
        action="store_true",
        help="Recompute madmom beat/downbeat analysis if madmom_downbeats.json is missing or stale.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip songs that already have a saved dataset .npz file.",
    )
    return parser.parse_args()


def _round_time(value: float) -> float:
    return round(float(value), 3)


def _load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_indexes(args: argparse.Namespace) -> list[int]:
    if args.indexes:
        return sorted(set(int(index) for index in args.indexes))
    return list(range(int(args.start), int(args.end) + 1))


def _probe_video_stream(video_path: Path) -> tuple[float, int, float]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,duration,nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    rate_text = stream["avg_frame_rate"]
    num_text, denom_text = rate_text.split("/")
    fps = float(num_text) / float(denom_text)
    num_frames = int(stream["nb_frames"])
    duration_sec = float(stream["duration"])
    return fps, num_frames, duration_sec


def _resolve_optional_local_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (WORKSPACE_ROOT / path).resolve()


def _load_cut_map(path: Path) -> dict[int, dict]:
    rows = _load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in cut-label file: {path}")
    return {int(row["index"]): row for row in rows}


def _load_or_build_summary(
    *,
    index: int,
    analysis_root: Path,
    refresh_analysis: bool,
) -> dict:
    summary_path = analysis_root / f"c{index}" / "summary.json"
    if summary_path.exists() and not refresh_analysis:
        return _load_json(summary_path)  # type: ignore[return-value]
    return run_analysis(index=index, output_root=analysis_root, beats_per_bar=4)


def _load_or_build_madmom(
    *,
    index: int,
    audio_path: Path,
    analysis_root: Path,
    refresh_madmom: bool,
) -> dict:
    madmom_path = analysis_root / f"c{index}" / "madmom_downbeats.json"
    if madmom_path.exists() and not refresh_madmom:
        return _load_json(madmom_path)  # type: ignore[return-value]
    return save_madmom_downbeats(audio_path, madmom_path)


def _load_demucs_summary(index: int, demucs_root: Path) -> dict:
    summary_path = demucs_root / f"c{index}" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Demucs summary not found: {summary_path}")
    return _load_json(summary_path)  # type: ignore[return-value]


def _load_aligned_audio(audio_path: Path, *, num_frames: int, fps: float) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    target_sample_count = int(round(float(num_frames) * float(sr) / float(fps)))
    if target_sample_count <= 0:
        raise ValueError(f"Invalid frame count for audio alignment: {audio_path}")
    if len(y) < target_sample_count:
        y = np.pad(y, (0, target_sample_count - len(y)))
    else:
        y = y[:target_sample_count]
    return np.asarray(y, dtype=np.float32), int(sr)


def _frame_edges(*, num_frames: int, sr: int, fps: float) -> np.ndarray:
    edges = np.round(np.arange(num_frames + 1, dtype=np.float64) * float(sr) / float(fps)).astype(np.int64)
    if edges[-1] <= edges[0]:
        raise ValueError("Invalid frame edges")
    return edges


def _frame_stats(y: np.ndarray, frame_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(y) < int(frame_edges[-1]):
        y = np.pad(y, (0, int(frame_edges[-1]) - len(y)))
    abs_y = np.abs(y).astype(np.float32)
    sq_y = np.square(y).astype(np.float32)
    lengths = np.maximum(1, np.diff(frame_edges).astype(np.int64))
    cumsum_abs = np.concatenate([np.zeros(1, dtype=np.float32), np.cumsum(abs_y, dtype=np.float32)])
    cumsum_sq = np.concatenate([np.zeros(1, dtype=np.float32), np.cumsum(sq_y, dtype=np.float32)])

    abs_mean = (cumsum_abs[frame_edges[1:]] - cumsum_abs[frame_edges[:-1]]) / lengths
    rms = np.sqrt((cumsum_sq[frame_edges[1:]] - cumsum_sq[frame_edges[:-1]]) / lengths)
    peak = np.maximum.reduceat(abs_y, frame_edges[:-1])[: len(lengths)]
    return abs_mean.astype(np.float32), rms.astype(np.float32), peak.astype(np.float32)


def _delta_feature(values: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros(1, dtype=np.float32), np.diff(values).astype(np.float32)])


def _event_frames_from_rows(rows: list[dict], *, fps: float, num_frames: int) -> np.ndarray:
    if not rows:
        return np.zeros(0, dtype=np.int32)
    frames = np.round(np.array([float(row["time_sec"]) for row in rows], dtype=np.float32) * float(fps)).astype(np.int32)
    frames = frames[(frames >= 0) & (frames < num_frames)]
    return np.unique(frames)


def _event_frames_from_cut_frames(cut_frames: list[int], *, num_frames: int) -> np.ndarray:
    frames = np.array([int(frame) for frame in cut_frames], dtype=np.int32)
    frames = frames[(frames >= 0) & (frames < num_frames)]
    return np.unique(frames)


def _pulse_track(event_frames: np.ndarray, num_frames: int) -> np.ndarray:
    pulses = np.zeros(num_frames, dtype=np.float32)
    if len(event_frames) > 0:
        pulses[event_frames] = 1.0
    return pulses


def _distance_tracks(event_frames: np.ndarray, num_frames: int, *, clip_frames: int) -> tuple[np.ndarray, np.ndarray]:
    since = np.full(num_frames, int(clip_frames), dtype=np.float32)
    until = np.full(num_frames, int(clip_frames), dtype=np.float32)
    if len(event_frames) == 0:
        scale = float(max(1, clip_frames))
        return since / scale, until / scale

    timeline = np.arange(num_frames, dtype=np.int32)
    prev_positions = np.searchsorted(event_frames, timeline, side="right") - 1
    next_positions = np.searchsorted(event_frames, timeline, side="left")

    prev_mask = prev_positions >= 0
    next_mask = next_positions < len(event_frames)

    since[prev_mask] = timeline[prev_mask] - event_frames[prev_positions[prev_mask]]
    until[next_mask] = event_frames[next_positions[next_mask]] - timeline[next_mask]

    scale = float(max(1, clip_frames))
    return np.clip(since, 0, clip_frames) / scale, np.clip(until, 0, clip_frames) / scale


def _build_soft_labels(cut_frames: np.ndarray, num_frames: int, radius: int) -> np.ndarray:
    labels = np.zeros(num_frames, dtype=np.float32)
    for cut_frame in cut_frames.tolist():
        start = max(0, int(cut_frame) - int(radius))
        end = min(num_frames - 1, int(cut_frame) + int(radius))
        span = np.arange(start, end + 1, dtype=np.int32)
        distance = np.abs(span - int(cut_frame)).astype(np.float32)
        ramp = np.maximum(0.0, 1.0 - distance / float(max(1, radius + 1)))
        labels[span] = np.maximum(labels[span], ramp)
    return labels


def _build_song_dataset(
    *,
    index: int,
    cut_row: dict,
    output_root: Path,
    analysis_root: Path,
    demucs_root: Path,
    video_root: Path,
    soft_radius: int,
    distance_clip_frames: int,
    refresh_analysis: bool,
    refresh_madmom: bool,
) -> dict:
    audio_path = resolve_audio_path(index=index)
    video_path = resolve_video_path(index=index, video_root=video_root)
    fps, num_frames, duration_sec = _probe_video_stream(video_path)

    summary = _load_or_build_summary(index=index, analysis_root=analysis_root, refresh_analysis=refresh_analysis)
    madmom = _load_or_build_madmom(
        index=index,
        audio_path=audio_path,
        analysis_root=analysis_root,
        refresh_madmom=refresh_madmom,
    )
    demucs_summary = _load_demucs_summary(index=index, demucs_root=demucs_root)

    y_mix, sr = _load_aligned_audio(audio_path, num_frames=num_frames, fps=fps)
    frame_edges = _frame_edges(num_frames=num_frames, sr=sr, fps=fps)

    mix_abs_mean, mix_rms, mix_peak = _frame_stats(y_mix, frame_edges)
    feature_columns = [mix_abs_mean, mix_rms, mix_peak]
    feature_names = ["mix_abs_mean", "mix_rms", "mix_peak"]

    stem_files = demucs_summary.get("stem_files", {})
    stem_rms_channels: dict[str, np.ndarray] = {"mix": mix_rms}
    for stem_name in ("vocals", "bass", "drums", "other"):
        stem_path_text = stem_files.get(stem_name)
        if not stem_path_text:
            raise FileNotFoundError(f"Missing Demucs stem `{stem_name}` for c{index}")
        stem_path = _resolve_optional_local_path(stem_path_text)
        if not stem_path.exists():
            raise FileNotFoundError(f"Stem file not found: {stem_path}")
        y_stem, stem_sr = _load_aligned_audio(stem_path, num_frames=num_frames, fps=fps)
        stem_frame_edges = _frame_edges(num_frames=num_frames, sr=stem_sr, fps=fps)
        stem_abs_mean, stem_rms, stem_peak = _frame_stats(y_stem, stem_frame_edges)
        feature_columns.extend([stem_abs_mean, stem_rms, stem_peak])
        feature_names.extend([f"{stem_name}_abs_mean", f"{stem_name}_rms", f"{stem_name}_peak"])
        stem_rms_channels[stem_name] = stem_rms

    for stem_name in ("mix", "vocals", "bass", "drums", "other"):
        delta = _delta_feature(stem_rms_channels[stem_name])
        feature_columns.append(delta)
        feature_names.append(f"{stem_name}_rms_delta")

    beat_frames = _event_frames_from_rows(madmom.get("beats", []), fps=fps, num_frames=num_frames)
    downbeat_frames = _event_frames_from_rows(madmom.get("downbeats", []), fps=fps, num_frames=num_frames)
    bar_frames = _event_frames_from_rows(summary.get("bars", []), fps=fps, num_frames=num_frames)

    for event_name, event_frames in (
        ("beat", beat_frames),
        ("downbeat", downbeat_frames),
        ("bar", bar_frames),
    ):
        pulses = _pulse_track(event_frames, num_frames)
        since, until = _distance_tracks(event_frames, num_frames, clip_frames=distance_clip_frames)
        feature_columns.extend([pulses, since, until])
        feature_names.extend([f"{event_name}_pulse", f"{event_name}_since", f"{event_name}_until"])

    cut_frames = _event_frames_from_cut_frames(cut_row.get("cut_frames", []), num_frames=num_frames)
    labels = _pulse_track(cut_frames, num_frames).astype(np.float32)
    soft_labels = _build_soft_labels(cut_frames, num_frames, soft_radius)

    features = np.stack(feature_columns, axis=1).astype(np.float32)
    frame_times_sec = (np.arange(num_frames, dtype=np.float32) / float(fps)).astype(np.float32)

    songs_dir = output_root / "songs"
    songs_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = songs_dir / f"c{index}.npz"
    np.savez_compressed(
        dataset_path,
        features=features,
        labels=labels,
        soft_labels=soft_labels,
        frame_times_sec=frame_times_sec,
        cut_frames=cut_frames.astype(np.int32),
    )

    return {
        "id": f"c{index}",
        "index": index,
        "fps": round(float(fps), 6),
        "num_frames": int(num_frames),
        "duration_sec": _round_time(duration_sec),
        "cut_count": int(len(cut_frames)),
        "positive_frame_count": int(np.sum(labels)),
        "positive_frame_ratio": round(float(np.mean(labels)) if len(labels) else 0.0, 6),
        "feature_path": str(dataset_path.relative_to(WORKSPACE_ROOT)),
        "feature_dim": int(features.shape[1]),
        "camera_vmd": cut_row.get("camera_vmd"),
        "beat_count": int(len(beat_frames)),
        "downbeat_count": int(len(downbeat_frames)),
        "bar_count": int(len(bar_frames)),
        "feature_names": feature_names,
    }


def build_dataset(
    *,
    indexes: list[int],
    output_root: Path,
    analysis_root: Path,
    demucs_root: Path,
    video_root: Path,
    cut_labels_path: Path,
    soft_radius: int,
    distance_clip_frames: int,
    refresh_analysis: bool,
    refresh_madmom: bool,
    skip_existing: bool,
) -> dict:
    cut_map = _load_cut_map(cut_labels_path)
    output_root.mkdir(parents=True, exist_ok=True)

    songs: list[dict] = []
    skipped: list[dict] = []
    build_times: list[dict] = []
    feature_names_reference: list[str] | None = None

    for index in indexes:
        dataset_path = output_root / "songs" / f"c{index}.npz"
        if skip_existing and dataset_path.exists():
            try:
                cached = np.load(dataset_path)
                songs.append(
                    {
                        "id": f"c{index}",
                        "index": index,
                        "fps": 30.0,
                        "num_frames": int(cached["features"].shape[0]),
                        "duration_sec": _round_time(int(cached["features"].shape[0]) / 30.0),
                        "cut_count": int(len(cached["cut_frames"])),
                        "positive_frame_count": int(np.sum(cached["labels"])),
                        "positive_frame_ratio": round(float(np.mean(cached["labels"])) if len(cached["labels"]) else 0.0, 6),
                        "feature_path": str(dataset_path.relative_to(WORKSPACE_ROOT)),
                        "feature_dim": int(cached["features"].shape[1]),
                        "camera_vmd": cut_map.get(index, {}).get("camera_vmd"),
                        "beat_count": None,
                        "downbeat_count": None,
                        "bar_count": None,
                    }
                )
                if feature_names_reference is None:
                    manifest_path = output_root / "manifest.json"
                    if manifest_path.exists():
                        manifest = _load_json(manifest_path)
                        feature_names_reference = manifest.get("feature_names")
                print(f"Skipped c{index}: existing dataset")
                continue
            except Exception:  # noqa: BLE001
                pass

        cut_row = cut_map.get(index)
        if cut_row is None:
            skipped.append({"id": f"c{index}", "index": index, "reason": "Missing GT cut-label row"})
            print(f"Skipped c{index}: missing GT cut-label row")
            continue

        start_time = perf_counter()
        try:
            song_row = _build_song_dataset(
                index=index,
                cut_row=cut_row,
                output_root=output_root,
                analysis_root=analysis_root,
                demucs_root=demucs_root,
                video_root=video_root,
                soft_radius=soft_radius,
                distance_clip_frames=distance_clip_frames,
                refresh_analysis=refresh_analysis,
                refresh_madmom=refresh_madmom,
            )
        except Exception as exc:  # noqa: BLE001
            skipped.append({"id": f"c{index}", "index": index, "reason": str(exc)})
            print(f"Skipped c{index}: {exc}")
            continue

        elapsed_sec = round(perf_counter() - start_time, 3)
        build_times.append({"id": song_row["id"], "elapsed_sec": elapsed_sec})
        songs.append(song_row)
        print(f"Built {song_row['id']} in {elapsed_sec:.3f}s")

        if feature_names_reference is None:
            feature_names_reference = list(song_row["feature_names"])

    if feature_names_reference is None and songs:
        first_song_path = WORKSPACE_ROOT / songs[0]["feature_path"]
        data = np.load(first_song_path)
        feature_names_reference = [f"feature_{i}" for i in range(int(data["features"].shape[1]))]

    positive_frames = int(sum(song["positive_frame_count"] for song in songs))
    total_frames = int(sum(song["num_frames"] for song in songs))
    total_cuts = int(sum(song["cut_count"] for song in songs))

    manifest = {
        "dataset_type": "frame_level_cut_transition",
        "version": 1,
        "label_source": str(cut_labels_path),
        "soft_radius_frames": int(soft_radius),
        "distance_clip_frames": int(distance_clip_frames),
        "feature_names": feature_names_reference or [],
        "song_count": len(songs),
        "skipped_count": len(skipped),
        "total_frames": total_frames,
        "total_cuts": total_cuts,
        "positive_frame_count": positive_frames,
        "positive_frame_ratio": round(float(positive_frames / total_frames) if total_frames else 0.0, 6),
        "songs": songs,
        "skipped": skipped,
        "build_times_sec": build_times,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_dataset(
        indexes=_resolve_indexes(args),
        output_root=args.output_root,
        analysis_root=args.analysis_root,
        demucs_root=args.demucs_root,
        video_root=args.video_root,
        cut_labels_path=args.cut_labels,
        soft_radius=max(0, int(args.soft_radius)),
        distance_clip_frames=max(1, int(args.distance_clip_frames)),
        refresh_analysis=args.refresh_analysis,
        refresh_madmom=args.refresh_madmom,
        skip_existing=args.skip_existing,
    )
    print(f"Saved manifest: {args.output_root / 'manifest.json'}")
    print(f"Songs: {', '.join(song['id'] for song in manifest['songs'])}")
    if manifest["skipped"]:
        print(f"Skipped: {', '.join(song['id'] for song in manifest['skipped'])}")


if __name__ == "__main__":
    main()
