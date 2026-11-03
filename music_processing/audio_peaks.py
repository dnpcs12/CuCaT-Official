from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "local_assets" / "amc_raw_data"
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_ROOT = WORKSPACE_ROOT / "saba_mmd" / "build" / "Release"
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class Peak:
    frame: int
    time_sec: float
    novelty: float


@dataclass(frozen=True)
class AudioAnalysisResult:
    summary: dict
    chroma: np.ndarray
    similarity: np.ndarray
    novelty: np.ndarray
    novelty_times: np.ndarray


def resolve_audio_path(index: int, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    wav_path = data_root / f"amc{index}" / f"a{index}.wav"
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV not found: {wav_path}")
    return wav_path


def resolve_video_path(index: int, video_root: Path = DEFAULT_VIDEO_ROOT) -> Path:
    video_path = video_root / f"c{index}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    return video_path


def _round_time(value: float) -> float:
    return round(float(value), 3)


def peak_to_dict(peak: Peak) -> dict:
    payload = asdict(peak)
    payload["time_sec"] = _round_time(payload["time_sec"])
    payload["novelty"] = round(float(payload["novelty"]), 4)
    return payload


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    serialized_rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(serialized_rows)


def write_peak_csv(path: Path, peaks: Iterable[dict]) -> None:
    write_csv(path, peaks, ["frame", "time_sec", "novelty"])


def write_timed_event_csv(path: Path, rows: Iterable[dict]) -> None:
    serialized_rows = list(rows)
    if serialized_rows:
        fieldnames = list(serialized_rows[0].keys())
    else:
        fieldnames = ["index", "frame", "time_sec", "novelty"]
    write_csv(path, serialized_rows, fieldnames)


def _tempo_to_float(tempo: np.ndarray | float) -> float:
    if isinstance(tempo, np.ndarray):
        return float(np.asarray(tempo).reshape(-1)[0])
    return float(tempo)


def _build_timed_rows(
    frames: np.ndarray,
    *,
    sr: int,
    novelty: np.ndarray,
    base_index: int = 1,
    beat_indices: np.ndarray | None = None,
) -> list[dict]:
    times = librosa.frames_to_time(frames, sr=sr)
    rows: list[dict] = []
    for idx, (frame, time_sec) in enumerate(zip(frames, times), start=base_index):
        row = {
            "index": idx,
            "frame": int(frame),
            "time_sec": _round_time(time_sec),
            "novelty": round(float(novelty[int(frame)]), 4),
        }
        if beat_indices is not None:
            row["beat_index"] = int(beat_indices[idx - base_index])
        rows.append(row)
    return rows


def _select_bar_offset(beat_frames: np.ndarray, novelty: np.ndarray, beats_per_bar: int) -> int:
    if len(beat_frames) == 0:
        return 0

    best_offset = 0
    best_score = float("-inf")
    max_offset = min(beats_per_bar, len(beat_frames))
    for offset in range(max_offset):
        sample = beat_frames[offset::beats_per_bar]
        if len(sample) == 0:
            continue
        score = float(np.mean(novelty[sample]))
        if score > best_score:
            best_score = score
            best_offset = offset
    return best_offset


def _compute_meter_rows(
    *,
    novelty: np.ndarray,
    sr: int,
    beats_per_bar: int,
) -> tuple[float, list[dict], list[dict], int]:
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=novelty, sr=sr, trim=False)
    tempo_bpm = _tempo_to_float(tempo)

    if len(beat_frames) == 0:
        return tempo_bpm, [], [], 0

    beat_rows = _build_timed_rows(beat_frames, sr=sr, novelty=novelty)
    bar_offset = _select_bar_offset(beat_frames, novelty, beats_per_bar=beats_per_bar)
    bar_frames = beat_frames[bar_offset::beats_per_bar]
    bar_beat_indices = np.arange(
        bar_offset + 1,
        bar_offset + 1 + len(bar_frames) * beats_per_bar,
        beats_per_bar,
    )
    bar_rows = _build_timed_rows(
        bar_frames,
        sr=sr,
        novelty=novelty,
        beat_indices=bar_beat_indices,
    )
    return tempo_bpm, beat_rows, bar_rows, bar_offset


def _rows_to_times(rows: list[dict], key: str = "time_sec") -> list[float]:
    return [float(row[key]) for row in rows]


def _safe_mean(values: np.ndarray, start: int, end: int) -> np.ndarray:
    if values.ndim == 1:
        values = values.reshape(1, -1)
    frame_count = values.shape[1]
    if frame_count == 0:
        return np.zeros(values.shape[0], dtype=np.float32)
    start = max(0, min(int(start), frame_count - 1))
    end = max(start + 1, min(int(end), frame_count))
    return np.mean(values[:, start:end], axis=1)


def _zscore_columns(values: np.ndarray) -> np.ndarray:
    mean = np.mean(values, axis=1, keepdims=True)
    std = np.std(values, axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (values - mean) / std


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float32)
    order = np.argsort(np.argsort(values))
    return order.astype(np.float32) / float(len(values) - 1)


def _estimate_structure_section_count(bar_count: int, duration_sec: float) -> int:
    if bar_count <= 2:
        return bar_count
    target = int(round(duration_sec / 10.0))
    target = max(4, target)
    target = min(target, 8, bar_count)
    return max(2, target)


def _build_bar_feature_payload(
    *,
    chroma: np.ndarray,
    spectral_contrast: np.ndarray,
    rms: np.ndarray,
    novelty: np.ndarray,
    bar_rows: list[dict],
    duration_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not bar_rows:
        return np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    start_frames = [int(row["frame"]) for row in bar_rows]
    end_frames = start_frames[1:] + [len(novelty)]

    feature_columns: list[np.ndarray] = []
    bar_energy: list[float] = []
    bar_novelty: list[float] = []

    for start, end in zip(start_frames, end_frames):
        chroma_mean = _safe_mean(chroma, start, end)
        contrast_mean = _safe_mean(spectral_contrast, start, end)
        rms_mean = float(_safe_mean(rms, start, end)[0])
        novelty_mean = float(_safe_mean(novelty, start, end)[0])
        feature_columns.append(np.concatenate([chroma_mean, contrast_mean, [rms_mean, novelty_mean]]))
        bar_energy.append(rms_mean)
        bar_novelty.append(novelty_mean)

    feature_matrix = np.stack(feature_columns, axis=1).astype(np.float32)
    normalized_matrix = _zscore_columns(feature_matrix)
    return normalized_matrix, np.asarray(bar_energy, dtype=np.float32), np.asarray(bar_novelty, dtype=np.float32)


def _build_section_ranges(boundary_indices: np.ndarray, total_bars: int) -> list[tuple[int, int]]:
    boundaries = sorted({0, total_bars, *[int(x) for x in np.asarray(boundary_indices).tolist()]})
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1) if boundaries[i] < boundaries[i + 1]]


def _section_centroid(feature_matrix: np.ndarray, start_bar: int, end_bar: int) -> np.ndarray:
    return np.mean(feature_matrix[:, start_bar:end_bar], axis=1)


def _merge_short_sections(
    ranges: list[tuple[int, int]],
    *,
    feature_matrix: np.ndarray,
    min_bars: int = 2,
) -> list[tuple[int, int]]:
    if len(ranges) <= 1:
        return ranges

    merged = list(ranges)
    while len(merged) > 1:
        short_index = next((i for i, (start, end) in enumerate(merged) if (end - start) < min_bars), None)
        if short_index is None:
            break

        start_bar, end_bar = merged[short_index]
        centroid = _section_centroid(feature_matrix, start_bar, end_bar)

        if short_index == 0:
            _, next_end = merged[1]
            merged[1] = (start_bar, next_end)
            del merged[0]
            continue
        if short_index == len(merged) - 1:
            prev_start, _ = merged[-2]
            merged[-2] = (prev_start, end_bar)
            del merged[-1]
            continue

        prev_range = merged[short_index - 1]
        next_range = merged[short_index + 1]
        prev_centroid = _section_centroid(feature_matrix, prev_range[0], prev_range[1])
        next_centroid = _section_centroid(feature_matrix, next_range[0], next_range[1])
        prev_distance = 1.0 - _cosine_similarity(centroid, prev_centroid)
        next_distance = 1.0 - _cosine_similarity(centroid, next_centroid)

        if prev_distance <= next_distance:
            merged[short_index - 1] = (prev_range[0], end_bar)
            del merged[short_index]
        else:
            merged[short_index + 1] = (start_bar, next_range[1])
            del merged[short_index]

    return merged


def _build_section_dicts(
    *,
    ranges: list[tuple[int, int]],
    bar_rows: list[dict],
    feature_matrix: np.ndarray,
    bar_energy: np.ndarray,
    bar_novelty: np.ndarray,
    duration_sec: float,
) -> list[dict]:
    sections: list[dict] = []
    for index, (start_bar, end_bar) in enumerate(ranges, start=1):
        start_time = float(bar_rows[start_bar]["time_sec"])
        end_time = float(bar_rows[end_bar]["time_sec"]) if end_bar < len(bar_rows) else float(duration_sec)
        centroid = _section_centroid(feature_matrix, start_bar, end_bar)
        sections.append(
            {
                "index": index,
                "start_bar": start_bar + 1,
                "end_bar": end_bar,
                "bar_count": end_bar - start_bar,
                "start_time_sec": _round_time(start_time),
                "end_time_sec": _round_time(end_time),
                "duration_sec": _round_time(end_time - start_time),
                "duration_bars": end_bar - start_bar,
                "mean_rms": round(float(np.mean(bar_energy[start_bar:end_bar])), 4),
                "mean_novelty": round(float(np.mean(bar_novelty[start_bar:end_bar])), 4),
                "_centroid": centroid,
            }
        )
    return sections


def _similarity_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    size = len(vectors)
    matrix = np.eye(size, dtype=np.float32)
    for i in range(size):
        for j in range(i + 1, size):
            value = _cosine_similarity(vectors[i], vectors[j])
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def _dynamic_repeat_threshold(similarity_matrix: np.ndarray) -> float:
    if similarity_matrix.size == 0 or similarity_matrix.shape[0] <= 1:
        return 1.0
    upper = similarity_matrix[np.triu_indices(similarity_matrix.shape[0], k=1)]
    if upper.size == 0:
        return 1.0
    threshold = float(np.quantile(upper, 0.85))
    return float(np.clip(max(0.68, threshold), 0.68, 0.92))


def _label_structure_sections(sections: list[dict], *, duration_sec: float) -> list[dict]:
    if not sections:
        return []

    centroids = [section["_centroid"] for section in sections]
    similarity_matrix = _similarity_matrix(centroids)
    repeat_threshold = _dynamic_repeat_threshold(similarity_matrix)
    energy = np.asarray([section["mean_rms"] for section in sections], dtype=np.float32)
    novelty = np.asarray([section["mean_novelty"] for section in sections], dtype=np.float32)
    duration_bars = np.asarray([section["duration_bars"] for section in sections], dtype=np.float32)
    positions = np.asarray(
        [
            ((float(section["start_time_sec"]) + float(section["end_time_sec"])) * 0.5) / max(duration_sec, 1e-6)
            for section in sections
        ],
        dtype=np.float32,
    )

    repeat_matches: list[list[int]] = []
    repeat_counts = np.zeros(len(sections), dtype=np.float32)
    for idx in range(len(sections)):
        matches = [
            other_index
            for other_index in range(len(sections))
            if other_index != idx and float(similarity_matrix[idx, other_index]) >= repeat_threshold
        ]
        repeat_matches.append(matches)
        repeat_counts[idx] = float(len(matches))

    energy_rank = _rank_normalize(energy)
    novelty_rank = _rank_normalize(novelty)
    position_bonus = np.clip(positions, 0.0, 1.0)
    intro_penalty = np.where(positions < 0.12, 0.75, 0.0)
    chorus_score = (repeat_counts * 1.8) + energy_rank + (novelty_rank * 0.7) + (position_bonus * 0.35) - intro_penalty

    chorus_seed = int(np.argmax(chorus_score))
    chorus_indices = {chorus_seed, *repeat_matches[chorus_seed]}
    chorus_energy = float(np.mean(energy[list(chorus_indices)]))
    chorus_bars = float(np.mean(duration_bars[list(chorus_indices)]))

    for idx in range(len(sections)):
        if idx in chorus_indices:
            continue
        similar_to_chorus = float(similarity_matrix[idx, chorus_seed]) >= max(0.55, repeat_threshold * 0.8)
        strong_late_section = (
            positions[idx] > 0.55
            and energy[idx] >= chorus_energy * 0.95
            and duration_bars[idx] >= max(2.0, chorus_bars * 0.75)
            and novelty_rank[idx] >= 0.4
        )
        if similar_to_chorus and strong_late_section:
            chorus_indices.add(idx)

    labels: list[str | None] = [None] * len(sections)
    for idx in chorus_indices:
        labels[idx] = "Chorus"

    if labels[0] != "Chorus" and float(sections[0]["end_time_sec"]) <= duration_sec * 0.2:
        labels[0] = "Intro"

    verse_candidates = [
        idx
        for idx in range(len(sections))
        if labels[idx] is None and repeat_counts[idx] > 0 and energy[idx] <= float(np.mean(energy[list(chorus_indices)])) * 1.15
    ]
    if verse_candidates:
        verse_seed = max(verse_candidates, key=lambda idx: (repeat_counts[idx], -energy_rank[idx], -positions[idx]))
        verse_indices = {verse_seed, *repeat_matches[verse_seed]} - chorus_indices
        for idx in verse_indices:
            if labels[idx] is None:
                labels[idx] = "Verse"

    for idx, label in enumerate(labels):
        if label != "Chorus":
            continue
        previous_idx = idx - 1
        if previous_idx <= 0 or labels[previous_idx] is not None:
            continue
        previous_energy = energy[previous_idx]
        prior_energy = energy[previous_idx - 1] if previous_idx - 1 >= 0 else previous_energy
        shorter_than_chorus = duration_bars[previous_idx] <= duration_bars[idx]
        rising_energy = previous_energy >= prior_energy * 0.98
        if shorter_than_chorus or rising_energy:
            labels[previous_idx] = "Pre-chorus"

    last_chorus_index = max(idx for idx, label in enumerate(labels) if label == "Chorus")
    bridge_candidates = [
        idx
        for idx in range(len(sections))
        if labels[idx] is None and positions[idx] > 0.45 and repeat_counts[idx] == 0 and idx < len(sections) - 1
    ]
    if bridge_candidates:
        bridge_index = max(
            bridge_candidates,
            key=lambda idx: (novelty_rank[idx], 1.0 - similarity_matrix[idx, chorus_seed], positions[idx]),
        )
        labels[bridge_index] = "Bridge"

    if (
        labels[-1] is None
        and positions[-1] > 0.85
        and len(labels) > 1
        and labels[-2] == "Chorus"
        and energy[-1] <= chorus_energy * 1.05
    ):
        labels[-1] = "Outro"

    for idx, label in enumerate(labels):
        if label is not None:
            continue
        labels[idx] = "Verse"

    serialized_sections: list[dict] = []
    for idx, section in enumerate(sections):
        payload = {
            key: value
            for key, value in section.items()
            if not key.startswith("_")
        }
        payload["label"] = labels[idx]
        payload["repeat_match_indices"] = [match + 1 for match in repeat_matches[idx]]
        payload["repeat_score"] = round(float(repeat_counts[idx]), 3)
        serialized_sections.append(payload)
    return serialized_sections


def _compute_structure_sections(
    *,
    chroma: np.ndarray,
    novelty: np.ndarray,
    spectral_contrast: np.ndarray,
    rms: np.ndarray,
    bar_rows: list[dict],
    duration_sec: float,
) -> list[dict]:
    if len(bar_rows) <= 1:
        if not bar_rows:
            return []
        return [
            {
                "index": 1,
                "label": "Intro",
                "start_bar": 1,
                "end_bar": len(bar_rows),
                "bar_count": len(bar_rows),
                "start_time_sec": 0.0,
                "end_time_sec": _round_time(duration_sec),
                "duration_sec": _round_time(duration_sec),
                "duration_bars": len(bar_rows),
                "mean_rms": round(float(np.mean(rms)) if rms.size else 0.0, 4),
                "mean_novelty": round(float(np.mean(novelty)) if novelty.size else 0.0, 4),
                "repeat_match_indices": [],
                "repeat_score": 0.0,
            }
        ]

    feature_matrix, bar_energy, bar_novelty = _build_bar_feature_payload(
        chroma=chroma,
        spectral_contrast=spectral_contrast,
        rms=rms,
        novelty=novelty,
        bar_rows=bar_rows,
        duration_sec=duration_sec,
    )
    target_section_count = _estimate_structure_section_count(len(bar_rows), duration_sec)
    boundary_indices = librosa.segment.agglomerative(feature_matrix, k=target_section_count)
    ranges = _build_section_ranges(boundary_indices, len(bar_rows))
    ranges = _merge_short_sections(ranges, feature_matrix=feature_matrix, min_bars=2)
    sections = _build_section_dicts(
        ranges=ranges,
        bar_rows=bar_rows,
        feature_matrix=feature_matrix,
        bar_energy=bar_energy,
        bar_novelty=bar_novelty,
        duration_sec=duration_sec,
    )
    return _label_structure_sections(sections, duration_sec=duration_sec)


def select_highlight_peaks(
    candidate_peaks: list[Peak],
    *,
    min_gap_sec: float = 1.25,
    peak_quantile: float = 0.9,
    target_count: int = 12,
) -> tuple[list[Peak], float]:
    if not candidate_peaks:
        return [], 0.0

    values = np.array([peak.novelty for peak in candidate_peaks], dtype=np.float32)
    threshold = float(np.quantile(values, peak_quantile))

    ordered = sorted(
        candidate_peaks,
        key=lambda peak: (peak.novelty, -peak.time_sec),
        reverse=True,
    )
    selected: list[Peak] = []

    def far_enough(candidate: Peak) -> bool:
        return all(abs(candidate.time_sec - peak.time_sec) >= min_gap_sec for peak in selected)

    for candidate in ordered:
        if candidate.novelty < threshold:
            continue
        if far_enough(candidate):
            selected.append(candidate)
        if len(selected) >= target_count:
            break

    if len(selected) < target_count:
        for candidate in ordered:
            if candidate in selected:
                continue
            if far_enough(candidate):
                selected.append(candidate)
            if len(selected) >= target_count:
                break

    selected.sort(key=lambda peak: peak.time_sec)
    return selected, threshold


def compute_audio_analysis(
    wav_path: Path,
    *,
    segments: int = 8,
    peak_quantile: float = 0.9,
    peak_min_gap_sec: float = 1.25,
    highlight_target_count: int = 12,
    beats_per_bar: int = 4,
) -> AudioAnalysisResult:
    y, sr = librosa.load(wav_path, sr=None)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    similarity = librosa.segment.recurrence_matrix(chroma, mode="affinity")
    novelty = librosa.onset.onset_strength(y=y, sr=sr)
    novelty_times = librosa.times_like(novelty, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    duration_sec = float(len(y) / sr)

    segment_count = max(2, min(int(segments), chroma.shape[1]))
    boundary_frames = librosa.segment.agglomerative(chroma, k=segment_count)
    boundary_times = librosa.frames_to_time(boundary_frames, sr=sr)

    candidate_frames = librosa.util.peak_pick(
        novelty,
        pre_max=16,
        post_max=16,
        pre_avg=16,
        post_avg=16,
        delta=max(float(novelty.max()) * 0.15, 0.1),
        wait=16,
    )
    candidate_times = librosa.frames_to_time(candidate_frames, sr=sr)
    candidate_peaks = [
        Peak(
            frame=int(frame),
            time_sec=float(time_sec),
            novelty=float(novelty[frame]),
        )
        for frame, time_sec in zip(candidate_frames, candidate_times)
    ]

    highlight_peaks, threshold = select_highlight_peaks(
        candidate_peaks,
        min_gap_sec=peak_min_gap_sec,
        peak_quantile=peak_quantile,
        target_count=highlight_target_count,
    )
    tempo_bpm, beat_rows, bar_rows, bar_offset = _compute_meter_rows(
        novelty=novelty,
        sr=sr,
        beats_per_bar=max(1, int(beats_per_bar)),
    )
    structure_sections = _compute_structure_sections(
        chroma=chroma,
        novelty=novelty,
        spectral_contrast=spectral_contrast,
        rms=rms,
        bar_rows=bar_rows,
        duration_sec=duration_sec,
    )

    summary = {
        "wav_path": str(wav_path),
        "sample_rate": int(sr),
        "duration_sec": duration_sec,
        "novelty_frame_count": int(len(novelty)),
        "segments_requested": int(segments),
        "estimated_segment_boundaries_sec": [_round_time(x) for x in boundary_times.tolist()],
        "candidate_peak_count": len(candidate_peaks),
        "candidate_peaks": [peak_to_dict(peak) for peak in candidate_peaks],
        "highlight_peak_params": {
            "min_gap_sec": peak_min_gap_sec,
            "peak_quantile": peak_quantile,
            "target_count": highlight_target_count,
            "threshold": round(threshold, 4),
        },
        "highlight_peak_count": len(highlight_peaks),
        "highlight_peaks": [peak_to_dict(peak) for peak in highlight_peaks],
        "tempo_bpm": round(tempo_bpm, 3),
        "meter_assumption": {
            "beats_per_bar": max(1, int(beats_per_bar)),
            "downbeat_offset": int(bar_offset),
            "description": "Approximate bars built by grouping detected beats into fixed-size measures.",
        },
        "beat_count": len(beat_rows),
        "beats": beat_rows,
        "beat_times_sec": _rows_to_times(beat_rows),
        "bar_count": len(bar_rows),
        "bars": bar_rows,
        "bar_times_sec": _rows_to_times(bar_rows),
        "structure_detection": {
            "description": "Approximate song-form inference using bar-synchronous chroma, spectral contrast, RMS, novelty, repetition, and position heuristics.",
            "labels": ["Intro", "Verse", "Pre-chorus", "Chorus", "Bridge", "Outro"],
        },
        "structure_section_count": len(structure_sections),
        "structure_sections": structure_sections,
        "structure_change_times_sec": _rows_to_times(structure_sections, key="start_time_sec")[1:],
    }

    return AudioAnalysisResult(
        summary=summary,
        chroma=chroma,
        similarity=similarity,
        novelty=novelty,
        novelty_times=novelty_times,
    )
