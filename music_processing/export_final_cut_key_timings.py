from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CUT_MANIFEST = (
    Path(__file__).resolve().parent
    / "results"
    / "cut_inference_viewer_dcm_body_demucs_events_no_rms_e30_density_all_keyframes_no_c47_c105"
    / "manifest.json"
)
DEFAULT_KEYFRAME_MANIFEST = Path(__file__).resolve().parent / "results" / "keyframe_inference_viewer" / "manifest.json"
DEFAULT_FEATURE_MANIFEST = Path(__file__).resolve().parent / "results" / "cut_keyframe_multimodal_dataset" / "manifest.json"
DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parent
    / "results"
    / "final_cut_key_timings_dcm_density40_global_no_c47_c105"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export final DCM cut+keyframe timings from cut probability and keyframe probability viewers."
    )
    parser.add_argument("--cut-manifest", type=Path, default=DEFAULT_CUT_MANIFEST)
    parser.add_argument("--keyframe-manifest", type=Path, default=DEFAULT_KEYFRAME_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--density-cut-percent", type=float, default=40.0)
    parser.add_argument(
        "--density-cut-percent-unit",
        choices=["cut_events", "cut_keyframes"],
        default="cut_events",
        help=(
            "cut_events keeps the legacy behavior where the percent directly sets the cut event budget. "
            "cut_keyframes interprets the percent as the share of total keyframe budget reserved for cut pairs, "
            "so selected cut event budget is floor(total_budget * percent / 2)."
        ),
    )
    parser.add_argument(
        "--cut-selection-mode",
        choices=["global", "adaptive", "uniform"],
        default="global",
        help=(
            "global picks top cuts across the whole song; adaptive distributes the total budget by per-window score mass; "
            "uniform spreads cuts over the covered time axis."
        ),
    )
    parser.add_argument("--cut-window-sec", type=float, default=10.0)
    parser.add_argument("--cut-min-gap-sec", type=float, default=1.0)
    parser.add_argument("--cut-min-score", type=float, default=0.15)
    parser.add_argument("--cut-prominence-weight", type=float, default=0.3)
    parser.add_argument(
        "--keyframe-selection-mode",
        choices=["probability", "heuristic", "random"],
        default="probability",
        help="probability uses the stage-2 keyframe score; heuristic uses dance/music/sync saliency; random samples eligible frames.",
    )
    parser.add_argument("--feature-manifest", type=Path, default=DEFAULT_FEATURE_MANIFEST)
    parser.add_argument(
        "--keyframe-threshold",
        type=float,
        default=None,
        help="Minimum keyframe probability. Defaults to no score floor when selecting the remaining density budget.",
    )
    parser.add_argument("--keyframe-min-gap-frames", type=int, default=None)
    parser.add_argument("--keyframe-prominence-weight", type=float, default=0.3)
    parser.add_argument(
        "--keyframe-placement-mode",
        choices=["score", "uniform", "density_uniform"],
        default="score",
        help=(
            "score greedily follows high scores; uniform assigns slots over eligible time; "
            "density_uniform allocates per-bin quotas from Stage-A density mixed with uniform duration."
        ),
    )
    parser.add_argument(
        "--noncut-budget-mode",
        choices=["split_remaining", "subtract_cut_pairs"],
        default="split_remaining",
        help=(
            "split_remaining uses the non-cut keyframe budget computed from density_cut_percent. "
            "subtract_cut_pairs recomputes non-cut keyframe budget after selected cuts as "
            "total_event_budget - number of reserved cut-pair keyframes."
        ),
    )
    parser.add_argument("--density-allocation-bin-sec", type=float, default=3.0)
    parser.add_argument("--density-uniform-alpha", type=float, default=0.6)
    parser.add_argument("--dance-motion-weight", type=float, default=0.55)
    parser.add_argument("--music-accent-weight", type=float, default=0.20)
    parser.add_argument("--music-dance-sync-weight", type=float, default=0.15)
    parser.add_argument("--beat-bonus-weight", type=float, default=0.10)
    parser.add_argument("--stage2-probability-weight", type=float, default=0.08)
    parser.add_argument("--sync-window-frames", type=int, default=3)
    parser.add_argument("--cut-near-penalty-weight", type=float, default=0.12)
    parser.add_argument("--cut-near-penalty-radius-frames", type=int, default=18)
    parser.add_argument("--overcrowding-penalty-weight", type=float, default=0.18)
    parser.add_argument("--overcrowding-penalty-radius-frames", type=int, default=24)
    parser.add_argument("--uniform-slot-distance-weight", type=float, default=0.6)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--keyframe-cut-exclusion-sec",
        type=float,
        default=None,
        help="When set, exclude keyframe candidates within +/- this many seconds of each selected cut frame.",
    )
    parser.add_argument(
        "--keyframe-cut-exclusion-radius-frames",
        type=int,
        default=1,
        help=(
            "Exclude keyframe candidates within this many frames of selected cut-pair frames. "
            "With the default 1, cut pair (f-1, f) also blocks f-2 and f+1."
        ),
    )
    parser.add_argument("--exclude-song-ids", default="")
    return parser.parse_args()


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_repo_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        candidate = path
    else:
        candidate = (WORKSPACE_ROOT / str(path_text).lstrip("/")).resolve()
    if candidate.exists():
        return candidate

    text = str(candidate)
    replacements = [
        (
            str(WORKSPACE_ROOT / "music_processing" / "results" / "keyframe_inference_viewer"),
            str(WORKSPACE_ROOT / "music_processing" / "results" / "03_viewer_data" / "keyframe_inference_viewer"),
        ),
        (
            str(WORKSPACE_ROOT / "music_processing" / "results" / "cut_inference_viewer"),
            str(WORKSPACE_ROOT / "music_processing" / "results" / "03_viewer_data" / "cut_inference_viewer"),
        ),
    ]
    for old, new in replacements:
        if old in text:
            repaired = Path(text.replace(old, new))
            if repaired.exists():
                return repaired
    return candidate


def repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def parse_id_set(text: str) -> set[str]:
    return {item.strip() for item in str(text).split(",") if item.strip()}


def robust_normalize(values: np.ndarray, *, low_percentile: float = 5.0, high_percentile: float = 95.0) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return array
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    low = float(np.percentile(finite, low_percentile))
    high = float(np.percentile(finite, high_percentile))
    if high <= low + 1e-6:
        return np.zeros_like(array, dtype=np.float32)
    normalized = (array - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def positive_delta(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return array
    delta = np.diff(array, prepend=array[:1])
    return np.maximum(delta, 0.0).astype(np.float32)


def local_max_filter(values: np.ndarray, *, radius: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return array
    radius = max(0, int(radius))
    if radius == 0:
        return array.copy()
    output = np.zeros_like(array, dtype=np.float32)
    for index in range(len(array)):
        start = max(0, index - radius)
        end = min(len(array), index + radius + 1)
        output[index] = float(np.max(array[start:end]))
    return output


def round_time(value: float) -> float:
    return round(float(value), 3)


def frame_5fps(frame: int, fps: float) -> int:
    return int(round(float(frame) * 5.0 / max(float(fps), 1e-6)))


def probability_series(song_data: dict) -> list[dict]:
    return song_data.get("probability_series") or song_data.get("probability_samples") or []


def load_feature_index(path: Path) -> tuple[dict[str, Path], list[str], list[str]]:
    manifest = load_json(path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Feature manifest is not a JSON object: {path}")
    feature_paths: dict[str, Path] = {}
    for song in manifest.get("songs", []):
        song_id = str(song.get("id") or song.get("source_song_id") or "")
        feature_path = song.get("feature_path")
        if song_id and feature_path:
            feature_paths[song_id] = resolve_repo_path(str(feature_path))
    return (
        feature_paths,
        [str(name) for name in manifest.get("audio_feature_names", [])],
        [str(name) for name in manifest.get("dance_feature_names", [])],
    )


def feature_column(names: list[str], name: str) -> int | None:
    try:
        return names.index(name)
    except ValueError:
        return None


def feature_values(features: np.ndarray, names: list[str], name: str) -> np.ndarray:
    index = feature_column(names, name)
    if index is None or index >= features.shape[1]:
        return np.zeros(features.shape[0], dtype=np.float32)
    return np.asarray(features[:, index], dtype=np.float32)


def build_heuristic_components(
    feature_path: Path,
    *,
    audio_feature_names: list[str],
    dance_feature_names: list[str],
    sync_window_frames: int,
) -> dict[str, np.ndarray]:
    payload = np.load(feature_path, allow_pickle=True)
    audio = np.asarray(payload["audio_features"], dtype=np.float32)
    dance = np.asarray(payload["dance_features"], dtype=np.float32)
    frame_count = min(len(audio), len(dance))
    audio = audio[:frame_count]
    dance = dance[:frame_count]

    motion_energy = feature_values(dance, dance_feature_names, "dance_motion_energy")
    speed_indices = [
        index for index, name in enumerate(dance_feature_names)
        if name.startswith("dance_joint_speed_") and index < dance.shape[1]
    ]
    if speed_indices:
        joint_speed = np.mean(np.abs(dance[:, speed_indices]), axis=1)
    else:
        joint_speed = np.linalg.norm(np.diff(dance, axis=0, prepend=dance[:1]), axis=1)
    root_velocity = np.sqrt(
        feature_values(dance, dance_feature_names, "dance_root_vel_x") ** 2
        + feature_values(dance, dance_feature_names, "dance_root_vel_y") ** 2
        + feature_values(dance, dance_feature_names, "dance_root_vel_z") ** 2
    )
    dance_raw = (
        0.50 * robust_normalize(motion_energy)
        + 0.35 * robust_normalize(joint_speed)
        + 0.15 * robust_normalize(root_velocity)
        + 0.15 * robust_normalize(positive_delta(motion_energy))
    )
    dance_motion = robust_normalize(dance_raw)

    drums = (
        0.35 * robust_normalize(feature_values(audio, audio_feature_names, "drums_peak"))
        + 0.25 * robust_normalize(feature_values(audio, audio_feature_names, "drums_rms"))
        + 0.20 * robust_normalize(positive_delta(feature_values(audio, audio_feature_names, "drums_rms")))
        + 0.10 * robust_normalize(feature_values(audio, audio_feature_names, "other_peak"))
        + 0.10 * robust_normalize(feature_values(audio, audio_feature_names, "bass_peak"))
    )
    music_accent = robust_normalize(drums)
    music_nearby = local_max_filter(music_accent, radius=int(sync_window_frames))
    sync_bonus = np.sqrt(np.clip(dance_motion * music_nearby, 0.0, 1.0)).astype(np.float32)

    beat_bonus = np.clip(
        0.60 * feature_values(audio, audio_feature_names, "beat_pulse")
        + 0.95 * feature_values(audio, audio_feature_names, "downbeat_pulse")
        + 0.80 * feature_values(audio, audio_feature_names, "bar_pulse"),
        0.0,
        1.0,
    ).astype(np.float32)
    return {
        "dance_motion_saliency": dance_motion,
        "music_accent_saliency": music_accent,
        "music_dance_sync_bonus": sync_bonus,
        "beat_or_downbeat_bonus": beat_bonus,
    }


def cut_candidate_peaks(
    series: list[dict],
    *,
    min_score: float,
    prominence_weight: float,
) -> list[dict]:
    if len(series) < 3:
        return []

    candidates: list[dict] = []
    for index in range(1, len(series) - 1):
        previous = float(series[index - 1].get("value", 0.0))
        current = float(series[index].get("value", 0.0))
        following = float(series[index + 1].get("value", 0.0))
        if current < float(min_score) or current < previous or current <= following:
            continue

        left = max(0, index - 6)
        right = min(len(series) - 1, index + 6)
        shoulders = [
            float(series[item].get("value", 0.0))
            for item in range(left, right + 1)
            if item != index
        ]
        shoulder = sum(shoulders) / len(shoulders) if shoulders else 0.0
        prominence = max(0.0, current - shoulder)
        candidates.append(
            {
                "frame": int(series[index]["frame"]),
                "time_sec": round_time(float(series[index]["time_sec"])),
                "probability": round(float(current), 6),
                "prominence": round(float(prominence), 6),
                "score": round(float(current + prominence_weight * prominence), 6),
            }
        )
    return candidates


def select_global_cuts(candidates: list[dict], *, total_cuts: int, min_gap_sec: float) -> list[dict]:
    selected: list[dict] = []
    select_top_cut_candidates(candidates, total_cuts, min_gap_sec=min_gap_sec, selected=selected)
    return sorted(selected, key=lambda item: float(item["time_sec"]))


def select_top_cut_candidates(
    candidates: list[dict],
    count: int,
    *,
    min_gap_sec: float,
    selected: list[dict],
) -> list[dict]:
    if count <= 0:
        return selected
    start_count = len(selected)
    ordered = sorted(candidates, key=lambda item: (-float(item["score"]), float(item["time_sec"])))
    for candidate in ordered:
        if all(abs(float(candidate["time_sec"]) - float(item["time_sec"])) >= float(min_gap_sec) for item in selected):
            selected.append(candidate)
        if len(selected) - start_count >= int(count):
            break
    return selected


def select_adaptive_cuts(
    candidates: list[dict],
    *,
    duration_sec: float,
    total_cuts: int,
    window_sec: float,
    min_gap_sec: float,
) -> list[dict]:
    if not candidates or total_cuts <= 0:
        return []

    buckets: list[dict] = []
    start = 0.0
    duration = max(float(duration_sec), 0.001)
    window = max(float(window_sec), 0.001)
    while start < duration:
        end = start + window
        bucket_candidates = [
            candidate for candidate in candidates if start <= float(candidate["time_sec"]) < end
        ]
        weight = sum(float(candidate["score"]) for candidate in bucket_candidates)
        buckets.append(
            {
                "start": start,
                "end": end,
                "candidates": bucket_candidates,
                "weight": weight,
                "quota": 0,
                "remainder": 0.0,
            }
        )
        start = end

    total_weight = sum(float(bucket["weight"]) for bucket in buckets)
    if total_weight <= 0:
        return select_global_cuts(candidates, total_cuts=total_cuts, min_gap_sec=min_gap_sec)

    for bucket in buckets:
        raw_quota = int(total_cuts) * float(bucket["weight"]) / total_weight
        bucket["quota"] = min(len(bucket["candidates"]), int(raw_quota // 1))
        bucket["remainder"] = raw_quota - int(raw_quota // 1)

    assigned = sum(int(bucket["quota"]) for bucket in buckets)
    while assigned < int(total_cuts):
        eligible = [bucket for bucket in buckets if int(bucket["quota"]) < len(bucket["candidates"])]
        if not eligible:
            break
        bucket = sorted(eligible, key=lambda item: (-float(item["remainder"]), -float(item["weight"])))[0]
        bucket["quota"] = int(bucket["quota"]) + 1
        bucket["remainder"] = 0.0
        assigned += 1

    selected: list[dict] = []
    for bucket in buckets:
        select_top_cut_candidates(
            bucket["candidates"],
            int(bucket["quota"]),
            min_gap_sec=min_gap_sec,
            selected=selected,
        )
    return sorted(selected, key=lambda item: float(item["time_sec"]))


def coverage_intervals(song_data: dict) -> list[tuple[float, float]]:
    duration = max(0.001, float(song_data.get("video_duration_sec", 0.0) or 0.0))
    segments = song_data.get("coverage_segments") or []
    intervals: list[tuple[float, float]] = []
    for segment in segments:
        start = max(0.0, float(segment.get("start_time_sec", 0.0) or 0.0))
        end = min(duration, float(segment.get("end_time_sec", segment.get("start_time_sec", 0.0)) or 0.0))
        if end > start:
            intervals.append((start, end))
    return intervals or [(0.0, duration)]


def candidate_uniform_offset(candidate: dict, intervals: list[tuple[float, float]]) -> float | None:
    time_sec = float(candidate["time_sec"])
    elapsed = 0.0
    for start, end in intervals:
        if start <= time_sec <= end:
            return elapsed + (time_sec - start)
        elapsed += max(0.0, end - start)
    return None


def cut_far_enough(candidate: dict, selected: list[dict], min_gap_sec: float) -> bool:
    return all(abs(float(candidate["time_sec"]) - float(item["time_sec"])) >= float(min_gap_sec) for item in selected)


def clean_candidate(candidate: dict) -> dict:
    return {key: value for key, value in candidate.items() if not str(key).startswith("_")}


def select_uniform_cuts(
    candidates: list[dict],
    *,
    intervals: list[tuple[float, float]],
    total_cuts: int,
    min_gap_sec: float,
) -> list[dict]:
    if not candidates or total_cuts <= 0:
        return []

    annotated: list[dict] = []
    for candidate in candidates:
        offset = candidate_uniform_offset(candidate, intervals)
        if offset is None:
            continue
        item = dict(candidate)
        item["_uniform_offset"] = float(offset)
        annotated.append(item)
    if not annotated:
        return select_global_cuts(candidates, total_cuts=total_cuts, min_gap_sec=min_gap_sec)

    total_length = sum(max(0.0, end - start) for start, end in intervals)
    slot_count = min(int(total_cuts), len(annotated))
    if total_length <= 0.0 or slot_count <= 0:
        return []

    slot_width = total_length / max(1, slot_count)
    selected: list[dict] = []
    selected_keys: set[tuple[int, float]] = set()
    missed_centers: list[float] = []

    def key_for(item: dict) -> tuple[int, float]:
        return int(item["frame"]), float(item["time_sec"])

    for slot_index in range(slot_count):
        slot_start = slot_index * slot_width
        slot_end = total_length if slot_index == slot_count - 1 else (slot_index + 1) * slot_width
        slot_center = (slot_start + slot_end) * 0.5
        bucket = [
            item
            for item in annotated
            if key_for(item) not in selected_keys
            and slot_start <= float(item["_uniform_offset"]) <= slot_end
        ]
        bucket.sort(
            key=lambda item: (
                -float(item["score"]),
                abs(float(item["_uniform_offset"]) - slot_center),
                float(item["time_sec"]),
            )
        )
        chosen = next((item for item in bucket if cut_far_enough(item, selected, min_gap_sec)), None)
        if chosen is None:
            missed_centers.append(slot_center)
            continue
        selected.append(chosen)
        selected_keys.add(key_for(chosen))

    for slot_center in missed_centers:
        if len(selected) >= int(total_cuts):
            break
        remaining = [item for item in annotated if key_for(item) not in selected_keys]
        remaining.sort(
            key=lambda item: (
                abs(float(item["_uniform_offset"]) - slot_center) / max(slot_width, 1e-6),
                -float(item["score"]),
                float(item["time_sec"]),
            )
        )
        chosen = next((item for item in remaining if cut_far_enough(item, selected, min_gap_sec)), None)
        if chosen is None:
            continue
        selected.append(chosen)
        selected_keys.add(key_for(chosen))

    if len(selected) < int(total_cuts):
        remaining = [item for item in annotated if key_for(item) not in selected_keys]
        remaining.sort(key=lambda item: (-float(item["score"]), float(item["time_sec"])))
        for item in remaining:
            if len(selected) >= int(total_cuts):
                break
            if not cut_far_enough(item, selected, min_gap_sec):
                continue
            selected.append(item)
            selected_keys.add(key_for(item))

    return sorted((clean_candidate(item) for item in selected), key=lambda item: float(item["time_sec"]))


def density_event_budgets(
    song_data: dict,
    *,
    density_cut_percent: float,
    density_cut_percent_unit: str = "cut_events",
) -> tuple[int, int, int, float]:
    density = song_data.get("density_budget") or {}
    ratio = max(0.0, float(density_cut_percent)) / 100.0

    def split_budget(total_budget: int) -> tuple[int, int, int]:
        total_budget = max(0, int(total_budget))
        bounded_ratio = min(ratio, 1.0)
        if density_cut_percent_unit == "cut_keyframes":
            cut_keyframe_budget = max(0, int(round(total_budget * bounded_ratio)))
            cut_event_budget = cut_keyframe_budget // 2
            noncut_keyframe_budget = max(0, total_budget - (2 * cut_event_budget))
            return cut_event_budget, noncut_keyframe_budget, total_budget

        cut_event_budget = max(0, int(round(total_budget * bounded_ratio)))
        noncut_keyframe_budget = max(0, total_budget - cut_event_budget)
        return cut_event_budget, noncut_keyframe_budget, total_budget

    pred_keyframe_mass = float(density.get("pred_keyframe_mass", 0.0) or 0.0)
    if pred_keyframe_mass > 0:
        total_budget = max(0, int(round(pred_keyframe_mass)))
        cut_budget, keyframe_budget, total_budget = split_budget(total_budget)
        return cut_budget, keyframe_budget, total_budget, pred_keyframe_mass

    fixed_budget = float(density.get("total_cut_budget", 0.0) or 0.0)
    original_ratio = float(density.get("budget_ratio", 0.0) or 0.0)
    if fixed_budget > 0 and original_ratio > 0:
        total_budget = max(0, int(round(fixed_budget / original_ratio)))
        cut_budget, keyframe_budget, total_budget = split_budget(total_budget)
        return cut_budget, keyframe_budget, total_budget, 0.0
    cut_budget = max(0, int(round(fixed_budget)))
    return cut_budget, 0, cut_budget, 0.0


def cut_pair_frames(cut_frame: int) -> list[int]:
    frame = int(cut_frame)
    if frame <= 0:
        return [0, 1]
    return [frame - 1, frame]


def expand_reserved_frames(frames: set[int], *, radius: int) -> set[int]:
    radius = max(0, int(radius))
    expanded: set[int] = set()
    for frame in frames:
        for offset in range(-radius, radius + 1):
            expanded.add(int(frame) + offset)
    return {frame for frame in expanded if frame >= 0}


def frames_near_events(events: list[dict], *, radius: int) -> set[int]:
    radius = max(0, int(radius))
    blocked: set[int] = set()
    for event in events:
        frame = int(event["frame"])
        for offset in range(-radius, radius + 1):
            candidate = frame + offset
            if candidate >= 0:
                blocked.add(candidate)
    return blocked


def keyframe_candidate_peaks(
    series: list[dict],
    *,
    min_score: float,
    min_gap_frames: int,
    prominence_weight: float,
    reserved_frames: set[int],
) -> list[dict]:
    if len(series) < 3:
        return []
    by_frame = {int(item["frame"]): item for item in series}
    frames = sorted(by_frame)
    candidates: list[dict] = []
    for index in range(1, len(frames) - 1):
        frame = frames[index]
        if frame in reserved_frames:
            continue
        previous = float(by_frame[frames[index - 1]].get("value", 0.0))
        current = float(by_frame[frame].get("value", 0.0))
        following = float(by_frame[frames[index + 1]].get("value", 0.0))
        if current < float(min_score) or current < previous or current <= following:
            continue
        left = max(0, index - 6)
        right = min(len(frames) - 1, index + 6)
        shoulders = [
            float(by_frame[frames[item]].get("value", 0.0))
            for item in range(left, right + 1)
            if item != index
        ]
        shoulder = sum(shoulders) / len(shoulders) if shoulders else 0.0
        prominence = max(0.0, current - shoulder)
        candidates.append(
            {
                "frame": int(frame),
                "probability": round(float(current), 6),
                "prominence": round(float(prominence), 6),
                "score": round(float(current + prominence_weight * prominence), 6),
            }
        )

    selected: list[dict] = []
    ordered = sorted(candidates, key=lambda item: (-float(item["score"]), int(item["frame"])))
    for candidate in ordered:
        frame = int(candidate["frame"])
        if all(abs(frame - int(event["frame"])) >= int(min_gap_frames) for event in selected):
            selected.append(candidate)
    return sorted(selected, key=lambda item: int(item["frame"]))


def select_top_keyframes(candidates: list[dict], *, total_keyframes: int) -> list[dict]:
    if total_keyframes <= 0:
        return []
    selected = sorted(
        candidates,
        key=lambda item: (-float(item.get("score", item.get("probability", 0.0))), int(item["frame"])),
    )[: int(total_keyframes)]
    return sorted([
        {
            "frame": int(item["frame"]),
            "time_sec": round_time(int(item["frame"]) / 30.0),
            "probability": round(float(item["probability"]), 6),
            "prominence": round(float(item.get("prominence", 0.0)), 6),
            "score": round(float(item.get("score", item["probability"])), 6),
        }
        for item in selected
    ], key=lambda item: int(item["frame"]))


def distance_to_frames(frame: int, frames: set[int]) -> int | None:
    if not frames:
        return None
    return min(abs(int(frame) - int(item)) for item in frames)


def heuristic_keyframe_candidates(
    series: list[dict],
    *,
    components: dict[str, np.ndarray],
    reserved_frames: set[int],
    cut_pair_reserved_frames: set[int],
    weights: dict[str, float],
    cut_penalty_radius_frames: int,
) -> tuple[list[dict], list[dict]]:
    if not series:
        return [], []
    probability_by_frame = {int(item["frame"]): float(item.get("value", 0.0)) for item in series}
    max_length = min((len(values) for values in components.values()), default=0)
    candidates: list[dict] = []
    score_series: list[dict] = []
    for item in series:
        frame = int(item["frame"])
        if frame < 0 or frame >= max_length:
            continue
        dance = float(components["dance_motion_saliency"][frame])
        music = float(components["music_accent_saliency"][frame])
        sync = float(components["music_dance_sync_bonus"][frame])
        beat = float(components["beat_or_downbeat_bonus"][frame])
        stage2 = float(probability_by_frame.get(frame, 0.0))
        cut_distance = distance_to_frames(frame, cut_pair_reserved_frames)
        if cut_distance is None or cut_penalty_radius_frames <= 0:
            cut_penalty = 0.0
        else:
            cut_penalty = max(0.0, 1.0 - float(cut_distance) / float(cut_penalty_radius_frames))

        base_score = (
            float(weights["dance_motion"]) * dance
            + float(weights["music_accent"]) * music
            + float(weights["music_dance_sync"]) * sync
            + float(weights["beat_bonus"]) * beat
            + float(weights["stage2_probability"]) * stage2
            - float(weights["cut_near_penalty"]) * cut_penalty
        )
        score = max(0.0, min(1.0, base_score))
        sample = {
            "frame": frame,
            "time_sec": round_time(frame / 30.0),
            "value": round(float(score), 6),
            "dance_motion_saliency": round(dance, 6),
            "music_accent_saliency": round(music, 6),
            "music_dance_sync_bonus": round(sync, 6),
            "beat_or_downbeat_bonus": round(beat, 6),
            "stage2_probability": round(stage2, 6),
            "cut_near_penalty": round(cut_penalty, 6),
        }
        score_series.append(sample)
        if frame in reserved_frames:
            continue
        candidates.append(
            {
                **sample,
                "probability": round(float(score), 6),
                "score": round(float(score), 6),
                "base_score": round(float(base_score), 6),
            }
        )
    return candidates, score_series


def select_heuristic_keyframes(
    candidates: list[dict],
    *,
    total_keyframes: int,
    min_gap_frames: int,
    overcrowding_penalty_weight: float,
    overcrowding_penalty_radius_frames: int,
) -> list[dict]:
    if total_keyframes <= 0 or not candidates:
        return []
    selected: list[dict] = []
    used_frames: set[int] = set()
    radius = max(1, int(overcrowding_penalty_radius_frames))

    while len(selected) < int(total_keyframes):
        best: dict | None = None
        best_adjusted = float("-inf")
        for candidate in candidates:
            frame = int(candidate["frame"])
            if frame in used_frames:
                continue
            if any(abs(frame - int(event["frame"])) < int(min_gap_frames) for event in selected):
                continue
            overcrowding = 0.0
            for event in selected:
                distance = abs(frame - int(event["frame"]))
                if distance < radius:
                    overcrowding += 1.0 - float(distance) / float(radius)
            adjusted = float(candidate["score"]) - float(overcrowding_penalty_weight) * overcrowding
            if adjusted > best_adjusted or (
                adjusted == best_adjusted
                and best is not None
                and int(candidate["frame"]) < int(best["frame"])
            ):
                best = candidate
                best_adjusted = adjusted
        if best is None:
            break
        used_frames.add(int(best["frame"]))
        selected.append(
            {
                **best,
                "adjusted_score": round(float(best_adjusted), 6),
                "overcrowding_penalty": round(float(float(best["score"]) - best_adjusted), 6),
            }
        )
    return sorted(selected, key=lambda item: int(item["frame"]))


def _keyframe_overcrowding(candidate_frame: int, selected: list[dict], radius: int) -> float:
    radius = max(1, int(radius))
    overcrowding = 0.0
    for event in selected:
        distance = abs(int(candidate_frame) - int(event["frame"]))
        if distance < radius:
            overcrowding += 1.0 - float(distance) / float(radius)
    return float(overcrowding)


def select_uniform_heuristic_keyframes(
    candidates: list[dict],
    *,
    total_keyframes: int,
    min_gap_frames: int,
    overcrowding_penalty_weight: float,
    overcrowding_penalty_radius_frames: int,
    slot_distance_weight: float,
) -> list[dict]:
    if total_keyframes <= 0 or not candidates:
        return []

    ordered_by_frame = sorted(candidates, key=lambda item: int(item["frame"]))
    target_count = min(int(total_keyframes), len(ordered_by_frame))
    if target_count <= 0:
        return []

    frames = np.asarray([int(item["frame"]) for item in ordered_by_frame], dtype=np.float32)
    slot_indices = np.linspace(0, len(ordered_by_frame) - 1, target_count + 2, dtype=np.float32)[1:-1]
    slot_centers = [float(frames[int(round(index))]) for index in slot_indices]
    ideal_gap = max(float((frames[-1] - frames[0]) / max(target_count, 1)), float(min_gap_frames), 1.0)

    selected: list[dict] = []
    used_frames: set[int] = set()
    candidate_pool = list(ordered_by_frame)

    for slot_index, center in enumerate(slot_centers, start=1):
        best: dict | None = None
        best_adjusted = float("-inf")
        for candidate in candidate_pool:
            frame = int(candidate["frame"])
            if frame in used_frames:
                continue
            if any(abs(frame - int(event["frame"])) < int(min_gap_frames) for event in selected):
                continue
            distance_norm = abs(float(frame) - center) / ideal_gap
            overcrowding = _keyframe_overcrowding(frame, selected, int(overcrowding_penalty_radius_frames))
            adjusted = (
                float(candidate["score"])
                - float(slot_distance_weight) * float(distance_norm)
                - float(overcrowding_penalty_weight) * overcrowding
            )
            if adjusted > best_adjusted or (
                adjusted == best_adjusted
                and best is not None
                and abs(frame - center) < abs(int(best["frame"]) - center)
            ):
                best = candidate
                best_adjusted = adjusted
        if best is None:
            continue
        frame = int(best["frame"])
        used_frames.add(frame)
        selected.append(
            {
                **best,
                "adjusted_score": round(float(best_adjusted), 6),
                "uniform_slot_index": int(slot_index),
                "uniform_slot_center_frame": int(round(center)),
                "uniform_slot_distance_frames": int(round(abs(frame - center))),
                "overcrowding_penalty": round(float(float(best["score"]) - best_adjusted), 6),
            }
        )

    if len(selected) < int(total_keyframes):
        fallback = select_heuristic_keyframes(
            [candidate for candidate in candidate_pool if int(candidate["frame"]) not in used_frames],
            total_keyframes=int(total_keyframes) - len(selected),
            min_gap_frames=int(min_gap_frames),
            overcrowding_penalty_weight=float(overcrowding_penalty_weight),
            overcrowding_penalty_radius_frames=int(overcrowding_penalty_radius_frames),
        )
        for candidate in fallback:
            frame = int(candidate["frame"])
            if any(abs(frame - int(event["frame"])) < int(min_gap_frames) for event in selected):
                continue
            selected.append(candidate)
            if len(selected) >= int(total_keyframes):
                break

    return sorted(selected, key=lambda item: int(item["frame"]))


def stagea_density_by_frame(song_data: dict) -> dict[int, float]:
    density = song_data.get("density_budget") or {}
    output: dict[int, float] = {}
    for segment in density.get("segments", []) or []:
        values = segment.get("pred_frame_density")
        if not values:
            continue
        start_frame = int(segment.get("source_start_frame", segment.get("start_frame", 0)) or 0)
        for offset, value in enumerate(values):
            output[start_frame + int(offset)] = max(0.0, float(value))
    return output


def _bin_quota_rows(
    *,
    candidates: list[dict],
    total_keyframes: int,
    density_by_frame: dict[int, float],
    blocked_frames: set[int],
    fps: float,
    bin_sec: float,
    alpha: float,
) -> list[dict]:
    if total_keyframes <= 0 or not candidates:
        return []

    candidate_frames = [int(candidate["frame"]) for candidate in candidates]
    density_frames = list(density_by_frame.keys())
    if not candidate_frames and not density_frames:
        return []

    start_frame = min(candidate_frames + density_frames)
    end_frame = max(candidate_frames + density_frames) + 1
    bin_frames = max(1, int(round(float(bin_sec) * float(fps))))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    total_eligible_frames = 0
    total_density_mass = 0.0
    bins: list[dict] = []
    for bin_index, bin_start in enumerate(range(start_frame, end_frame, bin_frames)):
        bin_end = min(end_frame, bin_start + bin_frames)
        bucket = [
            candidate
            for candidate in candidates
            if bin_start <= int(candidate["frame"]) < bin_end
        ]
        eligible_frames = [
            frame
            for frame in range(bin_start, bin_end)
            if frame not in blocked_frames
            and (frame in density_by_frame or any(int(candidate["frame"]) == frame for candidate in bucket))
        ]
        density_mass = float(
            sum(float(density_by_frame.get(frame, 0.0)) for frame in range(bin_start, bin_end) if frame not in blocked_frames)
        )
        total_eligible_frames += len(eligible_frames)
        total_density_mass += density_mass
        bins.append(
            {
                "index": bin_index,
                "start_frame": int(bin_start),
                "end_frame": int(bin_end),
                "candidates": bucket,
                "eligible_frame_count": int(len(eligible_frames)),
                "density_mass": float(density_mass),
                "quota": 0,
                "remainder": 0.0,
                "weight": 0.0,
            }
        )

    if not bins:
        return []

    for row in bins:
        density_weight = (
            float(row["density_mass"]) / total_density_mass
            if total_density_mass > 1e-8
            else 0.0
        )
        uniform_weight = (
            float(row["eligible_frame_count"]) / float(total_eligible_frames)
            if total_eligible_frames > 0
            else float(len(row["candidates"])) / max(1.0, float(len(candidates)))
        )
        row["weight"] = alpha * density_weight + (1.0 - alpha) * uniform_weight

    total_weight = sum(float(row["weight"]) for row in bins)
    if total_weight <= 1e-8:
        for row in bins:
            row["weight"] = float(len(row["candidates"])) / max(1.0, float(len(candidates)))
        total_weight = sum(float(row["weight"]) for row in bins)

    assigned = 0
    for row in bins:
        raw = int(total_keyframes) * float(row["weight"]) / max(total_weight, 1e-8)
        capacity = len(row["candidates"])
        quota = min(capacity, int(np.floor(raw)))
        row["quota"] = int(quota)
        row["remainder"] = float(raw - np.floor(raw))
        assigned += int(quota)

    while assigned < int(total_keyframes):
        eligible = [row for row in bins if int(row["quota"]) < len(row["candidates"])]
        if not eligible:
            break
        chosen = sorted(
            eligible,
            key=lambda row: (-float(row["remainder"]), -float(row["weight"]), int(row["start_frame"])),
        )[0]
        chosen["quota"] = int(chosen["quota"]) + 1
        chosen["remainder"] = 0.0
        assigned += 1

    return bins


def select_density_uniform_keyframes(
    candidates: list[dict],
    *,
    total_keyframes: int,
    min_gap_frames: int,
    density_by_frame: dict[int, float],
    blocked_frames: set[int],
    fps: float,
    bin_sec: float,
    alpha: float,
) -> tuple[list[dict], list[dict]]:
    if total_keyframes <= 0 or not candidates:
        return [], []
    bins = _bin_quota_rows(
        candidates=candidates,
        total_keyframes=int(total_keyframes),
        density_by_frame=density_by_frame,
        blocked_frames=blocked_frames,
        fps=fps,
        bin_sec=bin_sec,
        alpha=alpha,
    )
    if not bins:
        return select_top_keyframes(candidates, total_keyframes=total_keyframes), []

    selected: list[dict] = []
    selected_frames: set[int] = set()

    def far_enough(frame: int) -> bool:
        return all(abs(int(frame) - int(event["frame"])) >= int(min_gap_frames) for event in selected)

    def clean_candidate(candidate: dict, row: dict | None = None) -> dict:
        output = dict(candidate)
        frame = int(output["frame"])
        output["density_value"] = round(float(density_by_frame.get(frame, 0.0)), 6)
        if row is not None:
            output["density_bin_index"] = int(row["index"])
            output["density_bin_start_frame"] = int(row["start_frame"])
            output["density_bin_end_frame"] = int(row["end_frame"])
            output["density_bin_quota"] = int(row["quota"])
            output["density_bin_mass"] = round(float(row["density_mass"]), 6)
            output["density_uniform_weight"] = round(float(row["weight"]), 6)
        return output

    for row in bins:
        if int(row["quota"]) <= 0:
            continue
        bucket = sorted(
            row["candidates"],
            key=lambda item: (-float(item.get("score", item.get("probability", 0.0))), int(item["frame"])),
        )
        picked = 0
        for candidate in bucket:
            frame = int(candidate["frame"])
            if frame in selected_frames or not far_enough(frame):
                continue
            selected.append(clean_candidate(candidate, row))
            selected_frames.add(frame)
            picked += 1
            if picked >= int(row["quota"]):
                break

    if len(selected) < int(total_keyframes):
        remaining = sorted(
            [candidate for candidate in candidates if int(candidate["frame"]) not in selected_frames],
            key=lambda item: (-float(item.get("score", item.get("probability", 0.0))), int(item["frame"])),
        )
        for candidate in remaining:
            frame = int(candidate["frame"])
            if not far_enough(frame):
                continue
            selected.append(clean_candidate(candidate, None))
            selected_frames.add(frame)
            if len(selected) >= int(total_keyframes):
                break

    bin_summary = [
        {
            "index": int(row["index"]),
            "start_frame": int(row["start_frame"]),
            "end_frame": int(row["end_frame"]),
            "candidate_count": int(len(row["candidates"])),
            "eligible_frame_count": int(row["eligible_frame_count"]),
            "density_mass": round(float(row["density_mass"]), 6),
            "weight": round(float(row["weight"]), 6),
            "quota": int(row["quota"]),
        }
        for row in bins
    ]
    return sorted(selected, key=lambda item: int(item["frame"])), bin_summary


def random_keyframe_candidates(series: list[dict], *, reserved_frames: set[int]) -> list[dict]:
    candidates = []
    for item in series:
        frame = int(item["frame"])
        if frame in reserved_frames:
            continue
        candidates.append(
            {
                "frame": frame,
                "probability": 0.0,
                "score": 0.0,
            }
        )
    return candidates


def select_random_keyframes(
    candidates: list[dict],
    *,
    total_keyframes: int,
    min_gap_frames: int,
    rng: random.Random,
) -> list[dict]:
    if total_keyframes <= 0 or not candidates:
        return []
    selected: list[dict] = []
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    for candidate in shuffled:
        frame = int(candidate["frame"])
        if all(abs(frame - int(event["frame"])) >= int(min_gap_frames) for event in selected):
            selected.append(
                {
                    **candidate,
                    "random_rank": len(selected) + 1,
                }
            )
        if len(selected) >= int(total_keyframes):
            break
    return sorted(selected, key=lambda item: int(item["frame"]))


def event_payload(event: dict, *, index: int, fps: float, event_type: str) -> dict:
    frame = int(event["frame"])
    payload = {
        "index": int(index),
        "type": event_type,
        "frame": frame,
        "frame_5fps": frame_5fps(frame, fps),
        "time_sec": round_time(frame / fps),
    }
    for key in (
        "probability",
        "prominence",
        "score",
        "base_score",
        "adjusted_score",
        "dance_motion_saliency",
        "music_accent_saliency",
        "music_dance_sync_bonus",
        "beat_or_downbeat_bonus",
        "stage2_probability",
        "cut_near_penalty",
        "overcrowding_penalty",
        "uniform_slot_index",
        "uniform_slot_center_frame",
        "uniform_slot_distance_frames",
        "density_value",
        "density_bin_index",
        "density_bin_start_frame",
        "density_bin_end_frame",
        "density_bin_quota",
        "density_bin_mass",
        "density_uniform_weight",
        "random_rank",
    ):
        if key in event:
            payload[key] = round(float(event[key]), 6)
    return payload


def main() -> None:
    args = parse_args()
    exclude_song_ids = parse_id_set(args.exclude_song_ids)
    cut_manifest = load_json(args.cut_manifest)
    key_manifest = load_json(args.keyframe_manifest)
    key_song_by_id = {str(song["id"]): song for song in key_manifest.get("songs", [])}
    rng = random.Random(int(args.random_seed))
    feature_paths: dict[str, Path] = {}
    audio_feature_names: list[str] = []
    dance_feature_names: list[str] = []
    if args.keyframe_selection_mode == "heuristic":
        feature_paths, audio_feature_names, dance_feature_names = load_feature_index(args.feature_manifest)

    per_song: list[dict] = []
    csv_rows: list[dict] = []
    for cut_song in cut_manifest.get("songs", []):
        song_id = str(cut_song["id"])
        if song_id in exclude_song_ids:
            continue
        key_song = key_song_by_id.get(song_id)
        if key_song is None:
            continue

        cut_data = load_json(resolve_repo_path(str(cut_song["data_src"])))
        key_data = load_json(resolve_repo_path(str(key_song["data_src"])))
        fps = float(cut_data.get("fps", key_data.get("fps", 30.0)) or 30.0)

        target_cut_count, target_keyframe_count, total_event_budget, pred_keyframe_mass = density_event_budgets(
            cut_data,
            density_cut_percent=float(args.density_cut_percent),
            density_cut_percent_unit=str(args.density_cut_percent_unit),
        )
        candidates = cut_candidate_peaks(
            probability_series(cut_data),
            min_score=float(args.cut_min_score),
            prominence_weight=float(args.cut_prominence_weight),
        )
        if args.cut_selection_mode == "uniform":
            selected_cuts_raw = select_uniform_cuts(
                candidates,
                intervals=coverage_intervals(cut_data),
                total_cuts=target_cut_count,
                min_gap_sec=float(args.cut_min_gap_sec),
            )
        elif args.cut_selection_mode == "adaptive":
            selected_cuts_raw = select_adaptive_cuts(
                candidates,
                duration_sec=float(cut_data.get("video_duration_sec", key_data.get("video_duration_sec", 0.0)) or 0.0),
                total_cuts=target_cut_count,
                window_sec=float(args.cut_window_sec),
                min_gap_sec=float(args.cut_min_gap_sec),
            )
        else:
            selected_cuts_raw = select_global_cuts(
                candidates,
                total_cuts=target_cut_count,
                min_gap_sec=float(args.cut_min_gap_sec),
            )

        selected_cuts = []
        reserved_cut_frames: set[int] = set()
        for index, event in enumerate(selected_cuts_raw, start=1):
            payload = event_payload(event, index=index, fps=fps, event_type="cut")
            pair = [frame for frame in cut_pair_frames(payload["frame"]) if frame >= 0]
            payload["cut_pair_frames"] = pair
            payload["cut_pair_frames_5fps"] = [frame_5fps(frame, fps) for frame in pair]
            reserved_cut_frames.update(pair)
            selected_cuts.append(payload)

        split_target_keyframe_count = int(target_keyframe_count)
        if args.noncut_budget_mode == "subtract_cut_pairs":
            target_keyframe_count = max(0, int(total_event_budget) - len(reserved_cut_frames))

        if args.keyframe_cut_exclusion_sec is not None:
            keyframe_cut_exclusion_radius_frames = int(round(float(args.keyframe_cut_exclusion_sec) * fps))
            blocked_keyframe_frames = frames_near_events(
                selected_cuts,
                radius=keyframe_cut_exclusion_radius_frames,
            )
            keyframe_exclusion_description = (
                "selected cut frame +/- keyframe_cut_exclusion_sec seconds"
            )
        else:
            keyframe_cut_exclusion_radius_frames = int(args.keyframe_cut_exclusion_radius_frames)
            blocked_keyframe_frames = expand_reserved_frames(
                reserved_cut_frames,
                radius=keyframe_cut_exclusion_radius_frames,
            )
            keyframe_exclusion_description = (
                "selected cut pair frames (frame-1, frame), expanded by keyframe_cut_exclusion_radius_frames"
            )
        keyframe_min_score = float(args.keyframe_threshold) if args.keyframe_threshold is not None else 0.0
        keyframe_min_gap_frames = int(args.keyframe_min_gap_frames or key_data.get("min_peak_gap_frames", 6))
        keyframe_score_series: list[dict] = []
        keyframe_density_bin_summary: list[dict] = []
        if args.keyframe_selection_mode == "random":
            keyframe_candidates = random_keyframe_candidates(
                key_data.get("probability_series") or key_data.get("probability_samples") or [],
                reserved_frames=blocked_keyframe_frames,
            )
            selected_keyframes_raw = select_random_keyframes(
                keyframe_candidates,
                total_keyframes=int(target_keyframe_count),
                min_gap_frames=keyframe_min_gap_frames,
                rng=rng,
            )
        elif args.keyframe_selection_mode == "heuristic":
            feature_path = feature_paths.get(song_id)
            if feature_path is None or not feature_path.exists():
                raise FileNotFoundError(f"Heuristic feature NPZ not found for {song_id}: {feature_path}")
            heuristic_components = build_heuristic_components(
                feature_path,
                audio_feature_names=audio_feature_names,
                dance_feature_names=dance_feature_names,
                sync_window_frames=int(args.sync_window_frames),
            )
            heuristic_weights = {
                "dance_motion": float(args.dance_motion_weight),
                "music_accent": float(args.music_accent_weight),
                "music_dance_sync": float(args.music_dance_sync_weight),
                "beat_bonus": float(args.beat_bonus_weight),
                "stage2_probability": float(args.stage2_probability_weight),
                "cut_near_penalty": float(args.cut_near_penalty_weight),
            }
            keyframe_candidates, keyframe_score_series = heuristic_keyframe_candidates(
                key_data.get("probability_series") or key_data.get("probability_samples") or [],
                components=heuristic_components,
                reserved_frames=blocked_keyframe_frames,
                cut_pair_reserved_frames=reserved_cut_frames,
                weights=heuristic_weights,
                cut_penalty_radius_frames=int(args.cut_near_penalty_radius_frames),
            )
            if keyframe_min_score > 0:
                keyframe_candidates = [
                    candidate for candidate in keyframe_candidates
                    if float(candidate["score"]) >= keyframe_min_score
                ]
            if args.keyframe_placement_mode == "density_uniform":
                selected_keyframes_raw, keyframe_density_bin_summary = select_density_uniform_keyframes(
                    keyframe_candidates,
                    total_keyframes=int(target_keyframe_count),
                    min_gap_frames=keyframe_min_gap_frames,
                    density_by_frame=stagea_density_by_frame(cut_data),
                    blocked_frames=blocked_keyframe_frames,
                    fps=fps,
                    bin_sec=float(args.density_allocation_bin_sec),
                    alpha=float(args.density_uniform_alpha),
                )
            elif args.keyframe_placement_mode == "uniform":
                selected_keyframes_raw = select_uniform_heuristic_keyframes(
                    keyframe_candidates,
                    total_keyframes=int(target_keyframe_count),
                    min_gap_frames=keyframe_min_gap_frames,
                    overcrowding_penalty_weight=float(args.overcrowding_penalty_weight),
                    overcrowding_penalty_radius_frames=int(args.overcrowding_penalty_radius_frames),
                    slot_distance_weight=float(args.uniform_slot_distance_weight),
                )
            else:
                selected_keyframes_raw = select_heuristic_keyframes(
                    keyframe_candidates,
                    total_keyframes=int(target_keyframe_count),
                    min_gap_frames=keyframe_min_gap_frames,
                    overcrowding_penalty_weight=float(args.overcrowding_penalty_weight),
                    overcrowding_penalty_radius_frames=int(args.overcrowding_penalty_radius_frames),
                )
        else:
            keyframe_candidates = keyframe_candidate_peaks(
                key_data.get("probability_series") or key_data.get("probability_samples") or [],
                min_score=keyframe_min_score,
                min_gap_frames=keyframe_min_gap_frames,
                prominence_weight=float(args.keyframe_prominence_weight),
                reserved_frames=blocked_keyframe_frames,
            )
            if args.keyframe_placement_mode == "density_uniform":
                selected_keyframes_raw, keyframe_density_bin_summary = select_density_uniform_keyframes(
                    keyframe_candidates,
                    total_keyframes=int(target_keyframe_count),
                    min_gap_frames=keyframe_min_gap_frames,
                    density_by_frame=stagea_density_by_frame(cut_data),
                    blocked_frames=blocked_keyframe_frames,
                    fps=fps,
                    bin_sec=float(args.density_allocation_bin_sec),
                    alpha=float(args.density_uniform_alpha),
                )
            elif args.keyframe_placement_mode == "uniform":
                selected_keyframes_raw = select_uniform_heuristic_keyframes(
                    keyframe_candidates,
                    total_keyframes=int(target_keyframe_count),
                    min_gap_frames=keyframe_min_gap_frames,
                    overcrowding_penalty_weight=float(args.overcrowding_penalty_weight),
                    overcrowding_penalty_radius_frames=int(args.overcrowding_penalty_radius_frames),
                    slot_distance_weight=float(args.uniform_slot_distance_weight),
                )
            else:
                selected_keyframes_raw = select_top_keyframes(
                    keyframe_candidates,
                    total_keyframes=int(target_keyframe_count),
                )
        selected_keyframes = [
            event_payload(event, index=index, fps=fps, event_type="keyframe")
            for index, event in enumerate(selected_keyframes_raw, start=1)
        ]

        combined = sorted(
            [
                {**event, "index": None}
                for event in selected_cuts
            ]
            + [
                {**event, "index": None}
                for event in selected_keyframes
            ],
            key=lambda item: (int(item["frame"]), 0 if item["type"] == "cut" else 1),
        )
        for index, event in enumerate(combined, start=1):
            event["index"] = index

        gt_cut_count = int((cut_data.get("stats") or {}).get("covered_gt_cut_count", len(cut_data.get("gt_events", []))))
        gt_key_count = int(
            (key_data.get("stats") or {}).get("gt_keyframe_count", len(key_data.get("gt_keyframe_events", [])))
        )
        song_payload = {
            "id": song_id,
            "split": cut_manifest.get("split", "test"),
            "fps": fps,
            "video_src": cut_data.get("video_src", key_data.get("video_src")),
            "video_duration_sec": cut_data.get("video_duration_sec", key_data.get("video_duration_sec")),
            "source": {
                "cut_data_src": cut_song.get("data_src"),
                "keyframe_data_src": key_song.get("data_src"),
            },
            "selection": {
                "cut_count_source": "density_pred_keyframe_mass",
                "total_event_budget": int(total_event_budget),
            "density_cut_percent": float(args.density_cut_percent),
            "density_cut_percent_unit": str(args.density_cut_percent_unit),
            "target_cut_count": int(target_cut_count),
                "target_keyframe_count": int(target_keyframe_count),
                "split_remaining_target_keyframe_count": int(split_target_keyframe_count),
                "reserved_cut_keyframe_count": len(reserved_cut_frames),
                "noncut_budget_mode": args.noncut_budget_mode,
                "cut_selection_mode": args.cut_selection_mode,
                "cut_window_sec": float(args.cut_window_sec),
                "cut_min_gap_sec": float(args.cut_min_gap_sec),
                "cut_min_score": float(args.cut_min_score),
                "cut_prominence_weight": float(args.cut_prominence_weight),
                "keyframe_source": (
                    "random_eligible_frames_remaining_density_budget"
                    if args.keyframe_selection_mode == "random"
                    else (
                        "heuristic_dance_music_sync_topk_remaining_density_budget"
                        if args.keyframe_selection_mode == "heuristic"
                        else "stage2_probability_series_topk_remaining_density_budget"
                    )
                ),
                "keyframe_selection_mode": args.keyframe_selection_mode,
                "keyframe_placement_mode": args.keyframe_placement_mode,
                "density_allocation_bin_sec": float(args.density_allocation_bin_sec),
                "density_uniform_alpha": float(args.density_uniform_alpha),
                "keyframe_min_score": keyframe_min_score,
                "keyframe_min_gap_frames": keyframe_min_gap_frames,
                "keyframe_prominence_weight": float(args.keyframe_prominence_weight),
                "heuristic_weights": {
                    "dance_motion_saliency": float(args.dance_motion_weight),
                    "music_accent_saliency": float(args.music_accent_weight),
                    "music_dance_sync_bonus": float(args.music_dance_sync_weight),
                    "beat_or_downbeat_bonus": float(args.beat_bonus_weight),
                    "stage2_probability": float(args.stage2_probability_weight),
                    "cut_near_penalty": float(args.cut_near_penalty_weight),
                    "overcrowding_penalty": float(args.overcrowding_penalty_weight),
                },
                "sync_window_frames": int(args.sync_window_frames),
                "cut_near_penalty_radius_frames": int(args.cut_near_penalty_radius_frames),
                "overcrowding_penalty_radius_frames": int(args.overcrowding_penalty_radius_frames),
                "uniform_slot_distance_weight": float(args.uniform_slot_distance_weight),
                "keyframe_cut_exclusion_sec": args.keyframe_cut_exclusion_sec,
                "keyframe_cut_exclusion_radius_frames": int(keyframe_cut_exclusion_radius_frames),
                "excluded_keyframe_frames": keyframe_exclusion_description,
                "random_seed": int(args.random_seed),
            },
            "counts": {
                "pred_keyframe_mass": round(float(pred_keyframe_mass), 3),
                "total_event_budget": int(total_event_budget),
                "candidate_cut_count": len(candidates),
                "candidate_keyframe_count": len(keyframe_candidates),
                "target_cut_count": int(target_cut_count),
                "target_keyframe_count": int(target_keyframe_count),
                "split_remaining_target_keyframe_count": int(split_target_keyframe_count),
                "reserved_cut_keyframe_count": len(reserved_cut_frames),
                "selected_cut_count": len(selected_cuts),
                "selected_keyframe_count": len(selected_keyframes),
                "blocked_keyframe_frame_count": len(blocked_keyframe_frames),
                "combined_event_count": len(combined),
                "gt_cut_count": gt_cut_count,
                "gt_keyframe_count": gt_key_count,
            },
            "coverage_segments": key_data.get("coverage_segments") or cut_data.get("coverage_segments", []),
            "keyframe_score_series": keyframe_score_series,
            "keyframe_density_bin_summary": keyframe_density_bin_summary,
            "gt_cut_events": cut_data.get("gt_events") or key_data.get("gt_cut_events", []),
            "gt_keyframe_events": key_data.get("gt_keyframe_events") or cut_data.get("keyframe_events", []),
            "selected_cuts": selected_cuts,
            "selected_keyframes": selected_keyframes,
            "combined_events": combined,
        }
        per_song.append(song_payload)

        for event in combined:
            csv_rows.append(
                {
                    "song_id": song_id,
                    "index": event["index"],
                    "type": event["type"],
                    "frame": event["frame"],
                    "frame_5fps": event["frame_5fps"],
                    "time_sec": event["time_sec"],
                    "probability": event.get("probability", ""),
                    "score": event.get("score", ""),
                    "total_event_budget": total_event_budget,
                    "target_cut_count": target_cut_count,
                    "target_keyframe_count": target_keyframe_count,
                    "selected_cut_count": len(selected_cuts),
                    "selected_keyframe_count": len(selected_keyframes),
                }
            )

    aggregate = {
        "song_count": len(per_song),
        "total_event_budget": sum(int(song["counts"]["total_event_budget"]) for song in per_song),
        "selected_cut_count": sum(int(song["counts"]["selected_cut_count"]) for song in per_song),
        "target_cut_count": sum(int(song["counts"]["target_cut_count"]) for song in per_song),
        "candidate_cut_count": sum(int(song["counts"]["candidate_cut_count"]) for song in per_song),
        "selected_keyframe_count": sum(int(song["counts"]["selected_keyframe_count"]) for song in per_song),
        "target_keyframe_count": sum(int(song["counts"]["target_keyframe_count"]) for song in per_song),
        "split_remaining_target_keyframe_count": sum(
            int(song["counts"]["split_remaining_target_keyframe_count"]) for song in per_song
        ),
        "reserved_cut_keyframe_count": sum(int(song["counts"]["reserved_cut_keyframe_count"]) for song in per_song),
        "candidate_keyframe_count": sum(int(song["counts"]["candidate_keyframe_count"]) for song in per_song),
        "blocked_keyframe_frame_count": sum(int(song["counts"]["blocked_keyframe_frame_count"]) for song in per_song),
        "combined_event_count": sum(int(song["counts"]["combined_event_count"]) for song in per_song),
        "gt_cut_count": sum(int(song["counts"]["gt_cut_count"]) for song in per_song),
        "gt_keyframe_count": sum(int(song["counts"]["gt_keyframe_count"]) for song in per_song),
    }
    payload = {
        "dataset_type": "final_cut_key_timings",
        "version": 1,
        "cut_manifest": repo_path(args.cut_manifest),
        "keyframe_manifest": repo_path(args.keyframe_manifest),
        "excluded_song_ids": sorted(exclude_song_ids),
        "parameters": {
            "density_cut_percent": float(args.density_cut_percent),
            "density_cut_percent_unit": str(args.density_cut_percent_unit),
            "cut_selection_mode": args.cut_selection_mode,
            "cut_window_sec": float(args.cut_window_sec),
            "cut_min_gap_sec": float(args.cut_min_gap_sec),
            "cut_min_score": float(args.cut_min_score),
            "cut_prominence_weight": float(args.cut_prominence_weight),
            "keyframe_selection_mode": args.keyframe_selection_mode,
            "keyframe_placement_mode": args.keyframe_placement_mode,
            "noncut_budget_mode": args.noncut_budget_mode,
            "density_allocation_bin_sec": float(args.density_allocation_bin_sec),
            "density_uniform_alpha": float(args.density_uniform_alpha),
            "feature_manifest": repo_path(args.feature_manifest),
            "keyframe_threshold_override": args.keyframe_threshold,
            "keyframe_min_gap_frames_override": args.keyframe_min_gap_frames,
            "keyframe_prominence_weight": float(args.keyframe_prominence_weight),
            "dance_motion_weight": float(args.dance_motion_weight),
            "music_accent_weight": float(args.music_accent_weight),
            "music_dance_sync_weight": float(args.music_dance_sync_weight),
            "beat_bonus_weight": float(args.beat_bonus_weight),
            "stage2_probability_weight": float(args.stage2_probability_weight),
            "sync_window_frames": int(args.sync_window_frames),
            "cut_near_penalty_weight": float(args.cut_near_penalty_weight),
            "cut_near_penalty_radius_frames": int(args.cut_near_penalty_radius_frames),
            "overcrowding_penalty_weight": float(args.overcrowding_penalty_weight),
            "overcrowding_penalty_radius_frames": int(args.overcrowding_penalty_radius_frames),
            "uniform_slot_distance_weight": float(args.uniform_slot_distance_weight),
            "keyframe_cut_exclusion_sec": args.keyframe_cut_exclusion_sec,
            "keyframe_cut_exclusion_radius_frames": int(args.keyframe_cut_exclusion_radius_frames),
            "random_seed": int(args.random_seed),
        },
        "aggregate": aggregate,
        "songs": per_song,
    }

    output_root = args.output_root
    write_json(output_root / "final_cut_key_timings.json", payload)
    write_csv(
        output_root / "final_cut_key_timings.csv",
        csv_rows,
        [
            "song_id",
            "index",
            "type",
            "frame",
            "frame_5fps",
            "time_sec",
            "probability",
            "score",
            "total_event_budget",
            "target_cut_count",
            "target_keyframe_count",
            "selected_cut_count",
            "selected_keyframe_count",
        ],
    )
    by_song = {
        song["id"]: {
            "cuts": song["selected_cuts"],
            "keyframes": song["selected_keyframes"],
            "combined_events": song["combined_events"],
        }
        for song in per_song
    }
    write_json(output_root / "cut_key_timings_by_song.json", by_song)
    print(f"Output root: {output_root}")
    print(f"Songs: {aggregate['song_count']}")
    print(f"Cuts: {aggregate['selected_cut_count']} / target {aggregate['target_cut_count']}")
    print(f"Keyframes: {aggregate['selected_keyframe_count']}")
    print(f"Combined events: {aggregate['combined_event_count']}")
    print(f"Saved: {output_root / 'final_cut_key_timings.json'}")


if __name__ == "__main__":
    main()
