from __future__ import annotations

from pathlib import Path

import numpy as np

from music_processing.dance_motion_features import (
    BODY_PART_SPECS,
    DEFAULT_MOTION_ROOT,
    build_motion_features,
    build_window_starts,
    compute_body_part_speed_series,
    load_motion_positions,
    reduce_window,
    resolve_motion_path,
)


def json_ready_float(value: float) -> float:
    return round(float(value), 6)


def compute_window_body_part_score(
    payload: dict,
    *,
    start: int,
    end: int,
    reducer: str,
    score_mode: str,
) -> tuple[float, float, float]:
    raw_score = reduce_window(payload["raw_series"][start:end], str(reducer))
    global_reference_score = reduce_window(payload["global_reference_series"][start:end], str(reducer))
    if score_mode == "absolute":
        return raw_score, raw_score, global_reference_score
    if score_mode == "relative_to_global":
        floor = float(payload.get("global_reference_floor", 1e-6))
        return raw_score / max(global_reference_score, floor), raw_score, global_reference_score
    raise ValueError(f"Unsupported score_mode: {score_mode}")


def resolve_input_motion_path(
    *,
    index: int | None,
    motion_path: Path | None,
    motion_root: Path = DEFAULT_MOTION_ROOT,
) -> Path:
    if motion_path is not None:
        return Path(motion_path).resolve()
    if index is None:
        raise ValueError("Provide either index or motion_path.")
    return resolve_motion_path(int(index), motion_root=motion_root).resolve()


def frame_range_to_row_slice(
    frame_numbers: np.ndarray,
    *,
    window_frames: int,
    start_frame: int | None,
    end_frame: int | None,
) -> tuple[int, int]:
    first_frame = int(frame_numbers[0])
    last_frame = int(frame_numbers[-1])

    if start_frame is None and end_frame is None:
        raise ValueError("A specific window requires at least one of start_frame or end_frame.")

    if start_frame is None:
        start_frame = int(end_frame) - int(window_frames) + 1
    if end_frame is None:
        end_frame = int(start_frame) + int(window_frames) - 1

    if end_frame < start_frame:
        raise ValueError(f"window_end must be >= window_start, got {start_frame}..{end_frame}")
    if start_frame < first_frame or end_frame > last_frame:
        raise ValueError(f"Requested window {start_frame}..{end_frame} is outside available frame range {first_frame}..{last_frame}")

    start_row = int(start_frame - first_frame)
    end_row = int(end_frame - first_frame + 1)
    return start_row, end_row


def resolve_window_rows(
    frame_numbers: np.ndarray,
    *,
    window_frames: int,
    stride: int,
    window_start: int | None = None,
    window_end: int | None = None,
) -> tuple[list[tuple[int, int]], int]:
    if window_start is not None or window_end is not None:
        row_slice = frame_range_to_row_slice(
            frame_numbers,
            window_frames=int(window_frames),
            start_frame=window_start,
            end_frame=window_end,
        )
        return [row_slice], int(row_slice[1] - row_slice[0])

    effective_window_frames = int(window_frames)
    starts = build_window_starts(len(frame_numbers), effective_window_frames, int(stride))
    return (
        [(start, min(len(frame_numbers), start + effective_window_frames)) for start in starts],
        effective_window_frames,
    )


def build_body_part_activity_result(
    *,
    index: int | None,
    motion_path: Path | None,
    motion_root: Path = DEFAULT_MOTION_ROOT,
    fps: float = 30.0,
    window_frames: int = 45,
    stride: int = 15,
    window_start: int | None = None,
    window_end: int | None = None,
    reducer: str = "p75",
    activity_quantile: float = 75.0,
    min_active_score: float = 0.0,
    score_mode: str = "relative_to_global",
) -> dict:
    resolved_motion_path = resolve_input_motion_path(index=index, motion_path=motion_path, motion_root=motion_root)
    if not resolved_motion_path.exists():
        raise FileNotFoundError(f"Motion file not found: {resolved_motion_path}")

    frame_numbers, positions, bone_names = load_motion_positions(resolved_motion_path)
    dance_features, feature_names = build_motion_features(positions)
    body_part_series = compute_body_part_speed_series(dance_features, feature_names, bone_names)

    target_rows, effective_window_frames = resolve_window_rows(
        frame_numbers,
        window_frames=int(window_frames),
        stride=int(stride),
        window_start=window_start,
        window_end=window_end,
    )
    reference_starts = build_window_starts(len(frame_numbers), effective_window_frames, int(stride))
    reference_rows = [(start, min(len(frame_numbers), start + effective_window_frames)) for start in reference_starts]

    window_scores_by_key: dict[str, np.ndarray] = {}
    threshold_groups_by_key: dict[str, str] = {}
    threshold_group_scores: dict[str, list[np.ndarray]] = {}
    threshold_group_scales: dict[str, float] = {}

    for spec in BODY_PART_SPECS:
        payload = body_part_series[spec.key]
        window_scores = np.asarray(
            [
                compute_window_body_part_score(
                    payload,
                    start=start,
                    end=end,
                    reducer=str(reducer),
                    score_mode=str(score_mode),
                )[0]
                for start, end in reference_rows
            ],
            dtype=np.float32,
        )
        threshold_group = str(payload.get("threshold_group", spec.key))
        threshold_groups_by_key[spec.key] = threshold_group
        window_scores_by_key[spec.key] = window_scores
        threshold_group_scores.setdefault(threshold_group, []).append(window_scores)
        threshold_group_scales.setdefault(threshold_group, float(payload.get("threshold_scale", 1.0)))

    thresholds_by_group: dict[str, float] = {}
    for threshold_group, grouped_scores in threshold_group_scores.items():
        combined_scores = np.concatenate(grouped_scores).astype(np.float32)
        threshold_scale = float(threshold_group_scales.get(threshold_group, 1.0))
        thresholds_by_group[threshold_group] = max(
            float(np.percentile(combined_scores, float(activity_quantile))) * threshold_scale,
            float(min_active_score),
        )

    thresholds: dict[str, float] = {}
    body_part_metadata: dict[str, dict] = {}
    for spec in BODY_PART_SPECS:
        payload = body_part_series[spec.key]
        threshold_group = threshold_groups_by_key[spec.key]
        threshold_scale = float(payload.get("threshold_scale", 1.0))
        threshold = thresholds_by_group[threshold_group]
        thresholds[spec.key] = threshold
        body_part_metadata[spec.key] = {
            "display_name": payload["display_name"],
            "joint_names": payload["joint_names"],
            "joint_indices": [int(joint_index) for joint_index in payload["joint_indices"]],
            "speed_feature_names": payload["speed_feature_names"],
            "joint_speed_reducer": str(payload.get("joint_speed_reducer", "mean")),
            "global_reference_reducer": str(payload.get("global_reference_reducer", "mean")),
            "global_reference_joint_count": int(len(payload.get("global_reference_feature_names", []))),
            "global_reference_floor": json_ready_float(float(payload.get("global_reference_floor", 1e-6))),
            "threshold_group": threshold_group,
            "threshold_scale": json_ready_float(threshold_scale),
            "threshold": json_ready_float(threshold),
        }

    windows: list[dict] = []
    for window_index, (start_row, end_row) in enumerate(target_rows):
        ranked_parts: list[dict] = []
        active_keys: list[str] = []
        active_display_names: list[str] = []
        part_scores: dict[str, dict] = {}

        for spec in BODY_PART_SPECS:
            payload = body_part_series[spec.key]
            score, raw_score, global_reference_score = compute_window_body_part_score(
                payload,
                start=start_row,
                end=end_row,
                reducer=str(reducer),
                score_mode=str(score_mode),
            )
            threshold = thresholds[spec.key]
            active = bool(score >= threshold)
            relative_to_threshold = float(score / threshold) if threshold > 0.0 else 0.0

            if active:
                active_keys.append(spec.key)
                active_display_names.append(payload["display_name"])

            ranked_parts.append(
                {
                    "key": spec.key,
                    "display_name": payload["display_name"],
                    "score": score,
                    "raw_score": raw_score,
                    "global_reference_score": global_reference_score,
                    "threshold": threshold,
                    "relative_to_threshold": relative_to_threshold,
                    "active": active,
                }
            )
            part_scores[spec.key] = {
                "display_name": payload["display_name"],
                "score": json_ready_float(score),
                "raw_score": json_ready_float(raw_score),
                "global_reference_score": json_ready_float(global_reference_score),
                "threshold": json_ready_float(threshold),
                "relative_to_threshold": json_ready_float(relative_to_threshold),
                "active": active,
            }

        ranked_parts.sort(key=lambda item: item["relative_to_threshold"], reverse=True)
        windows.append(
            {
                "window_index": int(window_index),
                "start_frame": int(frame_numbers[start_row]),
                "end_frame": int(frame_numbers[end_row - 1]),
                "start_sec": json_ready_float(float(frame_numbers[start_row]) / float(fps)),
                "end_sec": json_ready_float(float(frame_numbers[end_row - 1]) / float(fps)),
                "num_frames": int(end_row - start_row),
                "active_labels": active_keys,
                "active_display_labels": active_display_names,
                "ranked_parts": [
                    {
                        "key": item["key"],
                        "display_name": item["display_name"],
                        "score": json_ready_float(item["score"]),
                        "raw_score": json_ready_float(item["raw_score"]),
                        "global_reference_score": json_ready_float(item["global_reference_score"]),
                        "threshold": json_ready_float(item["threshold"]),
                        "relative_to_threshold": json_ready_float(item["relative_to_threshold"]),
                        "active": bool(item["active"]),
                    }
                    for item in ranked_parts
                ],
                "body_parts": part_scores,
            }
        )

    frame_diffs = np.diff(frame_numbers)
    return {
        "index": int(index) if index is not None else None,
        "motion_path": str(resolved_motion_path),
        "fps": float(fps),
        "num_frames": int(len(frame_numbers)),
        "frame_range": {"start": int(frame_numbers[0]), "end": int(frame_numbers[-1])},
        "frame_numbers_are_contiguous": bool(frame_diffs.size == 0 or np.all(frame_diffs == 1)),
        "window_frames": int(effective_window_frames),
        "stride": int(stride),
        "reducer": str(reducer),
        "score_mode": str(score_mode),
        "activity_quantile": float(activity_quantile),
        "min_active_score": float(min_active_score),
        "body_parts": body_part_metadata,
        "windows": windows,
    }
