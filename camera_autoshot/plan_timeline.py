from __future__ import annotations

import json
from itertools import groupby
from pathlib import Path


def expand_sample_keyframes_to_timeline(
    keyframes: list[dict[str, object]],
    sample_fps: float,
    timeline_fps: float,
) -> list[dict[str, object]]:
    if sample_fps <= 0.0 or timeline_fps <= 0.0:
        return [
            {
                "source_frame": int(item["frame"]),
                "timeline_frame": int(item["frame"]),
                "original": dict(item),
            }
            for item in keyframes
        ]

    frame_scale = timeline_fps / sample_fps
    expanded: list[dict[str, object]] = []

    def source_frame(item: dict[str, object]) -> int:
        return int(item["frame"])

    for sample_frame, grouped_items_iter in groupby(keyframes, key=source_frame):
        grouped_items = [dict(item) for item in grouped_items_iter]
        count = len(grouped_items)
        base_timeline_frame = int(round(sample_frame * frame_scale))

        if count == 1:
            timeline_frames = [base_timeline_frame]
        else:
            start_frame = base_timeline_frame - (count - 1)
            if start_frame < 0:
                start_frame = 0
            timeline_frames = list(range(start_frame, start_frame + count))

        for timeline_frame, item in zip(timeline_frames, grouped_items):
            expanded.append(
                {
                    "source_frame": int(sample_frame),
                    "timeline_frame": int(timeline_frame),
                    "original": item,
                }
            )
    return expanded


def load_plan_payload(
    plan_path: Path,
    plan_sample_fps: float,
    timeline_fps: float,
) -> dict[str, object]:
    raw = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    sample_fps = float(plan_sample_fps or raw.get("sample_fps", 0.0) or 0.0)
    expanded = expand_sample_keyframes_to_timeline(
        list(raw.get("keyframes", [])),
        sample_fps=sample_fps,
        timeline_fps=timeline_fps,
    )
    return {
        "sample_fps": sample_fps,
        "timeline_fps": timeline_fps,
        "keyframes": expanded,
    }
