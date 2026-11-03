"""Shared helpers for DCM test-clip inference.

Feature windows, event timing, and model settings follow the paper's evaluation run exactly, so
regenerated shot plans match the released ones.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from camera_autoshot.reverse_v3.close_dense_schema import EXPORTED_DISTANCE_VALUES
from shotplan_transformer.build_dataset import _window_indices
from shotplan_transformer.compact_dance import compact_dance_from_full
from shotplan_transformer.v3_cut_context_model import V3CutContextConfig, V3CutContextShotPlanModel


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_CLIPS = REPO_ROOT / "datasets" / "dcm_test_clips.json"
DEFAULT_SHOTPLAN_RUN_DIR = REPO_ROOT / "final model" / "v3_1_visual_scale_005"
DEFAULT_INTERPOLATION_RUN_DIR = REPO_ROOT / "final model" / "v3_1_interpolation_stage"
DEFAULT_FEATURE_ROOT = REPO_ROOT / "music_processing" / "results" / "cut_keyframe_multimodal_dataset_dcm" / "songs"
DEFAULT_TARGETS_ROOT = REPO_ROOT / "camera_autoshot" / "reverses"
DEFAULT_TIMING_ROOT = REPO_ROOT / "evaluation" / "results" / "stage1_timing"
DEFAULT_TIMING_FILENAME = "cut_key_timings_by_song.json"

# Settings of the V3.1 shot-plan dataset the final model was trained on.
WINDOW_RADIUS = 24
DATASET_PREV_CONTEXT_SIZE = 8
# The paper's test-set plans were decoded with a 4-plan history (the evaluation script fell back to
# its default because the checkpoint does not store the context size), while the top-k export used
# for CP-SAT read the dataset's context size of 8. Both are kept so the paper outputs reproduce.
PLAN_HISTORY_SIZE = 4
TOPK_HISTORY_SIZE = DATASET_PREV_CONTEXT_SIZE


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def source_index(source_song_id: str) -> int:
    match = re.search(r"(\d+)", str(source_song_id))
    if not match:
        raise ValueError(f"cannot parse source index from {source_song_id!r}")
    return int(match.group(1))


def part_from_clip_id(clip_id: str) -> str:
    pieces = [piece for piece in str(clip_id).split("_") if piece]
    if len(pieces) >= 3:
        return str(pieces[2])
    return ""


def parse_indices(raw: str) -> set[int] | None:
    if str(raw).strip().lower() in ("", "all", "*"):
        return None
    values: set[int] = set()
    for token in str(raw).split(","):
        match = re.search(r"(\d+)", token.strip())
        if match:
            values.add(int(match.group(1)))
    return values


def select_test_clips(test_clips_path: Path, *, indices: str = "all", clip_ids: str = "") -> list[dict[str, Any]]:
    payload = load_json(test_clips_path)
    explicit = {item.strip() for item in str(clip_ids).split(",") if item.strip()}
    allowed_indices = parse_indices(indices)
    rows: list[dict[str, Any]] = []
    for row in payload.get("clips", []):
        if not isinstance(row, dict) or str(row.get("official_split", "")).lower() != "test":
            continue
        if explicit and str(row.get("clip_id", "")) not in explicit:
            continue
        if allowed_indices is not None and source_index(str(row["source_song_id"])) not in allowed_indices:
            continue
        rows.append(row)
    rows.sort(key=lambda item: (int(source_index(str(item["source_song_id"]))), str(item["clip_id"])))
    return rows


def external_timing_for_clip(
    timing_by_song: dict[str, Any],
    *,
    source_song_id: str,
    source_start_frame: int,
    source_end_frame: int,
    include_boundaries: bool,
) -> tuple[list[int], list[dict[str, Any]], dict[str, int]]:
    """Collect the cut-pair and transition frames of one clip from song-level timing."""

    song_timing = timing_by_song.get(source_song_id)
    if not isinstance(song_timing, dict):
        return [], [], {"timing_cut_count": 0, "timing_keyframe_count": 0}

    frames: set[int] = set()
    cut_pairs: list[dict[str, Any]] = []
    cut_count = 0
    key_count = 0
    if include_boundaries:
        frames.add(int(source_start_frame))
        frames.add(int(source_end_frame))

    for cut in song_timing.get("cuts", []):
        if not isinstance(cut, dict):
            continue
        pair = cut.get("cut_pair_frames")
        if not isinstance(pair, list) or not pair:
            pair = [cut.get("frame")]
        pair_frames = [int(frame) for frame in pair if frame is not None]
        clipped_pair = [frame for frame in pair_frames if int(source_start_frame) <= frame <= int(source_end_frame)]
        if not clipped_pair:
            continue
        cut_count += 1
        for frame in clipped_pair:
            frames.add(int(frame))
        if len(clipped_pair) >= 2:
            cut_pairs.append(
                {
                    "timing_frame": int(clipped_pair[-1]),
                    "source_frames": [int(clipped_pair[0]), int(clipped_pair[-1])],
                }
            )

    for keyframe in song_timing.get("keyframes", []):
        if not isinstance(keyframe, dict) or keyframe.get("frame") is None:
            continue
        frame = int(keyframe["frame"])
        if int(source_start_frame) <= frame <= int(source_end_frame):
            frames.add(frame)
            key_count += 1

    return sorted(frames), cut_pairs, {"timing_cut_count": cut_count, "timing_keyframe_count": key_count}


def build_feature_windows(
    *,
    feature_path: Path,
    source_manifest: dict[str, Any],
    local_frames: np.ndarray,
    window_radius: int,
    stats: dict[str, np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalized music and compact dance windows centred on each event frame."""

    arrays = np.load(feature_path, allow_pickle=False)
    audio = np.asarray(arrays["audio_features"], dtype=np.float32)
    dance_full = np.asarray(arrays["dance_features"], dtype=np.float32)
    dance, _names, _metadata = compact_dance_from_full(
        dance_full,
        [str(name) for name in source_manifest.get("dance_feature_names", [])],
    )
    length = min(int(audio.shape[0]), int(dance.shape[0]))
    if length <= 0:
        raise ValueError(f"feature sequence is empty: {feature_path}")
    audio = audio[:length]
    dance = dance[:length]

    window_size = int(window_radius) * 2 + 1
    audio_windows = np.zeros((local_frames.shape[0], window_size, audio.shape[1]), dtype=np.float32)
    dance_windows = np.zeros((local_frames.shape[0], window_size, dance.shape[1]), dtype=np.float32)
    for idx, frame in enumerate(local_frames.tolist()):
        center = min(max(int(frame), 0), length - 1)
        indices = _window_indices(center, length=length, radius=int(window_radius))
        audio_windows[idx] = audio[indices]
        dance_windows[idx] = dance[indices]

    audio_mean = np.asarray(stats["audio_mean"], dtype=np.float32).reshape(1, 1, -1)
    audio_std = np.asarray(stats["audio_std"], dtype=np.float32).reshape(1, 1, -1)
    dance_mean = np.asarray(stats["dance_mean"], dtype=np.float32).reshape(1, 1, -1)
    dance_std = np.asarray(stats["dance_std"], dtype=np.float32).reshape(1, 1, -1)
    audio_windows = (audio_windows - audio_mean) / audio_std
    dance_windows = (dance_windows - dance_mean) / dance_std
    return torch.from_numpy(audio_windows).float(), torch.from_numpy(dance_windows).float()


def make_model(checkpoint: dict[str, Any], device: torch.device) -> V3CutContextShotPlanModel:
    config = V3CutContextConfig(**dict(checkpoint["config"]))
    model = V3CutContextShotPlanModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def fields_and_vocabularies(checkpoint: dict[str, Any]) -> tuple[tuple[str, ...], dict[str, list[str]]]:
    fields = tuple(str(field) for field in checkpoint["fields"])
    vocabularies = {field: [str(value) for value in checkpoint["vocabularies"][field]] for field in fields}
    return fields, vocabularies


def distance_schema_for_vocabularies(vocabularies: dict[str, list[str]]) -> dict[str, object] | None:
    if "very_close" not in {str(value) for value in vocabularies.get("distance", [])}:
        return None
    return {
        "distance_values": list(EXPORTED_DISTANCE_VALUES),
        "notes": (
            "Reverse V3.1 close-dense distance labels. Exported labels are "
            "mapped to legacy internal V3 enum slots only while solving VMD."
        ),
    }


def feature_manifest(feature_root: Path) -> dict[str, Any]:
    path = Path(feature_root).resolve().parent / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Per-clip feature manifest not found: {path}. Build the DCM clip features first (README, Training)."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "index",
        "clip_id",
        "status",
        "official_split",
        "clip_start_frame",
        "clip_end_frame",
        "event_count",
        "timing_cut_count",
        "timing_keyframe_count",
        "cut_pair_count",
        "changed_pair_count",
        "changed_pair_rate",
        "plan",
        "vmd",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
