"""Stage-1 cut and transition timing for the DCM test clips from the released timing models.

    python -m evaluation.dcm_test_timing --output-root evaluation/results/stage1_timing

Writes ``cut_key_timings_by_song.json``, the input of ``evaluation.dcm_test_plans``, from each test
clip's music and dance features. The steps follow ``music_processing/run_unseen_mmd_timing_pipeline.py``:

1. the cut model gives per-frame cut probabilities;
2. Stage-A density without cut context sets the keyframe budget, 70% of which goes to cut pairs;
3. cuts are selected over the whole song, at least 1 s apart;
4. Stage-A density with the selected cuts as context sets the transition budget and the density map
   that spreads the transitions;
5. the cut-conditioned transition locator runs on the segments between the selected cut pairs, and
   transitions are picked with density-uniform placement.

Events are selected once per song across its test clips, with the selection functions and settings
of ``music_processing/export_final_cut_key_timings.py``.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.dcm_common import (
    DEFAULT_FEATURE_ROOT,
    DEFAULT_TEST_CLIPS,
    DEFAULT_TIMING_FILENAME,
    DEFAULT_TIMING_ROOT,
    REPO_ROOT,
    device_from_arg,
    feature_manifest,
    select_test_clips,
    source_index,
    write_json,
)
from music_processing import export_final_cut_key_timings as selection
from music_processing.build_audio_only_cut_dataset_from_multimodal import DEMUX_EVENTS_NO_RMS_AUDIO_FEATURES
from music_processing.build_cut_conditioned_keyframe_dataset import (
    _build_condition_features,
    _build_prediction_intervals,
    _split_long_interval,
)
from music_processing.build_keyframe_density_stagea_dataset import _build_window_starts, _previous_cut_frames
from music_processing.train_cut_conditioned_keyframe_conformer import _build_model as build_transition_model
from music_processing.train_cut_transition_baseline import _predict_song
from music_processing.train_film_cut_transition import AudioDanceFiLMCutPredictor
from music_processing.train_keyframe_density_stagea import _prev_cut_context, build_stagea_model
from shotplan_transformer.compact_dance import compact_dance_from_full


FINAL_MODEL_ROOT = REPO_ROOT / "final model"
FPS = 30.0
# Selection settings of the paper (export_final_cut_key_timings.py).
CUT_PERCENT = 70.0
CUT_MIN_GAP_SEC = 1.0
PEAK_PROMINENCE_WEIGHT = 0.3
TRANSITION_MIN_GAP_FRAMES = 5
TRANSITION_CUT_EXCLUSION_SEC = 0.5
DENSITY_BIN_SEC = 3.0
DENSITY_UNIFORM_ALPHA = 0.6
# Stage-A windows and transition-locator segments as in the models' training datasets.
STAGEA_WINDOW_FRAMES = 90
STAGEA_STRIDE_FRAMES = 15
STAGEA_BATCH_SIZE = 256
SEGMENT_MAX_FRAMES = 300


@dataclass
class Clip:
    clip_id: str
    song_id: str
    start: int
    audio: np.ndarray
    dance: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(len(self.audio))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-1 timing for the DCM test clips.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_TIMING_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--test-clips", type=Path, default=DEFAULT_TEST_CLIPS)
    parser.add_argument("--cut-model", type=Path, default=FINAL_MODEL_ROOT / "cut_transition" / "best_model.pt")
    parser.add_argument("--density-model", type=Path, default=FINAL_MODEL_ROOT / "density_stagea" / "best_model.pt")
    parser.add_argument("--transition-model", type=Path, default=FINAL_MODEL_ROOT / "keyframe_stage2" / "best_model.pt")
    parser.add_argument("--device", default="auto")
    return parser


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def std_of(checkpoint: dict[str, Any], key: str) -> np.ndarray:
    return np.maximum(np.asarray(checkpoint[key], dtype=np.float32), 1e-6)


def load_clips(test_clips: Path, feature_root: Path) -> list[Clip]:
    """Each test clip's 17 music and 168 compact dance features, the input of all three models."""

    manifest = feature_manifest(feature_root)
    audio_names = [str(name) for name in manifest["audio_feature_names"]]
    dance_names = [str(name) for name in manifest["dance_feature_names"]]
    columns = [audio_names.index(name) for name in DEMUX_EVENTS_NO_RMS_AUDIO_FEATURES]
    clips = []
    for row in select_test_clips(test_clips):
        arrays = np.load(Path(feature_root) / f"{row['clip_id']}.npz", allow_pickle=False)
        dance, _names, _metadata = compact_dance_from_full(np.asarray(arrays["dance_features"], dtype=np.float32), dance_names)
        clips.append(
            Clip(
                clip_id=str(row["clip_id"]),
                song_id=str(row["source_song_id"]),
                start=int(row["source_start_frame"]),
                audio=np.asarray(arrays["audio_features"], dtype=np.float32)[:, columns],
                dance=np.asarray(dance, dtype=np.float32),
            )
        )
    return clips


def frame_series(values: dict[int, float], *, digits: int) -> list[dict[str, Any]]:
    return [
        {"frame": int(frame), "time_sec": round(int(frame) / FPS, 3), "value": round(float(value), digits)}
        for frame, value in sorted(values.items())
    ]


# ---------------------------------------------------------------------------- models


def cut_probabilities(checkpoint: dict[str, Any], clips: list[Clip], device: torch.device) -> dict[str, list[dict[str, Any]]]:
    args = checkpoint.get("args") or {}
    model = AudioDanceFiLMCutPredictor(
        audio_dim=int(checkpoint.get("audio_dim", 17)),
        dance_dim=int(checkpoint.get("dance_dim", 168)),
        hidden_dim=int(args.get("hidden_dim", 128)),
        dropout=float(args.get("dropout", 0.1)),
        film_scale=float(args.get("film_scale", 0.5)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for clip in clips:
        probabilities = _predict_song(
            model,
            features=np.concatenate([clip.audio, clip.dance], axis=1),
            feature_mean=np.asarray(checkpoint["feature_mean"], dtype=np.float32),
            feature_std=std_of(checkpoint, "feature_std"),
            window_frames=int(args.get("window_frames", 240)),
            stride=int(args.get("stride", 60)),
            device=device,
        )
        song_end = max(other.start + other.num_frames for other in clips if other.song_id == clip.song_id)
        sums.setdefault(clip.song_id, np.zeros(song_end, dtype=np.float32))
        counts.setdefault(clip.song_id, np.zeros(song_end, dtype=np.float32))
        # Two clips of a song can share a boundary frame; it gets their average.
        sums[clip.song_id][clip.start : clip.start + clip.num_frames] += probabilities
        counts[clip.song_id][clip.start : clip.start + clip.num_frames] += 1.0
    series = {}
    for song_id, total in sums.items():
        covered = np.flatnonzero(counts[song_id] > 0)
        values = total[covered] / counts[song_id][covered]
        series[song_id] = frame_series(dict(zip(covered.tolist(), values.tolist())), digits=5)
    return series


def stagea_density(
    checkpoint: dict[str, Any],
    clips: list[Clip],
    device: torch.device,
    cuts_by_clip: dict[str, list[int]] | None,
) -> dict[str, dict[str, Any]]:
    """Per-song keyframe mass and frame density map; the cut context comes from ``cuts_by_clip``."""

    args = checkpoint.get("args") or {}
    dims = checkpoint.get("feature_dims") or {}
    model = build_stagea_model(
        model_type=str(args.get("model_type", "cnn")),
        audio_dim=int(dims.get("audio", 17)),
        dance_dim=int(dims.get("dance", 168)),
        cut_context_dim=int(dims.get("prev_cut", 4)),
        model_dim=int(args.get("model_dim", 192)),
        num_heads=int(args.get("num_heads", 4)),
        num_layers=int(args.get("num_layers", 3)),
        cnn_kernel_size=int(args.get("cnn_kernel_size", 5)),
        cnn_dilations=str(args.get("cnn_dilations", "1,2,4,8")),
        dropout=float(args.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    audio_mean, audio_std = np.asarray(checkpoint["audio_mean"], dtype=np.float32), std_of(checkpoint, "audio_std")
    dance_mean, dance_std = np.asarray(checkpoint["dance_mean"], dtype=np.float32), std_of(checkpoint, "dance_std")

    windows: list[tuple[Clip, int]] = []
    for clip in sorted(clips, key=lambda item: item.clip_id):
        starts = _build_window_starts(
            clip.num_frames, window_frames=STAGEA_WINDOW_FRAMES, stride_frames=STAGEA_STRIDE_FRAMES, include_tail=True
        )
        windows.extend((clip, int(start)) for start in starts)

    densities: list[float] = []
    with torch.no_grad():
        for batch_start in range(0, len(windows), STAGEA_BATCH_SIZE):
            audio, dance, context = [], [], []
            for clip, start in windows[batch_start : batch_start + STAGEA_BATCH_SIZE]:
                end = start + STAGEA_WINDOW_FRAMES
                local_cuts = np.asarray(sorted((cuts_by_clip or {}).get(clip.clip_id, [])), dtype=np.int32)
                last_cut, previous_cut = _previous_cut_frames(local_cuts, start)
                audio.append(((clip.audio[start:end] - audio_mean[None, :]) / audio_std[None, :]).astype(np.float32))
                dance.append(((clip.dance[start:end] - dance_mean[None, :]) / dance_std[None, :]).astype(np.float32))
                context.append(
                    _prev_cut_context(
                        start_frame=start,
                        window_frames=STAGEA_WINDOW_FRAMES,
                        fps=FPS,
                        last_cut_frame=last_cut,
                        previous_cut_frame=previous_cut,
                        max_age_sec=float(args.get("max_prev_cut_age_sec", 10.0)),
                        max_gap_sec=float(args.get("max_prev_cut_gap_sec", 10.0)),
                    ).astype(np.float32)
                )
            valid = np.ones((len(audio), STAGEA_WINDOW_FRAMES), dtype=np.float32)
            logits = model(
                torch.from_numpy(np.stack(audio)).to(device),
                torch.from_numpy(np.stack(dance)).to(device),
                torch.from_numpy(np.stack(context)).to(device),
                torch.from_numpy(valid).to(device),
            )
            densities.extend(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32).tolist())

    sums = {clip.clip_id: np.zeros(clip.num_frames, dtype=np.float32) for clip in clips}
    counts = {clip.clip_id: np.zeros(clip.num_frames, dtype=np.float32) for clip in clips}
    for (clip, start), density in zip(windows, densities):
        sums[clip.clip_id][start : start + STAGEA_WINDOW_FRAMES] += float(density)
        counts[clip.clip_id][start : start + STAGEA_WINDOW_FRAMES] += 1.0

    songs: dict[str, dict[str, Any]] = defaultdict(lambda: {"pred_keyframe_mass": 0.0, "density_by_frame": {}})
    for clip in sorted(clips, key=lambda item: item.clip_id):
        mask = counts[clip.clip_id] > 0
        frame_density = np.zeros(clip.num_frames, dtype=np.float32)
        frame_density[mask] = sums[clip.clip_id][mask] / counts[clip.clip_id][mask]
        song = songs[clip.song_id]
        song["pred_keyframe_mass"] += round(float(np.sum(frame_density)), 3)
        for offset, value in enumerate(frame_density.tolist()):
            song["density_by_frame"][clip.start + offset] = max(0.0, round(float(value), 6))
    for song in songs.values():
        song["pred_keyframe_mass"] = round(float(song["pred_keyframe_mass"]), 3)
    return dict(songs)


def transition_probabilities(
    checkpoint: dict[str, Any],
    clips: list[Clip],
    device: torch.device,
    cuts_by_clip: dict[str, list[int]],
) -> dict[str, list[dict[str, Any]]]:
    """Transition-locator probabilities on the segments between the selected cut pairs."""

    model = build_transition_model(
        args=argparse.Namespace(**dict(checkpoint["args"])),
        audio_input_dim=len(checkpoint["audio_mean"]),
        dance_input_dim=len(checkpoint["dance_mean"]),
        condition_input_dim=len(checkpoint["cond_mean"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    normalization = [
        (np.asarray(checkpoint[f"{name}_mean"], dtype=np.float32), std_of(checkpoint, f"{name}_std"))
        for name in ("audio", "dance", "cond")
    ]

    by_song: dict[str, dict[int, float]] = defaultdict(dict)
    for clip in clips:
        cut_frames = np.asarray(sorted(cuts_by_clip.get(clip.clip_id, [])), dtype=np.int32)
        pair_frames = np.asarray(
            sorted({frame for cut in cut_frames.tolist() for frame in selection.cut_pair_frames(cut)}), dtype=np.int32
        )
        for interval in _build_prediction_intervals(clip.num_frames, cut_frames, pair_frames):
            start, end = int(interval["interval_start_frame"]), int(interval["interval_end_frame"])
            if end < start:
                continue
            for chunk_start, chunk_end in _split_long_interval(start, end, max_frames=SEGMENT_MAX_FRAMES):
                length = chunk_end - chunk_start + 1
                condition = _build_condition_features(
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    interval_start=start,
                    interval_end=end,
                    has_left_cut=interval["left_cut_frame"] is not None,
                    has_right_cut=interval["right_cut_frame"] is not None,
                )
                inputs = []
                for values, (mean, std) in zip(
                    (clip.audio[chunk_start : chunk_end + 1], clip.dance[chunk_start : chunk_end + 1], condition),
                    normalization,
                ):
                    padded = np.zeros((1, SEGMENT_MAX_FRAMES, values.shape[1]), dtype=np.float32)
                    padded[0, :length] = (values - mean[None, :]) / std[None, :]
                    inputs.append(torch.from_numpy(padded).to(device))
                valid = np.zeros((1, SEGMENT_MAX_FRAMES), dtype=np.float32)
                valid[0, :length] = 1.0
                with torch.no_grad():
                    outputs = model(*inputs, valid_mask=torch.from_numpy(valid).to(device))
                probabilities = torch.sigmoid(outputs["keyframe_logits"])[0, :length].detach().cpu().numpy().astype(np.float32)
                song = by_song[clip.song_id]
                for offset, value in enumerate(probabilities.tolist()):
                    frame = clip.start + chunk_start + offset
                    song[frame] = max(float(value), song.get(frame, 0.0))
    return {song_id: frame_series(values, digits=6) for song_id, values in by_song.items()}


# ---------------------------------------------------------------------------- selection


def budget_split(pred_keyframe_mass: float) -> tuple[int, int]:
    """(cut count, total keyframe budget) for a song's keyframe mass."""

    cut_count, _, total, _ = selection.density_event_budgets(
        {"density_budget": {"pred_keyframe_mass": pred_keyframe_mass}},
        density_cut_percent=CUT_PERCENT,
        density_cut_percent_unit="cut_keyframes",
    )
    return int(cut_count), int(total)


def select_cuts(series: list[dict[str, Any]], cut_count: int) -> list[dict[str, Any]]:
    candidates = selection.cut_candidate_peaks(series, min_score=0.0, prominence_weight=PEAK_PROMINENCE_WEIGHT)
    cuts = []
    for index, event in enumerate(selection.select_global_cuts(candidates, total_cuts=cut_count, min_gap_sec=CUT_MIN_GAP_SEC), start=1):
        payload = selection.event_payload(event, index=index, fps=FPS, event_type="cut")
        payload["cut_pair_frames"] = [frame for frame in selection.cut_pair_frames(payload["frame"]) if frame >= 0]
        payload["cut_pair_frames_5fps"] = [selection.frame_5fps(frame, FPS) for frame in payload["cut_pair_frames"]]
        cuts.append(payload)
    return cuts


def select_transitions(
    series: list[dict[str, Any]],
    cuts: list[dict[str, Any]],
    *,
    count: int,
    density_by_frame: dict[int, float],
) -> list[dict[str, Any]]:
    blocked = selection.frames_near_events(cuts, radius=int(round(TRANSITION_CUT_EXCLUSION_SEC * FPS)))
    candidates = selection.keyframe_candidate_peaks(
        series,
        min_score=0.0,
        min_gap_frames=TRANSITION_MIN_GAP_FRAMES,
        prominence_weight=PEAK_PROMINENCE_WEIGHT,
        reserved_frames=blocked,
    )
    selected, _bins = selection.select_density_uniform_keyframes(
        candidates,
        total_keyframes=count,
        min_gap_frames=TRANSITION_MIN_GAP_FRAMES,
        density_by_frame=density_by_frame,
        blocked_frames=blocked,
        fps=FPS,
        bin_sec=DENSITY_BIN_SEC,
        alpha=DENSITY_UNIFORM_ALPHA,
    )
    return [selection.event_payload(event, index=index, fps=FPS, event_type="keyframe") for index, event in enumerate(selected, start=1)]


def local_cuts(cuts_by_song: dict[str, list[dict[str, Any]]], clips: list[Clip]) -> dict[str, list[int]]:
    return {
        clip.clip_id: [
            int(cut["frame"]) - clip.start
            for cut in cuts_by_song.get(clip.song_id, [])
            if clip.start <= int(cut["frame"]) < clip.start + clip.num_frames
        ]
        for clip in clips
    }


def main() -> None:
    args = build_parser().parse_args()
    device = device_from_arg(str(args.device))
    clips = load_clips(args.test_clips.resolve(), args.feature_root.resolve())
    song_ids = sorted({clip.song_id for clip in clips}, key=source_index)

    cut_series = cut_probabilities(load_checkpoint(args.cut_model), clips, device)
    density_checkpoint = load_checkpoint(args.density_model)
    before_cuts = stagea_density(density_checkpoint, clips, device, cuts_by_clip=None)
    budgets = {song_id: budget_split(before_cuts[song_id]["pred_keyframe_mass"]) for song_id in song_ids}
    cuts = {song_id: select_cuts(cut_series[song_id], budgets[song_id][0]) for song_id in song_ids}
    cuts_by_clip = local_cuts(cuts, clips)
    after_cuts = stagea_density(density_checkpoint, clips, device, cuts_by_clip=cuts_by_clip)
    transition_series = transition_probabilities(load_checkpoint(args.transition_model), clips, device, cuts_by_clip)

    timing: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    for song_id in song_ids:
        cut_count = budgets[song_id][0]
        _, total = budget_split(after_cuts[song_id]["pred_keyframe_mass"])
        transitions = select_transitions(
            transition_series[song_id],
            cuts[song_id],
            count=max(0, total - 2 * cut_count),
            density_by_frame=after_cuts[song_id]["density_by_frame"],
        )
        combined = sorted(
            [dict(event) for event in cuts[song_id] + transitions],
            key=lambda item: (int(item["frame"]), 0 if item["type"] == "cut" else 1),
        )
        for index, event in enumerate(combined, start=1):
            event["index"] = index
        timing[song_id] = {"cuts": cuts[song_id], "keyframes": transitions, "combined_events": combined}
        summary[song_id] = {
            "keyframe_mass_without_cut_context": before_cuts[song_id]["pred_keyframe_mass"],
            "keyframe_mass_with_cut_context": after_cuts[song_id]["pred_keyframe_mass"],
            "cut_count": len(cuts[song_id]),
            "transition_count": len(transitions),
        }

    output_root = args.output_root.resolve()
    write_json(output_root / DEFAULT_TIMING_FILENAME, timing)
    write_json(
        output_root / "stage1_summary.json",
        {
            "models": {"cut": str(args.cut_model), "density": str(args.density_model), "transition": str(args.transition_model)},
            "clip_count": len(clips),
            "cut_count": sum(row["cut_count"] for row in summary.values()),
            "transition_count": sum(row["transition_count"] for row in summary.values()),
            "songs": summary,
        },
    )
    print(f"songs: {len(song_ids)}  clips: {len(clips)}  cuts: {sum(len(c) for c in cuts.values())}  "
          f"transitions: {sum(row['transition_count'] for row in summary.values())}")
    print(f"Wrote {output_root / DEFAULT_TIMING_FILENAME}")


if __name__ == "__main__":
    main()
