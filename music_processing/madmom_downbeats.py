from __future__ import annotations

import json
from pathlib import Path

import importlib.metadata as importlib_metadata
import numpy as np


def _patch_madmom_distribution() -> None:
    original_distribution = importlib_metadata.distribution
    if getattr(importlib_metadata.distribution, "_music_processing_patched", False):
        return

    def patched_distribution(name: str):
        if name == "madmom":
            try:
                return original_distribution(name)
            except importlib_metadata.PackageNotFoundError:
                return original_distribution("madmom-prebuilt")
        return original_distribution(name)

    patched_distribution._music_processing_patched = True  # type: ignore[attr-defined]
    importlib_metadata.distribution = patched_distribution  # type: ignore[assignment]


def _import_madmom():
    _patch_madmom_distribution()
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

    return RNNDownBeatProcessor, DBNDownBeatTrackingProcessor


def _round_time(value: float) -> float:
    return round(float(value), 3)


def compute_madmom_downbeats(audio_path: Path, *, beats_per_bar: list[int] | None = None, fps: int = 100) -> dict:
    if beats_per_bar is None:
        beats_per_bar = [3, 4, 6]

    RNNDownBeatProcessor, DBNDownBeatTrackingProcessor = _import_madmom()
    activations = RNNDownBeatProcessor()(str(audio_path))
    beat_data = DBNDownBeatTrackingProcessor(beats_per_bar=beats_per_bar, fps=fps)(activations)
    beat_array = np.asarray(beat_data)

    beats: list[dict] = []
    downbeats: list[dict] = []
    for idx, row in enumerate(beat_array, start=1):
        time_sec = _round_time(row[0])
        beat_index = int(row[1])
        payload = {
            "index": idx,
            "time_sec": time_sec,
            "beat_index": beat_index,
            "label": f"{beat_index}",
        }
        beats.append(payload)
        if beat_index == 1:
            downbeats.append(
                {
                    "index": len(downbeats) + 1,
                    "time_sec": time_sec,
                    "beat_index": beat_index,
                    "label": "1",
                }
            )

    return {
        "audio_path": str(audio_path),
        "method": "madmom_rnn_dbn_downbeat",
        "fps": int(fps),
        "beats_per_bar_candidates": [int(x) for x in beats_per_bar],
        "activation_shape": [int(x) for x in activations.shape],
        "beat_count": len(beats),
        "beats": beats,
        "downbeat_count": len(downbeats),
        "downbeats": downbeats,
    }


def save_madmom_downbeats(
    audio_path: Path,
    output_path: Path,
    *,
    beats_per_bar: list[int] | None = None,
    fps: int = 100,
) -> dict:
    payload = compute_madmom_downbeats(audio_path, beats_per_bar=beats_per_bar, fps=fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
