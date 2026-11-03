from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


LABEL_FIELDS = (
    "target_body",
    "shot_size",
    "angle",
    "composition",
    "roll",
)
REMOVED_LABEL_FIELDS = {"vertical_composition"}


def label_fields_from_manifest(manifest: dict[str, Any]) -> tuple[str, ...]:
    """Return dataset label fields, preserving legacy defaults when absent."""

    fields = manifest.get("label_fields", LABEL_FIELDS)
    if not isinstance(fields, (list, tuple)) or not fields:
        return LABEL_FIELDS
    return tuple(str(field) for field in fields if str(field) not in REMOVED_LABEL_FIELDS)


def load_manifest(dataset_root: Path) -> dict[str, Any]:
    return json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))


def compute_normalization_stats(dataset_root: Path, split_name: str = "train") -> dict[str, np.ndarray]:
    manifest = load_manifest(dataset_root)
    song_lookup = {entry["song_id"]: entry for entry in manifest["songs"]}
    split_song_ids = manifest["splits"][split_name]

    audio_sum = None
    audio_sq_sum = None
    dance_sum = None
    dance_sq_sum = None
    token_count = 0

    for song_id in split_song_ids:
        entry = song_lookup.get(song_id)
        if entry is None:
            continue
        arrays = np.load(entry["path"], allow_pickle=False)
        audio = np.asarray(arrays["audio_windows"], dtype=np.float32)
        dance = np.asarray(arrays["dance_windows"], dtype=np.float32)
        flat_audio = audio.reshape(audio.shape[0] * audio.shape[1], audio.shape[-1])
        flat_dance = dance.reshape(dance.shape[0] * dance.shape[1], dance.shape[-1])

        if audio_sum is None:
            audio_sum = flat_audio.sum(axis=0, dtype=np.float64)
            audio_sq_sum = np.square(flat_audio, dtype=np.float64).sum(axis=0)
            dance_sum = flat_dance.sum(axis=0, dtype=np.float64)
            dance_sq_sum = np.square(flat_dance, dtype=np.float64).sum(axis=0)
        else:
            audio_sum += flat_audio.sum(axis=0, dtype=np.float64)
            audio_sq_sum += np.square(flat_audio, dtype=np.float64).sum(axis=0)
            dance_sum += flat_dance.sum(axis=0, dtype=np.float64)
            dance_sq_sum += np.square(flat_dance, dtype=np.float64).sum(axis=0)
        token_count += flat_audio.shape[0]

    if token_count <= 0 or audio_sum is None or dance_sum is None:
        raise ValueError(f"No training samples found in split {split_name}")

    audio_mean = audio_sum / float(token_count)
    dance_mean = dance_sum / float(token_count)
    audio_var = np.maximum(audio_sq_sum / float(token_count) - np.square(audio_mean), 1e-6)
    dance_var = np.maximum(dance_sq_sum / float(token_count) - np.square(dance_mean), 1e-6)

    return {
        "audio_mean": audio_mean.astype(np.float32),
        "audio_std": np.sqrt(audio_var).astype(np.float32),
        "dance_mean": dance_mean.astype(np.float32),
        "dance_std": np.sqrt(dance_var).astype(np.float32),
    }


def save_normalization_stats(path: Path, stats: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **stats)


def load_normalization_stats(path: Path) -> dict[str, np.ndarray]:
    arrays = np.load(path, allow_pickle=False)
    return {key: np.asarray(arrays[key], dtype=np.float32) for key in arrays.files}


class ShotPlanWindowDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        split_name: str,
        *,
        normalization_stats: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split_name = split_name
        self.manifest = load_manifest(self.dataset_root)
        self.label_fields = label_fields_from_manifest(self.manifest)
        song_lookup = {entry["song_id"]: entry for entry in self.manifest["songs"]}

        audio_chunks = []
        dance_chunks = []
        slot_count_chunks = []
        slot_mask_chunks = []
        sampled_frame_chunks = []
        source_center_chunks = []
        cut_flag_chunks = []
        prev_frame_delta_chunks = []
        prev_slot_mask_chunks = []
        future_frame_delta_chunks = []
        future_frame_mask_chunks = []
        bodypart_visible_chunks = []
        label_chunks = {field: [] for field in self.label_fields}
        prev_label_chunks = {field: [] for field in self.label_fields}
        sample_ids: list[str] = []
        song_ids: list[str] = []

        for song_id in self.manifest["splits"][split_name]:
            entry = song_lookup.get(song_id)
            if entry is None:
                continue
            arrays = np.load(entry["path"], allow_pickle=False)
            audio = np.asarray(arrays["audio_windows"], dtype=np.float32)
            dance = np.asarray(arrays["dance_windows"], dtype=np.float32)
            sampled_frames = np.asarray(arrays["sampled_frames"], dtype=np.int32)
            source_centers = np.asarray(arrays["source_center_frames"], dtype=np.int32)
            slot_count = np.asarray(arrays["slot_count"], dtype=np.int64)
            slot_mask = np.asarray(arrays["slot_mask"], dtype=np.float32)
            cut_flag = np.asarray(arrays["cut_flag"], dtype=np.float32)
            prev_frame_delta = np.asarray(arrays["prev_frame_delta"], dtype=np.float32)
            prev_slot_mask = np.asarray(arrays["prev_slot_mask"], dtype=np.float32)
            future_context_size = int(self.manifest.get("future_context_size", 0))
            future_frame_delta = (
                np.asarray(arrays["future_frame_delta"], dtype=np.float32)
                if "future_frame_delta" in arrays.files
                else np.zeros((audio.shape[0], future_context_size), dtype=np.float32)
            )
            future_frame_mask = (
                np.asarray(arrays["future_frame_mask"], dtype=np.float32)
                if "future_frame_mask" in arrays.files
                else np.zeros((audio.shape[0], future_context_size), dtype=np.float32)
            )
            bodypart_visible = (
                np.asarray(arrays["bodypart_visible_target"], dtype=np.float32)
                if "bodypart_visible_target" in arrays.files
                else np.zeros((audio.shape[0], 0), dtype=np.float32)
            )

            audio_chunks.append(audio)
            dance_chunks.append(dance)
            sampled_frame_chunks.append(sampled_frames)
            source_center_chunks.append(source_centers)
            slot_count_chunks.append(slot_count)
            slot_mask_chunks.append(slot_mask)
            cut_flag_chunks.append(cut_flag)
            prev_frame_delta_chunks.append(prev_frame_delta)
            prev_slot_mask_chunks.append(prev_slot_mask)
            future_frame_delta_chunks.append(future_frame_delta)
            future_frame_mask_chunks.append(future_frame_mask)
            bodypart_visible_chunks.append(bodypart_visible)

            for field in self.label_fields:
                label_chunks[field].append(np.asarray(arrays[f"label_{field}"], dtype=np.int64))
                prev_label_chunks[field].append(np.asarray(arrays[f"prev_label_{field}"], dtype=np.int64))

            sample_ids.extend(f"{song_id}_frame{int(frame)}" for frame in sampled_frames.tolist())
            song_ids.extend([song_id] * int(sampled_frames.shape[0]))

        if not audio_chunks:
            raise ValueError(f"No samples found for split {split_name}")

        self.audio = np.concatenate(audio_chunks, axis=0)
        self.dance = np.concatenate(dance_chunks, axis=0)
        self.sampled_frames = np.concatenate(sampled_frame_chunks, axis=0)
        self.source_center_frames = np.concatenate(source_center_chunks, axis=0)
        self.slot_count = np.concatenate(slot_count_chunks, axis=0)
        self.slot_mask = np.concatenate(slot_mask_chunks, axis=0)
        self.cut_flag = np.concatenate(cut_flag_chunks, axis=0)
        self.prev_frame_delta = np.concatenate(prev_frame_delta_chunks, axis=0)
        self.prev_slot_mask = np.concatenate(prev_slot_mask_chunks, axis=0)
        self.future_frame_delta = np.concatenate(future_frame_delta_chunks, axis=0)
        self.future_frame_mask = np.concatenate(future_frame_mask_chunks, axis=0)
        self.bodypart_visible_target = np.concatenate(bodypart_visible_chunks, axis=0)
        self.labels = {field: np.concatenate(chunks, axis=0) for field, chunks in label_chunks.items()}
        self.prev_labels = {field: np.concatenate(chunks, axis=0) for field, chunks in prev_label_chunks.items()}
        self.sample_ids = sample_ids
        self.song_ids = song_ids

        if normalization_stats is not None:
            audio_mean = normalization_stats["audio_mean"].reshape(1, 1, -1)
            audio_std = normalization_stats["audio_std"].reshape(1, 1, -1)
            dance_mean = normalization_stats["dance_mean"].reshape(1, 1, -1)
            dance_std = normalization_stats["dance_std"].reshape(1, 1, -1)
            self.audio = (self.audio - audio_mean) / audio_std
            self.dance = (self.dance - dance_mean) / dance_std

    def __len__(self) -> int:
        return int(self.audio.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {
            "audio": torch.from_numpy(self.audio[index]).float(),
            "dance": torch.from_numpy(self.dance[index]).float(),
            "slot_count": torch.tensor(int(self.slot_count[index]), dtype=torch.long),
            "slot_mask": torch.from_numpy(self.slot_mask[index]).float(),
            "cut_flag": torch.tensor(float(self.cut_flag[index]), dtype=torch.float32),
            "prev_frame_delta": torch.from_numpy(self.prev_frame_delta[index]).float(),
            "prev_slot_mask": torch.from_numpy(self.prev_slot_mask[index]).float(),
            "future_frame_delta": torch.from_numpy(self.future_frame_delta[index]).float(),
            "future_frame_mask": torch.from_numpy(self.future_frame_mask[index]).float(),
            "bodypart_visible_target": torch.from_numpy(self.bodypart_visible_target[index]).float(),
            "sampled_frame": torch.tensor(int(self.sampled_frames[index]), dtype=torch.long),
            "source_center_frame": torch.tensor(int(self.source_center_frames[index]), dtype=torch.long),
            "sample_id": self.sample_ids[index],
            "song_id": self.song_ids[index],
        }
        for field in self.label_fields:
            item[f"label_{field}"] = torch.from_numpy(self.labels[field][index]).long()
            item[f"prev_label_{field}"] = torch.from_numpy(self.prev_labels[field][index]).long()
        return item
