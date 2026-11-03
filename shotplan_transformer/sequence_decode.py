"""Greedy sequence decoder used for the paper's DCM test-set shot plans.

Cut frames are decoded first, then transition frames, each conditioned on the nearest predicted
cut plans and on the plans already predicted before it. Copied unchanged from the research
training code (``sample_sequence``) so that test-set plans match the paper bit for bit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from shotplan_transformer.v3_cut_context_model import V3CutContextShotPlanModel


PAD_VALUE = -100


def _position_tensors(
    current_frame: int,
    frames: np.ndarray,
    *,
    max_count: int,
    radius: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    nearby = [int(frame) for frame in frames.tolist() if abs(int(frame) - int(current_frame)) <= int(radius)]
    nearby.sort(key=lambda frame: abs(int(frame) - int(current_frame)))
    nearby = nearby[:max_count]
    delta = np.zeros((1, max_count), dtype=np.float32)
    mask = np.zeros((1, max_count), dtype=np.float32)
    for idx, frame in enumerate(nearby):
        delta[0, idx] = float(int(frame) - int(current_frame))
        mask[0, idx] = 1.0
    return torch.from_numpy(delta).to(device), torch.from_numpy(mask).to(device)


def _single_label_batch(
    row: dict[str, int] | None,
    *,
    fields: tuple[str, ...],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    out = {field: torch.full((1,), PAD_VALUE, dtype=torch.long, device=device) for field in fields}
    if row is None:
        return out
    for field in fields:
        out[field] = torch.tensor([int(row[field])], dtype=torch.long, device=device)
    return out


def _history_tensors(
    current_frame: int,
    predicted_rows: dict[int, dict[str, int]],
    *,
    fields: tuple[str, ...],
    history_size: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    prev_frames = [int(frame) for frame in sorted(predicted_rows) if int(frame) < int(current_frame)]
    prev_frames = prev_frames[-history_size:]
    labels = {field: np.full((1, history_size), PAD_VALUE, dtype=np.int64) for field in fields}
    mask = np.zeros((1, history_size), dtype=np.float32)
    delta = np.zeros((1, history_size), dtype=np.float32)
    start = history_size - len(prev_frames)
    for offset, frame in enumerate(prev_frames, start=start):
        row = predicted_rows[frame]
        mask[0, offset] = 1.0
        delta[0, offset] = float(int(current_frame) - int(frame))
        for field in fields:
            labels[field][0, offset] = int(row[field])
    return (
        {field: torch.from_numpy(values).to(device) for field, values in labels.items()},
        torch.from_numpy(mask).to(device),
        torch.from_numpy(delta).to(device),
    )


def _nearest_cut_rows(
    current_frame: int,
    cut_frames: np.ndarray,
    predicted_cut_rows: dict[int, dict[str, int]],
) -> tuple[int | None, dict[str, int] | None, int | None, dict[str, int] | None]:
    previous = [int(frame) for frame in cut_frames.tolist() if int(frame) < int(current_frame)]
    nexts = [int(frame) for frame in cut_frames.tolist() if int(frame) > int(current_frame)]
    prev_frame = max(previous) if previous else None
    next_frame = min(nexts) if nexts else None
    return (
        prev_frame,
        predicted_cut_rows.get(prev_frame) if prev_frame is not None else None,
        next_frame,
        predicted_cut_rows.get(next_frame) if next_frame is not None else None,
    )


def sample_sequence(
    model: V3CutContextShotPlanModel,
    clip: dict[str, Any],
    *,
    fields: tuple[str, ...],
    device: torch.device,
    max_position_tokens: int,
    window_radius: int,
    temperature: float,
    sample: bool,
    gt_labels: dict[str, np.ndarray] | None = None,
    decode_distance_fov_joint: bool = False,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor]:
    audio = clip["audio"].to(device)
    dance = clip["dance"].to(device)
    frames = np.asarray(clip["frames"], dtype=np.int32)
    if "cut_flag" in clip:
        cut_mask = np.asarray(clip["cut_flag"], dtype=bool)
    else:
        cut_mask = np.zeros((frames.shape[0],), dtype=bool)
        if frames.shape[0] > 1:
            cut_mask[1:] = np.diff(frames) == 1
    cut_frames = frames[cut_mask]
    generation_indices = np.concatenate([np.flatnonzero(cut_mask), np.flatnonzero(~cut_mask)])
    predicted_rows: dict[int, dict[str, int]] = {}
    predicted_cut_rows: dict[int, dict[str, int]] = {}
    token_ids = {field: np.zeros((frames.shape[0],), dtype=np.int64) for field in fields}
    total_log_prob = torch.zeros((), dtype=torch.float32, device=device)
    total_entropy = torch.zeros((), dtype=torch.float32, device=device)
    total_ce_loss = torch.zeros((), dtype=torch.float32, device=device)
    ce_count = 0

    for event_index in generation_indices.tolist():
        frame = int(frames[event_index])
        stage = "cut" if bool(cut_mask[event_index]) else "keyframe"
        prev_cut_frame, prev_cut_row, next_cut_frame, next_cut_row = _nearest_cut_rows(frame, cut_frames, predicted_cut_rows)
        prev_cut_labels = _single_label_batch(prev_cut_row, fields=fields, device=device)
        next_cut_labels = _single_label_batch(next_cut_row, fields=fields, device=device)
        prev_cut_mask = torch.tensor([1.0 if prev_cut_row is not None else 0.0], dtype=torch.float32, device=device)
        next_cut_mask = torch.tensor([1.0 if next_cut_row is not None else 0.0], dtype=torch.float32, device=device)
        prev_cut_delta = torch.tensor([float(frame - int(prev_cut_frame)) if prev_cut_frame is not None else 0.0], dtype=torch.float32, device=device)
        next_cut_delta = torch.tensor([float(int(next_cut_frame) - frame) if next_cut_frame is not None else 0.0], dtype=torch.float32, device=device)
        key_delta, key_mask = _position_tensors(frame, frames, max_count=max_position_tokens, radius=window_radius, device=device)
        cut_delta, cut_pos_mask = _position_tensors(frame, cut_frames, max_count=max_position_tokens, radius=window_radius, device=device)
        history_labels, history_mask, history_delta = _history_tensors(
            frame,
            predicted_rows,
            fields=fields,
            history_size=int(clip.get("history_size", 8)),
            device=device,
        )
        outputs = model(
            audio[event_index : event_index + 1],
            dance[event_index : event_index + 1],
            stage=stage,  # type: ignore[arg-type]
            prev_cut_labels=prev_cut_labels,
            prev_cut_mask=prev_cut_mask,
            prev_cut_delta=prev_cut_delta,
            next_cut_labels=next_cut_labels,
            next_cut_mask=next_cut_mask,
            next_cut_delta=next_cut_delta,
            key_position_delta=key_delta,
            key_position_mask=key_mask,
            cut_position_delta=cut_delta,
            cut_position_mask=cut_pos_mask,
            history_labels=history_labels,
            history_mask=history_mask,
            history_delta=history_delta,
        )
        row: dict[str, int] = {}
        decoded_joint_distance: int | None = None
        decoded_joint_fov: int | None = None
        if (
            bool(decode_distance_fov_joint)
            and "distance_fov_logits" in outputs
            and "distance" in fields
            and "fov" in fields
        ):
            distance_count = int(outputs["distance_logits"].shape[-1])
            fov_count = int(outputs["fov_logits"].shape[-1])
            joint_logits = outputs["distance_fov_logits"] / max(1e-6, float(temperature))
            if int(joint_logits.shape[-1]) == distance_count * fov_count:
                joint_dist = Categorical(logits=joint_logits)
                joint_action = joint_dist.sample() if sample else joint_logits.argmax(dim=-1)
                joint_id = int(joint_action.detach().cpu().item())
                decoded_joint_distance = joint_id // max(1, fov_count)
                decoded_joint_fov = joint_id % max(1, fov_count)
                if sample:
                    total_log_prob = total_log_prob + joint_dist.log_prob(joint_action).sum()
                    total_entropy = total_entropy + joint_dist.entropy().sum()
        for field in fields:
            if field == "distance" and decoded_joint_distance is not None:
                row[field] = int(decoded_joint_distance)
                token_ids[field][event_index] = row[field]
                if gt_labels is not None:
                    target_id = int(np.asarray(gt_labels[field], dtype=np.int64)[event_index])
                    if target_id >= 0:
                        target = torch.tensor([target_id], dtype=torch.long, device=device)
                        total_ce_loss = total_ce_loss + F.cross_entropy(outputs["distance_logits"], target)
                        ce_count += 1
                continue
            if field == "fov" and decoded_joint_fov is not None:
                row[field] = int(decoded_joint_fov)
                token_ids[field][event_index] = row[field]
                if gt_labels is not None:
                    target_id = int(np.asarray(gt_labels[field], dtype=np.int64)[event_index])
                    if target_id >= 0:
                        target = torch.tensor([target_id], dtype=torch.long, device=device)
                        total_ce_loss = total_ce_loss + F.cross_entropy(outputs["fov_logits"], target)
                        ce_count += 1
                continue
            raw_logits = outputs[f"{field}_logits"]
            logits = raw_logits / max(1e-6, float(temperature))
            dist = Categorical(logits=logits)
            action = dist.sample() if sample else logits.argmax(dim=-1)
            row[field] = int(action.detach().cpu().item())
            token_ids[field][event_index] = row[field]
            if gt_labels is not None:
                target_id = int(np.asarray(gt_labels[field], dtype=np.int64)[event_index])
                if target_id >= 0:
                    target = torch.tensor([target_id], dtype=torch.long, device=device)
                    total_ce_loss = total_ce_loss + F.cross_entropy(raw_logits, target)
                    ce_count += 1
            if sample:
                total_log_prob = total_log_prob + dist.log_prob(action).sum()
                total_entropy = total_entropy + dist.entropy().sum()
        predicted_rows[frame] = row
        if stage == "cut":
            predicted_cut_rows[frame] = row

    ce_loss = total_ce_loss / max(1, ce_count)
    return token_ids, total_log_prob, total_entropy, ce_loss
