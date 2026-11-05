from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, fields as dataclass_fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera_autoshot.providers import JsonFrameTargetProvider
from camera_autoshot.render_server_dcm_plans import (
    _load_render_variants_module,
    load_dcm_rows,
    resolve_render_assets,
)
from camera_autoshot.reverse_v3.close_dense_schema import (
    EXPORTED_DISTANCE_VALUES,
    apply_close_dense_distance_schema,
    apply_close_dense_distance_schema_from_payload,
    internalize_close_dense_row,
    uses_close_dense_distance_schema,
)
from camera_autoshot.reverse_v3.solver_v3 import CameraSolveConfigV3, solve_camera_timeline_v3
from camera_autoshot.reverse_v3.spec_v3 import ShotKeyframeRequestV3
from camera_autoshot.vmd import write_camera_vmd

from shotplan_transformer.compact_dance import compact_dance_feature_names
from shotplan_transformer.dataset import load_manifest, load_normalization_stats
from shotplan_transformer.infer_to_plans import _device_from_arg, _window_indices
from shotplan_transformer.infer_v3_dcm_from_timing import (
    _clip_rows_by_source_index,
    _compute_partial_render_stats,
    _cut_change_stats,
    _load_json,
    _parse_indices,
    _render_partial_jobs_batch,
    _select_dcm_clips_for_index,
    _timing_rows_to_source_frames,
    _write_manifest_csv,
)
from shotplan_transformer.infer_v3_from_timing import _prepare_features
from shotplan_transformer.v3_cut_context_model import V3CutContextConfig, V3CutContextShotPlanModel


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_ROOT = (
    WORKSPACE_ROOT
    / "music_processing"
    / "results"
    / "01_datasets"
    / "cut_keyframe_multimodal_dataset_dcm"
    / "songs"
)
DEFAULT_DCM_MANIFEST = (
    WORKSPACE_ROOT
    / "music_processing"
    / "results"
    / "01_datasets"
    / "cut_keyframe_multimodal_dataset_dcm"
    / "manifest.json"
)
DEFAULT_TIMING_ROOT = (
    WORKSPACE_ROOT
    / "camera_autoshot"
    / "results"
    / "shotplan_transformer_mlp_dropout015_cutdupw100_dcm_clip_cutplusrandom_p030_seed42_20cuts_firsthalf_corrected"
)
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "camera_autoshot" / "results" / "v3_cut_context_dcm_clip_timing"
PAD_VALUE = -100
REMOVED_OUTPUT_FIELDS = {"vertical_composition"}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _checkpoint_manifest(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the dataset manifest needed by feature prep and decoding."""

    args = checkpoint.get("args", {})
    dataset_root = Path(args.get("dataset_root", "")) if args.get("dataset_root") else None
    manifest: dict[str, Any] = {}
    if dataset_root is not None and (dataset_root / "manifest.json").exists():
        manifest = load_manifest(dataset_root)
    elif int(checkpoint["config"]["dance_dim"]) == len(compact_dance_feature_names()):
        # Released checkpoints ship without their training dataset. A 168-D dance
        # input is the compact set taken from the 427-D source features.
        manifest["dance_feature_set"] = "compact"
    manifest["label_fields"] = list(checkpoint["fields"])
    manifest["vocabularies"] = checkpoint["vocabularies"]
    manifest["window_size"] = int(checkpoint["config"]["window_size"])
    manifest["window_radius"] = int(
        manifest.get("window_radius", (int(checkpoint["config"]["window_size"]) - 1) // 2)
    )
    manifest["window_stride"] = int(manifest.get("window_stride", 1))
    manifest["prev_context_size"] = int(manifest.get("prev_context_size", 8))
    manifest["max_slots"] = int(manifest.get("max_slots", 1))
    manifest["position_context_mode"] = str(args.get("position_context_mode", manifest.get("position_context_mode", "radius")))
    manifest["position_context_side_count"] = int(
        args.get("position_context_side_count", manifest.get("position_context_side_count", 8))
    )
    return manifest


def _collect_train_priors(manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    """Estimate field priors from the training split in the checkpoint dataset."""

    fields = [str(field) for field in manifest["label_fields"]]
    vocabs = {str(field): list(values) for field, values in manifest["vocabularies"].items()}
    priors = {field: np.ones((len(vocabs[field]),), dtype=np.float64) for field in fields}
    dataset_root = Path(str(manifest.get("dataset_root", "")))
    song_entries = {entry["song_id"]: entry for entry in manifest.get("songs", [])}
    split_ids = manifest.get("splits", {}).get("train", [])
    if not dataset_root.exists() or not split_ids:
        return {field: values / values.sum() for field, values in priors.items()}

    for song_id in split_ids:
        entry = song_entries.get(song_id)
        if not entry:
            continue
        arrays = np.load(entry["path"], allow_pickle=False)
        for field in fields:
            key = f"label_{field}"
            if key not in arrays:
                continue
            labels = np.asarray(arrays[key], dtype=np.int64)[:, 0]
            for label in labels:
                label_id = int(label)
                if 0 <= label_id < priors[field].shape[0]:
                    priors[field][label_id] += 1.0
    return {field: values / values.sum() for field, values in priors.items()}


def _collect_prediction_priors(reference_root: Path, manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    """Estimate priors from a previous inference root for post-hoc calibration."""

    fields = [str(field) for field in manifest["label_fields"]]
    vocabs = {str(field): [str(value) for value in values] for field, values in manifest["vocabularies"].items()}
    value_to_id = {field: {value: idx for idx, value in enumerate(values)} for field, values in vocabs.items()}
    counts = {field: np.ones((len(vocabs[field]),), dtype=np.float64) for field in fields}

    for json_path in sorted(Path(reference_root).glob("c*/camera_poses*.json")):
        name = json_path.name
        if name.endswith("_debug.json") or name.endswith("_timing_debug.json") or name.endswith("_cut_change_stats.json"):
            continue
        try:
            payload = _load_json(json_path)
        except Exception:
            continue
        rows = payload.get("keyframes") or payload.get("camera_poses") or []
        if not isinstance(rows, list):
            continue
        for row in rows:
            for field in fields:
                label_id = value_to_id[field].get(str(row.get(field, "")))
                if label_id is not None:
                    counts[field][label_id] += 1.0
    return {field: values / values.sum() for field, values in counts.items()}


def _build_calibration_biases(
    *,
    manifest: dict[str, Any],
    reference_root: Path | None,
    alpha: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build field-wise logit biases that nudge predictions toward train priors."""

    if float(alpha) == 0:
        return {}
    train_priors = _collect_train_priors(manifest)
    if reference_root is not None:
        reference_priors = _collect_prediction_priors(reference_root, manifest)
    else:
        reference_priors = {
            field: np.full_like(prior, 1.0 / max(1, prior.shape[0]))
            for field, prior in train_priors.items()
        }

    biases: dict[str, torch.Tensor] = {}
    for field, train_prior in train_priors.items():
        ref_prior = reference_priors.get(field)
        if ref_prior is None or ref_prior.shape != train_prior.shape:
            ref_prior = np.full_like(train_prior, 1.0 / max(1, train_prior.shape[0]))
        bias = float(alpha) * np.log((train_prior + 1e-8) / (ref_prior + 1e-8))
        biases[field] = torch.tensor(bias, dtype=torch.float32, device=device).reshape(1, -1)
    return biases


def _build_cut_context_model(checkpoint: dict[str, Any]) -> V3CutContextShotPlanModel:
    allowed = {field.name for field in dataclass_fields(V3CutContextConfig)}
    config_payload = {key: value for key, value in checkpoint["config"].items() if key in allowed}
    config = V3CutContextConfig(**config_payload)
    model = V3CutContextShotPlanModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _empty_label_batch(fields: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    return {field: torch.full((1,), PAD_VALUE, dtype=torch.long, device=device) for field in fields}


def _single_label_batch(
    row: dict[str, object] | None,
    *,
    fields: list[str],
    field_to_id: dict[str, dict[str, int]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    values = _empty_label_batch(fields, device)
    if row is None:
        return values
    for field in fields:
        token = str(row.get(field))
        if token in field_to_id[field]:
            values[field] = torch.tensor([int(field_to_id[field][token])], dtype=torch.long, device=device)
    return values


def _history_tensors(
    *,
    current_frame: int,
    predicted_rows_by_frame: dict[int, dict[str, object]],
    fields: list[str],
    field_to_id: dict[str, dict[str, int]],
    history_size: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    frames = [frame for frame in sorted(predicted_rows_by_frame) if int(frame) < int(current_frame)]
    frames = frames[-history_size:]
    labels_np = {
        field: np.full((1, history_size), PAD_VALUE, dtype=np.int64)
        for field in fields
    }
    mask_np = np.zeros((1, history_size), dtype=np.float32)
    delta_np = np.zeros((1, history_size), dtype=np.float32)
    start = history_size - len(frames)
    for offset, frame in enumerate(frames, start=start):
        row = predicted_rows_by_frame[int(frame)]
        mask_np[0, offset] = 1.0
        delta_np[0, offset] = float(int(current_frame) - int(frame))
        for field in fields:
            token = str(row.get(field))
            if token in field_to_id[field]:
                labels_np[field][0, offset] = int(field_to_id[field][token])
    return (
        {field: torch.from_numpy(values).to(device) for field, values in labels_np.items()},
        torch.from_numpy(mask_np).to(device),
        torch.from_numpy(delta_np).to(device),
    )


def _position_tensors(
    *,
    current_frame: int,
    frames: list[int],
    max_count: int,
    radius: int,
    mode: str = "radius",
    side_count: int = 8,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = np.zeros((1, max_count), dtype=np.float32)
    mask = np.zeros((1, max_count), dtype=np.float32)
    if str(mode) == "prevnext":
        sorted_frames = sorted(int(frame) for frame in frames)
        prev_slots = min(max(0, int(side_count)), int(max_count))
        next_slots = min(max(0, int(side_count)), max(0, int(max_count) - prev_slots))
        previous = [frame for frame in sorted_frames if frame < int(current_frame)][-prev_slots:] if prev_slots else []
        future = [frame for frame in sorted_frames if frame > int(current_frame)][:next_slots] if next_slots else []
        prev_count = len(previous)
        if prev_count:
            start = prev_slots - prev_count
            for offset, frame in enumerate(previous):
                delta[0, start + offset] = float(int(frame) - int(current_frame))
                mask[0, start + offset] = 1.0
        for offset, frame in enumerate(future):
            idx = prev_slots + offset
            delta[0, idx] = float(int(frame) - int(current_frame))
            mask[0, idx] = 1.0
    else:
        nearby = [int(frame) for frame in frames if abs(int(frame) - int(current_frame)) <= int(radius)]
        nearby.sort(key=lambda frame: abs(int(frame) - int(current_frame)))
        nearby = nearby[:max_count]
        for idx, frame in enumerate(nearby):
            delta[0, idx] = float(int(frame) - int(current_frame))
            mask[0, idx] = 1.0
    return torch.from_numpy(delta).to(device), torch.from_numpy(mask).to(device)


def _predict_one(
    *,
    model: V3CutContextShotPlanModel,
    stage: str,
    audio: np.ndarray,
    dance: np.ndarray,
    full_frame: int,
    source_start_frame: int,
    selected_frames: list[int],
    cut_transition_frames: list[int],
    predicted_rows_by_frame: dict[int, dict[str, object]],
    predicted_cut_rows_by_frame: dict[int, dict[str, object]],
    manifest: dict[str, Any],
    device: torch.device,
    logit_temperature: float,
    calibration_biases: dict[str, torch.Tensor],
    decode_distance_fov_joint: bool,
) -> dict[str, object]:
    fields = [str(field) for field in manifest["label_fields"]]
    vocabularies = {
        str(field): [str(value) for value in values]
        for field, values in manifest["vocabularies"].items()
    }
    field_to_id = {
        field: {value: idx for idx, value in enumerate(values)}
        for field, values in vocabularies.items()
    }
    history_size = int(manifest.get("prev_context_size", 8))
    max_position_tokens = int(model.config.max_position_tokens)
    window_radius = int(manifest.get("window_radius", (int(model.config.window_size) - 1) // 2))
    window_stride = int(manifest.get("window_stride", 1))
    position_context_mode = str(manifest.get("position_context_mode", "radius"))
    position_context_side_count = int(manifest.get("position_context_side_count", 8))

    local_frame = int(full_frame) - int(source_start_frame)
    center = min(max(int(local_frame), 0), int(audio.shape[0]) - 1)
    indices = _window_indices(
        center,
        length=int(audio.shape[0]),
        radius=window_radius,
        stride=window_stride,
    )
    audio_batch = torch.from_numpy(audio[indices][None, :, :]).to(device)
    dance_batch = torch.from_numpy(dance[indices][None, :, :]).to(device)

    prev_cut_frame = max((frame for frame in cut_transition_frames if int(frame) < int(full_frame)), default=None)
    next_cut_frame = min((frame for frame in cut_transition_frames if int(frame) > int(full_frame)), default=None)
    prev_cut_row = predicted_cut_rows_by_frame.get(int(prev_cut_frame)) if prev_cut_frame is not None else None
    next_cut_row = predicted_cut_rows_by_frame.get(int(next_cut_frame)) if next_cut_frame is not None else None

    prev_cut_labels = _single_label_batch(prev_cut_row, fields=fields, field_to_id=field_to_id, device=device)
    next_cut_labels = _single_label_batch(next_cut_row, fields=fields, field_to_id=field_to_id, device=device)
    prev_cut_mask = torch.tensor([1.0 if prev_cut_row is not None else 0.0], dtype=torch.float32, device=device)
    next_cut_mask = torch.tensor([1.0 if next_cut_row is not None else 0.0], dtype=torch.float32, device=device)
    prev_cut_delta = torch.tensor(
        [float(int(full_frame) - int(prev_cut_frame)) if prev_cut_frame is not None else 0.0],
        dtype=torch.float32,
        device=device,
    )
    next_cut_delta = torch.tensor(
        [float(int(next_cut_frame) - int(full_frame)) if next_cut_frame is not None else 0.0],
        dtype=torch.float32,
        device=device,
    )
    key_position_delta, key_position_mask = _position_tensors(
        current_frame=full_frame,
        frames=selected_frames,
        max_count=max_position_tokens,
        radius=window_radius,
        mode=position_context_mode,
        side_count=position_context_side_count,
        device=device,
    )
    cut_position_delta, cut_position_mask = _position_tensors(
        current_frame=full_frame,
        frames=cut_transition_frames,
        max_count=max_position_tokens,
        radius=window_radius,
        mode=position_context_mode,
        side_count=position_context_side_count,
        device=device,
    )
    history_labels, history_mask, history_delta = _history_tensors(
        current_frame=full_frame,
        predicted_rows_by_frame=predicted_rows_by_frame,
        fields=fields,
        field_to_id=field_to_id,
        history_size=history_size,
        device=device,
    )

    with torch.no_grad():
        outputs = model(
            audio_batch,
            dance_batch,
            stage="cut" if stage == "cut" else "keyframe",
            prev_cut_labels=prev_cut_labels,
            prev_cut_mask=prev_cut_mask,
            prev_cut_delta=prev_cut_delta,
            next_cut_labels=next_cut_labels,
            next_cut_mask=next_cut_mask,
            next_cut_delta=next_cut_delta,
            key_position_delta=key_position_delta,
            key_position_mask=key_position_mask,
            cut_position_delta=cut_position_delta,
            cut_position_mask=cut_position_mask,
            history_labels=history_labels,
            history_mask=history_mask,
            history_delta=history_delta,
    )
    row: dict[str, object] = {"frame": int(full_frame)}
    temperature = max(1e-6, float(logit_temperature))
    use_distance_fov_joint = (
        bool(decode_distance_fov_joint)
        and "distance" in fields
        and "fov" in fields
        and "distance_fov_logits" in outputs
    )
    for field in fields:
        if use_distance_fov_joint and field in {"distance", "fov"}:
            continue
        logits = outputs[f"{field}_logits"] / temperature
        if field in calibration_biases:
            logits = logits + calibration_biases[field]
        token_id = int(logits.argmax(dim=-1).detach().cpu().numpy()[0])
        row[field] = str(vocabularies[field][token_id])
    if use_distance_fov_joint:
        distance_values = vocabularies["distance"]
        fov_values = vocabularies["fov"]
        joint_logits = outputs["distance_fov_logits"] / temperature
        joint_id = int(joint_logits.argmax(dim=-1).detach().cpu().numpy()[0])
        distance_id = joint_id // max(1, len(fov_values))
        fov_id = joint_id % max(1, len(fov_values))
        if 0 <= distance_id < len(distance_values) and 0 <= fov_id < len(fov_values):
            row["distance"] = str(distance_values[distance_id])
            row["fov"] = str(fov_values[fov_id])
        else:
            # Fallback should never trigger for a matching checkpoint, but keeps
            # older or partially incompatible checkpoints usable.
            for field in ("distance", "fov"):
                logits = outputs[f"{field}_logits"] / temperature
                if field in calibration_biases:
                    logits = logits + calibration_biases[field]
                token_id = int(logits.argmax(dim=-1).detach().cpu().numpy()[0])
                row[field] = str(vocabularies[field][token_id])
    row["transition"] = "linear"
    row["stage"] = "cut" if stage == "cut" else "keyframe"
    return row


def _predict_for_clip(
    *,
    model: V3CutContextShotPlanModel,
    manifest: dict[str, Any],
    norm_stats: dict[str, np.ndarray],
    feature_path: Path,
    full_frames: list[int],
    cut_pairs: list[dict[str, object]],
    source_start_frame: int,
    source_end_frame: int,
    device: torch.device,
    logit_temperature: float,
    calibration_biases: dict[str, torch.Tensor],
    decode_distance_fov_joint: bool,
) -> list[dict[str, object]]:
    audio, dance = _prepare_features(feature_path, manifest=manifest, norm_stats=norm_stats)
    selected_frames = [
        int(frame)
        for frame in sorted(set(full_frames))
        if int(source_start_frame) <= int(frame) <= int(source_end_frame)
    ]
    if int(source_start_frame) not in selected_frames:
        selected_frames.insert(0, int(source_start_frame))
    if int(source_end_frame) not in selected_frames:
        selected_frames.append(int(source_end_frame))
    selected_frames = sorted(set(selected_frames))

    cut_transition_frames = sorted(
        {
            int(pair["source_frames"][1])
            for pair in cut_pairs
            if len(pair.get("source_frames", [])) >= 2
            and int(source_start_frame) <= int(pair["source_frames"][1]) <= int(source_end_frame)
        }
    )
    predicted_rows_by_frame: dict[int, dict[str, object]] = {}
    predicted_cut_rows_by_frame: dict[int, dict[str, object]] = {}

    model.eval()
    for frame in cut_transition_frames:
        row = _predict_one(
            model=model,
            stage="cut",
            audio=audio,
            dance=dance,
            full_frame=frame,
            source_start_frame=source_start_frame,
            selected_frames=selected_frames,
            cut_transition_frames=cut_transition_frames,
            predicted_rows_by_frame=predicted_rows_by_frame,
            predicted_cut_rows_by_frame=predicted_cut_rows_by_frame,
            manifest=manifest,
            device=device,
            logit_temperature=logit_temperature,
            calibration_biases=calibration_biases,
            decode_distance_fov_joint=decode_distance_fov_joint,
        )
        predicted_rows_by_frame[int(frame)] = row
        predicted_cut_rows_by_frame[int(frame)] = row

    for frame in selected_frames:
        if int(frame) in predicted_rows_by_frame:
            continue
        row = _predict_one(
            model=model,
            stage="keyframe",
            audio=audio,
            dance=dance,
            full_frame=frame,
            source_start_frame=source_start_frame,
            selected_frames=selected_frames,
            cut_transition_frames=cut_transition_frames,
            predicted_rows_by_frame=predicted_rows_by_frame,
            predicted_cut_rows_by_frame=predicted_cut_rows_by_frame,
            manifest=manifest,
            device=device,
            logit_temperature=logit_temperature,
            calibration_biases=calibration_biases,
            decode_distance_fov_joint=decode_distance_fov_joint,
        )
        predicted_rows_by_frame[int(frame)] = row

    return [
        {key: value for key, value in predicted_rows_by_frame[int(frame)].items() if key not in REMOVED_OUTPUT_FIELDS}
        for frame in sorted(predicted_rows_by_frame)
    ]


def _write_vmd_from_plan(
    plan_path: Path,
    targets_path: Path,
    output_vmd: Path,
    debug_path: Path,
    solve_config: CameraSolveConfigV3 | None = None,
    stabilize_forward: bool = False,
) -> None:
    payload = _load_json(plan_path)
    provider = JsonFrameTargetProvider(targets_path, stabilize_forward=stabilize_forward)
    solved = _solve_plan_camera(payload, provider, solve_config)
    write_camera_vmd(solved, output_vmd, model_name="V3 Cut Context Shotplan")
    _write_json(debug_path, [asdict(key) for key in solved])


def _solve_plan_camera(
    payload: dict[str, Any],
    provider: JsonFrameTargetProvider,
    solve_config: CameraSolveConfigV3 | None = None,
) -> list[Any]:
    """Solve a shot-plan payload into camera keys; the web editor reuses this with a cached provider."""

    keyframe_rows = [row for row in payload.get("keyframes", []) if isinstance(row, dict)]
    if uses_close_dense_distance_schema(payload):
        apply_close_dense_distance_schema_from_payload(payload)
        keyframe_rows = [internalize_close_dense_row(row) for row in keyframe_rows]
    requests = [ShotKeyframeRequestV3.from_dict(row) for row in keyframe_rows]
    return solve_camera_timeline_v3(provider, requests, solve_config or CameraSolveConfigV3())


def _close_dense_distance_schema_for_manifest(manifest: dict[str, Any]) -> dict[str, object] | None:
    vocab = manifest.get("vocabularies", {})
    distance_values = vocab.get("distance", []) if isinstance(vocab, dict) else []
    if "very_close" not in {str(value) for value in distance_values}:
        return None
    multipliers = {
        "extreme_close": 0.55,
        "very_close": 0.80,
        "close": 1.20,
        "medium_close": 1.75,
        "medium": 2.55,
        "medium_far": 3.70,
        "far": 5.80,
    }
    provenance = "V3.1 close-dense default"
    manifest_text = " ".join(
        str(manifest.get(key, ""))
        for key in ("dataset_root", "label_root")
    ).lower()
    if "v3_1_1" in manifest_text or "vc095_close125" in manifest_text:
        multipliers["very_close"] = 0.95
        multipliers["close"] = 1.25
        provenance = "V3.1.1 close-dense calibration (very_close=0.95, close=1.25)"
    return {
        "name": "close_dense_no_wide",
        "exported_values": list(EXPORTED_DISTANCE_VALUES),
        "body_height_multipliers": multipliers,
        "note": (
            "V3.1 close-dense labels are stored in JSON. During VMD solving, "
            "they are mapped to legacy internal V3 distance enum slots. "
            f"{provenance}."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer ReverseV3 cut-context model on DCM clip timing.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--dcm-manifest", type=Path, default=DEFAULT_DCM_MANIFEST)
    parser.add_argument("--timing-root", type=Path, default=DEFAULT_TIMING_ROOT)
    parser.add_argument("--timing-name", default="camera_poses_timing.json")
    parser.add_argument("--targets-root", type=Path, default=WORKSPACE_ROOT / "camera_autoshot" / "reverses")
    parser.add_argument("--dcm-summary-csv", type=Path, default=WORKSPACE_ROOT / "saba_mmd" / "build" / "Release" / "camera_dcm_render_result.csv")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--indices", default="3,8,20,77")
    parser.add_argument("--clip-split", default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--render-width", type=int, default=0)
    parser.add_argument("--render-height", type=int, default=0)
    parser.add_argument("--viewer-session-size", type=int, default=0)
    parser.add_argument("--force-render", action="store_true")
    parser.add_argument("--vmd-only", action="store_true")
    parser.add_argument(
        "--logit-temperature",
        type=float,
        default=1.0,
        help="Decode with logits divided by this temperature before argmax.",
    )
    parser.add_argument(
        "--train-prior-alpha",
        type=float,
        default=0.0,
        help=(
            "Apply alpha * log(train_label_prior / reference_prediction_prior) "
            "as a field-wise decoding bias. Use with --calibration-reference-root."
        ),
    )
    parser.add_argument(
        "--calibration-reference-root",
        type=Path,
        default=None,
        help="Optional previous inference root used to estimate prediction priors for calibration.",
    )
    parser.add_argument(
        "--decode-distance-fov-joint",
        action="store_true",
        help=(
            "Decode distance/fov from the auxiliary joint distance_fov head "
            "instead of independent field argmaxes when the checkpoint has it."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = _device_from_arg(str(args.device))
    run_dir = args.run_dir.resolve()
    checkpoint_path = run_dir / str(args.checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    manifest = _checkpoint_manifest(checkpoint)
    label_fields = [str(field) for field in manifest["label_fields"] if str(field) not in REMOVED_OUTPUT_FIELDS]
    model = _build_cut_context_model(checkpoint).to(device)
    norm_stats = load_normalization_stats(run_dir / "normalization_stats.npz")
    calibration_biases = _build_calibration_biases(
        manifest=manifest,
        reference_root=args.calibration_reference_root.resolve() if args.calibration_reference_root else None,
        alpha=float(args.train_prior_alpha),
        device=device,
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dcm_manifest = _load_json(args.dcm_manifest.resolve())
    clip_rows_by_index = _clip_rows_by_source_index(dcm_manifest)
    dcm_render_rows = load_dcm_rows(args.dcm_summary_csv.resolve())

    variants = _load_render_variants_module()
    batch = variants.load_batch_module()
    default_model_path = Path(batch.MODEL_PATH)
    mapping = variants.load_mapping(batch)
    latest_ok_rows = variants.load_latest_ok_rows()

    manifest_rows: list[dict[str, object]] = []
    render_jobs: list[dict[str, object]] = []
    for index in _parse_indices(str(args.indices)):
        clips = _select_dcm_clips_for_index(clip_rows_by_index.get(index, []), split=str(args.clip_split))
        if not clips:
            manifest_rows.append({"index": index, "status": "missing-dcm-clip"})
            continue
        timing_path = args.timing_root.resolve() / f"c{index}" / str(args.timing_name)
        targets_path = args.targets_root.resolve() / f"c{index}" / "real_targets.json"
        if not timing_path.exists():
            manifest_rows.append({"index": index, "status": "missing-timing", "timing": str(timing_path)})
            continue
        if not targets_path.exists():
            manifest_rows.append({"index": index, "status": "missing-targets", "targets": str(targets_path)})
            continue

        timing_payload = _load_json(timing_path)
        timing_rows = [row for row in timing_payload.get("keyframes", []) if isinstance(row, dict) and "frame" in row]
        source_frames, timing_debug, cut_pairs = _timing_rows_to_source_frames(
            timing_rows,
            timing_sample_fps=float(timing_payload.get("sample_fps", 5.0)),
            timeline_fps=30.0,
        )

        for clip in clips:
            clip_id = str(clip["id"])
            source_start = int(clip["source_start_frame"])
            source_end = int(clip["source_end_frame"])
            feature_path = args.feature_root.resolve() / f"{clip_id}.npz"
            item_dir = output_root / f"c{index}"
            item_dir.mkdir(parents=True, exist_ok=True)
            suffix = "" if len(clips) == 1 else f"_{clip_id}"
            stem = f"camera_poses_v3_cutctx{suffix}"
            plan_path = item_dir / f"{stem}.json"
            timing_debug_path = item_dir / f"{stem}_timing_debug.json"
            vmd_path = item_dir / f"{stem}.vmd"
            debug_path = item_dir / f"{stem}_debug.json"
            mp4_path = item_dir / f"{stem}.mp4"
            row: dict[str, object] = {
                "index": index,
                "clip_id": clip_id,
                "status": "",
                "official_split": str(clip.get("official_split", "")),
                "source_start_frame": source_start,
                "source_end_frame": source_end,
                "feature": str(feature_path),
                "timing": str(timing_path),
                "plan": str(plan_path),
                "vmd": str(vmd_path),
                "mp4": str(mp4_path),
            }
            if not feature_path.exists():
                row["status"] = "missing-feature"
                manifest_rows.append(row)
                continue

            cut_stats_path = item_dir / f"{stem}_cut_change_stats.json"
            keyframes = _predict_for_clip(
                model=model,
                manifest=manifest,
                norm_stats=norm_stats,
                feature_path=feature_path,
                full_frames=source_frames,
                cut_pairs=cut_pairs,
                source_start_frame=source_start,
                source_end_frame=source_end,
                device=device,
                logit_temperature=float(args.logit_temperature),
                calibration_biases=calibration_biases,
                decode_distance_fov_joint=bool(args.decode_distance_fov_joint),
            )
            distance_schema = _close_dense_distance_schema_for_manifest(manifest)
            plan_payload = {
                "schema": (
                    "camera_autoshot.reverse_v3.close_dense_no_wide"
                    if distance_schema is not None
                    else "camera_autoshot.reverse_v3"
                ),
                "source_model": str(checkpoint_path),
                "clip_id": clip_id,
                "source_song_id": f"c{index}",
                "source_start_frame": source_start,
                "source_end_frame": source_end,
                "sample_fps": 30.0,
                "timing_source": str(timing_path),
                "decode_distance_fov_joint": bool(args.decode_distance_fov_joint),
                "inference_note": "Two-stage ReverseV3 discrete cut-context inference: cut-transition frames first, then ordinary keyframes.",
                "keyframes": keyframes,
            }
            if distance_schema is not None:
                plan_payload["distance_schema"] = distance_schema
            _write_json(plan_path, plan_payload)
            _write_json(timing_debug_path, {"converted_timing": timing_debug, "cut_pairs": cut_pairs})
            _write_vmd_from_plan(plan_path, targets_path, vmd_path, debug_path)
            cut_stats = _cut_change_stats(
                keyframes=keyframes,
                cut_pairs=cut_pairs,
                source_start_frame=source_start,
                source_end_frame=source_end,
                fields=label_fields,
            )
            _write_json(cut_stats_path, cut_stats)
            row.update(
                {
                    "cut_pair_count": int(cut_stats["cut_pair_count"]),
                    "changed_pair_count": int(cut_stats["changed_pair_count"]),
                    "changed_pair_rate": float(cut_stats["changed_pair_rate"]),
                    "cut_stats": str(cut_stats_path),
                }
            )
            if args.vmd_only:
                row["status"] = "ok-vmd-only"
                print(f"[ok] c{index}/{clip_id}: {vmd_path.name} (vmd-only)")
                manifest_rows.append(row)
                continue

            matched_rows = [
                candidate
                for candidate in dcm_render_rows.get(index, [])
                if int(candidate["segment_start_frame"]) == source_start
                and int(candidate["segment_end_frame"]) == source_end
            ]
            if not matched_rows:
                matched_rows = [{"segment_start_frame": source_start, "segment_end_frame": source_end, "motion": "", "wav": "", "output": ""}]
            try:
                model_path, motion_path, wav_path = resolve_render_assets(
                    variants=variants,
                    batch=batch,
                    mapping=mapping,
                    latest_ok_rows=latest_ok_rows,
                    index=index,
                    matched_rows=matched_rows,
                    override_model=None,
                    default_model_path=default_model_path,
                )
                batch.MODEL_PATH = model_path
                render_stats = _compute_partial_render_stats(
                    batch=batch,
                    motion=motion_path,
                    camera=vmd_path,
                    wav=wav_path,
                    clip_start_frame=source_start,
                    clip_end_frame=source_end,
                )
                row["motion"] = str(motion_path)
                row["wav"] = str(wav_path)
                row["model"] = str(model_path)
                row["clip_start_frame"] = int(render_stats["segment_start_frame"])
                row["clip_end_frame"] = int(render_stats["segment_end_frame"])
                row["segment_frame_count"] = int(render_stats["segment_frame_count"])
                row["segment_duration_sec"] = float(render_stats["segment_duration_sec"])
                if not bool(args.force_render) and mp4_path.exists() and mp4_path.stat().st_size > 0:
                    row["status"] = "ok-existing"
                    print(f"[skip] c{index}/{clip_id}: {mp4_path.name} already exists")
                else:
                    row["status"] = "planned-render"
                    render_jobs.append(
                        {
                            "manifest_row": row,
                            "index": index,
                            "clip_id": clip_id,
                            "model_path": model_path,
                            "motion_path": motion_path,
                            "wav_path": wav_path,
                            "vmd_path": vmd_path,
                            "output_path": mp4_path,
                            "render_stats": render_stats,
                        }
                    )
            except Exception as exc:
                row["status"] = "failed-render"
                row["error"] = str(exc)
                print(f"[fail] c{index}/{clip_id}: {exc}")
            manifest_rows.append(row)

    _render_partial_jobs_batch(
        batch=batch,
        jobs=render_jobs,
        default_model_path=default_model_path,
        viewer_session_size=int(args.viewer_session_size),
        render_width=int(args.render_width),
        render_height=int(args.render_height),
    )
    _write_json(output_root / "manifest.json", manifest_rows)
    _write_manifest_csv(manifest_rows, output_root / "manifest.csv")
    total = sum(int(row.get("cut_pair_count", 0) or 0) for row in manifest_rows)
    changed = sum(int(row.get("changed_pair_count", 0) or 0) for row in manifest_rows)
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "output_root": str(output_root),
        "clip_count": len([row for row in manifest_rows if str(row.get("status", "")).startswith("ok")]),
        "cut_pair_count": total,
        "changed_pair_count": changed,
        "changed_pair_rate": (changed / total) if total else 0.0,
    }
    _write_json(output_root / "cut_change_summary.json", summary)
    print(f"output root: {output_root}")
    print(f"cut change: {changed}/{total} = {summary['changed_pair_rate']*100:.2f}%")


if __name__ == "__main__":
    main()
