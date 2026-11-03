from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera_autoshot.reverse_v3.close_dense_schema import EXPORTED_DISTANCE_VALUES

from shotplan_transformer.dataset import load_manifest, load_normalization_stats
from shotplan_transformer.infer_to_plans import _device_from_arg, _window_indices
from shotplan_transformer.infer_v3_cut_context_dcm_from_timing import (
    REMOVED_OUTPUT_FIELDS,
    _build_calibration_biases,
    _build_cut_context_model,
    _checkpoint_manifest,
    _empty_label_batch,
    _history_tensors,
    _position_tensors,
    _single_label_batch,
)
from shotplan_transformer.infer_v3_from_timing import _prepare_features
from shotplan_transformer.v3_cut_context_model import V3CutContextShotPlanModel


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_ROOT = (
    WORKSPACE_ROOT
    / "music_processing"
    / "results"
    / "01_datasets"
    / "cut_keyframe_multimodal_dataset_dcm"
    / "songs"
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _candidate_plan_paths(source_root: Path, plan_glob: str) -> list[Path]:
    blocked = (
        "_debug.json",
        "_timing_debug.json",
        "_cut_change_stats.json",
        "_optimizer_debug.json",
    )
    paths: list[Path] = []
    for path in sorted(source_root.glob(f"c*/{plan_glob}")):
        name = path.name
        if any(name.endswith(suffix) for suffix in blocked):
            continue
        if "_top" in name or "_optimized" in name:
            continue
        paths.append(path)
    return paths


def _output_path_for_plan(source_root: Path, output_root: Path, plan_path: Path, top_k: int) -> Path:
    relative = plan_path.relative_to(source_root)
    stem = plan_path.stem
    return output_root / relative.parent / f"{stem}_top{int(top_k)}_candidates.json"


def _timing_debug_path_for_plan(plan_path: Path) -> Path:
    return plan_path.with_name(f"{plan_path.stem}_timing_debug.json")


def _close_dense_distance_schema(payload: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any] | None:
    if "distance_schema" in payload:
        return payload["distance_schema"]
    distance_values = {
        str(value)
        for value in manifest.get("vocabularies", {}).get("distance", [])
    }
    if "very_close" not in distance_values:
        return None
    return {
        "name": "close_dense_no_wide",
        "exported_values": list(EXPORTED_DISTANCE_VALUES),
        "note": (
            "V3.1 close-dense labels are stored in JSON. During VMD solving, "
            "they are mapped to legacy internal V3 distance enum slots."
        ),
    }


def _cut_metadata(timing_debug: dict[str, Any]) -> dict[int, dict[str, object]]:
    metadata: dict[int, dict[str, object]] = {}
    for row in timing_debug.get("converted_timing", []):
        if not isinstance(row, dict) or "source_frame" not in row:
            continue
        frame = int(row["source_frame"])
        metadata[frame] = {
            "timing_frame": int(row.get("timing_frame", frame)),
            "duplicate_offset": int(row.get("duplicate_offset", 0)),
            "duplicate_count": int(row.get("duplicate_count", 1)),
        }
    for pair_index, pair in enumerate(timing_debug.get("cut_pairs", [])):
        if not isinstance(pair, dict):
            continue
        source_frames = [int(frame) for frame in pair.get("source_frames", [])]
        for offset, frame in enumerate(source_frames):
            metadata.setdefault(frame, {})
            metadata[frame].update(
                {
                    "cut_pair_id": f"{pair.get('timing_frame', frame)}:{pair_index}",
                    "cut_pair_role": "first" if offset == 0 else "second",
                    "cut_flag": True,
                    "timing_frame": int(pair.get("timing_frame", frame)),
                    "duplicate_offset": int(offset),
                    "duplicate_count": int(pair.get("duplicate_count", len(source_frames))),
                }
            )
    return metadata


def _cut_transition_frames(timing_debug: dict[str, Any], source_start: int, source_end: int) -> list[int]:
    values = {
        int(pair["source_frames"][1])
        for pair in timing_debug.get("cut_pairs", [])
        if isinstance(pair, dict)
        and len(pair.get("source_frames", [])) >= 2
        and int(source_start) <= int(pair["source_frames"][1]) <= int(source_end)
    }
    return sorted(values)


def _model_outputs_one(
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
    checkpoint_args: dict[str, Any],
    device: torch.device,
    logit_temperature: float,
    calibration_biases: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
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
    position_context_mode = str(manifest.get("position_context_mode", checkpoint_args.get("position_context_mode", "radius")))
    position_context_side_count = int(
        manifest.get("position_context_side_count", checkpoint_args.get("position_context_side_count", 8))
    )

    local_frame = int(full_frame) - int(source_start_frame)
    center = min(max(int(local_frame), 0), int(audio.shape[0]) - 1)
    indices = _window_indices(center, length=int(audio.shape[0]), radius=window_radius, stride=window_stride)
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

    temperature = max(1e-6, float(logit_temperature))
    adjusted: dict[str, torch.Tensor] = {}
    for field in fields:
        logits = outputs[f"{field}_logits"] / temperature
        if field in calibration_biases:
            logits = logits + calibration_biases[field]
        adjusted[f"{field}_logits"] = logits
    return adjusted


def _candidate_beam(
    *,
    outputs: dict[str, torch.Tensor],
    manifest: dict[str, Any],
    stage: str,
    top_k: int,
) -> list[dict[str, Any]]:
    vocabularies = {
        str(field): [str(value) for value in values]
        for field, values in manifest["vocabularies"].items()
    }
    label_fields = [str(field) for field in manifest["label_fields"] if str(field) not in REMOVED_OUTPUT_FIELDS]
    beams: list[tuple[float, dict[str, str], dict[str, float]]] = [(0.0, {}, {})]
    for field in label_fields:
        logits = outputs[f"{field}_logits"][0]
        log_probs = F.log_softmax(logits, dim=-1)
        field_k = min(int(top_k), int(log_probs.numel()))
        values, indices = torch.topk(log_probs, k=field_k)
        next_beams: list[tuple[float, dict[str, str], dict[str, float]]] = []
        for base_logprob, base_plan, base_field_logprobs in beams:
            for value, token_id in zip(values.tolist(), indices.tolist()):
                token = str(vocabularies[field][int(token_id)])
                plan = dict(base_plan)
                field_logprobs = dict(base_field_logprobs)
                plan[field] = token
                field_logprobs[field] = float(value)
                next_beams.append((base_logprob + float(value), plan, field_logprobs))
        next_beams.sort(key=lambda item: item[0], reverse=True)
        beams = next_beams[: int(top_k)]

    candidates: list[dict[str, Any]] = []
    for rank, (logprob, plan, field_logprobs) in enumerate(beams, start=1):
        row = {field: str(plan[field]) for field in label_fields}
        row["transition"] = "linear"
        row["stage"] = "cut" if stage == "cut" else "keyframe"
        candidates.append(
            {
                "rank": int(rank),
                "logprob": float(logprob),
                "prob": float(math.exp(logprob)),
                "plan": row,
                "field_logprobs": field_logprobs,
            }
        )
    return candidates


def _export_plan_candidates(
    *,
    plan_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    model: V3CutContextShotPlanModel,
    manifest: dict[str, Any],
    norm_stats: dict[str, np.ndarray],
    feature_root: Path,
    device: torch.device,
    top_k: int,
    logit_temperature: float,
    calibration_biases: dict[str, torch.Tensor],
) -> dict[str, Any]:
    payload = _load_json(plan_path)
    timing_debug_path = _timing_debug_path_for_plan(plan_path)
    timing_debug = _load_json(timing_debug_path) if timing_debug_path.exists() else {"converted_timing": [], "cut_pairs": []}
    source_start = int(payload["source_start_frame"])
    source_end = int(payload["source_end_frame"])
    clip_id = str(payload["clip_id"])
    feature_path = feature_root / f"{clip_id}.npz"
    if not feature_path.exists():
        raise FileNotFoundError(f"feature not found: {feature_path}")

    keyframes = [row for row in payload.get("keyframes", []) if isinstance(row, dict) and "frame" in row]
    selected_frames = sorted({int(row["frame"]) for row in keyframes})
    if source_start not in selected_frames:
        selected_frames.insert(0, source_start)
    if source_end not in selected_frames:
        selected_frames.append(source_end)
    selected_frames = sorted(set(selected_frames))
    cut_transition_frames = _cut_transition_frames(timing_debug, source_start, source_end)
    frame_meta = _cut_metadata(timing_debug)
    audio, dance = _prepare_features(feature_path, manifest=manifest, norm_stats=norm_stats)

    predicted_rows_by_frame: dict[int, dict[str, object]] = {}
    predicted_cut_rows_by_frame: dict[int, dict[str, object]] = {}
    exported_by_frame: dict[int, dict[str, Any]] = {}
    model.eval()
    checkpoint_config_path = checkpoint_path.parent / "config.json"
    checkpoint_args = (
        dict(_load_json(checkpoint_config_path).get("args", {}))
        if checkpoint_config_path.exists()
        else {}
    )

    def export_one(frame: int, stage: str) -> None:
        outputs = _model_outputs_one(
            model=model,
            stage=stage,
            audio=audio,
            dance=dance,
            full_frame=frame,
            source_start_frame=source_start,
            selected_frames=selected_frames,
            cut_transition_frames=cut_transition_frames,
            predicted_rows_by_frame=predicted_rows_by_frame,
            predicted_cut_rows_by_frame=predicted_cut_rows_by_frame,
            manifest=manifest,
            checkpoint_args=checkpoint_args,
            device=device,
            logit_temperature=logit_temperature,
            calibration_biases=calibration_biases,
        )
        candidates = _candidate_beam(outputs=outputs, manifest=manifest, stage=stage, top_k=top_k)
        top1 = dict(candidates[0]["plan"]) if candidates else {}
        predicted_rows_by_frame[int(frame)] = top1
        if stage == "cut":
            predicted_cut_rows_by_frame[int(frame)] = top1
        meta = dict(frame_meta.get(int(frame), {}))
        meta.setdefault("duplicate_count", 1)
        meta.setdefault("duplicate_offset", 0)
        meta.setdefault("cut_flag", bool(meta.get("cut_pair_id")))
        exported_by_frame[int(frame)] = {
            "frame": int(frame),
            "time_sec": float((int(frame) - source_start) / 30.0),
            "slot_count": 1,
            "stage": "cut" if stage == "cut" else "keyframe",
            "slots": [
                {
                    "slot_index": 0,
                    "top1": top1,
                    "candidates": candidates,
                }
            ],
            **meta,
        }

    for frame in cut_transition_frames:
        if frame in selected_frames:
            export_one(frame, "cut")
    for frame in selected_frames:
        if frame in predicted_rows_by_frame:
            continue
        export_one(frame, "keyframe")

    out_payload: dict[str, Any] = {
        "schema": str(payload.get("schema", "camera_autoshot.reverse_v3")),
        "source_model": str(checkpoint_path),
        "model_checkpoint": str(checkpoint_path),
        "source_plan": str(plan_path),
        "clip_id": clip_id,
        "source_song_id": str(payload.get("source_song_id", plan_path.parent.name)),
        "source_start_frame": source_start,
        "source_end_frame": source_end,
        "sample_fps": float(payload.get("sample_fps", 30.0)),
        "timeline_fps": 30.0,
        "timing_source": str(payload.get("timing_source", "")),
        "label_fields": [str(field) for field in manifest["label_fields"] if str(field) not in REMOVED_OUTPUT_FIELDS],
        "top_k": int(top_k),
        "score_definition": "sum of per-field log_softmax probabilities; transition is fixed to linear",
        "converted_timing": timing_debug.get("converted_timing", []),
        "cut_pairs": timing_debug.get("cut_pairs", []),
        "frames": [exported_by_frame[frame] for frame in sorted(exported_by_frame)],
    }
    distance_schema = _close_dense_distance_schema(payload, manifest)
    if distance_schema is not None:
        out_payload["distance_schema"] = distance_schema
    _write_json(output_path, out_payload)
    return {
        "plan": str(plan_path),
        "output": str(output_path),
        "clip_id": clip_id,
        "frame_bucket_count": len(out_payload["frames"]),
        "candidate_count": sum(len(slot["candidates"]) for frame in out_payload["frames"] for slot in frame["slots"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export top-k candidates for existing V3 cut-context result JSONs.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None, help="Defaults to --source-root.")
    parser.add_argument("--plan-glob", default="camera_poses_v3_cutctx*.json")
    parser.add_argument("--run-dir", type=Path, default=None, help="Defaults to the source_model parent from the first plan.")
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--logit-temperature", type=float, default=1.0)
    parser.add_argument("--train-prior-alpha", type=float, default=0.0)
    parser.add_argument("--calibration-reference-root", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_root = args.source_root.resolve()
    output_root = (args.output_root or args.source_root).resolve()
    plan_paths = _candidate_plan_paths(source_root, str(args.plan_glob))
    if not plan_paths:
        raise FileNotFoundError(f"No plan JSONs found under {source_root} matching {args.plan_glob!r}")

    first_payload = _load_json(plan_paths[0])
    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        checkpoint_path = run_dir / str(args.checkpoint_name)
    else:
        checkpoint_path = Path(str(first_payload["source_model"])).resolve()
        run_dir = checkpoint_path.parent

    device = _device_from_arg(str(args.device))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    manifest = _checkpoint_manifest(checkpoint)
    model = _build_cut_context_model(checkpoint).to(device)
    norm_stats = load_normalization_stats(run_dir / "normalization_stats.npz")
    calibration_biases = _build_calibration_biases(
        manifest=manifest,
        reference_root=args.calibration_reference_root.resolve() if args.calibration_reference_root else None,
        alpha=float(args.train_prior_alpha),
        device=device,
    )

    rows: list[dict[str, Any]] = []
    for plan_path in plan_paths:
        output_path = _output_path_for_plan(source_root, output_root, plan_path, int(args.top_k))
        try:
            row = _export_plan_candidates(
                plan_path=plan_path,
                output_path=output_path,
                checkpoint_path=checkpoint_path,
                model=model,
                manifest=manifest,
                norm_stats=norm_stats,
                feature_root=args.feature_root.resolve(),
                device=device,
                top_k=int(args.top_k),
                logit_temperature=float(args.logit_temperature),
                calibration_biases=calibration_biases,
            )
            row["status"] = "ok"
            print(f"[ok] {plan_path.parent.name}/{plan_path.stem}: frames={row['frame_bucket_count']}")
        except Exception as exc:
            row = {"plan": str(plan_path), "status": "failed", "error": str(exc)}
            print(f"[fail] {plan_path}: {exc}")
        rows.append(row)

    manifest_path = output_root / "v3_cutctx_topk_export_manifest.json"
    _write_json(manifest_path, rows)
    if rows:
        csv_path = manifest_path.with_suffix(".csv")
        fieldnames = sorted({key for row in rows for key in row})
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
