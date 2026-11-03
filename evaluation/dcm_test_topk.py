"""Export top-k shot-plan candidates per event for the DCM test clips (input of CP-SAT).

Each event gets the k best joint plans (sum of per-field log-probabilities); the top-1 plan is fed
back as context, as in greedy decoding.

    python -m evaluation.dcm_test_topk --plan-root <dcm_test_plans output> --output-root <dir>
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.dcm_common import (
    DEFAULT_FEATURE_ROOT,
    DEFAULT_SHOTPLAN_RUN_DIR,
    DEFAULT_TIMING_FILENAME,
    DEFAULT_TIMING_ROOT,
    TOPK_HISTORY_SIZE,
    WINDOW_RADIUS,
    build_feature_windows,
    device_from_arg,
    distance_schema_for_vocabularies,
    external_timing_for_clip,
    feature_manifest,
    fields_and_vocabularies,
    load_json,
    make_model,
    source_index,
    write_json,
)
from shotplan_transformer.sequence_decode import (
    _history_tensors,
    _nearest_cut_rows,
    _position_tensors,
    _single_label_batch,
)


def candidate_beam(
    *,
    outputs: dict[str, torch.Tensor],
    fields: tuple[str, ...],
    vocabularies: dict[str, list[str]],
    top_k: int,
    stage: str,
) -> list[dict[str, Any]]:
    beams: list[tuple[float, dict[str, str], dict[str, float]]] = [(0.0, {}, {})]
    for field in fields:
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
        row = {field: str(plan[field]) for field in fields}
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


def cut_metadata(source_frames: list[int], cut_pairs: list[dict[str, Any]]) -> dict[int, dict[str, object]]:
    metadata: dict[int, dict[str, object]] = {
        int(frame): {
            "timing_frame": int(frame),
            "duplicate_offset": 0,
            "duplicate_count": 1,
            "cut_flag": False,
        }
        for frame in source_frames
    }
    for pair_index, pair in enumerate(cut_pairs):
        pair_frames = [int(frame) for frame in pair.get("source_frames", [])]
        for offset, frame in enumerate(pair_frames):
            metadata.setdefault(int(frame), {})
            metadata[int(frame)].update(
                {
                    "cut_pair_id": f"{pair.get('timing_frame', frame)}:{pair_index}",
                    "cut_pair_role": "first" if offset == 0 else "second",
                    "cut_flag": True,
                    "timing_frame": int(pair.get("timing_frame", frame)),
                    "duplicate_offset": int(offset),
                    "duplicate_count": int(pair.get("duplicate_count", len(pair_frames))),
                }
            )
    return metadata


def converted_timing(source_frames: list[int], metadata: dict[int, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame in source_frames:
        meta = metadata.get(int(frame), {})
        rows.append(
            {
                "source_frame": int(frame),
                "timing_frame": int(meta.get("timing_frame", frame)),
                "duplicate_offset": int(meta.get("duplicate_offset", 0)),
                "duplicate_count": int(meta.get("duplicate_count", 1)),
                "cut_flag": bool(meta.get("cut_flag", False)),
            }
        )
    return rows


def export_candidates_for_row(
    *,
    row: dict[str, Any],
    timing_by_song: dict[str, Any],
    source_manifest: dict[str, Any],
    model: torch.nn.Module,
    stats: dict[str, np.ndarray],
    fields: tuple[str, ...],
    vocabularies: dict[str, list[str]],
    distance_schema: dict[str, object] | None,
    feature_root: Path,
    output_root: Path,
    checkpoint_path: Path,
    top_k: int,
    history_size: int,
    device: torch.device,
    include_boundaries: bool,
) -> dict[str, object]:
    clip_id = str(row["clip_id"])
    source_song_id = str(row.get("source_song_id") or f"c{int(row['index'])}")
    index = source_index(source_song_id)
    source_start = int(row["source_start_frame"])
    source_end = int(row["source_end_frame"])
    feature_path = feature_root / f"{clip_id}.npz"
    source_plan = Path(str(row.get("plan", "")))
    item_dir = output_root / f"c{index}"
    output_path = item_dir / f"{source_plan.stem}_top{int(top_k)}_candidates.json"

    out: dict[str, object] = {
        "index": index,
        "clip_id": clip_id,
        "source_plan": str(source_plan),
        "output": str(output_path),
        "status": "",
    }
    if not source_plan.exists():
        out["status"] = "missing-plan"
        return out
    if not feature_path.exists():
        out["status"] = "missing-feature"
        return out

    source_frames, cut_pairs, timing_counts = external_timing_for_clip(
        timing_by_song,
        source_song_id=source_song_id,
        source_start_frame=source_start,
        source_end_frame=source_end,
        include_boundaries=include_boundaries,
    )
    if len(source_frames) < 2:
        out["status"] = "too-few-events"
        return out

    source_frames_np = np.asarray(source_frames, dtype=np.int32)
    local_frames = (source_frames_np - int(source_start)).astype(np.int32)
    audio, dance = build_feature_windows(
        feature_path=feature_path,
        source_manifest=source_manifest,
        local_frames=local_frames,
        window_radius=WINDOW_RADIUS,
        stats=stats,
    )

    cut_mask = np.zeros((local_frames.shape[0],), dtype=bool)
    if local_frames.shape[0] > 1:
        cut_mask[1:] = np.diff(local_frames) == 1
    cut_frames = local_frames[cut_mask]
    generation_indices = np.concatenate([np.flatnonzero(cut_mask), np.flatnonzero(~cut_mask)])
    predicted_rows: dict[int, dict[str, int]] = {}
    predicted_cut_rows: dict[int, dict[str, int]] = {}
    frame_meta = cut_metadata(source_frames, cut_pairs)
    exported_by_frame: dict[int, dict[str, Any]] = {}
    max_position_tokens = int(model.config.max_position_tokens)

    model.eval()
    with torch.no_grad():
        for event_index in generation_indices.tolist():
            local_frame = int(local_frames[event_index])
            full_frame = int(source_frames_np[event_index])
            stage = "cut" if bool(cut_mask[event_index]) else "keyframe"
            prev_cut_frame, prev_cut_row, next_cut_frame, next_cut_row = _nearest_cut_rows(
                local_frame,
                cut_frames,
                predicted_cut_rows,
            )
            prev_cut_labels = _single_label_batch(prev_cut_row, fields=fields, device=device)
            next_cut_labels = _single_label_batch(next_cut_row, fields=fields, device=device)
            prev_cut_mask = torch.tensor([1.0 if prev_cut_row is not None else 0.0], dtype=torch.float32, device=device)
            next_cut_mask = torch.tensor([1.0 if next_cut_row is not None else 0.0], dtype=torch.float32, device=device)
            prev_cut_delta = torch.tensor(
                [float(local_frame - int(prev_cut_frame)) if prev_cut_frame is not None else 0.0],
                dtype=torch.float32,
                device=device,
            )
            next_cut_delta = torch.tensor(
                [float(int(next_cut_frame) - local_frame) if next_cut_frame is not None else 0.0],
                dtype=torch.float32,
                device=device,
            )
            key_delta, key_mask = _position_tensors(
                local_frame,
                local_frames,
                max_count=max_position_tokens,
                radius=WINDOW_RADIUS,
                device=device,
            )
            cut_delta, cut_pos_mask = _position_tensors(
                local_frame,
                cut_frames,
                max_count=max_position_tokens,
                radius=WINDOW_RADIUS,
                device=device,
            )
            history_labels, history_mask, history_delta = _history_tensors(
                local_frame,
                predicted_rows,
                fields=fields,
                history_size=history_size,
                device=device,
            )
            outputs = model(
                audio[event_index : event_index + 1].to(device),
                dance[event_index : event_index + 1].to(device),
                stage=stage,
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
            candidates = candidate_beam(
                outputs=outputs,
                fields=fields,
                vocabularies=vocabularies,
                top_k=top_k,
                stage=stage,
            )
            top1_plan = dict(candidates[0]["plan"]) if candidates else {}
            top1_ids = {field: int(vocabularies[field].index(str(top1_plan[field]))) for field in fields}
            predicted_rows[local_frame] = top1_ids
            if stage == "cut":
                predicted_cut_rows[local_frame] = top1_ids

            meta = dict(frame_meta.get(full_frame, {}))
            meta.setdefault("duplicate_count", 1)
            meta.setdefault("duplicate_offset", 0)
            meta.setdefault("cut_flag", bool(meta.get("cut_pair_id")))
            exported_by_frame[full_frame] = {
                "frame": full_frame,
                "time_sec": float((full_frame - source_start) / 30.0),
                "slot_count": 1,
                "stage": stage,
                "slots": [{"slot_index": 0, "top1": top1_plan, "candidates": candidates}],
                **meta,
            }

    source_payload = load_json(source_plan)
    payload: dict[str, Any] = {
        "schema": str(source_payload.get("schema", "camera_autoshot.reverse_v3")),
        "source_model": str(checkpoint_path),
        "model_checkpoint": str(checkpoint_path),
        "source_plan": str(source_plan),
        "clip_id": clip_id,
        "source_song_id": source_song_id,
        "source_start_frame": source_start,
        "source_end_frame": source_end,
        "sample_fps": float(source_payload.get("sample_fps", 30.0)),
        "timeline_fps": float(source_payload.get("timeline_fps", 30.0)),
        "timing_source": str(source_payload.get("timing_source", "")),
        "label_fields": list(fields),
        "top_k": int(top_k),
        "score_definition": "sum of per-field log_softmax probabilities; transition is fixed to linear",
        "converted_timing": converted_timing(source_frames, frame_meta),
        "cut_pairs": cut_pairs,
        "frames": [exported_by_frame[frame] for frame in sorted(exported_by_frame)],
    }
    if distance_schema is not None:
        payload["distance_schema"] = distance_schema
    write_json(output_path, payload)
    out.update(
        {
            "status": "ok",
            "frame_bucket_count": len(payload["frames"]),
            "candidate_count": sum(len(slot["candidates"]) for frame in payload["frames"] for slot in frame["slots"]),
            "cut_pair_count": len(cut_pairs),
            **timing_counts,
        }
    )
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export top-k shot-plan candidates for the DCM test clips.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_SHOTPLAN_RUN_DIR)
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--timing-root", type=Path, default=DEFAULT_TIMING_ROOT)
    parser.add_argument("--timing-filename", default=DEFAULT_TIMING_FILENAME)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--history-size", type=int, default=TOPK_HISTORY_SIZE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--include-boundaries", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = device_from_arg(str(args.device))
    checkpoint_path = Path(args.run_dir).resolve() / str(args.checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    fields, vocabularies = fields_and_vocabularies(checkpoint)
    model = make_model(checkpoint, device)
    distance_schema = distance_schema_for_vocabularies(vocabularies)
    feature_root = Path(args.feature_root).resolve()
    source_manifest = feature_manifest(feature_root)
    timing_by_song = load_json(Path(args.timing_root).resolve() / str(args.timing_filename))
    plan_root = Path(args.plan_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = load_json(plan_root / "manifest.json")

    rows: list[dict[str, object]] = []
    for manifest_row in manifest_rows:
        if not isinstance(manifest_row, dict):
            continue
        if not str(manifest_row.get("status", "")).startswith("ok"):
            rows.append(dict(manifest_row))
            continue
        row = export_candidates_for_row(
            row=dict(manifest_row),
            timing_by_song=timing_by_song,
            source_manifest=source_manifest,
            model=model,
            stats=checkpoint["stats"],
            fields=fields,
            vocabularies=vocabularies,
            distance_schema=distance_schema,
            feature_root=feature_root,
            output_root=output_root,
            checkpoint_path=checkpoint_path,
            top_k=int(args.top_k),
            history_size=int(args.history_size),
            device=device,
            include_boundaries=bool(args.include_boundaries),
        )
        rows.append(row)
        print(f"[{row.get('status')}] {row.get('clip_id')}: candidates={row.get('candidate_count', 0)}", flush=True)
    write_json(output_root / "topk_export_manifest.json", rows)
    failed = [row for row in rows if row.get("status") != "ok"]
    if failed:
        raise SystemExit(f"{len(failed)} clip(s) failed; see topk_export_manifest.json")


if __name__ == "__main__":
    main()
