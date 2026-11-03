from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shotplan_transformer.infer_v3_cut_context_dcm_from_timing import _write_vmd_from_plan
from shotplan_transformer.v3_interpolation_stage import (
    V3InterpolationStageConfig,
    V3InterpolationStageModel,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS_ROOT = WORKSPACE_ROOT / "camera_autoshot" / "reverses"
DEFAULT_PLAN_NAME = "camera_poses_v3_cutctx.json"
DEFAULT_OUTPUT_PLAN_NAME = "camera_poses_v3_cutctx_interpstage.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply a trained interpolation stage to V3 shot-plan result folders.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan-name", default=DEFAULT_PLAN_NAME)
    parser.add_argument("--output-plan-name", default=DEFAULT_OUTPUT_PLAN_NAME)
    parser.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS_ROOT)
    parser.add_argument(
        "--actor-forward-stabilization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep the actor forward direction sign-stable over time when solving the camera VMDs.",
    )
    parser.add_argument("--device", default="auto")
    return parser


def _device_from_arg(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def _build_model(checkpoint: dict[str, Any]) -> V3InterpolationStageModel:
    allowed = {field.name for field in dataclass_fields(V3InterpolationStageConfig)}
    config = V3InterpolationStageConfig(**{k: v for k, v in checkpoint["config"].items() if k in allowed})
    model = V3InterpolationStageModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _label_id(row: dict[str, object] | None, field: str, value_to_id: dict[str, dict[str, int]]) -> int:
    pad_id = len(value_to_id[field])
    if row is None:
        return pad_id
    return value_to_id[field].get(str(row.get(field, "")), pad_id)


def _predict_interpolations(
    rows: list[dict[str, object]],
    *,
    model: V3InterpolationStageModel,
    fields: list[str],
    value_to_id: dict[str, dict[str, int]],
    interpolation_vocab: list[str],
    device: torch.device,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    ordered = sorted([dict(row) for row in rows if "frame" in row], key=lambda row: int(row["frame"]))
    for index, row in enumerate(ordered):
        prev_row = ordered[index - 1] if index > 0 else None
        next_row = ordered[index + 1] if index + 1 < len(ordered) else None
        labels: dict[str, torch.Tensor] = {}
        for prefix, source in (("prev", prev_row), ("current", row), ("next", next_row)):
            for field in fields:
                labels[f"{prefix}_{field}"] = torch.tensor(
                    [_label_id(source, field, value_to_id)],
                    dtype=torch.long,
                    device=device,
                )
        prev_delta = 0 if prev_row is None else max(0, int(row["frame"]) - int(prev_row["frame"]))
        next_delta = 0 if next_row is None else max(0, int(next_row["frame"]) - int(row["frame"]))
        with torch.no_grad():
            logits = model(
                labels,
                prev_delta=torch.tensor([float(prev_delta)], dtype=torch.float32, device=device),
                next_delta=torch.tensor([float(next_delta)], dtype=torch.float32, device=device),
            )
            pred_id = int(logits.argmax(dim=-1).item())
        row["interpolation"] = interpolation_vocab[pred_id] if 0 <= pred_id < len(interpolation_vocab) else "linear"
        output.append(row)
    return output


def _plan_path_from_manifest_row(row: dict[str, object], plan_root: Path, plan_name: str) -> Path:
    existing = Path(str(row.get("plan", "")))
    if existing.exists():
        return existing
    index = int(row["index"])
    clip_id = str(row.get("clip_id", ""))
    cdir = plan_root / f"c{index}"
    candidates = sorted(cdir.glob(plan_name))
    if len(candidates) == 1:
        return candidates[0]
    if clip_id:
        safe = clip_id.replace("/", "_").replace("\\", "_")
        direct = cdir / plan_name.replace(".json", f"_{safe}.json")
        if direct.exists():
            return direct
    return cdir / plan_name


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir.resolve()
    plan_root = args.plan_root.resolve()
    output_root = args.output_root.resolve()
    checkpoint_path = run_dir / str(args.checkpoint_name)
    device = _device_from_arg(str(args.device))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    fields = [str(field) for field in checkpoint["fields"]]
    vocabularies = {str(field): [str(value) for value in values] for field, values in checkpoint["vocabularies"].items()}
    value_to_id = {field: {value: idx for idx, value in enumerate(values)} for field, values in vocabularies.items()}
    interpolation_vocab = [str(value) for value in checkpoint["interpolation_vocab"]]
    model = _build_model(checkpoint).to(device)

    manifest_path = plan_root / "manifest.json"
    rows = _load_json(manifest_path)
    if not isinstance(rows, list):
        raise TypeError(f"manifest must be a list: {manifest_path}")

    output_rows: list[dict[str, object]] = []
    changed_counts: dict[str, int] = {label: 0 for label in interpolation_vocab}
    for manifest_row in rows:
        if not isinstance(manifest_row, dict):
            continue
        row = dict(manifest_row)
        if not str(row.get("status", "")).startswith("ok"):
            output_rows.append(row)
            continue
        source_plan = _plan_path_from_manifest_row(row, plan_root, str(args.plan_name))
        if not source_plan.exists():
            row["status"] = "missing-plan"
            row["plan"] = str(source_plan)
            output_rows.append(row)
            continue
        payload = _load_json(source_plan)
        keyframes = payload.get("keyframes", []) if isinstance(payload, dict) else []
        if not isinstance(keyframes, list):
            row["status"] = "bad-plan"
            output_rows.append(row)
            continue
        predicted = _predict_interpolations(
            [frame for frame in keyframes if isinstance(frame, dict)],
            model=model,
            fields=fields,
            value_to_id=value_to_id,
            interpolation_vocab=interpolation_vocab,
            device=device,
        )
        for item in predicted:
            changed_counts[str(item.get("interpolation", "linear"))] = changed_counts.get(str(item.get("interpolation", "linear")), 0) + 1

        index = int(row["index"])
        item_dir = output_root / f"c{index}"
        source_stem = source_plan.stem
        if source_stem == Path(str(args.plan_name)).stem:
            stem = Path(str(args.output_plan_name)).stem
        else:
            stem = source_stem.replace(Path(str(args.plan_name)).stem, Path(str(args.output_plan_name)).stem)
            if stem == source_stem:
                stem = f"{source_stem}_interpstage"
        output_plan = item_dir / f"{stem}.json"
        output_vmd = item_dir / f"{stem}.vmd"
        output_debug = item_dir / f"{stem}_debug.json"
        output_mp4 = item_dir / f"{stem}.mp4"

        new_payload = dict(payload)
        new_payload["interpolation_stage_model"] = str(checkpoint_path)
        new_payload["inference_note"] = str(new_payload.get("inference_note", "")) + " Interpolation is predicted by a separate post-stage model."
        new_payload["keyframes"] = predicted
        _write_json(output_plan, new_payload)
        timing_debug_src = source_plan.with_name(f"{source_plan.stem}_timing_debug.json")
        if timing_debug_src.exists():
            shutil.copy2(timing_debug_src, output_plan.with_name(f"{output_plan.stem}_timing_debug.json"))

        targets_path = args.targets_root.resolve() / f"c{index}" / "real_targets.json"
        _write_vmd_from_plan(
            output_plan,
            targets_path,
            output_vmd,
            output_debug,
            stabilize_forward=bool(args.actor_forward_stabilization),
        )
        row.update(
            {
                "status": "ok-vmd-only",
                "plan": str(output_plan),
                "vmd": str(output_vmd),
                "mp4": str(output_mp4),
                "interpolation_stage_model": str(checkpoint_path),
            }
        )
        output_rows.append(row)
        print(f"[ok] c{index}: {output_vmd.name}", flush=True)

    _write_json(output_root / "manifest.json", output_rows)
    _write_json(
        output_root / "interpolation_stage_summary.json",
        {
            "source_plan_root": str(plan_root),
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint_path),
            "fields": fields,
            "interpolation_vocab": interpolation_vocab,
            "predicted_counts": changed_counts,
        },
    )
    print(f"output root: {output_root}", flush=True)


if __name__ == "__main__":
    main()
