from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .shot_sequence_cp_sat import FIELDS, V3_1_FIELDS, fields_for_schema, optimize_shot_sequence_cp_sat


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = (
    WORKSPACE_ROOT
    / "camera_autoshot"
    / "results"
    / "shotplan_transformer_mlp_dropout015_cutdupw100_dcm_film_demucs_no_rms_density40_budgeted_cutgap7_6_keygap5"
)
DEFAULT_GT_ROOT = WORKSPACE_ROOT / "camera_autoshot" / "reverses_roll_feet_5fps"
DEFAULT_REAL_TARGETS_ROOT = WORKSPACE_ROOT / "camera_autoshot" / "reverses"


def _wrap_delta_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _unwrap_degrees(values: list[float]) -> list[float]:
    if not values:
        return []
    out = [float(values[0])]
    for value in values[1:]:
        out.append(out[-1] + _wrap_delta_deg(float(value) - out[-1]))
    return out


def _heading_from_forward(forward: list[float]) -> float:
    if len(forward) < 3:
        return 0.0
    return math.degrees(math.atan2(float(forward[0]), float(forward[2])))


def _load_heading_track(targets_path: Path) -> tuple[list[int], list[float]]:
    payload = _load_json(targets_path)
    frames: list[int] = []
    headings: list[float] = []
    for row in payload.get("frames", []):
        if not isinstance(row, dict) or "forward" not in row:
            continue
        frames.append(int(row.get("frame", len(frames))))
        headings.append(_heading_from_forward(row.get("forward") or [0.0, 0.0, 1.0]))
    return frames, _unwrap_degrees(headings)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run CP-SAT sequence-stat optimization over exported top-k shot-plan "
            "candidates and write optimized camera_poses JSON files."
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--indices", default="", help="Comma-separated cNN indices. Empty means all top-k folders found.")
    parser.add_argument("--candidate-name", default="camera_poses_top10_candidates.json")
    parser.add_argument("--gt-name", default="camera_poses_inferred.json")
    parser.add_argument("--output-name", default="camera_poses_optimized_cp_sat.json")
    parser.add_argument("--debug-name", default="camera_poses_optimized_cp_sat_optimizer_debug.json")
    parser.add_argument("--manifest-name", default="cp_sat_optimization_manifest.json")
    parser.add_argument(
        "--schema",
        choices=["auto", "v2", "v3_1"],
        default="auto",
        help="Shot-plan schema for optimized fields and output metadata.",
    )
    parser.add_argument(
        "--allow-missing-gt",
        action="store_true",
        help=(
            "Allow missing GT stats when --disable-sequence-stat-objective is set; "
            "top-1 candidate rows are used only as a stats/debug fallback."
        ),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-time-sec", type=float, default=30.0)
    parser.add_argument("--local-cost-scale", type=int, default=100)
    parser.add_argument(
        "--disable-sequence-stat-objective",
        action="store_true",
        help=(
            "Disable marginal/transition/change GT-stat matching. This keeps "
            "the optimizer close to model top-k local costs and only applies "
            "explicit penalties such as distance bounce or post-cut jumps."
        ),
    )
    parser.add_argument(
        "--distance-bounce-weight",
        type=int,
        default=0,
        help=(
            "Penalty weight for short close-far-close or far-close-far patterns "
            "in the target_body+shot_size distance proxy. 0 disables it."
        ),
    )
    parser.add_argument(
        "--distance-bounce-min-rank-delta",
        type=int,
        default=80,
        help="Minimum distance-rank change on both sides to count as a bounce.",
    )
    parser.add_argument(
        "--distance-bounce-max-frame-gap",
        type=int,
        default=10,
        help="Maximum source-frame gap on each side of the V/逆V bounce pattern.",
    )
    parser.add_argument(
        "--composition-bounce-weight",
        type=int,
        default=0,
        help=(
            "Penalty weight for short left-center-left, center-left-center, "
            "right-center-right, or center-right-center composition reversals. 0 disables it."
        ),
    )
    parser.add_argument(
        "--composition-bounce-min-rank-delta",
        type=int,
        default=80,
        help="Minimum left/right composition-rank change on both sides to count as a bounce.",
    )
    parser.add_argument(
        "--composition-bounce-max-frame-gap",
        type=int,
        default=10,
        help="Maximum source-frame gap on each side of the composition bounce pattern.",
    )
    parser.add_argument(
        "--vertical-bounce-weight",
        type=int,
        default=0,
        help=(
            "Penalty weight for short up-down-up or down-up-down target/pitch "
            "framing reversals. 0 disables it."
        ),
    )
    parser.add_argument(
        "--vertical-bounce-min-rank-delta",
        type=int,
        default=35,
        help="Minimum vertical-rank change on both sides to count as a bounce.",
    )
    parser.add_argument(
        "--vertical-bounce-max-frame-gap",
        type=int,
        default=10,
        help="Maximum source-frame gap on each side of the vertical bounce pattern.",
    )
    parser.add_argument("--top-pitch-bounce-weight", type=int, default=0)
    parser.add_argument("--top-pitch-bounce-max-frame-gap", type=int, default=30)
    parser.add_argument("--top-pitch-bounce-neighbor-pitches", default="low,level")
    parser.add_argument("--top-pitch-bounce-return-pitches", default="low,level,high")
    parser.add_argument("--turn-yaw-jitter-weight", type=int, default=0)
    parser.add_argument("--turn-yaw-jitter-targets-root", type=Path, default=DEFAULT_REAL_TARGETS_ROOT)
    parser.add_argument("--turn-yaw-jitter-fps", type=float, default=30.0)
    parser.add_argument("--turn-yaw-jitter-min-frame-gap", type=int, default=1)
    parser.add_argument("--turn-yaw-jitter-max-frame-gap", type=int, default=90)
    parser.add_argument("--turn-yaw-jitter-min-heading-delta", type=float, default=25.0)
    parser.add_argument("--turn-yaw-jitter-min-heading-speed", type=float, default=9999.0)
    parser.add_argument("--turn-yaw-jitter-min-relative-yaw-delta", type=float, default=45.0)
    parser.add_argument("--turn-yaw-jitter-min-world-yaw-delta", type=float, default=60.0)
    parser.add_argument("--turn-yaw-jitter-penalty-degrees-per-unit", type=float, default=15.0)
    parser.add_argument("--turn-yaw-jitter-allow-opposite-direction", action="store_true")
    parser.add_argument("--turn-yaw-jitter-include-cut-pairs", action="store_true")
    parser.add_argument(
        "--cut-identity-weight",
        type=int,
        default=0,
        help="Penalty weight when two same-frame cut slots select identical shot labels. 0 disables it.",
    )
    parser.add_argument(
        "--cut-contrast-weight",
        type=int,
        default=0,
        help="Penalty weight when two same-frame cut slots are only weakly different. 0 disables it.",
    )
    parser.add_argument(
        "--cut-contrast-min-score",
        type=int,
        default=3,
        help="Minimum contrast score required for same-frame cut slot pairs.",
    )
    parser.add_argument(
        "--cut-contrast-medium-rank-delta",
        type=int,
        default=100,
        help="Distance-rank delta that contributes one contrast point.",
    )
    parser.add_argument(
        "--cut-contrast-strong-rank-delta",
        type=int,
        default=220,
        help="Distance-rank delta that contributes two contrast points.",
    )
    parser.add_argument(
        "--post-cut-distance-jump-weight",
        type=int,
        default=0,
        help="Penalty weight for a large distance-rank jump immediately after a cut slot pair. 0 disables it.",
    )
    parser.add_argument(
        "--post-cut-distance-jump-min-rank-delta",
        type=int,
        default=250,
        help="Minimum target_body+shot_size distance-rank delta to count as a post-cut jump.",
    )
    parser.add_argument(
        "--post-cut-distance-jump-max-frame-gap",
        type=int,
        default=10,
        help="Maximum source-frame gap after a cut slot pair to penalize a distance jump.",
    )
    parser.add_argument(
        "--cut-context-speed-weight",
        type=int,
        default=0,
        help=(
            "Penalty weight for too-fast distance-rank moves immediately "
            "before or after a same-frame cut pair. 0 disables it."
        ),
    )
    parser.add_argument(
        "--cut-context-speed-min-speed",
        type=float,
        default=250.0,
        help="Minimum distance-rank speed per second to penalize around cut pairs.",
    )
    parser.add_argument(
        "--cut-context-speed-max-frame-gap",
        type=int,
        default=10,
        help="Maximum source-frame gap before/after a cut pair for the speed penalty.",
    )
    parser.add_argument(
        "--cut-context-speed-fps",
        type=float,
        default=5.0,
        help="FPS used to convert source-frame gaps into seconds for cut-context speed.",
    )
    parser.add_argument(
        "--repeated-distance-bounce-weight",
        type=int,
        default=0,
        help=(
            "Penalty weight for repeated close-far-close-far-close or "
            "far-close-far-close-far distance-rank motifs. 0 disables it."
        ),
    )
    parser.add_argument(
        "--repeated-distance-bounce-min-rank-delta",
        type=int,
        default=180,
        help="Minimum distance-rank change for each leg of a repeated bounce motif.",
    )
    parser.add_argument(
        "--repeated-distance-bounce-max-frame-gap",
        type=int,
        default=15,
        help="Maximum source-frame gap for each leg of a repeated bounce motif; same-frame cut legs are allowed.",
    )
    parser.add_argument(
        "--rare-duration-stats",
        type=Path,
        default=None,
        help=(
            "Optional rare_duration_stats.json with class-specific GT next-distinct-frame "
            "duration statistics. When provided with --rare-duration-weight, rare candidates "
            "outside the selected duration band receive extra cost."
        ),
    )
    parser.add_argument(
        "--rare-duration-weight",
        type=int,
        default=0,
        help="Penalty weight per rare-duration violation unit. 0 disables it.",
    )
    parser.add_argument(
        "--rare-duration-fps",
        type=float,
        default=5.0,
        help="FPS used to convert source-frame gaps to seconds for rare-duration penalties.",
    )
    parser.add_argument(
        "--rare-duration-lower-key",
        default="p25_sec",
        help="Lower duration stat key to use as the acceptable rare duration bound.",
    )
    parser.add_argument(
        "--rare-duration-upper-key",
        default="p75_sec",
        help="Upper duration stat key to use as the acceptable rare duration bound.",
    )
    parser.add_argument(
        "--rare-duration-missing-penalty-units",
        type=int,
        default=0,
        help="Penalty units for rare candidates with no following distinct frame.",
    )
    parser.add_argument(
        "--drop-near-cut-keyframes",
        action="store_true",
        help=(
            "Drop non-cut candidate frame rows that are within "
            "--near-cut-frame-radius frames of a same-frame cut row before optimization."
        ),
    )
    parser.add_argument(
        "--near-cut-frame-radius",
        type=int,
        default=1,
        help="Frame distance used by --drop-near-cut-keyframes.",
    )
    parser.add_argument("--crop-gt-to-candidate-span", action="store_true", default=True)
    parser.add_argument("--no-crop-gt-to-candidate-span", dest="crop_gt_to_candidate_span", action="store_false")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_indices(raw: str, source_root: Path, candidate_name: str) -> list[int]:
    if raw.strip():
        values: list[int] = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            values.append(int(token[1:] if token.startswith("c") else token))
        return sorted(set(values))
    indices: list[int] = []
    for path in sorted(source_root.glob(f"c*/{candidate_name}")):
        name = path.parent.name
        if name.startswith("c") and name[1:].isdigit():
            indices.append(int(name[1:]))
    return sorted(set(indices))


def _payload_keyframes(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = payload.get("keyframes", [])
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def _infer_schema_name(payload: dict[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    label_fields = {str(field) for field in payload.get("label_fields", [])}
    if label_fields & set(V3_1_FIELDS):
        return "v3_1"
    for frame_row in payload.get("frames", []):
        if not isinstance(frame_row, dict):
            continue
        for slot in frame_row.get("slots", []):
            if not isinstance(slot, dict):
                continue
            for candidate in slot.get("candidates", []):
                plan = candidate.get("plan", {}) if isinstance(candidate, dict) else {}
                if isinstance(plan, dict) and ("target_anchor" in plan or "distance" in plan or "yaw" in plan):
                    return "v3_1"
    return "v2"


def _metadata_for_output(candidate_payload: dict[str, Any], schema_name: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "sample_fps": float(candidate_payload.get("sample_fps", 30.0 if schema_name == "v3_1" else 5.0)),
        "source_topk_candidates": None,
    }
    if schema_name == "v3_1":
        passthrough_keys = [
            "schema",
            "source_model",
            "model_checkpoint",
            "clip_id",
            "source_song_id",
            "source_start_frame",
            "source_end_frame",
            "timeline_fps",
            "timing_source",
            "distance_schema",
        ]
        for key in passthrough_keys:
            if key in candidate_payload:
                metadata[key] = candidate_payload[key]
        metadata.setdefault(
            "schema",
            "camera_autoshot.reverse_v3.close_dense_no_wide"
            if "distance_schema" in candidate_payload
            else "camera_autoshot.reverse_v3",
        )
    return metadata


def _resolve_gt_path(
    *,
    gt_root: Path,
    song_id: str,
    gt_name: str,
    candidate_payload: dict[str, Any],
) -> Path:
    song_path = gt_root / song_id / gt_name
    if song_path.exists():
        return song_path
    clip_id = str(candidate_payload.get("clip_id", "")).strip()
    if clip_id:
        clip_path = gt_root / clip_id / gt_name
        if clip_path.exists():
            return clip_path
    return song_path


def _candidate_sequence_from_topk(payload: dict[str, Any], top_k: int) -> tuple[list[dict], list[dict[str, Any]]]:
    """Flatten top-k frame/slot candidates into one sequence timestep per slot."""
    candidates: list[dict] = []
    step_meta: list[dict[str, Any]] = []
    sequence_step = 0
    for frame_row in payload.get("frames", []):
        source_frame = int(frame_row["frame"])
        frame_meta = {
            key: frame_row[key]
            for key in (
                "timing_frame",
                "duplicate_offset",
                "duplicate_count",
                "cut_pair_id",
                "cut_pair_role",
                "cut_flag",
                "clip_id",
            )
            if key in frame_row
        }
        for slot in frame_row.get("slots", []):
            slot_index = int(slot.get("slot_index", 0))
            slot_candidates = slot.get("candidates", [])[: int(top_k)]
            step_meta.append(
                {
                    "sequence_step": sequence_step,
                    "source_frame": source_frame,
                    "slot_index": slot_index,
                    "cut_flag": bool(frame_row.get("cut_flag", False)),
                    "slot_count": int(frame_row.get("slot_count", 1)),
                    **frame_meta,
                }
            )
            for candidate in slot_candidates:
                plan = dict(candidate.get("plan", {}))
                plan["frame"] = sequence_step
                plan["source_frame"] = source_frame
                plan["slot_index"] = slot_index
                plan.update(frame_meta)
                plan["rank"] = int(candidate.get("rank", 0))
                plan["model_logprob"] = float(candidate.get("logprob", 0.0))
                plan["model_prob"] = float(candidate.get("prob", 0.0))
                # CP-SAT minimizes local_cost, so negative log probability is the cost.
                plan["local_cost"] = -float(candidate.get("logprob", 0.0))
                candidates.append(plan)
            sequence_step += 1
    return candidates, step_meta


def _is_cut_frame_row(frame_row: dict[str, Any]) -> bool:
    """Return whether a top-k frame row represents a cut with two shot slots."""
    if bool(frame_row.get("cut_flag", False)):
        return True
    if frame_row.get("cut_pair_id") is not None:
        return True
    try:
        if int(frame_row.get("duplicate_count", 1)) >= 2:
            return True
    except (TypeError, ValueError):
        pass
    try:
        return int(frame_row.get("slot_count", 1)) >= 2
    except (TypeError, ValueError):
        return len(frame_row.get("slots", [])) >= 2


def _drop_near_cut_keyframe_rows(
    payload: dict[str, Any],
    *,
    radius: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Drop non-cut frame rows immediately before/after cut rows.

    This removes tiny helper keyframes such as F-1 or F+1 around a cut F before
    CP-SAT sees them, preventing the optimizer from changing shot labels only to
    smooth a one-frame distance bounce.
    """
    if radius <= 0:
        return payload, []

    frame_rows = payload.get("frames", [])
    if not isinstance(frame_rows, list):
        return payload, []

    cut_frames = {
        int(row["frame"])
        for row in frame_rows
        if isinstance(row, dict) and "frame" in row and _is_cut_frame_row(row)
    }
    if not cut_frames:
        return payload, []

    kept_rows: list[dict[str, Any]] = []
    removed_rows: list[dict[str, Any]] = []
    for row in frame_rows:
        if not isinstance(row, dict) or "frame" not in row:
            kept_rows.append(row)
            continue
        frame = int(row["frame"])
        is_cut = _is_cut_frame_row(row)
        nearest_cut_distance = min((abs(frame - cut_frame) for cut_frame in cut_frames), default=radius + 1)
        if not is_cut and 0 < nearest_cut_distance <= radius:
            removed_rows.append(
                {
                    "frame": frame,
                    "nearest_cut_distance": nearest_cut_distance,
                    "slot_count": int(row.get("slot_count", len(row.get("slots", [])) or 1)),
                    "cut_flag": bool(row.get("cut_flag", False)),
                }
            )
            continue
        kept_rows.append(row)

    filtered = dict(payload)
    filtered["frames"] = kept_rows
    return filtered, removed_rows


def _crop_gt_sequence(gt_sequence: list[dict], min_frame: int, max_frame: int) -> list[dict]:
    """Crop GT rows to the candidate frame span; fallback is handled by caller."""
    cropped: list[dict] = []
    for row in gt_sequence:
        if "frame" not in row and "source_frame" not in row:
            continue
        frame = int(row.get("source_frame", row.get("frame", 0)))
        if min_frame <= frame <= max_frame:
            cropped.append(row)
    return cropped


def _strip_optimizer_fields(row: dict, fields: list[str]) -> dict:
    """Convert an optimizer-selected row back to camera_poses JSON shape."""
    output = {
        "frame": int(row.get("source_frame", row.get("frame", 0))),
    }
    for field in fields:
        output[field] = row.get(field)
    if "transition" in row:
        output["transition"] = row["transition"]
    if "stage" in row:
        output["stage"] = row["stage"]
    return output


def _rank_summary(selected_sequence: list[dict]) -> dict[str, Any]:
    ranks = [int(row.get("rank", 0)) for row in selected_sequence]
    return {
        "count": len(ranks),
        "rank_1": sum(1 for rank in ranks if rank == 1),
        "rank_gt_1": sum(1 for rank in ranks if rank > 1),
        "max_rank": max(ranks) if ranks else 0,
        "avg_rank": sum(ranks) / len(ranks) if ranks else 0.0,
    }


def optimize_one_song(
    *,
    source_root: Path,
    gt_root: Path,
    index: int,
    candidate_name: str,
    gt_name: str,
    output_name: str,
    debug_name: str,
    top_k: int,
    max_time_sec: float,
    local_cost_scale: int,
    disable_sequence_stat_objective: bool,
    distance_bounce_weight: int,
    distance_bounce_min_rank_delta: int,
    distance_bounce_max_frame_gap: int,
    cut_identity_weight: int,
    cut_contrast_weight: int,
    cut_contrast_min_score: int,
    cut_contrast_medium_rank_delta: int,
    cut_contrast_strong_rank_delta: int,
    post_cut_distance_jump_weight: int,
    post_cut_distance_jump_min_rank_delta: int,
    post_cut_distance_jump_max_frame_gap: int,
    cut_context_speed_weight: int,
    cut_context_speed_min_speed: float,
    cut_context_speed_max_frame_gap: int,
    cut_context_speed_fps: float,
    repeated_distance_bounce_weight: int,
    repeated_distance_bounce_min_rank_delta: int,
    repeated_distance_bounce_max_frame_gap: int,
    composition_bounce_weight: int,
    composition_bounce_min_rank_delta: int,
    composition_bounce_max_frame_gap: int,
    vertical_bounce_weight: int,
    vertical_bounce_min_rank_delta: int,
    vertical_bounce_max_frame_gap: int,
    top_pitch_bounce_weight: int,
    top_pitch_bounce_max_frame_gap: int,
    top_pitch_bounce_neighbor_pitches: str,
    top_pitch_bounce_return_pitches: str,
    turn_yaw_jitter_weight: int,
    turn_yaw_jitter_targets_root: Path,
    turn_yaw_jitter_fps: float,
    turn_yaw_jitter_min_frame_gap: int,
    turn_yaw_jitter_max_frame_gap: int,
    turn_yaw_jitter_min_heading_delta: float,
    turn_yaw_jitter_min_heading_speed: float,
    turn_yaw_jitter_min_relative_yaw_delta: float,
    turn_yaw_jitter_min_world_yaw_delta: float,
    turn_yaw_jitter_penalty_degrees_per_unit: float,
    turn_yaw_jitter_require_same_direction: bool,
    turn_yaw_jitter_exclude_cut_pairs: bool,
    rare_duration_weight: int,
    rare_duration_stats: dict[str, Any] | None,
    rare_duration_fps: float,
    rare_duration_lower_key: str,
    rare_duration_upper_key: str,
    rare_duration_missing_penalty_units: int,
    drop_near_cut_keyframes: bool,
    near_cut_frame_radius: int,
    crop_gt_to_candidate_span: bool,
    schema: str,
    allow_missing_gt: bool,
    verbose: bool,
) -> dict[str, Any]:
    song_id = f"c{index}"
    candidate_path = source_root / song_id / candidate_name
    gt_path = gt_root / song_id / gt_name
    output_path = source_root / song_id / output_name
    debug_path = source_root / song_id / debug_name
    row: dict[str, Any] = {
        "index": index,
        "song_id": song_id,
        "candidate_path": str(candidate_path),
        "gt_path": str(gt_path),
        "output_path": str(output_path),
        "debug_path": str(debug_path),
        "status": "",
    }
    if not candidate_path.exists():
        row["status"] = "missing-candidates"
        return row
    candidate_payload = _load_json(candidate_path)
    schema_name = _infer_schema_name(candidate_payload, schema)
    optimized_fields = fields_for_schema(schema_name)
    gt_path = _resolve_gt_path(
        gt_root=gt_root,
        song_id=song_id,
        gt_name=gt_name,
        candidate_payload=candidate_payload,
    )
    row["gt_path"] = str(gt_path)
    if not gt_path.exists() and not (allow_missing_gt and disable_sequence_stat_objective):
        row["status"] = "missing-gt"
        return row
    removed_near_cut_keyframes: list[dict[str, Any]] = []
    if drop_near_cut_keyframes:
        candidate_payload, removed_near_cut_keyframes = _drop_near_cut_keyframe_rows(
            candidate_payload,
            radius=int(near_cut_frame_radius),
        )
    optimizer_candidates, step_meta = _candidate_sequence_from_topk(candidate_payload, top_k)
    if not optimizer_candidates:
        row["status"] = "empty-candidates"
        return row

    source_frames = [int(meta["source_frame"]) for meta in step_meta]
    min_frame = min(source_frames)
    max_frame = max(source_frames)
    gt_stats_fallback = ""
    if gt_path.exists():
        gt_full = _payload_keyframes(_load_json(gt_path))
        gt_used = _crop_gt_sequence(gt_full, min_frame, max_frame) if crop_gt_to_candidate_span else gt_full
        if not gt_used:
            gt_used = gt_full
    else:
        gt_full = [_strip_optimizer_fields(candidate, optimized_fields) for candidate in optimizer_candidates if int(candidate.get("rank", 0)) == 1]
        gt_used = list(gt_full)
        gt_stats_fallback = "top1_candidates"

    weights = None
    if disable_sequence_stat_objective:
        weights = {
            "marginal": {field: 0 for field in optimized_fields},
            "transition": {field: 0 for field in optimized_fields},
            "change": {field: 0 for field in optimized_fields},
            "local": 1,
        }

    turn_yaw_heading_frames: list[int] | None = None
    turn_yaw_heading_values: list[float] | None = None
    turn_yaw_targets_path: Path | None = None
    if int(turn_yaw_jitter_weight) > 0:
        turn_yaw_targets_path = Path(turn_yaw_jitter_targets_root) / song_id / "real_targets.json"
        if turn_yaw_targets_path.exists():
            turn_yaw_heading_frames, turn_yaw_heading_values = _load_heading_track(turn_yaw_targets_path)

    result = optimize_shot_sequence_cp_sat(
        optimizer_candidates,
        gt_used,
        fields=optimized_fields,
        weights=weights,
        max_time_sec=max_time_sec,
        local_cost_scale=local_cost_scale,
        distance_bounce_weight=distance_bounce_weight,
        distance_bounce_min_rank_delta=distance_bounce_min_rank_delta,
        distance_bounce_max_frame_gap=distance_bounce_max_frame_gap,
        cut_identity_weight=cut_identity_weight,
        cut_contrast_weight=cut_contrast_weight,
        cut_contrast_min_score=cut_contrast_min_score,
        cut_contrast_medium_rank_delta=cut_contrast_medium_rank_delta,
        cut_contrast_strong_rank_delta=cut_contrast_strong_rank_delta,
        post_cut_distance_jump_weight=post_cut_distance_jump_weight,
        post_cut_distance_jump_min_rank_delta=post_cut_distance_jump_min_rank_delta,
        post_cut_distance_jump_max_frame_gap=post_cut_distance_jump_max_frame_gap,
        cut_context_speed_weight=cut_context_speed_weight,
        cut_context_speed_min_speed=cut_context_speed_min_speed,
        cut_context_speed_max_frame_gap=cut_context_speed_max_frame_gap,
        cut_context_speed_fps=cut_context_speed_fps,
        repeated_distance_bounce_weight=repeated_distance_bounce_weight,
        repeated_distance_bounce_min_rank_delta=repeated_distance_bounce_min_rank_delta,
        repeated_distance_bounce_max_frame_gap=repeated_distance_bounce_max_frame_gap,
        composition_bounce_weight=composition_bounce_weight,
        composition_bounce_min_rank_delta=composition_bounce_min_rank_delta,
        composition_bounce_max_frame_gap=composition_bounce_max_frame_gap,
        vertical_bounce_weight=vertical_bounce_weight,
        vertical_bounce_min_rank_delta=vertical_bounce_min_rank_delta,
        vertical_bounce_max_frame_gap=vertical_bounce_max_frame_gap,
        top_pitch_bounce_weight=top_pitch_bounce_weight,
        top_pitch_bounce_max_frame_gap=top_pitch_bounce_max_frame_gap,
        top_pitch_bounce_neighbor_pitches=top_pitch_bounce_neighbor_pitches,
        top_pitch_bounce_return_pitches=top_pitch_bounce_return_pitches,
        turn_yaw_jitter_weight=turn_yaw_jitter_weight,
        turn_yaw_jitter_heading_frames=turn_yaw_heading_frames,
        turn_yaw_jitter_heading_values=turn_yaw_heading_values,
        turn_yaw_jitter_fps=turn_yaw_jitter_fps,
        turn_yaw_jitter_min_frame_gap=turn_yaw_jitter_min_frame_gap,
        turn_yaw_jitter_max_frame_gap=turn_yaw_jitter_max_frame_gap,
        turn_yaw_jitter_min_heading_delta=turn_yaw_jitter_min_heading_delta,
        turn_yaw_jitter_min_heading_speed=turn_yaw_jitter_min_heading_speed,
        turn_yaw_jitter_min_relative_yaw_delta=turn_yaw_jitter_min_relative_yaw_delta,
        turn_yaw_jitter_min_world_yaw_delta=turn_yaw_jitter_min_world_yaw_delta,
        turn_yaw_jitter_penalty_degrees_per_unit=turn_yaw_jitter_penalty_degrees_per_unit,
        turn_yaw_jitter_require_same_direction=turn_yaw_jitter_require_same_direction,
        turn_yaw_jitter_exclude_cut_pairs=turn_yaw_jitter_exclude_cut_pairs,
        rare_duration_weight=rare_duration_weight,
        rare_duration_stats=rare_duration_stats,
        rare_duration_fps=rare_duration_fps,
        rare_duration_lower_key=rare_duration_lower_key,
        rare_duration_upper_key=rare_duration_upper_key,
        rare_duration_missing_penalty_units=rare_duration_missing_penalty_units,
        verbose=verbose,
    )
    if result["status"] not in {"OPTIMAL", "FEASIBLE"}:
        row["status"] = result["status"]
        row["objective"] = result.get("objective")
        _write_json(debug_path, result)
        return row

    optimized_keyframes = [_strip_optimizer_fields(selected, optimized_fields) for selected in result["selected_sequence"]]
    output_payload = _metadata_for_output(candidate_payload, schema_name)
    output_payload.update(
        {
        "source_topk_candidates": str(candidate_path),
        "gt_stats_source": str(gt_path) if gt_path.exists() else None,
        "gt_stats_fallback": gt_stats_fallback,
        "gt_stats_frame_span": [min_frame, max_frame],
        "optimizer": {
            "name": "ortools_cp_sat_sequence_stats",
            "schema": schema_name,
            "fields": optimized_fields,
            "top_k": int(top_k),
            "max_time_sec": float(max_time_sec),
            "local_cost_scale": int(local_cost_scale),
            "disable_sequence_stat_objective": bool(disable_sequence_stat_objective),
            "distance_bounce_weight": int(distance_bounce_weight),
            "distance_bounce_min_rank_delta": int(distance_bounce_min_rank_delta),
            "distance_bounce_max_frame_gap": int(distance_bounce_max_frame_gap),
            "cut_identity_weight": int(cut_identity_weight),
            "cut_contrast_weight": int(cut_contrast_weight),
            "cut_contrast_min_score": int(cut_contrast_min_score),
            "cut_contrast_medium_rank_delta": int(cut_contrast_medium_rank_delta),
            "cut_contrast_strong_rank_delta": int(cut_contrast_strong_rank_delta),
            "post_cut_distance_jump_weight": int(post_cut_distance_jump_weight),
            "post_cut_distance_jump_min_rank_delta": int(post_cut_distance_jump_min_rank_delta),
            "post_cut_distance_jump_max_frame_gap": int(post_cut_distance_jump_max_frame_gap),
            "cut_context_speed_weight": int(cut_context_speed_weight),
            "cut_context_speed_min_speed": float(cut_context_speed_min_speed),
            "cut_context_speed_max_frame_gap": int(cut_context_speed_max_frame_gap),
            "cut_context_speed_fps": float(cut_context_speed_fps),
            "repeated_distance_bounce_weight": int(repeated_distance_bounce_weight),
            "repeated_distance_bounce_min_rank_delta": int(repeated_distance_bounce_min_rank_delta),
            "repeated_distance_bounce_max_frame_gap": int(repeated_distance_bounce_max_frame_gap),
            "composition_bounce_weight": int(composition_bounce_weight),
            "composition_bounce_min_rank_delta": int(composition_bounce_min_rank_delta),
            "composition_bounce_max_frame_gap": int(composition_bounce_max_frame_gap),
            "vertical_bounce_weight": int(vertical_bounce_weight),
            "vertical_bounce_min_rank_delta": int(vertical_bounce_min_rank_delta),
            "vertical_bounce_max_frame_gap": int(vertical_bounce_max_frame_gap),
            "top_pitch_bounce_weight": int(top_pitch_bounce_weight),
            "top_pitch_bounce_max_frame_gap": int(top_pitch_bounce_max_frame_gap),
            "top_pitch_bounce_neighbor_pitches": str(top_pitch_bounce_neighbor_pitches),
            "top_pitch_bounce_return_pitches": str(top_pitch_bounce_return_pitches),
            "turn_yaw_jitter_weight": int(turn_yaw_jitter_weight),
            "turn_yaw_jitter_targets_path": str(turn_yaw_targets_path) if turn_yaw_targets_path else None,
            "turn_yaw_jitter_enabled": bool(turn_yaw_heading_frames and turn_yaw_heading_values),
            "turn_yaw_jitter_fps": float(turn_yaw_jitter_fps),
            "turn_yaw_jitter_min_frame_gap": int(turn_yaw_jitter_min_frame_gap),
            "turn_yaw_jitter_max_frame_gap": int(turn_yaw_jitter_max_frame_gap),
            "turn_yaw_jitter_min_heading_delta": float(turn_yaw_jitter_min_heading_delta),
            "turn_yaw_jitter_min_heading_speed": float(turn_yaw_jitter_min_heading_speed),
            "turn_yaw_jitter_min_relative_yaw_delta": float(turn_yaw_jitter_min_relative_yaw_delta),
            "turn_yaw_jitter_min_world_yaw_delta": float(turn_yaw_jitter_min_world_yaw_delta),
            "turn_yaw_jitter_penalty_degrees_per_unit": float(turn_yaw_jitter_penalty_degrees_per_unit),
            "turn_yaw_jitter_require_same_direction": bool(turn_yaw_jitter_require_same_direction),
            "turn_yaw_jitter_exclude_cut_pairs": bool(turn_yaw_jitter_exclude_cut_pairs),
            "rare_duration_weight": int(rare_duration_weight),
            "rare_duration_enabled": bool(rare_duration_stats and rare_duration_weight > 0),
            "rare_duration_fps": float(rare_duration_fps),
            "rare_duration_lower_key": str(rare_duration_lower_key),
            "rare_duration_upper_key": str(rare_duration_upper_key),
            "rare_duration_missing_penalty_units": int(rare_duration_missing_penalty_units),
            "drop_near_cut_keyframes": bool(drop_near_cut_keyframes),
            "near_cut_frame_radius": int(near_cut_frame_radius),
            "removed_near_cut_keyframe_count": len(removed_near_cut_keyframes),
            "status": result["status"],
            "schema": schema_name,
            "objective": result["objective"],
            "objective_breakdown": result["objective_breakdown"],
            "rank_summary": _rank_summary(result["selected_sequence"]),
        },
        "keyframes": optimized_keyframes,
        }
    )
    debug_payload = {
        "output": output_payload,
        "step_meta": step_meta,
        "removed_near_cut_keyframes": removed_near_cut_keyframes,
        "selected_indices": result["selected_indices"],
        "selected_sequence_with_scores": result["selected_sequence"],
        "stats": result["stats"],
    }
    _write_json(output_path, output_payload)
    _write_json(debug_path, debug_payload)

    row.update(
        {
            "status": result["status"],
            "objective": result["objective"],
            "frame_bucket_count": len(candidate_payload.get("frames", [])),
            "slot_count": len(optimized_keyframes),
            "gt_full_count": len(gt_full),
            "gt_used_count": len(gt_used),
            "gt_min_frame": min_frame,
            "gt_max_frame": max_frame,
            "removed_near_cut_keyframe_count": len(removed_near_cut_keyframes),
            **{f"rank_summary_{key}": value for key, value in _rank_summary(result["selected_sequence"]).items()},
            **{f"objective_{key}": value for key, value in result["objective_breakdown"].items()},
        }
    )
    return row


def main() -> None:
    args = build_parser().parse_args()
    source_root = args.source_root.resolve()
    gt_root = args.gt_root.resolve()
    indices = _parse_indices(str(args.indices), source_root, str(args.candidate_name))
    rare_duration_stats = _load_json(args.rare_duration_stats) if args.rare_duration_stats else None
    manifest: list[dict[str, Any]] = []
    for index in indices:
        row = optimize_one_song(
            source_root=source_root,
            gt_root=gt_root,
            index=index,
            candidate_name=str(args.candidate_name),
            gt_name=str(args.gt_name),
            output_name=str(args.output_name),
            debug_name=str(args.debug_name),
            top_k=int(args.top_k),
            max_time_sec=float(args.max_time_sec),
            local_cost_scale=int(args.local_cost_scale),
            disable_sequence_stat_objective=bool(args.disable_sequence_stat_objective),
            distance_bounce_weight=int(args.distance_bounce_weight),
            distance_bounce_min_rank_delta=int(args.distance_bounce_min_rank_delta),
            distance_bounce_max_frame_gap=int(args.distance_bounce_max_frame_gap),
            cut_identity_weight=int(args.cut_identity_weight),
            cut_contrast_weight=int(args.cut_contrast_weight),
            cut_contrast_min_score=int(args.cut_contrast_min_score),
            cut_contrast_medium_rank_delta=int(args.cut_contrast_medium_rank_delta),
            cut_contrast_strong_rank_delta=int(args.cut_contrast_strong_rank_delta),
            post_cut_distance_jump_weight=int(args.post_cut_distance_jump_weight),
            post_cut_distance_jump_min_rank_delta=int(args.post_cut_distance_jump_min_rank_delta),
            post_cut_distance_jump_max_frame_gap=int(args.post_cut_distance_jump_max_frame_gap),
            cut_context_speed_weight=int(args.cut_context_speed_weight),
            cut_context_speed_min_speed=float(args.cut_context_speed_min_speed),
            cut_context_speed_max_frame_gap=int(args.cut_context_speed_max_frame_gap),
            cut_context_speed_fps=float(args.cut_context_speed_fps),
            repeated_distance_bounce_weight=int(args.repeated_distance_bounce_weight),
            repeated_distance_bounce_min_rank_delta=int(args.repeated_distance_bounce_min_rank_delta),
            repeated_distance_bounce_max_frame_gap=int(args.repeated_distance_bounce_max_frame_gap),
            composition_bounce_weight=int(args.composition_bounce_weight),
            composition_bounce_min_rank_delta=int(args.composition_bounce_min_rank_delta),
            composition_bounce_max_frame_gap=int(args.composition_bounce_max_frame_gap),
            vertical_bounce_weight=int(args.vertical_bounce_weight),
            vertical_bounce_min_rank_delta=int(args.vertical_bounce_min_rank_delta),
            vertical_bounce_max_frame_gap=int(args.vertical_bounce_max_frame_gap),
            top_pitch_bounce_weight=int(args.top_pitch_bounce_weight),
            top_pitch_bounce_max_frame_gap=int(args.top_pitch_bounce_max_frame_gap),
            top_pitch_bounce_neighbor_pitches=str(args.top_pitch_bounce_neighbor_pitches),
            top_pitch_bounce_return_pitches=str(args.top_pitch_bounce_return_pitches),
            turn_yaw_jitter_weight=int(args.turn_yaw_jitter_weight),
            turn_yaw_jitter_targets_root=args.turn_yaw_jitter_targets_root.resolve(),
            turn_yaw_jitter_fps=float(args.turn_yaw_jitter_fps),
            turn_yaw_jitter_min_frame_gap=int(args.turn_yaw_jitter_min_frame_gap),
            turn_yaw_jitter_max_frame_gap=int(args.turn_yaw_jitter_max_frame_gap),
            turn_yaw_jitter_min_heading_delta=float(args.turn_yaw_jitter_min_heading_delta),
            turn_yaw_jitter_min_heading_speed=float(args.turn_yaw_jitter_min_heading_speed),
            turn_yaw_jitter_min_relative_yaw_delta=float(args.turn_yaw_jitter_min_relative_yaw_delta),
            turn_yaw_jitter_min_world_yaw_delta=float(args.turn_yaw_jitter_min_world_yaw_delta),
            turn_yaw_jitter_penalty_degrees_per_unit=float(args.turn_yaw_jitter_penalty_degrees_per_unit),
            turn_yaw_jitter_require_same_direction=not bool(args.turn_yaw_jitter_allow_opposite_direction),
            turn_yaw_jitter_exclude_cut_pairs=not bool(args.turn_yaw_jitter_include_cut_pairs),
            rare_duration_weight=int(args.rare_duration_weight),
            rare_duration_stats=rare_duration_stats,
            rare_duration_fps=float(args.rare_duration_fps),
            rare_duration_lower_key=str(args.rare_duration_lower_key),
            rare_duration_upper_key=str(args.rare_duration_upper_key),
            rare_duration_missing_penalty_units=int(args.rare_duration_missing_penalty_units),
            drop_near_cut_keyframes=bool(args.drop_near_cut_keyframes),
            near_cut_frame_radius=int(args.near_cut_frame_radius),
            crop_gt_to_candidate_span=bool(args.crop_gt_to_candidate_span),
            schema=str(args.schema),
            allow_missing_gt=bool(args.allow_missing_gt),
            verbose=bool(args.verbose),
        )
        manifest.append(row)
        print(
            f"[{row.get('status')}] {row.get('song_id')}: "
            f"slots={row.get('slot_count', 0)} rank>1={row.get('rank_summary_rank_gt_1', 0)} "
            f"objective={row.get('objective')}"
        )

    manifest_path = source_root / str(args.manifest_name)
    _write_json(manifest_path, manifest)
    csv_path = manifest_path.with_suffix(".csv")
    if manifest:
        fieldnames = sorted({key for row in manifest for key in row})
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest)
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
