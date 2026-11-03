"""Predict shot plans for the DCM test clips from stage-1 cut/transition timing.

This is the shot-plan step of the paper's DCM evaluation. Each clip's cut pairs and transition
frames come from the song-level timing JSON; plans are decoded greedily with the paper decoder
(cut frames first, then transition frames) and solved into a camera VMD with the body targets.

    python -m evaluation.dcm_test_plans --timing-root <stage-1 timing folder> --output-root <dir>
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera_autoshot.providers import JsonFrameTargetProvider
from camera_autoshot.reverse_v3.close_dense_schema import (
    apply_close_dense_distance_schema,
    internalize_close_dense_row,
)
from camera_autoshot.reverse_v3.solver_v3 import CameraSolveConfigV3, solve_camera_timeline_v3
from camera_autoshot.reverse_v3.spec_v3 import ShotKeyframeRequestV3
from camera_autoshot.vmd import write_camera_vmd
from evaluation.dcm_common import (
    DEFAULT_FEATURE_ROOT,
    DEFAULT_SHOTPLAN_RUN_DIR,
    DEFAULT_TARGETS_ROOT,
    DEFAULT_TEST_CLIPS,
    DEFAULT_TIMING_FILENAME,
    DEFAULT_TIMING_ROOT,
    PLAN_HISTORY_SIZE,
    WINDOW_RADIUS,
    build_feature_windows,
    device_from_arg,
    distance_schema_for_vocabularies,
    external_timing_for_clip,
    feature_manifest,
    fields_and_vocabularies,
    load_json,
    make_model,
    part_from_clip_id,
    select_test_clips,
    source_index,
    write_json,
    write_manifest_csv,
)
from shotplan_transformer.infer_v3_dcm_from_timing import _cut_change_stats
from shotplan_transformer.sequence_decode import sample_sequence


PLAN_STEM = "camera_poses"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict shot plans for the DCM test clips from stage-1 timing.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_SHOTPLAN_RUN_DIR)
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--test-clips", type=Path, default=DEFAULT_TEST_CLIPS)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--timing-root", type=Path, default=DEFAULT_TIMING_ROOT)
    parser.add_argument("--timing-filename", default=DEFAULT_TIMING_FILENAME)
    parser.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--indices", default="all", help="Comma-separated song indices, or 'all'.")
    parser.add_argument("--clip-ids", default="", help="Comma-separated DCM clip ids, e.g. C_8_1,C_81_3.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--history-size", type=int, default=PLAN_HISTORY_SIZE)
    parser.add_argument("--include-boundaries", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-events", type=int, default=2)
    return parser


def decode_keyframes(
    *,
    source_frames: np.ndarray,
    token_ids: dict[str, np.ndarray],
    fields: tuple[str, ...],
    vocabularies: dict[str, list[str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, frame in enumerate(source_frames.tolist()):
        row = {field: str(vocabularies[field][int(token_ids[field][index])]) for field in fields}
        rows.append({"frame": int(frame), "transition": "linear", **row})
    return rows


def write_vmd_from_plan(plan_payload: dict[str, Any], targets_path: Path, output_vmd: Path, debug_path: Path) -> int:
    keyframe_rows = [row for row in plan_payload.get("keyframes", []) if isinstance(row, dict)]
    if plan_payload.get("distance_schema") is not None:
        apply_close_dense_distance_schema()
        keyframe_rows = [internalize_close_dense_row(row) for row in keyframe_rows]
    requests = [ShotKeyframeRequestV3.from_dict(row) for row in keyframe_rows]
    provider = JsonFrameTargetProvider(targets_path, stabilize_forward=True)
    solved = solve_camera_timeline_v3(
        provider,
        requests,
        CameraSolveConfigV3(synthesize_start_keyframe=False),
    )
    output_vmd.parent.mkdir(parents=True, exist_ok=True)
    write_camera_vmd(solved, output_vmd, model_name="CuCaT")
    write_json(debug_path, [asdict(key) for key in solved])
    frames = [int(key.frame) for key in solved]
    return max(frames) if frames else 0


def main() -> None:
    args = build_parser().parse_args()
    device = device_from_arg(str(args.device))
    checkpoint_path = Path(args.run_dir).resolve() / str(args.checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    fields, vocabularies = fields_and_vocabularies(checkpoint)
    stats = checkpoint["stats"]
    decode_distance_fov_joint = bool(checkpoint.get("args", {}).get("predict_distance_fov_joint_as_field", False))
    max_position_tokens = int(checkpoint.get("config", {}).get("max_position_tokens", 16))
    distance_schema = distance_schema_for_vocabularies(vocabularies)
    if distance_schema is not None:
        apply_close_dense_distance_schema()
    model = make_model(checkpoint, device)

    feature_root = Path(args.feature_root).resolve()
    source_manifest = feature_manifest(feature_root)
    timing_root = Path(args.timing_root).resolve()
    timing_by_song = load_json(timing_root / str(args.timing_filename))
    selected_rows = select_test_clips(args.test_clips, indices=str(args.indices), clip_ids=str(args.clip_ids))
    if not selected_rows:
        raise SystemExit("no DCM test clips selected")

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []

    with torch.no_grad():
        for meta in selected_rows:
            clip_id = str(meta["clip_id"])
            source_song_id = str(meta["source_song_id"])
            index = source_index(source_song_id)
            source_start = int(meta["source_start_frame"])
            source_end = int(meta["source_end_frame"])
            item_dir = output_root / f"c{index}"
            stem = f"{PLAN_STEM}_{clip_id}"
            plan_path = item_dir / f"{stem}.json"
            vmd_path = item_dir / f"{stem}.vmd"
            debug_path = item_dir / f"{stem}_debug.json"
            cut_stats_path = item_dir / f"{stem}_cut_change_stats.json"
            targets_path = Path(args.targets_root).resolve() / f"c{index}" / "real_targets.json"
            feature_path = feature_root / f"{clip_id}.npz"
            row: dict[str, object] = {
                "index": index,
                "part": "",
                "clip_id": clip_id,
                "status": "",
                "official_split": "test",
                "source_start_frame": source_start,
                "source_end_frame": source_end,
                "clip_start_frame": source_start,
                "clip_end_frame": source_end,
                "plan": str(plan_path),
                "vmd": str(vmd_path),
            }
            try:
                if not targets_path.exists():
                    raise FileNotFoundError(f"missing targets: {targets_path}")
                if not feature_path.exists():
                    raise FileNotFoundError(f"missing features: {feature_path}")
                source_frames, cut_pairs, timing_counts = external_timing_for_clip(
                    timing_by_song,
                    source_song_id=source_song_id,
                    source_start_frame=source_start,
                    source_end_frame=source_end,
                    include_boundaries=bool(args.include_boundaries),
                )
                row.update(timing_counts)
                if len(source_frames) < int(args.min_events):
                    raise ValueError(f"too few timing events for {clip_id}: {len(source_frames)}")
                source_frames_np = np.asarray(source_frames, dtype=np.int32)
                local_frames = (source_frames_np - int(source_start)).astype(np.int32)
                audio, dance = build_feature_windows(
                    feature_path=feature_path,
                    source_manifest=source_manifest,
                    local_frames=local_frames,
                    window_radius=WINDOW_RADIUS,
                    stats=stats,
                )
                cut_flag = np.zeros((local_frames.shape[0],), dtype=bool)
                if local_frames.shape[0] > 1:
                    cut_flag[1:] = np.diff(local_frames) == 1
                clip = {
                    "clip_id": clip_id,
                    "audio": audio,
                    "dance": dance,
                    "frames": local_frames,
                    "cut_flag": cut_flag,
                    "history_size": int(args.history_size),
                }
                token_ids, _log_prob, _entropy, _ce_loss = sample_sequence(
                    model,
                    clip,
                    fields=fields,
                    device=device,
                    max_position_tokens=max_position_tokens,
                    window_radius=WINDOW_RADIUS,
                    temperature=1.0,
                    sample=False,
                    gt_labels=None,
                    decode_distance_fov_joint=decode_distance_fov_joint,
                )
                keyframes = decode_keyframes(
                    source_frames=source_frames_np,
                    token_ids=token_ids,
                    fields=fields,
                    vocabularies=vocabularies,
                )
                plan_payload: dict[str, Any] = {
                    "schema": (
                        "camera_autoshot.reverse_v3.close_dense_no_wide"
                        if distance_schema is not None
                        else "camera_autoshot.reverse_v3"
                    ),
                    "source_model": str(checkpoint_path),
                    "clip_id": clip_id,
                    "source_song_id": source_song_id,
                    "source_start_frame": source_start,
                    "source_end_frame": source_end,
                    "sample_fps": 30.0,
                    "timeline_fps": 30.0,
                    "timing_source": str(timing_root),
                    "prev_context_size": int(args.history_size),
                    "keyframes": keyframes,
                }
                if distance_schema is not None:
                    plan_payload["distance_schema"] = distance_schema
                write_json(plan_path, plan_payload)
                camera_max_frame = write_vmd_from_plan(plan_payload, targets_path, vmd_path, debug_path)
                cut_stats = _cut_change_stats(
                    keyframes=keyframes,
                    cut_pairs=cut_pairs,
                    source_start_frame=source_start,
                    source_end_frame=source_end,
                    fields=list(fields),
                )
                write_json(cut_stats_path, cut_stats)
                row.update(
                    {
                        "status": "ok",
                        "event_count": int(len(source_frames)),
                        "camera_max_frame": int(max(camera_max_frame, source_end)),
                        "cut_pair_count": int(cut_stats["cut_pair_count"]),
                        "changed_pair_count": int(cut_stats["changed_pair_count"]),
                        "changed_pair_rate": float(cut_stats["changed_pair_rate"]),
                    }
                )
                part = part_from_clip_id(clip_id)
                row["split_sample_name"] = f"c{index}_{part}" if part else f"c{index}"
                print(
                    f"[ok] {clip_id}: events={len(source_frames)} "
                    f"cuts={row.get('timing_cut_count', 0)} keys={row.get('timing_keyframe_count', 0)}",
                    flush=True,
                )
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
                print(f"[fail] {clip_id}: {exc}", flush=True)
            manifest_rows.append(row)

    write_json(output_root / "manifest.json", manifest_rows)
    write_manifest_csv(manifest_rows, output_root / "manifest.csv")
    ok = [row for row in manifest_rows if row.get("status") == "ok"]
    total = sum(int(row.get("cut_pair_count", 0) or 0) for row in manifest_rows)
    changed = sum(int(row.get("changed_pair_count", 0) or 0) for row in manifest_rows)
    write_json(
        output_root / "summary.json",
        {
            "checkpoint": str(checkpoint_path),
            "timing_root": str(timing_root),
            "output_root": str(output_root),
            "history_size": int(args.history_size),
            "selected_clip_count": len(selected_rows),
            "ok_clip_count": len(ok),
            "failed_clip_count": len(manifest_rows) - len(ok),
            "cut_pair_count": total,
            "changed_pair_count": changed,
            "changed_pair_rate": (changed / total) if total else 0.0,
        },
    )
    print(f"ok clips: {len(ok)}/{len(selected_rows)}", flush=True)
    if len(ok) != len(selected_rows):
        raise SystemExit("some clips failed; see manifest.json")


if __name__ == "__main__":
    main()
