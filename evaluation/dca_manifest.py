"""Match generated test-clip VMDs to the DCM++ Test ground truth for the DCA evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.dcm_common import DEFAULT_TEST_CLIPS, load_json, write_json


def split_sample_name_from_clip_id(clip_id: str) -> str:
    """DCM clip id -> DCM++ sample name, e.g. C_9_4 -> c9_4."""

    pieces = [piece for piece in clip_id.split("_") if piece]
    if len(pieces) >= 3:
        return f"c{int(pieces[1])}_{pieces[2]}"
    if len(pieces) >= 2:
        return f"c{int(pieces[1])}"
    raise ValueError(f"Cannot parse clip_id: {clip_id}")


def split_camera_len(split_gt_root: Path, sample_name: str) -> int | None:
    path = split_gt_root / "CameraCentric" / f"{sample_name}.json"
    if not path.exists():
        return None
    payload = load_json(path)
    fov = payload.get("Fov") if isinstance(payload, dict) else None
    return len(fov) if isinstance(fov, list) else None


def build_split_lookup(test_clips_path: Path = DEFAULT_TEST_CLIPS) -> dict[int, list[dict[str, Any]]]:
    payload = load_json(test_clips_path)
    lookup: dict[int, list[dict[str, Any]]] = {}
    for row in payload.get("clips", []):
        if row.get("official_split") != "test":
            continue
        index = int(str(row["source_song_id"]).lstrip("c"))
        lookup.setdefault(index, []).append(row)
    for rows in lookup.values():
        rows.sort(key=lambda row: (int(row["source_start_frame"]), int(row["source_end_frame"])))
    return lookup


def resolve_manifest_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_dir / path).resolve()


def build_dca_manifest(
    vmd_root: Path,
    split_lookup: dict[int, list[dict[str, Any]]],
    split_gt_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Write ``manifest_dca_eval_input.json``: one row per test clip with its VMD and clip range.

    ``vmd_root/manifest.json`` lists the generated clips (``index``, ``clip_start_frame``,
    ``clip_end_frame``, ``vmd``; relative VMD paths are resolved against ``vmd_root``).
    """

    manifest_path = vmd_root / "manifest.json"
    rows = load_json(manifest_path)
    if not isinstance(rows, list):
        raise TypeError(f"manifest must be a JSON list: {manifest_path}")

    used_clip_ids: set[str] = set()
    converted: list[dict[str, Any]] = []
    missing_matches: list[dict[str, Any]] = []
    missing_vmd: list[str] = []
    missing_split_files: list[str] = []

    def row_sort_key(row: dict[str, Any]) -> tuple[int, int]:
        part = row.get("part") or 0
        try:
            part_i = int(part)
        except Exception:
            part_i = 0
        return int(row.get("index", 0)), part_i

    for row in sorted([r for r in rows if isinstance(r, dict)], key=row_sort_key):
        if row.get("status") not in ("ok", "ok-vmd-only", "ok-existing"):
            continue
        index = int(row["index"])
        start = int(row.get("clip_start_frame", row.get("source_start_frame")))
        end = int(row.get("clip_end_frame", row.get("source_end_frame")))
        frame_count = int(row.get("segment_frame_count") or (end - start + 1))
        candidates = [
            candidate
            for candidate in split_lookup.get(index, [])
            if str(candidate["clip_id"]) not in used_clip_ids
        ]
        scored: list[tuple[int, int, dict[str, Any], str, int | None]] = []
        for candidate in candidates:
            sample_name = split_sample_name_from_clip_id(str(candidate["clip_id"]))
            score = abs(int(candidate["source_start_frame"]) - start) + abs(int(candidate["source_end_frame"]) - end)
            split_len = split_camera_len(split_gt_root, sample_name)
            len_delta = 999999 if split_len is None else abs(split_len - frame_count)
            scored.append((score, len_delta, candidate, sample_name, split_len))
        scored.sort(key=lambda item: (item[0], item[1]))
        if not scored:
            missing_matches.append({"index": index, "part": row.get("part", ""), "reason": "no split candidate"})
            continue
        best_score, len_delta, best, split_sample_name, split_len = scored[0]
        if not (best_score <= 4 or len_delta <= 2 or len(candidates) == 1):
            missing_matches.append(
                {
                    "index": index,
                    "part": row.get("part", ""),
                    "clip_start_frame": start,
                    "clip_end_frame": end,
                    "best_score": best_score,
                    "len_delta": len_delta,
                    "best_clip_id": best["clip_id"],
                }
            )
            continue
        used_clip_ids.add(str(best["clip_id"]))
        vmd_path = resolve_manifest_path(str(row["vmd"]), vmd_root)
        if not vmd_path.exists():
            missing_vmd.append(str(vmd_path))
        if not (split_gt_root / "CameraCentric" / f"{split_sample_name}.json").exists():
            missing_split_files.append(split_sample_name)

        out_row = dict(row)
        out_row["status"] = "ok"
        out_row["vmd"] = str(vmd_path)
        out_row["clip_start_frame"] = start
        out_row["clip_end_frame"] = end
        out_row["clip_id"] = str(best["clip_id"])
        out_row["split_sample_name"] = split_sample_name
        out_row["split_source_start_frame"] = int(best["source_start_frame"])
        out_row["split_source_end_frame"] = int(best["source_end_frame"])
        out_row["split_match_frame_delta"] = int(best_score)
        out_row["split_camera_frame_count"] = split_len
        converted.append(out_row)

    unused = [
        str(row["clip_id"])
        for rows_for_index in split_lookup.values()
        for row in rows_for_index
        if str(row["clip_id"]) not in used_clip_ids
    ]
    summary = {
        "source_manifest": str(manifest_path),
        "record_count": len(converted),
        "unique_indices": len({int(row["index"]) for row in converted}),
        "missing_matches": missing_matches,
        "missing_vmd": missing_vmd,
        "missing_split_files": sorted(set(missing_split_files)),
        "unused_test_split_clip_ids": unused,
    }
    if missing_matches or missing_vmd or missing_split_files or unused:
        raise RuntimeError(f"DCA manifest mapping incomplete for {vmd_root}: {summary}")

    out_path = vmd_root / "manifest_dca_eval_input.json"
    write_json(out_path, converted)
    write_json(vmd_root / "manifest_dca_eval_input_summary.json", summary)
    return out_path, summary
