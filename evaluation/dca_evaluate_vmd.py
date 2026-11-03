"""Evaluate camera VMDs on the DCM test clips with DanceCamAnimator's metrics.

Run this file with the Python of a DanceCamAnimator environment (it imports DanceCamAnimator's
``scripts`` modules and calls its ``scripts/evaluate.py``). ``evaluation/run_dca_eval.py`` does
that for you. Each manifest row gives a VMD and the clip's frame range in the full song; the VMD is
converted to DanceCamAnimator's camera-centric format, cropped to the clip, paired with the DCM++
Test ground truth of the same clip, and the metrics are written to
``<output_root>/results_<generated_root name>/metrics_new.json``.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

FPS = 30


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_json(data, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(data, fout, indent=2, ensure_ascii=False)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as fin:
        return json.load(fin)


def normalize_part(part_value):
    if part_value in (None, ""):
        return None
    if isinstance(part_value, int):
        return part_value
    part_text = str(part_value).strip()
    if not part_text:
        return None
    if part_text.isdigit():
        return int(part_text)
    return part_text


def record_sort_key(record):
    part_value = normalize_part(record.get("part"))
    if isinstance(part_value, int):
        part_key = (0, part_value)
    elif part_value is None:
        part_key = (0, 0)
    else:
        part_key = (1, str(part_value))
    return (int(record["index"]),) + part_key


def sample_name_for_record(record):
    index = int(record["index"])
    part_value = normalize_part(record.get("part"))
    if part_value is None:
        clip_id = str(record.get("clip_id", "")).strip()
        if clip_id:
            pieces = [piece for piece in clip_id.split("_") if piece != ""]
            if len(pieces) >= 3 and pieces[1].isdigit():
                return f"c{int(pieces[1])}_{pieces[2]}"
        return f"c{index}"
    return f"c{index}_part{part_value}"


def split_sample_name_for_record(record):
    """Return the DCM++ sample name of a record, e.g. C_9_4 -> c9_4."""
    if record.get("split_sample_name"):
        return str(record["split_sample_name"])

    clip_id = str(record.get("clip_id", "")).strip()
    if clip_id:
        pieces = [piece for piece in clip_id.split("_") if piece != ""]
        if len(pieces) >= 2 and pieces[1].isdigit():
            index = int(pieces[1])
            if len(pieces) >= 3:
                return f"c{index}_{pieces[2]}"
            return f"c{index}"

    index = int(record["index"])
    part_value = normalize_part(record.get("part"))
    if part_value is None:
        return f"c{index}"
    return f"c{index}_{part_value}"


def motion_name_for_sample(sample_name: str):
    return f"m{sample_name[1:]}_gt"


def load_manifest(manifest_path: Path):
    records = load_json(manifest_path)
    if not isinstance(records, list):
        raise TypeError(f"manifest_path must contain a JSON list: {manifest_path}")
    ok_records = [record for record in records if record.get("status", "ok") == "ok"]
    ok_records.sort(key=record_sort_key)
    if not ok_records:
        raise RuntimeError(f"No manifest entries with status=ok in {manifest_path}")
    return ok_records


def convert_generated_full(records, generated_full_root: Path):
    from camera_preprocess import camera_interpolation_vmd, camera_vmd2centric
    from my_utils.vmd import Vmd

    reset_dir(generated_full_root)
    generated_keyframe_dir = generated_full_root / "CameraKeyframe"
    ensure_dir(generated_keyframe_dir)

    summaries = []
    for record in records:
        sample_name = sample_name_for_record(record)
        vmd_path = Path(record["vmd"])
        if not vmd_path.exists():
            raise FileNotFoundError(f"Missing VMD file for {sample_name}: {vmd_path}")
        print(f"[GEN] Converting {sample_name}", flush=True)
        vmd_dict = Vmd.from_file(str(vmd_path)).dict
        camera_keyframe_json = {
            "CameraKeyFrameNumber": vmd_dict["CameraKeyFrameNumber"],
            "CameraKeyFrameRecord": vmd_dict["CameraKeyFrameRecord"],
        }
        save_json(camera_keyframe_json, generated_keyframe_dir / f"{sample_name}.json")
        max_frame = 0
        if camera_keyframe_json["CameraKeyFrameRecord"]:
            max_frame = max(int(keyframe["FrameTime"]) for keyframe in camera_keyframe_json["CameraKeyFrameRecord"])
        summaries.append(
            {
                "sample_name": sample_name,
                "index": int(record["index"]),
                "part": record.get("part", ""),
                "camera_keyframe_count": int(camera_keyframe_json["CameraKeyFrameNumber"]),
                "camera_max_frame_in_vmd": max_frame,
            }
        )

    generated_interpolated_dir = generated_full_root / "CameraInterpolated"
    generated_centric_dir = generated_full_root / "CameraCentric"
    ensure_dir(generated_interpolated_dir)
    ensure_dir(generated_centric_dir)

    print("[GEN] Interpolating full generated camera keyframes", flush=True)
    camera_interpolation_vmd(str(generated_keyframe_dir), str(generated_interpolated_dir))
    print("[GEN] Converting full generated cameras to centric coordinates", flush=True)
    camera_vmd2centric(str(generated_interpolated_dir), str(generated_centric_dir))
    return summaries


def crop_camera_centric(camera_data, start_frame: int, stop_frame: int):
    cropped = {}
    for key, value in camera_data.items():
        if key == "FrameTime":
            continue
        if isinstance(value, list):
            cropped[key] = value[start_frame:stop_frame]
        else:
            cropped[key] = value
    frame_count = len(cropped.get("Fov", []))
    cropped["FrameTime"] = frame_count
    return cropped


def record_start_frame(record) -> int:
    return int(record.get("clip_start_frame", record.get("source_start_frame")))


def record_end_frame(record) -> int:
    return int(record.get("clip_end_frame", record.get("source_end_frame")))


def trim_camera_segment(camera_data, frame_count: int):
    trimmed = {}
    for key, value in camera_data.items():
        if key == "FrameTime":
            continue
        if isinstance(value, list):
            trimmed[key] = value[:frame_count]
        else:
            trimmed[key] = value
    trimmed["FrameTime"] = frame_count
    return trimmed


def trim_motion_segment(motion_data, frame_count: int):
    trimmed = dict(motion_data)
    trimmed["BoneKeyFrameTransformRecord"] = motion_data["BoneKeyFrameTransformRecord"][:frame_count]
    trimmed["BoneKeyFrameNumber"] = len(trimmed["BoneKeyFrameTransformRecord"])
    return trimmed


def build_segment_subsets_from_split_gt(records, split_gt_root: Path, generated_full_root: Path, gt_root: Path, result_root: Path):
    """Pair each generated clip with the pre-split DCM++ ground truth of the same clip."""
    from my_utils.detect_bone_mask import DetactBoneMask

    split_camera_root = split_gt_root / "CameraCentric"
    split_motion_root = split_gt_root / "Simplified_MotionGlobalTransform"
    for required_dir in (split_camera_root, split_motion_root):
        if not required_dir.exists():
            raise FileNotFoundError(f"Missing split GT directory: {required_dir}")

    reset_dir(gt_root)
    reset_dir(result_root)
    ensure_dir(gt_root / "CameraCentric")
    ensure_dir(gt_root / "Simplified_MotionGlobalTransform")
    ensure_dir(gt_root / "BoneMask")
    ensure_dir(result_root / "CameraCentric")
    ensure_dir(result_root / "BoneMask")

    gt_camera_cache = {}
    gt_motion_cache = {}
    generated_camera_cache = {}
    segment_summaries = []

    for record in records:
        sample_name = sample_name_for_record(record)
        split_sample_name = split_sample_name_for_record(record)
        output_motion_name = motion_name_for_sample(sample_name)
        split_motion_name = motion_name_for_sample(split_sample_name)

        if split_sample_name not in gt_camera_cache:
            gt_camera_cache[split_sample_name] = load_json(split_camera_root / f"{split_sample_name}.json")
        if split_sample_name not in gt_motion_cache:
            gt_motion_cache[split_sample_name] = load_json(split_motion_root / f"{split_motion_name}.json")
        if sample_name not in generated_camera_cache:
            generated_camera_cache[sample_name] = load_json(generated_full_root / "CameraCentric" / f"{sample_name}.json")

        start_frame = record_start_frame(record)
        clip_end_frame = record_end_frame(record)
        generated_end_frame = int(record.get("camera_max_frame", clip_end_frame))
        end_frame = min(clip_end_frame, generated_end_frame)
        stop_frame = end_frame + 1

        gt_camera_segment = gt_camera_cache[split_sample_name]
        gt_motion_segment = gt_motion_cache[split_sample_name]
        generated_camera_segment = crop_camera_centric(generated_camera_cache[sample_name], start_frame, stop_frame)

        actual_frame_count = min(
            len(gt_camera_segment.get("Fov", [])),
            int(gt_motion_segment["BoneKeyFrameNumber"]),
            len(generated_camera_segment.get("Fov", [])),
        )
        if actual_frame_count <= 0:
            raise RuntimeError(
                f"Segment {sample_name} has no overlapping frames after cropping: "
                f"split_sample={split_sample_name}, start={start_frame}, end={end_frame}"
            )

        gt_camera_segment = trim_camera_segment(gt_camera_segment, actual_frame_count)
        gt_motion_segment = trim_motion_segment(gt_motion_segment, actual_frame_count)
        generated_camera_segment = trim_camera_segment(generated_camera_segment, actual_frame_count)

        save_json(gt_camera_segment, gt_root / "CameraCentric" / f"{sample_name}.json")
        save_json(gt_motion_segment, gt_root / "Simplified_MotionGlobalTransform" / f"{output_motion_name}.json")
        save_json(generated_camera_segment, result_root / "CameraCentric" / f"{sample_name}.json")

        segment_summaries.append(
            {
                "sample_name": sample_name,
                "split_sample_name": split_sample_name,
                "index": int(record["index"]),
                "part": record.get("part", ""),
                "clip_id": record.get("clip_id", ""),
                "clip_start_frame": start_frame,
                "clip_end_frame": clip_end_frame,
                "camera_max_frame": generated_end_frame,
                "requested_stop_exclusive": stop_frame,
                "actual_frame_count": actual_frame_count,
                "segment_duration_sec": actual_frame_count / float(FPS),
            }
        )

    print("[GT] Building GT bone masks from split GT", flush=True)
    DetactBoneMask(
        str(gt_root / "CameraCentric"),
        str(gt_root / "Simplified_MotionGlobalTransform"),
        str(gt_root / "BoneMask"),
    )
    print("[GEN] Building generated bone masks", flush=True)
    DetactBoneMask(
        str(result_root / "CameraCentric"),
        str(gt_root / "Simplified_MotionGlobalTransform"),
        str(result_root / "BoneMask"),
    )
    return segment_summaries


def main():
    parser = argparse.ArgumentParser(description="Evaluate camera VMDs on the DCM test clips with DanceCamAnimator metrics.")
    parser.add_argument("--dca-root", required=True, help="DanceCamAnimator-Official checkout (with evaluation fixes applied).")
    parser.add_argument("--generated-root", required=True, help="Folder of the generated VMDs; its name tags the outputs.")
    parser.add_argument("--manifest-path", required=True, help="manifest_dca_eval_input.json from evaluation/dca_manifest.py.")
    parser.add_argument("--split-gt-root", required=True, help="DCM++ Test folder (CameraCentric, Simplified_MotionGlobalTransform).")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--skip-eval", action="store_true", help="Only build the converted inputs.")
    args = parser.parse_args()

    dca_root = Path(args.dca_root).resolve()
    scripts_dir = dca_root / "scripts"
    evaluate_script = scripts_dir / "evaluate.py"
    if not evaluate_script.exists():
        raise FileNotFoundError(f"DanceCamAnimator evaluate.py not found: {evaluate_script}")
    sys.path.insert(0, str(scripts_dir))

    generated_root = Path(args.generated_root).resolve()
    manifest_path = Path(args.manifest_path).resolve()
    split_gt_root = Path(args.split_gt_root).resolve()
    output_root = Path(args.output_root).resolve()
    for path in (generated_root, manifest_path, split_gt_root):
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}")

    records = load_manifest(manifest_path)
    subset_tag = generated_root.name
    gt_root = output_root / f"gt_subset_{subset_tag}"
    result_root = output_root / f"results_{subset_tag}"
    generated_full_root = output_root / "cache" / "generated_full"

    generated_full_summary = convert_generated_full(records, generated_full_root)
    segment_summary = build_segment_subsets_from_split_gt(records, split_gt_root, generated_full_root, gt_root, result_root)
    save_json(
        {
            "generated_root": str(generated_root),
            "manifest_path": str(manifest_path),
            "split_gt_root": str(split_gt_root),
            "record_count": len(records),
            "unique_index_count": len({int(record["index"]) for record in records}),
            "generated_full_summary": generated_full_summary,
            "segment_summary": segment_summary,
        },
        output_root / "manifest_eval_summary.json",
    )

    if args.skip_eval:
        print("[SKIP] Evaluation skipped", flush=True)
        return
    subprocess.run(
        [sys.executable, str(evaluate_script), "--result_dir", str(result_root), "--test_dir", str(gt_root)],
        cwd=str(dca_root),
        check=True,
    )


if __name__ == "__main__":
    main()
