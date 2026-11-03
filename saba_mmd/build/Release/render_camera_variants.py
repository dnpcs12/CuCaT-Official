from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import shutil
import time
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from model_aliases import resolve_model_choice

RELEASE_DIR = Path(__file__).resolve().parent
BATCH_SCRIPT = RELEASE_DIR / "batch_render_gt_range.py"
INIT_PATH = RELEASE_DIR / "init.lua"
INIT_BACKUP_PATH = RELEASE_DIR / "init.camera_variant.backup.lua"
SUMMARY_CSV = RELEASE_DIR / "camera_variant_render_result.csv"
SUMMARY_JSON = RELEASE_DIR / "camera_variant_render_result.json"
LATEST_OK_CSV = RELEASE_DIR / "batch_render_result_latest_ok.csv"

RESULT_FIELDS = [
    "run_at",
    "range_start",
    "range_end",
    "index",
    "variant",
    "status",
    "model",
    "motion",
    "base_camera",
    "camera",
    "wav",
    "wav_duration_sec",
    "motion_max_frame",
    "camera_max_frame",
    "export_duration_sec",
    "export_frame_count",
    "fps",
    "output",
]

CAMERA_NAME_TOKENS = ("camera", "cam", "car.", "镜头", "カメラ")


def load_batch_module():
    spec = importlib.util.spec_from_file_location("batch_render_gt_range", BATCH_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load: {BATCH_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_name(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def load_latest_ok_rows() -> dict[int, dict[str, str]]:
    if not LATEST_OK_CSV.exists():
        return {}
    rows: dict[int, dict[str, str]] = {}
    with LATEST_OK_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            idx = row.get("index", "")
            if idx.isdigit():
                rows[int(idx)] = row
    return rows


def load_mapping(batch) -> dict[int, dict[str, str]]:
    mapping = batch.load_mapping()
    if mapping:
        return mapping

    xlsx_path = batch.XLSX_PATH
    if not xlsx_path.exists():
        return {}

    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    loaded: dict[int, dict[str, str]] = {}

    with zipfile.ZipFile(xlsx_path) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                shared.append("".join((t.text or "") for t in si.findall(".//a:t", ns)))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}

        first_sheet = workbook.find("a:sheets/a:sheet", ns)
        if first_sheet is None:
            return {}

        rel_id = first_sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        target = rel_map[rel_id]
        if not target.startswith("xl/"):
            target = f"xl/{target}"

        sheet = ET.fromstring(zf.read(target))
        for row in sheet.findall("a:sheetData/a:row", ns):
            values: dict[str, str] = {}
            for cell in row.findall("a:c", ns):
                ref = cell.attrib.get("r", "")
                col = "".join(ch for ch in ref if ch.isalpha())
                cell_type = cell.attrib.get("t")
                value_node = cell.find("a:v", ns)
                if value_node is None:
                    value = ""
                elif cell_type == "s":
                    value = shared[int(value_node.text)]
                else:
                    value = value_node.text or ""
                values[col] = value

            index_text = values.get("A", "")
            if not index_text.startswith("amc"):
                continue
            try:
                index = int(index_text[3:])
            except ValueError:
                continue
            loaded[index] = {
                "motion": values.get("B", "").strip(),
                "camera": values.get("C", "").strip(),
            }

    return loaded


def find_base_assets(
    batch,
    mapping: dict[int, dict[str, str]],
    latest_ok_rows: dict[int, dict[str, str]],
    index: int,
    override_model: Path | None,
) -> tuple[Path, Path, Path, Path, Path]:
    folder = batch.BASE_DIR / f"amc{index}"
    if not folder.exists():
        raise FileNotFoundError(f"missing folder: {folder}")

    latest_row = latest_ok_rows.get(index)

    model = override_model
    if model is None and latest_row:
        candidate = Path(latest_row.get("model", ""))
        if str(candidate).strip() and candidate.exists():
            model = candidate
    if model is None:
        model = batch.MODEL_PATH

    motion: Path | None = None
    base_camera: Path | None = None
    wav: Path | None = None

    if latest_row:
        motion_text = latest_row.get("motion", "")
        if motion_text:
            candidate = Path(motion_text)
            if candidate.exists():
                motion = candidate

        camera_text = latest_row.get("camera", "")
        if camera_text:
            candidate = Path(camera_text)
            if candidate.exists():
                base_camera = candidate

        wav_text = latest_row.get("wav", "")
        if wav_text:
            candidate = Path(wav_text)
            if candidate.exists():
                wav = candidate

    pref = mapping.get(index, {"motion": "", "camera": ""})

    if motion is None:
        motion = batch.resolve_vmd(folder, pref.get("motion", ""), "motion", index)
    if base_camera is None:
        base_camera = batch.resolve_vmd(folder, pref.get("camera", ""), "camera", index)
    if wav is None:
        wav = folder / f"a{index}.wav"

    if motion is None:
        raise FileNotFoundError(f"missing base motion in amc{index}")
    if base_camera is None:
        raise FileNotFoundError(f"missing base camera in amc{index}")
    if not wav.exists():
        raise FileNotFoundError(f"missing wav: {wav}")
    if not model.exists():
        raise FileNotFoundError(f"model not found: {model}")

    return folder, model, motion, base_camera, wav


def collect_variant_entries(folder: Path, version: int) -> list[Path]:
    prefix = f"ver{version}"
    items: list[Path] = []
    for entry in folder.iterdir():
        if normalize_name(entry.name).startswith(prefix):
            if entry.suffix.lower() in (".zip", ".rar"):
                continue
            items.append(entry)
    return items


def candidate_score(candidate: Path, stats: dict[str, int], base_motion: Path, base_camera: Path) -> tuple[int, int, int, int, int]:
    name = normalize_name(candidate.name)
    base_motion_name = normalize_name(base_motion.name)
    base_camera_name = normalize_name(base_camera.name)

    keyfix_match = 1 if "키보정" in candidate.name else 0
    needs_tda = "tda" in base_motion_name or "tda" in base_camera_name
    tda_match = 1 if "tda" in name else 0

    base_numbers = set(re.findall(r"\d+", base_camera_name))
    number_match = 1 if base_numbers and any(number in name for number in base_numbers) else 0

    camera_name_match = 1 if any(token in name for token in CAMERA_NAME_TOKENS) else 0

    return (
        keyfix_match,
        camera_name_match,
        tda_match if needs_tda else 0,
        number_match,
        stats["camera_max_frame"],
        candidate.stat().st_size,
    )


def select_variant_camera(batch, entries: list[Path], base_motion: Path, base_camera: Path) -> Path | None:
    all_vmd: list[Path] = []
    for entry in entries:
        if entry.is_dir():
            all_vmd.extend(p for p in entry.rglob("*.vmd") if p.is_file())
        elif entry.is_file() and entry.suffix.lower() == ".vmd":
            all_vmd.append(entry)

    if not all_vmd:
        return None

    stats_map = {path: batch.read_vmd_stats(path) for path in all_vmd}
    camera_candidates = [
        path
        for path in all_vmd
        if stats_map[path]["camera_count"] > 0 or any(token in normalize_name(path.name) for token in CAMERA_NAME_TOKENS)
    ]
    if not camera_candidates:
        return None

    camera_candidates.sort(
        key=lambda path: candidate_score(path, stats_map[path], base_motion, base_camera),
        reverse=True,
    )
    return camera_candidates[0]


def render_variant(
    batch,
    run_at: str,
    start: int,
    end: int,
    index: int,
    variant_version: int,
    model: Path,
    motion: Path,
    base_camera: Path,
    camera: Path,
    wav: Path,
) -> dict[str, str]:
    motion_stats = batch.read_vmd_stats(motion)
    camera_stats = batch.read_vmd_stats(camera)
    wav_duration_sec = batch.ffprobe_duration_sec(wav)
    wav_duration_float_sec = batch.ffprobe_duration_float_sec(wav)

    dance_end_frame = motion_stats["motion_max_frame"]
    camera_end_frame = camera_stats["camera_max_frame"]
    music_end_frame = max(0, int(math.floor(wav_duration_float_sec * batch.FPS)) - 1)

    output_name = f"c{index}-{variant_version - 1}.mp4"
    output_path = batch.RELEASE_DIR / output_name

    row = {
        "run_at": run_at,
        "range_start": str(start),
        "range_end": str(end),
        "index": str(index),
        "variant": f"ver{variant_version}",
        "status": "planned",
        "model": str(model),
        "motion": str(motion),
        "base_camera": str(base_camera),
        "camera": str(camera),
        "wav": str(wav),
        "wav_duration_sec": str(wav_duration_sec),
        "motion_max_frame": str(dance_end_frame),
        "camera_max_frame": str(camera_end_frame),
        "export_duration_sec": "",
        "export_frame_count": "",
        "fps": str(batch.FPS),
        "output": str(output_path),
    }

    if dance_end_frame <= 0 or camera_end_frame <= 0:
        row["status"] = "missing-vmd"
        return row

    shortest_end_frame = min(dance_end_frame, camera_end_frame, music_end_frame)
    export_frame_count = max(1, shortest_end_frame + 1)
    duration_sec = export_frame_count / batch.FPS
    row["export_duration_sec"] = f"{duration_sec:.6f}"
    row["export_frame_count"] = str(export_frame_count)

    print(f"===== c{index} ver{variant_version} planned =====")
    print(f"[plan] model={model}")
    print(f"[plan] motion={motion}")
    print(f"[plan] base_camera={base_camera}")
    print(f"[plan] camera={camera}")
    print(
        f"[plan] wav={wav} wav_duration={wav_duration_sec}s "
        f"dance_end_frame={dance_end_frame} "
        f"camera_end_frame={camera_end_frame} "
        f"music_end_frame={music_end_frame} "
        f"export_frames={export_frame_count} duration={duration_sec:.6f}s"
    )

    return row


def write_results(rows: list[dict[str, str]]) -> None:
    csv_exists = SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not csv_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})

    if SUMMARY_JSON.exists():
        try:
            existing = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
    else:
        existing = []

    existing.extend(rows)
    SUMMARY_JSON.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


def backup_init() -> bool:
    if INIT_PATH.exists():
        shutil.copy2(INIT_PATH, INIT_BACKUP_PATH)
        return True
    return False


def restore_init(had_init: bool) -> None:
    if INIT_BACKUP_PATH.exists():
        shutil.copy2(INIT_BACKUP_PATH, INIT_PATH)
        INIT_BACKUP_PATH.unlink()
    elif not had_init and INIT_PATH.exists():
        INIT_PATH.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-alias", type=str, default="")
    parser.add_argument("--versions", type=str, default="2,3,4,5")
    parser.add_argument(
        "--viewer-session-size",
        type=int,
        default=0,
        help=(
            "Number of variant clips to render per saba_viewer process. "
            "Use 0 to render all planned variants in one viewer session."
        ),
    )
    args = parser.parse_args()

    batch = load_batch_module()
    if not batch.VIEWER_EXE.exists():
        raise FileNotFoundError(f"Viewer not found: {batch.VIEWER_EXE}")

    override_model: Path | None = None
    if args.model_path or args.model_alias:
        override_model = resolve_model_choice(args.model_path, args.model_alias, batch.MODEL_PATH)
        if not override_model.exists():
            raise FileNotFoundError(f"Model not found: {override_model}")

    versions = [int(part.strip()) for part in args.versions.split(",") if part.strip()]
    latest_ok_rows = load_latest_ok_rows()
    mapping = load_mapping(batch)

    had_init = backup_init()
    run_at = time.strftime("%Y-%m-%d %H:%M:%S")
    results: list[dict[str, str]] = []

    batch.kill_viewer()
    try:
        for index in range(args.start, args.end + 1):
            try:
                folder, model, motion, base_camera, wav = find_base_assets(
                    batch,
                    mapping,
                    latest_ok_rows,
                    index,
                    override_model,
                )
            except FileNotFoundError as e:
                print(f"[fail] c{index}: {e}")
                results.append(
                    {
                        "run_at": run_at,
                        "range_start": str(args.start),
                        "range_end": str(args.end),
                        "index": str(index),
                        "variant": "",
                        "status": "missing-base",
                        "model": str(
                            override_model
                            or latest_ok_rows.get(index, {}).get("model", "")
                            or batch.MODEL_PATH
                        ),
                    }
                )
                continue

            found_any_variant = False
            for variant_version in versions:
                entries = collect_variant_entries(folder, variant_version)
                if not entries:
                    continue

                found_any_variant = True
                camera = select_variant_camera(batch, entries, motion, base_camera)
                if camera is None:
                    print(f"[fail] c{index} ver{variant_version}: no variant camera found")
                    results.append(
                        {
                            "run_at": run_at,
                            "range_start": str(args.start),
                            "range_end": str(args.end),
                            "index": str(index),
                            "variant": f"ver{variant_version}",
                            "status": "missing-camera-variant",
                            "model": str(model),
                            "motion": str(motion),
                            "base_camera": str(base_camera),
                            "wav": str(wav),
                        }
                    )
                    continue

                row = render_variant(
                    batch,
                    run_at,
                    args.start,
                    args.end,
                    index,
                    variant_version,
                    model,
                    motion,
                    base_camera,
                    camera,
                    wav,
                )
                results.append(row)

            if not found_any_variant:
                print(f"[skip] c{index}: no ver2~ver5 assets found")
                results.append(
                    {
                        "run_at": run_at,
                        "range_start": str(args.start),
                        "range_end": str(args.end),
                        "index": str(index),
                        "variant": "",
                        "status": "no-variant",
                        "model": str(model),
                        "motion": str(motion),
                        "base_camera": str(base_camera),
                        "wav": str(wav),
                    }
                )

        batch.render_planned_rows(
            results,
            override_model or batch.MODEL_PATH,
            viewer_session_size=args.viewer_session_size,
        )
    finally:
        restore_init(had_init)
        batch.kill_viewer()

    write_results(results)
    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"Summary JSON: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
