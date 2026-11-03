from __future__ import annotations

import csv
import json
import math
import os
import argparse
import subprocess
import time
import zipfile
import struct
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from model_aliases import resolve_model_choice

try:
    import openpyxl
except ImportError:
    openpyxl = None

RELEASE_DIR = Path(__file__).resolve().parent
LOCAL_ASSETS_DIR = RELEASE_DIR.parents[2] / "local_assets"  # repository-root local_assets/
BASE_DIR = LOCAL_ASSETS_DIR / "amc_raw_data"
XLSX_PATH = LOCAL_ASSETS_DIR / "LinkOfRawData.xlsx"
MODEL_PATH = LOCAL_ASSETS_DIR / "default_model.pmx"
DOME_PATH = LOCAL_ASSETS_DIR / "dome12.pmx"
GRID_PATH = LOCAL_ASSETS_DIR / "grid.pmx"
VIEWER_EXE = RELEASE_DIR / "saba_viewer.exe"
FPS = 30
WIDTH = 1280
HEIGHT = 720
SHADOW_COLOR = (0.72, 0.72, 0.72, 0.45)
SUMMARY_CSV = RELEASE_DIR / "batch_render_result.csv"
SUMMARY_JSON = RELEASE_DIR / "batch_render_result.json"
LATEST_OK_CSV = RELEASE_DIR / "batch_render_result_latest_ok.csv"
DEFAULT_PLAN_XLSX = RELEASE_DIR / "export_frame_plan.xlsx"
WORKERS_DIR = RELEASE_DIR / "workers"
DEFAULT_WORKER_IDS = ("agent01", "agent02", "agent03")
DEFAULT_WORKER_LOCK_WAIT_SEC = 600.0
DEFAULT_WORKER_LOCK_STALE_SEC = 12 * 60 * 60

RESULT_FIELDS = [
    "run_at",
    "range_start",
    "range_end",
    "index",
    "status",
    "model",
    "motion",
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


def _normalize_latest_ok_output_path(output_text: str) -> str:
    output_text = (output_text or "").strip()
    if not output_text:
        return output_text
    candidate = Path(output_text)
    local_candidate = RELEASE_DIR / candidate.name
    if local_candidate.exists():
        return str(local_candidate)
    return str(candidate)


def _merge_result_fieldnames(*fieldname_lists: list[str]) -> list[str]:
    merged = list(RESULT_FIELDS)
    for fieldnames in fieldname_lists:
        for name in fieldnames or []:
            if name not in merged:
                merged.append(name)
    return merged


def _candidate_latest_ok_csvs(latest_ok_csv: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in sorted(latest_ok_csv.parent.glob("batch_render_result_latest_ok*.csv")):
        if path == latest_ok_csv:
            continue
        candidates.append(path)
    candidates.append(latest_ok_csv)
    return candidates


def _row_output_exists(row: dict[str, str]) -> bool:
    output_text = _normalize_latest_ok_output_path(row.get("output", ""))
    if not output_text:
        return False
    return Path(output_text).exists()


def _path_is_valid_motion_vmd(path: Path | None) -> bool:
    if path is None or not path.exists():
        return False
    stats = read_vmd_stats(path)
    return stats.get("motion_max_frame", 0) > 0 or stats.get("camera_count", 0) == 0


def _path_is_valid_camera_vmd(path: Path | None) -> bool:
    if path is None or not path.exists():
        return False
    stats = read_vmd_stats(path)
    return stats.get("camera_count", 0) > 0


def _row_motion_is_valid(row: dict[str, str]) -> bool:
    motion_text = (row.get("motion") or "").strip()
    if not motion_text:
        return False
    return _path_is_valid_motion_vmd(Path(motion_text))


def _normalize_latest_ok_row(row: dict[str, str], fieldnames: list[str]) -> dict[str, str]:
    normalized = {key: row.get(key, "") for key in fieldnames}
    normalized["status"] = "ok"
    normalized["output"] = _normalize_latest_ok_output_path(normalized.get("output", ""))
    return normalized


def _read_ok_rows_from_csv(path: Path, fieldnames: list[str]) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}

    rows_by_index: dict[int, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("status") or "").strip().lower() != "ok":
                continue
            index_text = (row.get("index") or row.get("idx") or "").strip()
            if not index_text.isdigit():
                continue
            normalized = _normalize_latest_ok_row(row, fieldnames)
            if not _row_output_exists(normalized):
                continue
            rows_by_index[int(index_text)] = normalized
    return rows_by_index


def _existing_render_indices() -> list[int]:
    indices: list[int] = []
    for path in RELEASE_DIR.glob("c*.mp4"):
        if not path.is_file():
            continue
        stem = path.stem
        if len(stem) < 2 or stem[0].lower() != "c" or not stem[1:].isdigit():
            continue
        indices.append(int(stem[1:]))
    return sorted(set(indices))


def _rebuild_row_from_assets(
    index: int,
    fieldnames: list[str],
    template_row: dict[str, str] | None,
    mapping: dict[int, dict[str, str]],
) -> dict[str, str] | None:
    folder = BASE_DIR / f"amc{index}"
    output_path = RELEASE_DIR / f"c{index}.mp4"
    if not folder.exists() or not output_path.exists():
        return None

    wav = folder / f"a{index}.wav"
    pref = mapping.get(index, {"motion": "", "camera": ""})
    template_motion_text = (template_row or {}).get("motion", "").strip()
    template_camera_text = (template_row or {}).get("camera", "").strip()
    template_motion = Path(template_motion_text) if template_motion_text else None
    template_camera = Path(template_camera_text) if template_camera_text else None

    motion = (
        template_motion
        if _path_is_valid_motion_vmd(template_motion)
        else resolve_vmd(folder, pref.get("motion", ""), "motion", index)
    )
    camera = (
        template_camera
        if _path_is_valid_camera_vmd(template_camera)
        else resolve_vmd(folder, pref.get("camera", ""), "camera", index)
    )
    if motion is None or camera is None or not wav.exists():
        return None

    row = {key: "" for key in fieldnames}
    if template_row:
        for key in fieldnames:
            row[key] = template_row.get(key, "")

    row["run_at"] = row.get("run_at", "") or time.strftime("%Y-%m-%d %H:%M:%S")
    row["range_start"] = row.get("range_start", "") or str(index)
    row["range_end"] = row.get("range_end", "") or str(index)
    row["index"] = str(index)
    row["status"] = "ok"
    row["model"] = row.get("model", "") or str(MODEL_PATH)
    row["motion"] = str(motion)
    row["camera"] = str(camera)
    row["wav"] = str(wav)
    row["fps"] = row.get("fps", "") or str(FPS)
    row["output"] = str(output_path)

    motion_stats = read_vmd_stats(motion)
    camera_stats = read_vmd_stats(camera)
    row["motion_max_frame"] = str(motion_stats.get("motion_max_frame", 0))
    row["camera_max_frame"] = str(camera_stats.get("camera_max_frame", 0))

    try:
        wav_duration_sec = ffprobe_duration_sec(wav)
        wav_duration_float_sec = ffprobe_duration_float_sec(wav)
        row["wav_duration_sec"] = str(wav_duration_sec)

        dance_end_frame = motion_stats.get("motion_max_frame", 0)
        camera_end_frame = camera_stats.get("camera_max_frame", 0)
        music_end_frame = max(0, int(math.floor(wav_duration_float_sec * FPS)) - 1)
        shortest_end_frame = min(dance_end_frame, camera_end_frame, music_end_frame)
        export_frame_count = max(1, shortest_end_frame + 1)
        row["export_frame_count"] = str(export_frame_count)
        row["export_duration_sec"] = f"{export_frame_count / FPS:.6f}"
    except Exception:
        pass

    return row


def rewrite_latest_ok_csv(summary_csv: Path = SUMMARY_CSV, latest_ok_csv: Path = LATEST_OK_CSV) -> None:
    if not summary_csv.exists() and not latest_ok_csv.exists():
        return

    fieldnames = list(RESULT_FIELDS)
    if summary_csv.exists():
        with summary_csv.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = _merge_result_fieldnames(fieldnames, list(reader.fieldnames or []))
    if latest_ok_csv.exists():
        with latest_ok_csv.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = _merge_result_fieldnames(fieldnames, list(reader.fieldnames or []))

    latest_rows: dict[int, dict[str, str]] = {}

    for candidate_csv in _candidate_latest_ok_csvs(latest_ok_csv):
        latest_rows.update(_read_ok_rows_from_csv(candidate_csv, fieldnames))

    if summary_csv.exists():
        latest_rows.update(_read_ok_rows_from_csv(summary_csv, fieldnames))

    mapping = load_mapping()
    for index in _existing_render_indices():
        current_row = latest_rows.get(index)
        current_camera_text = (current_row or {}).get("camera", "").strip()
        current_camera_exists = bool(current_camera_text) and Path(current_camera_text).exists()
        current_motion_valid = current_row is not None and _row_motion_is_valid(current_row)
        if current_row is not None and current_camera_exists and current_motion_valid:
            continue
        rebuilt_row = _rebuild_row_from_assets(index, fieldnames, current_row, mapping)
        if rebuilt_row is not None:
            latest_rows[index] = rebuilt_row

    with latest_ok_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in sorted(latest_rows):
            writer.writerow(latest_rows[index])


def to_lua_path(p: Path) -> str:
    return str(p).replace("\\", "/")


def kill_viewer() -> None:
    subprocess.run(["taskkill", "/F", "/IM", "saba_viewer.exe"], capture_output=True, text=True)


def terminate_viewer_process(proc: subprocess.Popen, timeout_sec: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            text=True,
        )
    else:
        proc.terminate()
    try:
        proc.wait(timeout=timeout_sec)
    except Exception:
        pass


def _is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
            )
        except Exception:
            return False
        stdout = result.stdout.strip()
        return str(pid) in stdout and "No tasks" not in stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _safe_worker_id(worker_id: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in worker_id.strip())
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"invalid worker id: {worker_id!r}")
    return cleaned


def _worker_candidates(worker_id: str | None) -> list[str]:
    requested = (worker_id or "auto").strip()
    if requested.lower() != "auto":
        return [_safe_worker_id(requested)]

    candidates: list[str] = []
    if WORKERS_DIR.exists():
        for path in sorted(WORKERS_DIR.iterdir()):
            if path.is_dir() and path.name not in candidates:
                candidates.append(path.name)
    for default_id in DEFAULT_WORKER_IDS:
        if default_id not in candidates:
            candidates.append(default_id)
    return candidates


def _read_json_file(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _remove_stale_worker_lock(worker_dir: Path, stale_sec: float) -> bool:
    lock_path = worker_dir / "worker.lock"
    lease_path = worker_dir / "lease.json"
    if not lock_path.exists():
        return False

    lock_payload = _read_json_file(lock_path)
    lease_payload = _read_json_file(lease_path)
    owner_pid = int(lock_payload.get("owner_pid") or lease_payload.get("owner_pid") or 0)
    viewer_pid = int(lease_payload.get("viewer_pid") or 0)
    started_at = float(lock_payload.get("started_at") or lease_payload.get("started_at") or 0.0)
    age_sec = max(0.0, time.time() - started_at) if started_at > 0 else stale_sec + 1

    owner_alive = _is_pid_running(owner_pid)
    if owner_pid > 0 and owner_alive:
        return False
    stale = (owner_pid > 0 and not owner_alive) or age_sec >= stale_sec
    if not stale:
        return False

    if viewer_pid > 0 and _is_pid_running(viewer_pid):
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(viewer_pid)],
            capture_output=True,
            text=True,
        )
    for path in (lock_path, lease_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except PermissionError:
            return False
    return True


def prepare_saba_worker_dir(worker_dir: Path) -> Path:
    worker_dir.mkdir(parents=True, exist_ok=True)
    resource_src = RELEASE_DIR / "resource"
    resource_dst = worker_dir / "resource"
    if resource_src.exists() and not resource_dst.exists():
        try:
            os.symlink(resource_src, resource_dst, target_is_directory=True)
        except Exception:
            shutil.copytree(resource_src, resource_dst)
    return worker_dir


class SabaWorkerLease:
    def __init__(self, worker_id: str, worker_dir: Path, token: str):
        self.worker_id = worker_id
        self.worker_dir = worker_dir
        self.token = token
        self.lock_path = worker_dir / "worker.lock"
        self.lease_path = worker_dir / "lease.json"

    def update(self, **extra: object) -> None:
        payload = _read_json_file(self.lease_path)
        payload.update(extra)
        payload["worker_id"] = self.worker_id
        payload["worker_dir"] = str(self.worker_dir)
        payload["token"] = self.token
        payload["owner_pid"] = os.getpid()
        self.lease_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def release(self) -> None:
        lock_payload = _read_json_file(self.lock_path)
        if lock_payload.get("token") != self.token:
            return
        for path in (self.lease_path, self.lock_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> "SabaWorkerLease":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def _try_acquire_worker_lock(worker_id: str, worker_dir: Path, lease_info: dict[str, object]) -> SabaWorkerLease | None:
    worker_dir.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{int(time.time() * 1000)}"
    payload = {
        "worker_id": worker_id,
        "worker_dir": str(worker_dir),
        "token": token,
        "owner_pid": os.getpid(),
        "started_at": time.time(),
        **lease_info,
    }
    try:
        fd = os.open(str(worker_dir / "worker.lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    except PermissionError:
        return None
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    lease = SabaWorkerLease(worker_id, worker_dir, token)
    try:
        prepare_saba_worker_dir(worker_dir)
        lease.update(**payload)
    except Exception:
        lease.release()
        raise
    return lease


def acquire_saba_worker(
    worker_id: str | None = "auto",
    *,
    worker_dir: Path | None = None,
    wait_sec: float = DEFAULT_WORKER_LOCK_WAIT_SEC,
    stale_sec: float = DEFAULT_WORKER_LOCK_STALE_SEC,
    lease_info: dict[str, object] | None = None,
) -> SabaWorkerLease:
    lease_payload = lease_info or {}
    deadline = time.time() + max(0.0, wait_sec)
    while True:
        if worker_dir is not None:
            candidate_dir = Path(worker_dir).resolve()
            candidate_id = _safe_worker_id(candidate_dir.name)
            _remove_stale_worker_lock(candidate_dir, stale_sec)
            lease = _try_acquire_worker_lock(candidate_id, candidate_dir, lease_payload)
            if lease is not None:
                return lease
        else:
            for candidate_id in _worker_candidates(worker_id):
                candidate_dir = WORKERS_DIR / candidate_id
                _remove_stale_worker_lock(candidate_dir, stale_sec)
                lease = _try_acquire_worker_lock(candidate_id, candidate_dir, lease_payload)
                if lease is not None:
                    return lease

        if time.time() >= deadline:
            requested = str(worker_dir) if worker_dir is not None else (worker_id or "auto")
            raise TimeoutError(f"no free Saba worker available: {requested}")
        time.sleep(2.0)


def _rar_extract_cmds(archive: Path, out_dir: Path) -> list[list[str]]:
    cmds: list[list[str]] = []
    candidates = [
        shutil.which("7z"),
        shutil.which("7za"),
        shutil.which("unrar"),
        shutil.which("WinRAR"),
        str(Path(r"C:\Program Files\7-Zip\7z.exe")),
        str(Path(r"C:\Program Files (x86)\7-Zip\7z.exe")),
        str(Path(r"C:\Program Files\WinRAR\WinRAR.exe")),
        str(Path(r"C:\Program Files (x86)\WinRAR\WinRAR.exe")),
    ]
    seen: set[str] = set()
    for c in candidates:
        if not c:
            continue
        if c in seen:
            continue
        seen.add(c)
        p = Path(c)
        if p.is_absolute() and not p.exists():
            continue
        name = p.name.lower()
        if name in ("7z", "7z.exe", "7za", "7za.exe"):
            cmds.append([str(c), "x", "-y", f"-o{out_dir}", str(archive)])
        elif name in ("unrar", "unrar.exe"):
            cmds.append([str(c), "x", "-o+", str(archive), str(out_dir)])
        elif name in ("winrar", "winrar.exe"):
            cmds.append([str(c), "x", "-y", str(archive), str(out_dir)])
    return cmds


def _extract_rar(archive: Path, out_dir: Path) -> bool:
    cmds = _rar_extract_cmds(archive, out_dir)
    if not cmds:
        print(f"[warn] no extractor found for rar: {archive}")
        return False
    for cmd in cmds:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                return True
        except Exception:
            pass
    return False


def unzip_all(folder: Path) -> None:
    archives = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".zip", ".rar")]
    for z in archives:
        try:
            if z.suffix.lower() == ".zip":
                with zipfile.ZipFile(z, "r") as zf:
                    zf.extractall(folder)
                print(f"[unzip] {z.name}")
            else:
                ok = _extract_rar(z, folder)
                if ok:
                    print(f"[unrar] {z.name}")
                else:
                    print(f"[warn] unrar failed: {z}")
        except Exception as e:
            print(f"[warn] unzip failed: {z} ({e})")


def load_mapping() -> dict[int, dict[str, str]]:
    if openpyxl is None:
        if not XLSX_PATH.exists():
            print(f"[warn] mapping XLSX not found: {XLSX_PATH}")
            return {}

        ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        mapping: dict[int, dict[str, str]] = {}
        with zipfile.ZipFile(XLSX_PATH) as zf:
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

                idx = values.get("A", "")
                if not isinstance(idx, str) or not idx.startswith("amc"):
                    continue
                try:
                    n = int(idx[3:])
                except ValueError:
                    continue
                mapping[n] = {
                    "motion": values.get("B", "").strip(),
                    "camera": values.get("C", "").strip(),
                }

        print("[warn] openpyxl not installed; using XLSX XML fallback for motion/camera mapping")
        return mapping
    if not XLSX_PATH.exists():
        print(f"[warn] mapping XLSX not found: {XLSX_PATH}")
        return {}
    wb = openpyxl.load_workbook(XLSX_PATH, data_only=True)
    ws = wb[wb.sheetnames[0]]
    mapping: dict[int, dict[str, str]] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        idx, motion, camera = row[0], row[1], row[2]
        if not isinstance(idx, str) or not idx.startswith("amc"):
            continue
        try:
            n = int(idx[3:])
        except ValueError:
            continue
        mapping[n] = {
            "motion": (motion or "").strip() if isinstance(motion, str) else "",
            "camera": (camera or "").strip() if isinstance(camera, str) else "",
        }
    return mapping


def _read_u32(f) -> int:
    b = f.read(4)
    if len(b) != 4:
        raise EOFError("unexpected EOF while reading u32")
    return struct.unpack("<I", b)[0]


def read_vmd_stats(vmd_path: Path) -> dict[str, int]:
    stats = {
        "bone_count": 0,
        "morph_count": 0,
        "camera_count": 0,
        "motion_max_frame": 0,
        "camera_max_frame": 0,
    }
    try:
        with vmd_path.open("rb") as f:
            _ = f.read(30)  # header
            _ = f.read(20)  # model name

            bone_count = _read_u32(f)
            stats["bone_count"] = bone_count
            bone_max = 0
            for _i in range(bone_count):
                _ = f.read(15)
                frame = _read_u32(f)
                if frame > bone_max:
                    bone_max = frame
                _ = f.read(12 + 16 + 64)

            morph_count = _read_u32(f)
            stats["morph_count"] = morph_count
            morph_max = 0
            for _i in range(morph_count):
                _ = f.read(15)
                frame = _read_u32(f)
                if frame > morph_max:
                    morph_max = frame
                _ = f.read(4)

            camera_count = _read_u32(f)
            stats["camera_count"] = camera_count
            camera_max = 0
            for _i in range(camera_count):
                frame = _read_u32(f)
                if frame > camera_max:
                    camera_max = frame
                _ = f.read(57)

            stats["motion_max_frame"] = max(bone_max, morph_max)
            stats["camera_max_frame"] = camera_max
    except Exception:
        pass
    return stats


def resolve_vmd(folder: Path, preferred: str, kind: str, idx: int) -> Path | None:
    all_vmd = [p for p in folder.rglob("*.vmd") if p.is_file()]
    if not all_vmd:
        return None

    stats_map: dict[Path, dict[str, int]] = {p: read_vmd_stats(p) for p in all_vmd}

    if kind == "motion":
        typed_candidates = [
            p
            for p in all_vmd
            if stats_map[p]["motion_max_frame"] > 0 or stats_map[p]["camera_count"] == 0
        ]
    else:
        typed_candidates = [p for p in all_vmd if stats_map[p]["camera_count"] > 0]

    if kind == "camera" and not typed_candidates:
        # Avoid selecting arbitrary motion VMD as camera when camera keyframes are absent.
        named_cam_candidates = [
            p
            for p in all_vmd
            if any((token in p.name.lower()) for token in ("camera", "cam", "car."))
            or ("镜头" in p.name)
            or ("カメラ" in p.name)
        ]
        if named_cam_candidates:
            named_cam_candidates.sort(
                key=lambda p: (stats_map[p]["camera_max_frame"], p.stat().st_size),
                reverse=True,
            )
            return named_cam_candidates[0]
        return None

    if not typed_candidates:
        typed_candidates = all_vmd

    if preferred and preferred != "/":
        for p in typed_candidates:
            if p.name == preferred:
                return p

    if kind == "motion":
        expected = f"m{idx}.vmd".lower()
    else:
        expected = f"c{idx}.vmd".lower()
    for p in typed_candidates:
        if p.name.lower() == expected:
            return p

    if kind == "motion":
        motion_candidates = [
            p
            for p in typed_candidates
            if not any(token in p.name.lower() for token in ("camera", "cam", "car."))
            and ("镜头" not in p.name)
            and ("カメラ" not in p.name)
        ]
        if motion_candidates:
            motion_candidates.sort(
                key=lambda p: (stats_map[p]["motion_max_frame"], p.stat().st_size),
                reverse=True,
            )
            return motion_candidates[0]
    else:
        cam_candidates = [
            p
            for p in typed_candidates
            if any((token in p.name.lower()) for token in ("camera", "cam", "car."))
            or ("镜头" in p.name)
            or ("カメラ" in p.name)
        ]
        if cam_candidates:
            cam_candidates.sort(
                key=lambda p: (stats_map[p]["camera_max_frame"], p.stat().st_size),
                reverse=True,
            )
            return cam_candidates[0]

    typed_candidates.sort(
        key=lambda p: (
            stats_map[p]["camera_max_frame"] if kind == "camera" else stats_map[p]["motion_max_frame"],
            p.stat().st_size,
        ),
        reverse=True,
    )
    return typed_candidates[0]


def ffprobe_duration_sec(wav_path: Path) -> int:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(wav_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    d = float(r.stdout.strip().splitlines()[0])
    return int(math.ceil(d))


def ffprobe_duration_float_sec(wav_path: Path) -> float:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(wav_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(r.stdout.strip().splitlines()[0])


def write_plan_xlsx(rows: list[dict[str, str]], xlsx_path: Path) -> None:
    if openpyxl is None:
        print(f"[warn] openpyxl not installed; skipping plan XLSX write: {xlsx_path}")
        return
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "export_plan"
    headers = [
        "index",
        "status",
        "model",
        "motion",
        "camera",
        "wav",
        "wav_duration_sec",
        "dance_end_frame",
        "camera_end_frame",
        "motion_max_frame",
        "camera_max_frame",
        "export_duration_sec",
        "export_frame_count",
        "fps",
        "output",
    ]
    ws.append(headers)
    for r in rows:
        ws.append([r.get(h, "") for h in headers])
    wb.save(xlsx_path)


def cleanup_export_frame_dirs(work_dir: Path | None = None) -> int:
    base_dir = Path(work_dir) if work_dir is not None else RELEASE_DIR
    removed = 0
    for p in base_dir.glob("export_frames_*"):
        if not p.is_dir():
            continue
        try:
            shutil.rmtree(p)
            removed += 1
        except Exception as e:
            print(f"[warn] cleanup failed: {p} ({e})")
    return removed


def write_init_lua(
    motion: Path,
    camera: Path,
    wav: Path,
    output_name: str,
    duration_sec: float,
    *,
    fps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    msaa_enable: bool = True,
    msaa_count: int = 8,
    work_dir: Path | None = None,
) -> None:
    lua_dir = Path(work_dir) if work_dir is not None else RELEASE_DIR
    lua_dir.mkdir(parents=True, exist_ok=True)
    render_fps = int(fps or FPS)
    render_width = int(width or WIDTH)
    render_height = int(height or HEIGHT)
    msaa_enable_lua = "true" if msaa_enable else "false"
    stage_assets = "".join(_lua_command("open", (to_lua_path(path),)) + "\n" for path in _stage_asset_paths())
    lua = f'''MSAA = {{
    Enable = {msaa_enable_lua},
    Count  = {int(msaa_count)}
}}

Commands = {{
    {{
        Cmd  = "open",
        Args = {{"{to_lua_path(MODEL_PATH)}"}},
    }},
    {{
        Cmd  = "open",
        Args = {{"{to_lua_path(motion)}"}},
    }},
{stage_assets}    {{
        Cmd  = "select",
        Args = {{"model_001"}},
    }},
    {{
        Cmd  = "open",
        Args = {{"{to_lua_path(camera)}"}},
    }},
    {{
        Cmd  = "setGroundShadowColor",
        Args = {{"{SHADOW_COLOR[0]}", "{SHADOW_COLOR[1]}", "{SHADOW_COLOR[2]}", "{SHADOW_COLOR[3]}"}},
    }},
    {{
        Cmd  = "exportVideo",
        Args = {{
            "{output_name}",
            "{duration_sec}",
            "{render_fps}",
            "{to_lua_path(wav)}",
            "{render_width}",
            "{render_height}"
        }},
    }},
}}
'''
    (lua_dir / "init.lua").write_text(lua, encoding="utf-8")


def _stage_asset_paths() -> list[Path]:
    """The dome and floor grid only decorate the background; renders work without them."""

    return [path for path in (DOME_PATH, GRID_PATH) if path.exists()]


def _lua_quote(text: str) -> str:
    return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _lua_command(cmd: str, args: list[str] | tuple[str, ...] = ()) -> str:
    args_text = ", ".join(_lua_quote(arg) for arg in args)
    return f'''    {{
        Cmd  = {_lua_quote(cmd)},
        Args = {{{args_text}}},
    }},'''


def write_batch_init_lua(
    rows: list[dict[str, str]],
    model_path: Path,
    *,
    fps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    msaa_enable: bool = True,
    msaa_count: int = 8,
    quit_when_done: bool = True,
    work_dir: Path | None = None,
) -> None:
    if not rows:
        raise ValueError("write_batch_init_lua requires at least one render row")

    lua_dir = Path(work_dir) if work_dir is not None else RELEASE_DIR
    lua_dir.mkdir(parents=True, exist_ok=True)
    render_fps = int(fps or FPS)
    render_width = int(width or WIDTH)
    render_height = int(height or HEIGHT)
    msaa_enable_lua = "true" if msaa_enable else "false"
    commands: list[str] = []

    for idx, row in enumerate(rows):
        motion = Path(row["motion"])
        camera = Path(row["camera"])
        wav = Path(row["wav"])
        output_name = Path(row["output"]).name
        duration_sec = row["export_duration_sec"]

        if idx == 0:
            commands.extend(
                [
                    _lua_command("open", (to_lua_path(model_path),)),
                    _lua_command("open", (to_lua_path(motion),)),
                    *(_lua_command("open", (to_lua_path(path),)) for path in _stage_asset_paths()),
                    _lua_command("select", ("model_001",)),
                    _lua_command("open", (to_lua_path(camera),)),
                    _lua_command(
                        "setGroundShadowColor",
                        tuple(str(component) for component in SHADOW_COLOR),
                    ),
                ]
            )
        else:
            commands.extend(
                [
                    _lua_command("clearSceneAnimation"),
                    _lua_command("select", ("model_001",)),
                    _lua_command("clearAnimation"),
                    _lua_command("open", (to_lua_path(motion),)),
                    _lua_command("open", (to_lua_path(camera),)),
                ]
            )

        commands.append(
            _lua_command(
                "exportVideo",
                (
                    output_name,
                    duration_sec,
                    str(render_fps),
                    to_lua_path(wav),
                    str(render_width),
                    str(render_height),
                ),
            )
        )

    if quit_when_done:
        commands.append(_lua_command("quit"))

    lua = f'''MSAA = {{
    Enable = {msaa_enable_lua},
    Count  = {int(msaa_count)}
}}

Commands = {{
{chr(10).join(commands)}
}}
'''
    (lua_dir / "init.lua").write_text(lua, encoding="utf-8")


def _chunk_rows(rows: list[dict[str, str]], chunk_size: int) -> list[list[dict[str, str]]]:
    if chunk_size <= 0:
        return [rows]
    return [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]


def _session_timeout_sec(rows: list[dict[str, str]]) -> int:
    # Keep the old per-render timeout budget, but apply it to the whole viewer session.
    return max(1800, 1800 * len(rows))


def _render_row_label(row: dict[str, str]) -> str:
    output = (row.get("output") or "").strip()
    if output:
        return Path(output).stem
    index = (row.get("index") or "").strip()
    return f"c{index}" if index else "unknown"


def _group_rows_by_model(
    rows: list[dict[str, str]],
    fallback_model_path: Path,
) -> list[tuple[Path, list[dict[str, str]]]]:
    groups: list[tuple[Path, list[dict[str, str]]]] = []
    current_model: Path | None = None
    current_rows: list[dict[str, str]] = []

    for row in rows:
        model_text = (row.get("model") or "").strip()
        model_path = Path(model_text) if model_text else fallback_model_path
        if current_model is None or model_path != current_model:
            if current_rows and current_model is not None:
                groups.append((current_model, current_rows))
            current_model = model_path
            current_rows = []
        current_rows.append(row)

    if current_rows and current_model is not None:
        groups.append((current_model, current_rows))
    return groups


def render_planned_rows(
    rows: list[dict[str, str]],
    model_path: Path,
    *,
    viewer_session_size: int,
    fps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    msaa_enable: bool = True,
    msaa_count: int = 8,
    work_dir: Path | None = None,
    worker_lease: SabaWorkerLease | None = None,
) -> None:
    planned_rows = [row for row in rows if row.get("status") == "planned"]
    if not planned_rows:
        return

    session_work_dir = Path(work_dir).resolve() if work_dir is not None else RELEASE_DIR
    if session_work_dir != RELEASE_DIR:
        prepare_saba_worker_dir(session_work_dir)

    session_size = viewer_session_size if viewer_session_size > 0 else len(planned_rows)
    session_chunks: list[tuple[Path, list[dict[str, str]]]] = []
    for group_model_path, model_rows in _group_rows_by_model(planned_rows, model_path):
        for chunk in _chunk_rows(model_rows, session_size):
            session_chunks.append((group_model_path, chunk))

    print(
        f"[batch] rendering {len(planned_rows)} clip(s) in "
        f"{len(session_chunks)} viewer session(s), session_size={session_size}, "
        f"work_dir={session_work_dir}"
    )

    for session_idx, (session_model_path, session_rows) in enumerate(session_chunks, start=1):
        label = ", ".join(_render_row_label(row) for row in session_rows)
        print(f"===== viewer session {session_idx}/{len(session_chunks)} start: {label} =====")

        for row in session_rows:
            output_path = Path(row["output"])
            if output_path.exists():
                output_path.unlink()

        write_batch_init_lua(
            session_rows,
            session_model_path,
            fps=fps,
            width=width,
            height=height,
            msaa_enable=msaa_enable,
            msaa_count=msaa_count,
            work_dir=session_work_dir,
        )
        proc = subprocess.Popen([str(VIEWER_EXE)], cwd=str(session_work_dir))
        if worker_lease is not None:
            worker_lease.update(
                viewer_pid=proc.pid,
                viewer_session=f"{session_idx}/{len(session_chunks)}",
                viewer_session_label=label,
            )
        timed_out = False
        try:
            proc.wait(timeout=_session_timeout_sec(session_rows))
        except subprocess.TimeoutExpired:
            timed_out = True
            print(f"[warn] viewer session timed out: {label}")
            terminate_viewer_process(proc)
        finally:
            if proc.poll() is None:
                terminate_viewer_process(proc)
            if worker_lease is not None:
                worker_lease.update(viewer_pid=0, viewer_session="", viewer_session_label="")

        removed_dirs = cleanup_export_frame_dirs(session_work_dir)
        if removed_dirs > 0:
            print(f"[cleanup] removed {removed_dirs} export_frames_* dirs")

        for row in session_rows:
            output_path = Path(row["output"])
            if not timed_out and output_path.exists() and output_path.stat().st_size > 0:
                row["status"] = "ok"
                print(f"===== {_render_row_label(row)} done =====")
            else:
                row["status"] = "failed"
                print(f"===== {_render_row_label(row)} failed =====")


def wait_until_output_stable(proc: subprocess.Popen, output_path: Path, timeout_sec: int = 1800) -> bool:
    start = time.time()
    last_size = -1
    stable_count = 0
    exited_at: float | None = None
    no_output_grace_sec = 180
    while time.time() - start < timeout_sec:
        if output_path.exists():
            size = output_path.stat().st_size
            if size > 0 and size == last_size:
                stable_count += 1
            else:
                stable_count = 0
                last_size = size
            if stable_count >= 4:
                return True
        elif proc.poll() is not None:
            if exited_at is None:
                exited_at = time.time()
            if time.time() - exited_at >= no_output_grace_sec:
                return False
        time.sleep(5)
    return False


def main() -> None:
    global MODEL_PATH
    parser = argparse.ArgumentParser(description="Batch-render GT camera clips for an arbitrary index range.")
    parser.add_argument("--start", type=int, default=21, help="Start index (inclusive). Historical default is 21.")
    parser.add_argument("--end", type=int, default=30, help="End index (inclusive). Historical default is 30.")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--plan-xlsx", type=str, default="")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-alias", type=str, default="")
    parser.add_argument(
        "--viewer-session-size",
        type=int,
        default=0,
        help=(
            "Number of clips to render per saba_viewer process. "
            "Use 0 to render the whole range in one viewer session; use 1 for legacy per-clip sessions."
        ),
    )
    args = parser.parse_args()

    run_at = time.strftime("%Y-%m-%d %H:%M:%S")

    MODEL_PATH = resolve_model_choice(args.model_path, args.model_alias, MODEL_PATH)
    selected_model_path = MODEL_PATH

    if not VIEWER_EXE.exists():
        raise FileNotFoundError(f"Viewer not found: {VIEWER_EXE}")
    if not selected_model_path.exists():
        raise FileNotFoundError(f"Model not found: {selected_model_path}")

    mapping = load_mapping()
    results: list[dict[str, str]] = []

    kill_viewer()

    for n in range(args.start, args.end + 1):
        print(f"===== c{n} start =====")
        folder = BASE_DIR / f"amc{n}"
        if not folder.exists():
            print(f"[fail] missing folder: {folder}")
            results.append(
                {
                    "run_at": run_at,
                    "range_start": str(args.start),
                    "range_end": str(args.end),
                    "index": str(n),
                    "status": "missing-folder",
                    "model": str(selected_model_path),
                    "output": "",
                }
            )
            continue

        wav = folder / f"a{n}.wav"
        if not wav.exists():
            print(f"[fail] missing wav: {wav}")
            results.append(
                {
                    "run_at": run_at,
                    "range_start": str(args.start),
                    "range_end": str(args.end),
                    "index": str(n),
                    "status": "missing-wav",
                    "model": str(selected_model_path),
                    "output": "",
                }
            )
            continue

        pref = mapping.get(n, {"motion": "", "camera": ""})
        motion = resolve_vmd(folder, pref.get("motion", ""), "motion", n)
        camera = resolve_vmd(folder, pref.get("camera", ""), "camera", n)
        if motion is None or camera is None:
            print(f"[fail] missing vmd in amc{n}")
            results.append(
                {
                    "run_at": run_at,
                    "range_start": str(args.start),
                    "range_end": str(args.end),
                    "index": str(n),
                    "status": "missing-vmd",
                    "model": str(selected_model_path),
                    "output": "",
                }
            )
            continue

        motion_stats = read_vmd_stats(motion)
        camera_stats = read_vmd_stats(camera)
        wav_duration_sec = ffprobe_duration_sec(wav)
        wav_duration_float_sec = ffprobe_duration_float_sec(wav)

        dance_end_frame = motion_stats["motion_max_frame"]
        camera_end_frame = camera_stats["camera_max_frame"]
        music_end_frame = max(0, int(math.floor(wav_duration_float_sec * FPS)) - 1)

        if dance_end_frame <= 0 or camera_end_frame <= 0:
            print(f"[fail] invalid end frame(s): dance={dance_end_frame} camera={camera_end_frame}")
            results.append(
                {
                    "run_at": run_at,
                    "range_start": str(args.start),
                    "range_end": str(args.end),
                    "index": str(n),
                    "status": "missing-vmd",
                    "model": str(selected_model_path),
                    "output": "",
                }
            )
            continue

        shortest_end_frame = min(dance_end_frame, camera_end_frame, music_end_frame)
        max_vmd_frame = max(dance_end_frame, camera_end_frame)
        export_frame_count = max(1, shortest_end_frame + 1)
        duration_sec = export_frame_count / FPS
        output_name = f"c{n}.mp4"
        output_path = RELEASE_DIR / output_name

        print(f"[plan] motion={motion}")
        print(f"[plan] camera={camera}")
        print(
            f"[plan] wav={wav} wav_duration={wav_duration_sec}s "
            f"dance_end_frame={dance_end_frame} "
            f"camera_end_frame={camera_end_frame} "
            f"music_end_frame={music_end_frame} "
            f"shortest_end_frame={shortest_end_frame} "
            f"vmd_max_frame={max_vmd_frame} export_frames={export_frame_count} duration={duration_sec:.6f}s"
        )

        row = {
            "run_at": run_at,
            "range_start": str(args.start),
            "range_end": str(args.end),
            "index": str(n),
            "status": "planned",
            "model": str(selected_model_path),
            "motion": str(motion),
            "camera": str(camera),
            "wav": str(wav),
            "wav_duration_sec": str(wav_duration_sec),
            "dance_end_frame": str(dance_end_frame),
            "camera_end_frame": str(camera_end_frame),
            "motion_max_frame": str(motion_stats["motion_max_frame"]),
            "camera_max_frame": str(camera_stats["camera_max_frame"]),
            "export_duration_sec": f"{duration_sec:.6f}",
            "export_frame_count": str(export_frame_count),
            "fps": str(FPS),
            "output": str(output_path),
        }

        results.append(row)

    if not args.plan_only:
        render_planned_rows(
            results,
            selected_model_path,
            viewer_session_size=args.viewer_session_size,
        )

    summary_csv = SUMMARY_CSV
    csv_exists = summary_csv.exists()
    with summary_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not csv_exists:
            writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in RESULT_FIELDS})

    summary_json = SUMMARY_JSON
    if summary_json.exists():
        try:
            existing = json.loads(summary_json.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
    else:
        existing = []
    existing.extend(results)
    summary_json.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")

    rewrite_latest_ok_csv()

    plan_xlsx = Path(args.plan_xlsx) if args.plan_xlsx else DEFAULT_PLAN_XLSX
    write_plan_xlsx(results, plan_xlsx)

    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Latest OK CSV: {LATEST_OK_CSV}")
    print(f"Plan XLSX: {plan_xlsx}")


if __name__ == "__main__":
    main()
