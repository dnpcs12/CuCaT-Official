"""Local web demo for the CuCaT camera-generation pipeline.

Run from the repository root:

    python -m music_processing.serve_unseen_mmd_pipeline

then open http://127.0.0.1:8785/ . The page uploads a song, a dance VMD, and a PMX/PMD model,
runs ``run_unseen_mmd_timing_pipeline`` in the background, and links the camera VMD, shot plan,
and preview MP4. The server is meant for local use: it reads local paths typed into the page and
serves files from the repository folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, unquote, urlsplit

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        import cgi  # Multipart uploads. Removed in Python 3.13; `pip install legacy-cgi` restores it.
    except ImportError as exc:  # pragma: no cover - depends on the Python version
        raise SystemExit(
            "The web demo needs the `cgi` module, which Python 3.13 removed. "
            "Use Python 3.11 (see README) or run `pip install legacy-cgi`."
        ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PIPELINE_SCRIPT = REPO_ROOT / "music_processing" / "run_unseen_mmd_timing_pipeline.py"
PIPELINE_MODULE = "music_processing.run_unseen_mmd_timing_pipeline"
VIEWER_PATH = REPO_ROOT / "music_processing" / "unseen_mmd_pipeline_viewer.html"
WEB_DEMO_ROOT = REPO_ROOT / "music_processing" / "results" / "web_demo"
DEFAULT_UPLOAD_ROOT = WEB_DEMO_ROOT / "uploads"
DEFAULT_OUTPUT_ROOT = WEB_DEMO_ROOT / "outputs"
PIPELINE_STAGE_PREFIX = "__PIPELINE_STAGE__ "
TIMING_EVENT_IDS = {
    "input_setup",
    "motion_conversion",
    "frame_alignment",
    "demucs",
    "beat_tracking",
    "audio_features",
    "dance_features",
    "feature_bundle",
    "model_load",
    "cut_inference",
    "density_pre_cut",
    "cut_selection",
    "density_after_cut",
    "stage2_inference",
    "keyframe_selection",
}
TIMING_COMPLETION_EVENT_IDS = {"keyframe_selection"}
PIPELINE_STAGE_ALIASES = {
    "shotplan_features": "shot_plan",
    "shot_plan": "shot_plan",
    "optimization": "optimization",
    "camera_trajectory": "camera_trajectory",
    "preview_render": "render_review",
}
PIPELINE_PROGRESS_STEPS = [
    {"id": "input_setup", "label": "Input setup", "weight": 2.0},
    {"id": "motion_conversion", "label": "VMD motion conversion", "weight": 14.0},
    {"id": "frame_alignment", "label": "Frame alignment", "weight": 1.0},
    {"id": "demucs", "label": "Demucs stem separation", "weight": 90.0},
    {"id": "beat_tracking", "label": "Beat/downbeat extraction", "weight": 18.0},
    {"id": "audio_features", "label": "Audio features", "weight": 16.0},
    {"id": "dance_features", "label": "Dance features", "weight": 4.0},
    {"id": "feature_bundle", "label": "Feature bundle", "weight": 2.0},
    {"id": "model_load", "label": "Timing model load", "weight": 4.0},
    {"id": "cut_inference", "label": "Cut inference", "weight": 4.0},
    {"id": "density_pre_cut", "label": "Density before cuts", "weight": 3.0},
    {"id": "cut_selection", "label": "Cut selection", "weight": 1.0},
    {"id": "density_after_cut", "label": "Density after cuts", "weight": 3.0},
    {"id": "stage2_inference", "label": "Keyframe probability", "weight": 8.0},
    {"id": "keyframe_selection", "label": "Keyframe selection", "weight": 2.0},
    {"id": "shotplan_features", "label": "Shot-plan features", "weight": 2.0},
    {"id": "shot_plan", "label": "Shot-plan prediction", "weight": 10.0},
    {"id": "camera_trajectory", "label": "Camera VMD export", "weight": 8.0},
    {"id": "optimization", "label": "Shot-plan optimization", "weight": 18.0},
    {"id": "preview_render", "label": "Preview render", "weight": 80.0},
    {"id": "output_write", "label": "Output write", "weight": 1.0},
]
PROGRESS_STEP_BY_ID = {str(step["id"]): step for step in PIPELINE_PROGRESS_STEPS}
TIMING_ONLY_PROGRESS_IDS = TIMING_EVENT_IDS | {"output_write"}
TERMINAL_PROGRESS_STATUSES = {"completed", "ok", "skipped", "missing"}
QUIET_LOG_PREFIXES = ("/api/unseen/jobs",)


@dataclass
class Job:
    id: str
    song_id: str
    pipeline_mode: str = "timing_only"
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    command: list[str] = field(default_factory=list)
    returncode: int | None = None
    log: list[str] = field(default_factory=list)
    error: str | None = None
    upload_dir: str | None = None
    output_dir: str | None = None
    result_url: str | None = None
    summary_url: str | None = None
    viewer_url: str | None = None
    summary: dict[str, Any] | None = None
    stages: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    progress_state: dict[str, Any] = field(default_factory=dict)


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def configure_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def utf8_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def repo_url(path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError:
        return str(resolved)
    return "/" + quote(relative.as_posix(), safe="/:?=&")


def slug(text: str) -> str:
    import re

    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    normalized = re.sub(r"_+", "_", normalized).strip("._-")
    return normalized[:80] or "unseen"


def new_job_id(seed: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha1(f"{seed}-{time.time_ns()}".encode("utf-8")).hexdigest()[:8]
    return f"{stamp}_{digest}"


def progress_steps_for_job(job: Job) -> list[dict[str, Any]]:
    if job.pipeline_mode == "timing_only":
        return [step for step in PIPELINE_PROGRESS_STEPS if str(step["id"]) in TIMING_ONLY_PROGRESS_IDS]
    return PIPELINE_PROGRESS_STEPS


def build_progress_payload(job: Job, *, now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    state = job.progress_state or {}
    step_status = dict(state.get("step_status") or {})
    step_started_at = dict(state.get("step_started_at") or {})
    steps = progress_steps_for_job(job)
    effective_steps = [step for step in steps if step_status.get(str(step["id"])) != "skipped"] or steps
    total_weight = sum(float(step["weight"]) for step in effective_steps) or 1.0

    completed_weight = 0.0
    active_step: dict[str, Any] | None = None
    active_partial = 0.0
    for step in effective_steps:
        step_id = str(step["id"])
        status = str(step_status.get(step_id) or "")
        weight = float(step["weight"])
        if status in TERMINAL_PROGRESS_STATUSES:
            completed_weight += weight
        elif status == "running" and active_step is None:
            active_step = step
            started_at = float(step_started_at.get(step_id) or now)
            active_elapsed = max(0.0, now - started_at)
            active_partial = min(weight * 0.85, weight * active_elapsed / max(weight, 1.0) * 0.85)

    if active_step is None:
        active_step = next(
            (
                step
                for step in effective_steps
                if str(step_status.get(str(step["id"])) or "") not in TERMINAL_PROGRESS_STATUSES
            ),
            None,
        )

    fraction = max(0.0, min(0.99, (completed_weight + active_partial) / total_weight))
    if job.status == "completed":
        fraction = 1.0
        active_step = None
    elif job.status == "failed":
        fraction = max(0.0, min(1.0, completed_weight / total_weight))

    elapsed_sec = max(0.0, now - float(job.created_at))
    eta_sec: float | None = None
    if job.status in {"running", "queued"} and fraction > 0.03:
        eta_sec = max(0.0, elapsed_sec * (1.0 - fraction) / max(fraction, 0.03))

    current_label = str(state.get("last_label") or "")
    if active_step is not None:
        current_label = str(active_step.get("label") or current_label or active_step.get("id"))
    elif job.status == "completed":
        current_label = "Completed"
    elif job.status == "failed":
        current_label = "Failed"
    elif not current_label:
        current_label = "Waiting to start"

    return {
        "percent": round(fraction * 100.0, 1),
        "elapsed_sec": round(elapsed_sec, 3),
        "eta_sec": round(float(eta_sec), 3) if eta_sec is not None else None,
        "current_step": current_label,
        "current_step_id": str(active_step.get("id")) if active_step is not None else state.get("last_event_id"),
        "last_event_id": state.get("last_event_id"),
        "last_status": state.get("last_status"),
        "completed_steps": sum(
            1
            for step in effective_steps
            if str(step_status.get(str(step["id"])) or "") in TERMINAL_PROGRESS_STATUSES
        ),
        "total_steps": len(effective_steps),
        "is_estimate": job.status not in {"completed", "failed"},
    }


def json_ready_job(job: Job) -> dict[str, Any]:
    payload = asdict(job)
    payload.pop("progress_state", None)
    payload["progress"] = build_progress_payload(job)
    payload["created_at_text"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job.created_at))
    payload["updated_at_text"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job.updated_at))
    return payload


def default_stages(*, timing_status: str = "queued") -> list[dict[str, Any]]:
    return [
        {
            "id": "timing",
            "name": "Keyframe timing",
            "status": timing_status,
            "description": "Extract music and dance features, then select cut pairs and transition frames.",
        },
        {
            "id": "shot_plan",
            "name": "Shot-plan prediction",
            "status": "queued",
            "description": "Predict a shot plan for every cut and transition frame.",
        },
        {
            "id": "optimization",
            "name": "Shot-plan optimization",
            "status": "queued",
            "description": "Choose among the top-k shot-plan candidates with CP-SAT.",
        },
        {
            "id": "camera_trajectory",
            "name": "Camera VMD",
            "status": "queued",
            "description": "Solve the shot plan into MMD camera keyframes and write the camera VMD.",
        },
        {
            "id": "render_review",
            "name": "Preview video",
            "status": "planned",
            "description": "Render the camera VMD with the dance into an MP4 preview.",
        },
    ]


def set_stage(job_id: str, stage_id: str, status: str, **extra: Any) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        stages = job.stages or default_stages()
        for stage in stages:
            if stage.get("id") == stage_id:
                stage["status"] = status
                stage.update(extra)
                break
        job.stages = stages
        job.updated_at = time.time()


def planned_artifacts() -> dict[str, dict[str, Any]]:
    return {
        "timing_json": {
            "label": "final_cut_key_timings.json",
            "stage": "timing",
            "status": "pending",
            "url": None,
        },
        "summary_json": {
            "label": "summary.json",
            "stage": "timing",
            "status": "pending",
            "url": None,
        },
        "timing_viewer": {
            "label": "Timing viewer",
            "stage": "timing",
            "status": "pending",
            "url": None,
        },
        "shot_plan_json": {
            "label": "final shot-plan JSON",
            "stage": "shot_plan",
            "status": "pending",
            "url": None,
        },
        "topk_candidates_json": {
            "label": "top-k candidates JSON",
            "stage": "optimization",
            "status": "pending",
            "url": None,
        },
        "optimized_shot_plan_json": {
            "label": "optimized shot-plan JSON",
            "stage": "optimization",
            "status": "pending",
            "url": None,
        },
        "optimizer_debug_json": {
            "label": "optimizer debug JSON",
            "stage": "optimization",
            "status": "pending",
            "url": None,
        },
        "camera_vmd": {
            "label": "final camera VMD",
            "stage": "camera_trajectory",
            "status": "pending",
            "url": None,
        },
        "preview_video": {
            "label": "preview MP4",
            "stage": "render_review",
            "status": "pending",
            "url": None,
        },
    }


def set_job(job_id: str, **updates: Any) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        for key, value in updates.items():
            setattr(job, key, value)
        job.updated_at = time.time()


def append_log(job_id: str, line: str) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        job.log.append(line.rstrip())
        if len(job.log) > 2000:
            job.log = job.log[-2000:]
        job.updated_at = time.time()


def pipeline_stage_for_event(event_id: str) -> str | None:
    if event_id in TIMING_EVENT_IDS:
        return "timing"
    return PIPELINE_STAGE_ALIASES.get(event_id)


def describe_stage_event(label: str, status: str, duration_sec: Any | None) -> str:
    if isinstance(duration_sec, (int, float)):
        return f"{label}: {status} in {float(duration_sec):.2f}s"
    return f"{label}: {status}"


def record_progress_event(job_id: str, event_id: str, status: str, label: str) -> None:
    if event_id not in PROGRESS_STEP_BY_ID:
        return
    normalized_status = "completed" if status == "ok" else status
    with JOBS_LOCK:
        job = JOBS[job_id]
        state = dict(job.progress_state or {})
        step_status = dict(state.get("step_status") or {})
        step_started_at = dict(state.get("step_started_at") or {})
        if normalized_status == "running":
            step_started_at.setdefault(event_id, time.time())
        step_status[event_id] = normalized_status
        state.update(
            {
                "step_status": step_status,
                "step_started_at": step_started_at,
                "last_event_id": event_id,
                "last_status": normalized_status,
                "last_label": label,
            }
        )
        job.progress_state = state
        job.updated_at = time.time()


def apply_pipeline_stage_event(job_id: str, event: dict[str, Any]) -> bool:
    event_id = str(event.get("id") or "")
    raw_status = str(event.get("status") or "running")
    status = "completed" if raw_status == "ok" else raw_status
    label = str(event.get("label") or event_id)
    description = describe_stage_event(label, status, event.get("duration_sec"))
    record_progress_event(job_id, event_id, status, label)

    stage_id = pipeline_stage_for_event(event_id)
    if stage_id is None:
        # e.g. output_write: counted in the progress bar, but no stage card of its own.
        if event_id in PROGRESS_STEP_BY_ID:
            append_log(job_id, f"[stage] {description}")
            return True
        return False

    # The timing card represents the full timing block, not every sub-step. Keep it running while
    # Demucs/features/models progress, and complete it once the transition frames are selected.
    if stage_id == "timing" and status == "completed" and event_id not in TIMING_COMPLETION_EVENT_IDS:
        status = "running"

    # The shot-plan feature bundle is the first sub-step of shot planning; the card completes only
    # after the labels are predicted.
    if stage_id == "shot_plan" and event_id != "shot_plan" and status == "completed":
        status = "running"

    set_stage(job_id, stage_id, status, description=description)
    append_log(job_id, f"[stage] {description}")
    return True


def maybe_apply_pipeline_stage_line(job_id: str, line: str) -> bool:
    stripped = line.rstrip()
    if not stripped.startswith(PIPELINE_STAGE_PREFIX):
        return False
    try:
        payload = json.loads(stripped[len(PIPELINE_STAGE_PREFIX) :])
    except json.JSONDecodeError:
        append_log(job_id, stripped)
        return True
    if not isinstance(payload, dict):
        append_log(job_id, stripped)
        return True
    if not apply_pipeline_stage_event(job_id, payload):
        append_log(job_id, stripped)
    return True


def field_text(form: cgi.FieldStorage, name: str, default: str = "") -> str:
    if name not in form:
        return default
    value = form[name]
    if isinstance(value, list):
        value = value[0]
    if getattr(value, "filename", None):
        return default
    raw = value.value
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace").strip()
    return str(raw).strip()


def field_bool(form: cgi.FieldStorage, name: str) -> bool:
    return field_text(form, name, "").lower() in {"1", "true", "yes", "on"}


def save_upload(form: cgi.FieldStorage, name: str, dest_dir: Path) -> Path | None:
    if name not in form:
        return None
    value = form[name]
    if isinstance(value, list):
        value = value[0]
    filename = Path(getattr(value, "filename", "") or "").name
    if not filename:
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    output_path = dest_dir / filename
    with output_path.open("wb") as handle:
        shutil.copyfileobj(value.file, handle)
    return output_path


def safe_upload_relative_path(filename: str) -> Path | None:
    normalized = str(filename or "").replace("\\", "/").strip("/")
    if not normalized:
        return None
    pure = PurePosixPath(normalized)
    parts = [part for part in pure.parts if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        return None
    if len(parts) >= 2 and parts[0].endswith(":"):
        parts = parts[1:]
    return Path(*parts) if parts else None


def save_uploads_preserve_relative(form: cgi.FieldStorage, name: str, dest_dir: Path) -> list[Path]:
    if name not in form:
        return []
    values = form[name]
    if not isinstance(values, list):
        values = [values]
    saved: list[Path] = []
    for value in values:
        filename = getattr(value, "filename", "") or ""
        relative = safe_upload_relative_path(filename)
        if relative is None:
            continue
        output_path = (dest_dir / relative).resolve()
        try:
            output_path.relative_to(dest_dir.resolve())
        except ValueError:
            continue
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("wb") as handle:
                shutil.copyfileobj(value.file, handle)
        except OSError:
            continue  # e.g. a name that is not a valid Windows file name; skip that file only
        saved.append(output_path)
    return saved


def first_model_file(paths: list[Path]) -> Path | None:
    models = [path for path in paths if path.suffix.lower() in {".pmx", ".pmd"}]
    if not models:
        return None
    return sorted(models, key=lambda path: (-path.stat().st_size, len(path.parts), str(path).lower()))[0]


def matching_uploaded_model_in_folder(model_upload: Path, paths: list[Path]) -> Path | None:
    upload_name = model_upload.name.lower()
    matches = [
        path
        for path in paths
        if path.suffix.lower() in {".pmx", ".pmd"} and path.name.lower() == upload_name
    ]
    if not matches:
        return None
    return sorted(matches, key=lambda path: (len(path.parts), str(path).lower()))[0]


def existing_or_none(path_text: str) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text.strip().strip('"')).expanduser()
    return path.resolve()


def add_optional_arg(command: list[str], flag: str, value: str) -> None:
    if value != "":
        command.extend([flag, value])


def path_from_url_or_text(value: str | None) -> Path | None:
    if not value:
        return None
    text = unquote(str(value).strip())
    if not text:
        return None
    parsed = urlsplit(text)
    if parsed.scheme in {"http", "https"}:
        text = unquote(parsed.path)
    if len(text) >= 2 and text[1] == ":":
        return Path(text).resolve()
    if text.startswith("/"):
        return (REPO_ROOT / text.lstrip("/")).resolve()
    return Path(text).expanduser().resolve()


def read_json_file(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def first_existing_path(*values: Any) -> Path | None:
    for value in values:
        path = path_from_url_or_text(str(value)) if value else None
        if path is not None and path.exists():
            return path
    return None


def render_edited_shotplan(payload: dict[str, Any]) -> dict[str, Any]:
    """Save an edited shot plan from the editor page, solve it into a camera VMD, and re-render."""

    summary_path = path_from_url_or_text(payload.get("summary_url") or payload.get("summary_path"))
    if summary_path is None or not summary_path.exists():
        raise ValueError("A valid summary_url or summary_path is required.")
    output_dir = summary_path.parent
    summary = read_json_file(summary_path)
    shotplan_summary = summary.get("shotplan") if isinstance(summary.get("shotplan"), dict) else {}

    source_plan_path = first_existing_path(
        payload.get("source_plan_url"),
        payload.get("source_plan_path"),
        shotplan_summary.get("final_shotplan_json"),
        shotplan_summary.get("optimization_json"),
        summary.get("shot_plan_json"),
    )
    if source_plan_path is None:
        raise ValueError("Could not resolve source shot-plan JSON.")
    source_plan = read_json_file(source_plan_path)

    keyframes = payload.get("keyframes")
    if not isinstance(keyframes, list) or not keyframes:
        raise ValueError("Payload must include a non-empty keyframes list.")
    edited_keyframes = sorted(
        [dict(row) for row in keyframes],
        key=lambda row: int(row.get("frame", 0) or 0),
    )

    final_timing_path = first_existing_path(summary.get("final_cut_key_timings"))
    final_timing = read_json_file(final_timing_path)
    final_input = final_timing.get("input") if isinstance(final_timing.get("input"), dict) else {}
    preprocess_path = output_dir / "preprocess_summary.json"
    preprocess = read_json_file(preprocess_path)
    motion_conversion = preprocess.get("motion_conversion") if isinstance(preprocess.get("motion_conversion"), dict) else {}
    demucs = preprocess.get("demucs") if isinstance(preprocess.get("demucs"), dict) else {}

    default_target_probe_path = output_dir / "intermediate" / f"{summary.get('song_id', output_dir.name)}_target_probe.json"
    target_probe_path = first_existing_path(
        payload.get("target_probe_path"),
        shotplan_summary.get("target_probe_json"),
        default_target_probe_path,
    )
    model_path = first_existing_path(
        payload.get("render_model_path"),
        summary.get("render_model_path"),
        final_input.get("model_path"),
        motion_conversion.get("model_path"),
    )
    motion_vmd_path = first_existing_path(
        payload.get("motion_vmd_path"),
        final_input.get("vmd_path"),
        motion_conversion.get("vmd_path"),
    )
    wav_path = first_existing_path(
        payload.get("wav_path"),
        final_input.get("wav_path"),
        demucs.get("audio_path"),
    )
    if target_probe_path is None:
        raise ValueError("Could not resolve the target probe JSON needed for camera VMD export.")
    if model_path is None:
        raise ValueError("Could not resolve render model path.")
    if motion_vmd_path is None:
        raise ValueError("Could not resolve source motion VMD path.")
    if wav_path is None:
        raise ValueError("Could not resolve source WAV path.")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha1(json.dumps(edited_keyframes, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
    nonce = hashlib.sha1(str(time.time_ns()).encode("utf-8")).hexdigest()[:6]
    edit_dir = output_dir / "shotplan_edits"
    edit_dir.mkdir(parents=True, exist_ok=True)
    edited_plan_path = edit_dir / f"shotplan_edit_{stamp}_{digest}_{nonce}.json"
    edited_vmd_path = edited_plan_path.with_suffix(".vmd")
    edited_debug_path = edited_plan_path.with_name(f"{edited_plan_path.stem}_debug.json")

    edited_plan = json.loads(json.dumps(source_plan))
    edited_plan["keyframes"] = edited_keyframes
    edited_plan["edited_from"] = str(source_plan_path)
    edited_plan["edit_metadata"] = {
        "created_at": stamp,
        "source_summary": str(summary_path),
        "keyframe_count": len(edited_keyframes),
    }
    edited_plan_path.write_text(json.dumps(edited_plan, indent=2, ensure_ascii=False), encoding="utf-8")

    from music_processing.run_unseen_mmd_timing_pipeline import (
        _render_preview_video,
        _shotplan_height_calibration_config,
        _write_vmd_from_plan,
    )

    # Reuse the height calibration recorded by the original run.
    calibration = shotplan_summary.get("height_calibration") if isinstance(shotplan_summary.get("height_calibration"), dict) else {}
    face_offset = -0.035
    chest_offset = -0.025
    for rule in calibration.get("exported_rules") or []:
        if rule.get("target_anchor") == "face":
            face_offset = float(rule.get("y_offset_fraction", face_offset))
            break
    for rule in calibration.get("exported_rules") or []:
        if rule.get("target_anchor") == "chest":
            chest_offset = float(rule.get("y_offset_fraction", chest_offset))
            break
    solve_args = argparse.Namespace(
        shotplan_height_calibration=bool(calibration.get("enabled", True)),
        shotplan_height_calibration_face_offset=face_offset,
        shotplan_height_calibration_chest_offset=chest_offset,
    )
    _write_vmd_from_plan(
        edited_plan_path,
        target_probe_path,
        edited_vmd_path,
        edited_debug_path,
        solve_config=_shotplan_height_calibration_config(solve_args),
        stabilize_forward=bool(shotplan_summary.get("actor_forward_stabilization", True)),
    )

    render_preview = bool(payload.get("render_preview", True))
    render_summary: dict[str, Any] = {"status": "skipped"}
    if render_preview:
        duration_sec = float(
            summary.get("duration_sec")
            or final_input.get("duration_sec")
            or (
                float(final_input.get("num_frames", 1) or 1)
                / max(float(final_input.get("fps", 30) or 30), 1.0)
            )
            or 1.0
        )
        render_summary = _render_preview_video(
            output_dir=edit_dir,
            song_id=f"{summary.get('song_id', output_dir.name)}_shotplan_edit",
            model_path=model_path,
            motion_vmd_path=motion_vmd_path,
            camera_vmd_path=edited_vmd_path,
            wav_path=wav_path,
            duration_sec=duration_sec,
            fps=int(payload.get("render_fps") or 30),
            width=int(payload.get("render_width") or 1280),
            height=int(payload.get("render_height") or 720),
            viewer_session_size=int(payload.get("render_session_size") or 1),
        )

    preview_path = path_from_url_or_text(render_summary.get("preview_video")) if render_summary.get("preview_video") else None
    return {
        "status": "ok",
        "edited_plan": str(edited_plan_path),
        "edited_plan_url": repo_url(edited_plan_path),
        "edited_vmd": str(edited_vmd_path),
        "edited_vmd_url": repo_url(edited_vmd_path),
        "edited_debug": str(edited_debug_path),
        "edited_debug_url": repo_url(edited_debug_path),
        "target_probe": str(target_probe_path),
        "target_probe_url": repo_url(target_probe_path),
        "preview_video": str(preview_path) if preview_path else None,
        "preview_video_url": repo_url(preview_path) if preview_path and preview_path.exists() else None,
        "render_summary": render_summary,
        "source_plan": str(source_plan_path),
        "summary": str(summary_path),
    }


def run_pipeline_job(job_id: str, command: list[str], output_dir: Path) -> None:
    set_job(job_id, status="running", command=command, output_dir=str(output_dir))
    set_stage(job_id, "timing", "running")
    append_log(job_id, "Starting pipeline...")
    append_log(job_id, "Command: " + " ".join(f'"{part}"' if " " in part else part for part in command))
    try:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=utf8_subprocess_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            if not maybe_apply_pipeline_stage_line(job_id, line):
                append_log(job_id, line)
        returncode = process.wait()
        set_job(job_id, returncode=returncode)
        if returncode != 0:
            set_job(job_id, status="failed", error=f"Pipeline exited with code {returncode}. See the log for details.")
            with JOBS_LOCK:
                running = [stage["id"] for stage in JOBS[job_id].stages if stage.get("status") == "running"]
            for stage_id in running or ["timing"]:
                set_stage(job_id, stage_id, "failed")
            append_log(job_id, f"Pipeline failed with code {returncode}")
            return

        summary_path = output_dir / "summary.json"
        result_path = output_dir / "final_cut_key_timings.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        result_url = repo_url(result_path) if result_path.exists() else None
        summary_url = repo_url(summary_path) if summary_path.exists() else None
        shotplan_summary = summary.get("shotplan") if isinstance(summary.get("shotplan"), dict) else {}
        shot_plan_path = Path(str(summary.get("shot_plan_json") or "")) if summary.get("shot_plan_json") else None
        camera_vmd_path = Path(str(summary.get("camera_vmd") or "")) if summary.get("camera_vmd") else None
        preview_video_path = (
            Path(str(summary.get("preview_video") or shotplan_summary.get("preview_video") or ""))
            if summary.get("preview_video") or shotplan_summary.get("preview_video")
            else None
        )
        topk_path = (
            Path(str(shotplan_summary.get("topk_candidates_json")))
            if shotplan_summary.get("topk_candidates_json")
            else None
        )
        optimized_path = (
            Path(str(shotplan_summary.get("optimization_json")))
            if shotplan_summary.get("optimization_json")
            else None
        )
        optimizer_debug_path = (
            Path(str(shotplan_summary.get("optimization_debug")))
            if shotplan_summary.get("optimization_debug")
            else None
        )
        shot_plan_url = repo_url(shot_plan_path) if shot_plan_path is not None and shot_plan_path.exists() else None
        camera_vmd_url = repo_url(camera_vmd_path) if camera_vmd_path is not None and camera_vmd_path.exists() else None
        preview_video_url = (
            repo_url(preview_video_path)
            if preview_video_path is not None and preview_video_path.exists()
            else None
        )
        topk_url = repo_url(topk_path) if topk_path is not None and topk_path.exists() else None
        optimized_url = repo_url(optimized_path) if optimized_path is not None and optimized_path.exists() else None
        optimizer_debug_url = (
            repo_url(optimizer_debug_path)
            if optimizer_debug_path is not None and optimizer_debug_path.exists()
            else None
        )
        viewer_url = None
        if result_url is not None:
            viewer_url = f"/music_processing/final_cut_keyframe_viewer.html?result={quote(result_url, safe='/:')}"
            if preview_video_url is not None:
                viewer_url += f"&video={quote(preview_video_url, safe='/:')}"
        artifacts = planned_artifacts()
        artifacts["timing_json"].update({"status": "ready" if result_url else "missing", "url": result_url})
        artifacts["summary_json"].update({"status": "ready" if summary_url else "missing", "url": summary_url})
        artifacts["timing_viewer"].update({"status": "ready" if viewer_url else "missing", "url": viewer_url})
        timing_only = job_pipeline_mode(job_id) == "timing_only"
        optimization_status = str(shotplan_summary.get("optimization_status", "skipped"))
        optimization_artifact_status = "skipped" if optimization_status == "skipped" else "missing"
        render_preview = shotplan_summary.get("render_preview") if isinstance(shotplan_summary.get("render_preview"), dict) else {}
        render_status = str(render_preview.get("status") or "")
        render_artifact_status = "skipped" if timing_only or render_status == "skipped" else "missing"
        camera_artifact_status = "skipped" if timing_only else "missing"
        artifacts["shot_plan_json"].update(
            {"status": "ready" if shot_plan_url else camera_artifact_status, "url": shot_plan_url}
        )
        artifacts["topk_candidates_json"].update(
            {"status": "ready" if topk_url else optimization_artifact_status, "url": topk_url}
        )
        artifacts["optimized_shot_plan_json"].update(
            {"status": "ready" if optimized_url else optimization_artifact_status, "url": optimized_url}
        )
        artifacts["optimizer_debug_json"].update(
            {"status": "ready" if optimizer_debug_url else optimization_artifact_status, "url": optimizer_debug_url}
        )
        artifacts["camera_vmd"].update({"status": "ready" if camera_vmd_url else camera_artifact_status, "url": camera_vmd_url})
        artifacts["preview_video"].update({"status": "ready" if preview_video_url else render_artifact_status, "url": preview_video_url})
        set_job(
            job_id,
            status="completed",
            summary=summary,
            result_url=result_url,
            summary_url=summary_url,
            viewer_url=viewer_url,
            artifacts=artifacts,
        )
        timing_extra: dict[str, Any] = {}
        if summary.get("selected_cut_count") is not None and summary.get("selected_noncut_keyframe_count") is not None:
            timing_extra["description"] = (
                f"Selected {summary['selected_cut_count']} cut pairs and "
                f"{summary['selected_noncut_keyframe_count']} transition frames."
            )
        set_stage(job_id, "timing", "completed", **timing_extra)
        set_stage(job_id, "shot_plan", "completed" if shot_plan_url else "skipped")
        set_stage(
            job_id,
            "optimization",
            "completed" if optimized_url else ("skipped" if optimization_status == "skipped" else "missing"),
            description=f"Optimization status: {optimization_status}",
        )
        camera_extra = {"description": str(shotplan_summary["warning"])} if shotplan_summary.get("warning") else {}
        set_stage(job_id, "camera_trajectory", "completed" if camera_vmd_url else camera_artifact_status, **camera_extra)
        render_extra: dict[str, Any] = {}
        if render_preview.get("error") or render_preview.get("warning"):
            render_extra["description"] = f"Preview {render_status}: {render_preview.get('error') or render_preview.get('warning')}"
        set_stage(job_id, "render_review", "completed" if preview_video_url else render_artifact_status, **render_extra)
        append_log(job_id, "Pipeline completed.")
    except Exception as exc:  # Keep browser-visible failures readable.
        set_job(job_id, status="failed", error=repr(exc))
        set_stage(job_id, "timing", "failed")
        append_log(job_id, traceback.format_exc())


def job_pipeline_mode(job_id: str) -> str:
    with JOBS_LOCK:
        return JOBS[job_id].pipeline_mode


def build_command_from_form(
    form: cgi.FieldStorage,
    *,
    pipeline_python: Path,
    upload_root: Path,
    output_root: Path,
    default_model_path: Path | None,
) -> tuple[str, Path, list[str], Path, str]:
    seed_name = field_text(form, "song_id") or field_text(form, "wav_path") or "unseen"
    job_id = new_job_id(seed_name)
    input_dir = upload_root / job_id / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    wav_upload = save_upload(form, "wav_file", input_dir)
    vmd_upload = save_upload(form, "vmd_file", input_dir)
    model_upload = save_upload(form, "model_file", input_dir)
    motion_upload = save_upload(form, "motion_json_file", input_dir)
    model_folder_dir = input_dir / "model_folder"
    model_folder_uploads = save_uploads_preserve_relative(form, "model_folder_files", model_folder_dir)
    model_folder_model = first_model_file(model_folder_uploads) if model_upload is None else None
    model_upload_in_texture_folder: Path | None = None
    if model_upload is not None and model_folder_uploads:
        matched_folder_model = matching_uploaded_model_in_folder(model_upload, model_folder_uploads)
        if matched_folder_model is not None:
            model_upload_in_texture_folder = matched_folder_model
        else:
            # The separately uploaded PMX should sit next to the uploaded texture folder, not inside
            # it. Many PMX files reference textures as e.g. "tex/foo.png"; copying the PMX into
            # "tex/" would make the renderer look for "tex/tex/foo.png".
            model_folder_dir.mkdir(parents=True, exist_ok=True)
            model_upload_in_texture_folder = model_folder_dir / model_upload.name
            shutil.copy2(model_upload, model_upload_in_texture_folder)

    wav_path = wav_upload or existing_or_none(field_text(form, "wav_path"))
    vmd_path = vmd_upload or existing_or_none(field_text(form, "vmd_path"))
    model_path = (
        model_upload_in_texture_folder
        or model_upload
        or existing_or_none(field_text(form, "model_path"))
        or model_folder_model
    )
    motion_json_path = motion_upload or existing_or_none(field_text(form, "motion_json_path"))
    if model_path is None and default_model_path is not None:
        model_path = default_model_path

    if wav_path is None:
        raise ValueError("A WAV upload or a WAV local path is required.")
    if vmd_path is None and motion_json_path is None:
        raise ValueError("A dance VMD upload or local path is required (or a GlobalTransform JSON).")
    if motion_json_path is None and model_path is None:
        raise ValueError(
            "A PMX/PMD model whose skeleton matches the dance is required to evaluate the VMD. "
            "Upload it (with its texture folder for a textured preview), type its local path, "
            "or start the server with --default-model-path."
        )

    song_base = field_text(form, "song_id") or wav_path.stem
    song_id = f"{slug(song_base)}_{job_id[-8:]}"
    output_dir = output_root / song_id
    pipeline_mode = field_text(form, "pipeline_mode", "timing_only") or "timing_only"

    command = [
        str(pipeline_python),
        "-m",
        PIPELINE_MODULE,
        str(input_dir),
        "--song-id",
        song_id,
        "--output-root",
        str(output_root),
        "--wav",
        str(wav_path),
    ]
    if vmd_path is not None:
        command.extend(["--vmd", str(vmd_path)])
    if motion_json_path is not None:
        command.extend(["--motion-json", str(motion_json_path)])
    if model_path is not None:
        command.extend(["--model-path", str(model_path)])

    for field_name, flag in (
        ("target_dump_tool", "--target-dump-tool"),
        ("demucs_python", "--demucs-python"),
        ("demucs_model", "--demucs-model"),
        ("demucs_device", "--demucs-device"),
        ("device", "--device"),
        ("frame_policy", "--frame-policy"),
        ("cut_selection_mode", "--cut-selection-mode"),
        ("fps", "--fps"),
        ("density_cut_percent", "--density-cut-percent"),
        ("cut_window_sec", "--cut-window-sec"),
        ("cut_min_gap_sec", "--cut-min-gap-sec"),
        ("cut_min_score", "--cut-min-score"),
        ("keyframe_cut_exclusion_sec", "--keyframe-cut-exclusion-sec"),
        ("keyframe_min_gap_frames", "--keyframe-min-gap-frames"),
        ("density_allocation_bin_sec", "--density-allocation-bin-sec"),
        ("density_uniform_alpha", "--density-uniform-alpha"),
        ("demucs_shifts", "--demucs-shifts"),
        ("demucs_jobs", "--demucs-jobs"),
        ("demucs_segment", "--demucs-segment"),
        ("optimization_top_k", "--optimization-top-k"),
        ("optimization_max_time_sec", "--optimization-max-time-sec"),
        ("optimization_local_cost_scale", "--optimization-local-cost-scale"),
        ("optimization_distance_bounce_weight", "--optimization-distance-bounce-weight"),
        ("optimization_composition_bounce_weight", "--optimization-composition-bounce-weight"),
        ("optimization_top_pitch_bounce_weight", "--optimization-top-pitch-bounce-weight"),
        ("optimization_top_pitch_bounce_max_frame_gap", "--optimization-top-pitch-bounce-max-frame-gap"),
        ("optimization_cut_identity_weight", "--optimization-cut-identity-weight"),
        ("optimization_cut_contrast_weight", "--optimization-cut-contrast-weight"),
        ("optimization_cut_contrast_min_score", "--optimization-cut-contrast-min-score"),
        ("optimization_cut_context_speed_weight", "--optimization-cut-context-speed-weight"),
        ("optimization_cut_context_speed_min_speed", "--optimization-cut-context-speed-min-speed"),
        ("optimization_repeated_distance_bounce_weight", "--optimization-repeated-distance-bounce-weight"),
        ("optimization_distance_level_duration_weight", "--optimization-distance-level-duration-weight"),
        ("optimization_distance_level_duration_stats", "--optimization-distance-level-duration-stats"),
        ("optimization_distance_level_duration_include_levels", "--optimization-distance-level-duration-include-levels"),
        ("optimization_distance_level_duration_upper_key", "--optimization-distance-level-duration-upper-key"),
        ("shotplan_height_calibration_face_offset", "--shotplan-height-calibration-face-offset"),
        ("shotplan_height_calibration_chest_offset", "--shotplan-height-calibration-chest-offset"),
        ("render_model_path", "--render-model-path"),
        ("render_preview_width", "--render-preview-width"),
        ("render_preview_height", "--render-preview-height"),
        ("render_preview_fps", "--render-preview-fps"),
        ("render_preview_session_size", "--render-preview-session-size"),
    ):
        add_optional_arg(command, flag, field_text(form, field_name))

    if field_bool(form, "skip_demucs"):
        command.append("--skip-demucs")
    if field_bool(form, "allow_stem_fallback"):
        command.append("--allow-stem-fallback")
    if field_bool(form, "force"):
        command.append("--force")
    if pipeline_mode == "timing_only":
        command.append("--skip-shotplan")
    if field_bool(form, "skip_optimization"):
        command.append("--skip-optimization")
    if field_bool(form, "optimization_enable_sequence_stats"):
        command.append("--optimization-enable-sequence-stat-objective")
    if field_bool(form, "optimization_allow_new_top_pitch_candidates"):
        command.append("--optimization-allow-new-top-pitch-candidates")
    if field_text(form, "shotplan_height_calibration_mode", "on").lower() in {"0", "false", "no", "off"}:
        command.append("--no-shotplan-height-calibration")
    if field_bool(form, "skip_render_preview"):
        command.append("--skip-render-preview")

    return job_id, input_dir, command, output_dir, pipeline_mode


def app_config(args: argparse.Namespace) -> dict[str, Any]:
    default_model = args.default_model_path
    return {
        "repo_root": REPO_ROOT.as_posix().rstrip("/") + "/",
        "default_model": default_model.name if default_model is not None else None,
        "output_root": repo_url(args.output_root),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the CuCaT web demo locally.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8785)
    parser.add_argument(
        "--pipeline-python",
        type=Path,
        default=Path(sys.executable),
        help="Python used to run the pipeline. Defaults to the Python running this server.",
    )
    parser.add_argument("--upload-root", type=Path, default=DEFAULT_UPLOAD_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--default-model-path",
        type=Path,
        default=None,
        help="PMX/PMD used when the page gets no model. Without it, every run needs a model.",
    )
    return parser.parse_args()


class Handler(SimpleHTTPRequestHandler):
    server_version = "CuCaTDemoHTTP/0.1"

    def __init__(self, *handler_args: Any, **handler_kwargs: Any):
        super().__init__(*handler_args, directory=str(REPO_ROOT), **handler_kwargs)

    @property
    def app_args(self) -> argparse.Namespace:
        return self.server.app_args  # type: ignore[attr-defined]

    def log_request(self, code: Any = "-", size: Any = "-") -> None:
        # The page polls job status every couple of seconds; keep those out of the console.
        status = code.value if isinstance(code, HTTPStatus) else code
        if (
            self.command == "GET"
            and str(status).startswith("2")
            and urlsplit(self.path).path.startswith(QUIET_LOG_PREFIXES)
        ):
            return
        super().log_request(code, size)

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def resolve_static_path(self, url_path: str) -> Path | None:
        relative = unquote(url_path.lstrip("/"))
        candidate = (REPO_ROOT / relative).resolve()
        try:
            candidate.relative_to(REPO_ROOT)
        except ValueError:
            return None
        if not candidate.exists() or not candidate.is_file():
            return None
        return candidate

    def parse_range_header(self, size: int) -> tuple[int, int] | None:
        header = self.headers.get("Range")
        if not header or not header.startswith("bytes="):
            return None
        spec = header[len("bytes=") :].strip()
        if "," in spec:
            return None
        start_text, _, end_text = spec.partition("-")
        if not start_text and not end_text:
            return None
        try:
            if start_text:
                start = int(start_text)
                end = int(end_text) if end_text else size - 1
            else:
                suffix = int(end_text)
                if suffix <= 0:
                    return None
                start = max(0, size - suffix)
                end = size - 1
        except ValueError:
            return None
        if start < 0 or end < start or start >= size:
            return None
        return start, min(end, size - 1)

    def serve_static_file(self, path: Path, head_only: bool = False) -> None:
        size = path.stat().st_size
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        range_tuple = self.parse_range_header(size)
        if range_tuple is None:
            self.send_response(HTTPStatus.OK.value)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if head_only:
                return
            with path.open("rb") as handle:
                shutil.copyfileobj(handle, self.wfile)
            return

        start, end = range_tuple
        length = end - start + 1
        self.send_response(HTTPStatus.PARTIAL_CONTENT.value)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if head_only:
            return
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(1024 * 64, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_POST(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path == "/api/unseen/shotplan/render":
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw_body = self.rfile.read(length).decode("utf-8")
                payload = json.loads(raw_body) if raw_body.strip() else {}
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object payload.")
                self.send_json(render_edited_shotplan(payload))
            except Exception as exc:
                self.send_json({"error": str(exc), "traceback": traceback.format_exc()}, status=HTTPStatus.BAD_REQUEST)
            return

        if parsed.path != "/api/unseen/jobs":
            self.send_error(HTTPStatus.NOT_FOUND.value, "Unknown API endpoint")
            return
        try:
            ctype = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in ctype:
                raise ValueError("Expected multipart/form-data.")
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": ctype,
                    "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                },
            )
            job_id, input_dir, command, output_dir, pipeline_mode = build_command_from_form(
                form,
                pipeline_python=self.app_args.pipeline_python,
                upload_root=self.app_args.upload_root.resolve(),
                output_root=self.app_args.output_root.resolve(),
                default_model_path=self.app_args.default_model_path,
            )
            song_id = output_dir.name
            job = Job(
                id=job_id,
                song_id=song_id,
                pipeline_mode=pipeline_mode,
                upload_dir=str(input_dir),
                output_dir=str(output_dir),
                stages=default_stages(),
                artifacts=planned_artifacts(),
            )
            if pipeline_mode == "timing_only":
                for stage in job.stages:
                    if stage["id"] != "timing":
                        stage["status"] = "skipped"
            with JOBS_LOCK:
                JOBS[job_id] = job
            thread = threading.Thread(target=run_pipeline_job, args=(job_id, command, output_dir), daemon=True)
            thread.start()
            self.send_json(json_ready_job(job), status=HTTPStatus.ACCEPTED)
        except Exception as exc:
            self.send_json({"error": str(exc), "traceback": traceback.format_exc()}, status=HTTPStatus.BAD_REQUEST)

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path
        if path in {"", "/", "/unseen_mmd_pipeline_viewer.html", "/music_processing/unseen_mmd_pipeline_viewer.html"}:
            self.serve_file(VIEWER_PATH)
            return
        if path == "/api/unseen/config":
            self.send_json(app_config(self.app_args))
            return
        if path == "/api/unseen/jobs":
            with JOBS_LOCK:
                jobs = sorted(JOBS.values(), key=lambda job: job.created_at, reverse=True)
                payload = [json_ready_job(job) for job in jobs[:50]]
            self.send_json(payload)
            return
        if path.startswith("/api/unseen/jobs/"):
            job_id = unquote(path.rsplit("/", 1)[-1])
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                payload = json_ready_job(job) if job is not None else None
            if payload is None:
                self.send_json({"error": f"Unknown job: {job_id}"}, status=HTTPStatus.NOT_FOUND)
                return
            self.send_json(payload)
            return
        static_path = self.resolve_static_path(path)
        if static_path is not None:
            self.serve_static_file(static_path)
            return
        self.send_error(HTTPStatus.NOT_FOUND.value, "File not found")

    def do_HEAD(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path in {"", "/", "/unseen_mmd_pipeline_viewer.html", "/music_processing/unseen_mmd_pipeline_viewer.html"}:
            self.serve_file(VIEWER_PATH, head_only=True)
            return
        static_path = self.resolve_static_path(parsed.path)
        if static_path is not None:
            self.serve_static_file(static_path, head_only=True)
            return
        self.send_error(HTTPStatus.NOT_FOUND.value, "File not found")

    def serve_file(self, path: Path, head_only: bool = False) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND.value, "File not found")
            return
        data_size = path.stat().st_size
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(data_size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if head_only:
            return
        with path.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile)


def main() -> None:
    configure_utf8_stdio()
    args = parse_args()
    args.upload_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not PIPELINE_SCRIPT.exists():
        raise SystemExit(f"Pipeline script not found: {PIPELINE_SCRIPT}")
    if not VIEWER_PATH.exists():
        raise SystemExit(f"Viewer HTML not found: {VIEWER_PATH}")
    if args.default_model_path is not None:
        args.default_model_path = args.default_model_path.expanduser().resolve()
        if not args.default_model_path.exists():
            raise SystemExit(f"--default-model-path not found: {args.default_model_path}")
    try:
        args.output_root.resolve().relative_to(REPO_ROOT)
    except ValueError:
        print("Warning: --output-root is outside the repository, so the page cannot link or play the results.", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.app_args = args  # type: ignore[attr-defined]
    print(f"CuCaT demo: http://{args.host}:{args.port}/", flush=True)
    print(f"Pipeline Python: {args.pipeline_python}", flush=True)
    print(f"Output root: {args.output_root}", flush=True)
    if args.default_model_path is not None:
        print(f"Default model: {args.default_model_path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
