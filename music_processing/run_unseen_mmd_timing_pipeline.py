from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from music_processing.build_audio_only_cut_dataset_from_multimodal import (
    DEMUX_EVENTS_NO_RMS_AUDIO_FEATURES,
    _compact_dance_payload,
)
from music_processing.build_cut_conditioned_keyframe_dataset import (
    _build_condition_features,
    _build_prediction_intervals,
)
from music_processing.build_cut_transition_dataset import (
    _delta_feature,
    _distance_tracks,
    _event_frames_from_rows,
    _frame_edges,
    _frame_stats,
    _load_aligned_audio,
    _pulse_track,
)
from music_processing.dance_motion_features import (
    align_motion_positions,
    align_motion_rotations,
    build_motion_features,
    load_motion_transforms,
)
from music_processing.export_final_cut_key_timings import (
    coverage_intervals,
    cut_candidate_peaks,
    cut_pair_frames,
    event_payload,
    frames_near_events,
    keyframe_candidate_peaks,
    select_adaptive_cuts,
    select_density_uniform_keyframes,
    select_global_cuts,
    select_uniform_cuts,
)
from music_processing.madmom_downbeats import compute_madmom_downbeats
from music_processing.train_cut_conditioned_keyframe_conformer import _build_model as build_stage2_model
from music_processing.train_cut_transition_baseline import _build_window_starts, _device_from_arg, _predict_song
from music_processing.train_film_cut_transition import AudioDanceFiLMCutPredictor
from music_processing.train_keyframe_density_stagea import _prev_cut_context, build_stagea_model
from shotplan_transformer.apply_v3_interpolation_stage import (
    _build_model as build_interpolation_stage_model,
    _predict_interpolations,
)
from camera_autoshot.reverse_v3.solver_v3 import CameraSolveConfigV3
from shotplan_transformer.dataset import load_normalization_stats
from shotplan_transformer.infer_v3_cut_context_dcm_from_timing import (
    _build_cut_context_model,
    _checkpoint_manifest,
    _close_dense_distance_schema_for_manifest,
    _predict_for_clip as predict_v3_cut_context_for_clip,
    _write_vmd_from_plan,
)
from shotplan_transformer.export_topk_candidates_v3_cut_context_from_results import _export_plan_candidates


REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_MODEL_ROOT = REPO_ROOT / "final model"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "unseen_mmd_pipeline"
DEFAULT_SHOTPLAN_RUN_DIR = FINAL_MODEL_ROOT / "v3_1_visual_scale_005"
DEFAULT_INTERPOLATION_RUN_DIR = FINAL_MODEL_ROOT / "v3_1_interpolation_stage"
DEFAULT_DISTANCE_LEVEL_DURATION_STATS = (
    REPO_ROOT / "camera_autoshot" / "analysis" / "v31_distance_level_duration_stats_midpoint.json"
)
DEFAULT_SABA_TOOL_CANDIDATES = (REPO_ROOT / "saba_mmd" / "build" / "Release" / "mmd_target_dump.exe",)

FULL_AUDIO_FEATURE_NAMES = [
    "mix_abs_mean",
    "mix_rms",
    "mix_peak",
    "vocals_abs_mean",
    "vocals_rms",
    "vocals_peak",
    "bass_abs_mean",
    "bass_rms",
    "bass_peak",
    "drums_abs_mean",
    "drums_rms",
    "drums_peak",
    "other_abs_mean",
    "other_rms",
    "other_peak",
    "mix_rms_delta",
    "vocals_rms_delta",
    "bass_rms_delta",
    "drums_rms_delta",
    "other_rms_delta",
    "beat_pulse",
    "beat_since",
    "beat_until",
    "downbeat_pulse",
    "downbeat_since",
    "downbeat_until",
    "bar_pulse",
    "bar_since",
    "bar_until",
]
STEM_NAMES = ("vocals", "bass", "drums", "other")
# Lines starting with this prefix are progress events read by serve_unseen_mmd_pipeline.py.
PIPELINE_STAGE_PREFIX = "__PIPELINE_STAGE__ "


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


def emit_stage_event(stage_id: str, status: str, label: str, duration_sec: float | None = None) -> None:
    payload: dict[str, Any] = {
        "id": stage_id,
        "label": label,
        "status": status,
    }
    if duration_sec is not None:
        payload["duration_sec"] = round(float(duration_sec), 4)
    print(PIPELINE_STAGE_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


@contextmanager
def timed_stage(timings: list[dict[str, Any]], stage_id: str, label: str, *, emit_events: bool = True):
    started = time.perf_counter()
    status = "ok"
    if emit_events:
        emit_stage_event(stage_id, "running", label)
    try:
        yield
    except Exception:
        status = "failed"
        raise
    finally:
        elapsed = time.perf_counter() - started
        if emit_events:
            emit_stage_event(
                stage_id,
                "failed" if status == "failed" else "completed",
                label,
                duration_sec=elapsed,
            )
        timings.append(
            {
                "id": stage_id,
                "label": label,
                "status": status,
                "duration_sec": round(float(elapsed), 4),
            }
        )


def timing_payload(timings: list[dict[str, Any]], *, total_started_at: float) -> dict[str, Any]:
    total_sec = time.perf_counter() - total_started_at
    rows = list(timings)
    return {
        "total_sec": round(float(total_sec), 4),
        "stages": rows,
        "stages_by_id": {str(row["id"]): row for row in rows},
    }


# Demo-pipeline-only VMD calibration.  These exported V3.1 conditions are
# internalized before solving as:
#   very_close -> internal extreme_close, close -> internal close.
DEFAULT_SHOTPLAN_HEIGHT_CALIBRATION_OFFSETS = (
    ("face", "extreme_close", -0.035),
    ("face", "close", -0.035),
    ("chest", "extreme_close", -0.025),
    ("chest", "close", -0.025),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run raw unseen MMD audio/motion preprocessing and export final cut/keyframe timings. "
            "Raw VMD conversion requires a matching PMX/PMD model; pass --motion-json to skip that step."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing one .wav and one .vmd file.")
    parser.add_argument("--wav", type=Path, default=None, help="Explicit WAV path. Defaults to first .wav in input_dir.")
    parser.add_argument("--vmd", type=Path, default=None, help="Explicit VMD path. Defaults to first .vmd in input_dir.")
    parser.add_argument(
        "--motion-json",
        type=Path,
        default=None,
        help="Optional precomputed DCA-style *_GlobalTransform.json. If omitted, --model-path is required.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="PMX/PMD model used to evaluate the raw VMD into global bone transforms.",
    )
    parser.add_argument(
        "--target-dump-tool",
        type=Path,
        default=None,
        help="mmd_target_dump.exe with -global-transform support. Defaults to saba_mmd build locations.",
    )
    parser.add_argument("--song-id", default=None, help="Output song id. Defaults to a sanitized input folder name.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--frame-policy",
        choices=["audio", "motion", "min"],
        default="min",
        help="How to choose final frame count when audio and motion lengths differ.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate cached intermediate files.")
    parser.add_argument(
        "--reuse-from",
        type=Path,
        default=None,
        help=(
            "Output folder of an earlier run. Its motion conversion, Demucs stems, and beat tracking are reused "
            "when they were made from the same WAV, VMD, and model with the same settings."
        ),
    )

    parser.add_argument("--demucs-model", default="htdemucs")
    parser.add_argument(
        "--demucs-python",
        type=Path,
        default=None,
        help="Optional Python executable that has demucs installed. Defaults to the current Python.",
    )
    parser.add_argument("--demucs-device", default="cuda", help="cuda, cpu, or auto (cuda when available).")
    parser.add_argument("--demucs-shifts", type=int, default=1)
    parser.add_argument("--demucs-jobs", type=int, default=0)
    parser.add_argument("--demucs-segment", type=int, default=None)
    parser.add_argument("--skip-demucs", action="store_true", help="Use existing Demucs summary only.")
    parser.add_argument(
        "--allow-stem-fallback",
        action="store_true",
        help="If a Demucs stem is missing, use the mix waveform for that stem instead of failing.",
    )

    parser.add_argument("--device", default="auto")
    parser.add_argument("--cut-model", type=Path, default=FINAL_MODEL_ROOT / "cut_transition" / "best_model.pt")
    parser.add_argument("--density-model", type=Path, default=FINAL_MODEL_ROOT / "density_stagea" / "best_model.pt")
    parser.add_argument("--keyframe-model", type=Path, default=FINAL_MODEL_ROOT / "keyframe_stage2" / "best_model.pt")
    parser.add_argument("--skip-shotplan", action="store_true", help="Stop after cut/key timing prediction.")
    parser.add_argument("--skip-interpolation", action="store_true", help="Skip the separate interpolation post-stage.")
    parser.add_argument("--shotplan-run-dir", type=Path, default=DEFAULT_SHOTPLAN_RUN_DIR)
    parser.add_argument("--shotplan-checkpoint-name", default="best.pt")
    parser.add_argument("--interpolation-run-dir", type=Path, default=DEFAULT_INTERPOLATION_RUN_DIR)
    parser.add_argument("--interpolation-checkpoint-name", default="best.pt")
    parser.add_argument("--shotplan-logit-temperature", type=float, default=1.0)
    parser.add_argument(
        "--decode-distance-fov-joint",
        action="store_true",
        default=True,
        help="Decode distance/fov with the trained joint head when available.",
    )
    parser.add_argument(
        "--no-decode-distance-fov-joint",
        dest="decode_distance_fov_joint",
        action="store_false",
        help="Use independent distance and fov heads instead of the joint head.",
    )
    parser.add_argument("--skip-optimization", action="store_true", help="Skip CP-SAT shot-plan optimization.")
    parser.add_argument(
        "--shotplan-height-calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply demo VMD height calibration for near face/chest shots "
            "(face/chest + very_close/close look-at anchor slightly lower)."
        ),
    )
    parser.add_argument("--shotplan-height-calibration-face-offset", type=float, default=-0.035)
    parser.add_argument("--shotplan-height-calibration-chest-offset", type=float, default=-0.025)
    parser.add_argument(
        "--actor-forward-stabilization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the actor forward direction sign-stable over time when solving the camera VMDs.",
    )
    parser.add_argument("--optimization-top-k", type=int, default=10)
    parser.add_argument("--optimization-max-time-sec", type=float, default=30.0)
    parser.add_argument("--optimization-local-cost-scale", type=int, default=100)
    parser.add_argument("--optimization-disable-sequence-stat-objective", action="store_true", default=True)
    parser.add_argument(
        "--optimization-enable-sequence-stat-objective",
        dest="optimization_disable_sequence_stat_objective",
        action="store_false",
        help="Enable GT sequence-stat objective. For unseen inputs this usually requires matching GT labels.",
    )
    parser.add_argument("--optimization-distance-bounce-weight", type=int, default=400)
    parser.add_argument("--optimization-distance-bounce-min-rank-delta", type=int, default=80)
    parser.add_argument("--optimization-distance-bounce-max-frame-gap", type=int, default=60)
    parser.add_argument("--optimization-composition-bounce-weight", type=int, default=100)
    parser.add_argument("--optimization-composition-bounce-min-rank-delta", type=int, default=80)
    parser.add_argument("--optimization-composition-bounce-max-frame-gap", type=int, default=60)
    parser.add_argument("--optimization-top-pitch-bounce-weight", type=int, default=400)
    parser.add_argument("--optimization-top-pitch-bounce-max-frame-gap", type=int, default=30)
    parser.add_argument("--optimization-cut-identity-weight", type=int, default=400)
    parser.add_argument("--optimization-cut-contrast-weight", type=int, default=100)
    parser.add_argument("--optimization-cut-contrast-min-score", type=int, default=3)
    parser.add_argument("--optimization-cut-contrast-medium-rank-delta", type=int, default=100)
    parser.add_argument("--optimization-cut-contrast-strong-rank-delta", type=int, default=220)
    parser.add_argument("--optimization-cut-context-speed-weight", type=int, default=400)
    parser.add_argument("--optimization-cut-context-speed-min-speed", type=float, default=250.0)
    parser.add_argument("--optimization-cut-context-speed-max-frame-gap", type=int, default=60)
    parser.add_argument("--optimization-repeated-distance-bounce-weight", type=int, default=1200)
    parser.add_argument("--optimization-repeated-distance-bounce-min-rank-delta", type=int, default=180)
    parser.add_argument("--optimization-repeated-distance-bounce-max-frame-gap", type=int, default=90)
    parser.add_argument("--optimization-distance-level-duration-weight", type=int, default=160)
    parser.add_argument("--optimization-distance-level-duration-stats", type=Path, default=DEFAULT_DISTANCE_LEVEL_DURATION_STATS)
    parser.add_argument("--optimization-distance-level-duration-include-levels", default="tight_close,close,medium_close")
    parser.add_argument("--optimization-distance-level-duration-upper-key", default="p99_sec")
    parser.add_argument("--optimization-distance-level-duration-penalty-sec-per-unit", type=float, default=0.5)
    parser.add_argument(
        "--optimization-disallow-new-top-pitch-candidates",
        action="store_true",
        default=True,
        help="Prevent optimization from introducing top pitch when the top-1 candidate was not top.",
    )
    parser.add_argument(
        "--optimization-allow-new-top-pitch-candidates",
        dest="optimization_disallow_new_top_pitch_candidates",
        action="store_false",
    )
    parser.add_argument(
        "--skip-render-preview",
        action="store_true",
        help="Skip SABA MP4 preview rendering after camera VMD export.",
    )
    parser.add_argument(
        "--render-model-path",
        type=Path,
        default=None,
        help="Optional textured PMX/PMD used only for preview rendering.",
    )
    parser.add_argument("--render-preview-width", type=int, default=1280)
    parser.add_argument("--render-preview-height", type=int, default=720)
    parser.add_argument("--render-preview-fps", type=int, default=30)
    parser.add_argument("--render-preview-session-size", type=int, default=1)

    # Defaults follow the paper setting (density 70% of keyframes for cut pairs, global cut selection,
    # no minimum cut score).
    parser.add_argument("--density-cut-percent", type=float, default=70.0)
    parser.add_argument(
        "--density-budget-scale",
        type=float,
        default=1.0,
        help="Multiply the Stage-A keyframe budget (both passes) before it is split into cuts and transitions.",
    )
    parser.add_argument(
        "--density-cut-percent-unit",
        choices=["cut_keyframes", "cut_events"],
        default="cut_keyframes",
        help="cut_keyframes: percent of the keyframe budget used by cut pairs (2 keyframes each). "
        "cut_events: percent of the budget used as the number of cuts.",
    )
    parser.add_argument("--cut-selection-mode", choices=["global", "adaptive", "uniform"], default="global")
    parser.add_argument("--cut-window-sec", type=float, default=10.0)
    parser.add_argument("--cut-min-gap-sec", type=float, default=1.0)
    parser.add_argument("--cut-min-score", type=float, default=0.0)
    parser.add_argument("--cut-prominence-weight", type=float, default=0.3)
    parser.add_argument("--density-window-frames", type=int, default=90)
    parser.add_argument("--density-stride-frames", type=int, default=15)
    parser.add_argument("--density-distance-clip-frames", type=int, default=90)
    parser.add_argument("--density-allocation-bin-sec", type=float, default=3.0)
    parser.add_argument("--density-uniform-alpha", type=float, default=0.6)
    parser.add_argument("--keyframe-min-score", type=float, default=0.0)
    parser.add_argument("--keyframe-min-gap-frames", type=int, default=5)
    parser.add_argument("--keyframe-cut-exclusion-sec", type=float, default=0.5)
    parser.add_argument("--keyframe-prominence-weight", type=float, default=0.3)
    parser.add_argument("--stage2-max-frames", type=int, default=300)
    return parser.parse_args()


def _slug(text: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("._-")
    if normalized:
        return normalized[:80]
    return "unseen_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_file(recorded: Any, current: Path | None) -> bool:
    if not recorded or current is None:
        return False
    earlier = Path(str(recorded))
    if not (earlier.is_file() and current.is_file()):
        return False
    if earlier.resolve() == current.resolve():
        return True
    return earlier.stat().st_size == current.stat().st_size and _file_sha1(earlier) == _file_sha1(current)


def _link_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _reuse_earlier_preprocessing(
    source_dir: Path,
    *,
    song_id: str,
    intermediate_dir: Path,
    inputs: dict[str, Any],
    demucs_settings: dict[str, Any] | None,
) -> dict[str, str]:
    """Link an earlier run's motion conversion, Demucs stems, and beats into this run's intermediate folder.

    Each one is reused only when it was made from the same files with the same settings; the stages then find
    them as cached files. Anything else is computed as usual.
    """

    source_intermediate = source_dir / "intermediate"
    earlier = _read_json_or_empty(source_intermediate / "inputs.json")
    if not earlier:
        # Runs made before inputs.json was written: recover the same facts from their summaries.
        preprocess = _read_json_or_empty(source_dir / "preprocess_summary.json")
        motion = preprocess.get("motion_conversion") or {}
        command = [str(item) for item in motion.get("command") or []]
        earlier = {
            "wav": (preprocess.get("demucs") or {}).get("audio_path"),
            "vmd": motion.get("vmd_path"),
            "model": motion.get("model_path"),
            "motion_end_frame": int(command[command.index("-end") + 1]) if "-end" in command[:-1] else None,
        }

    reused: dict[str, str] = {}
    same_wav = _same_file(earlier.get("wav"), inputs["wav"])
    if (
        same_wav
        and earlier.get("motion_end_frame") == inputs["motion_end_frame"]
        and _same_file(earlier.get("vmd"), inputs["vmd"])
        and _same_file(earlier.get("model"), inputs["model"])
    ):
        pairs = [
            (source_intermediate / f"{source_dir.name}_{suffix}", intermediate_dir / f"{song_id}_{suffix}")
            for suffix in ("GlobalTransform.json", "target_probe.json")
        ]
        if all(source.is_file() for source, _ in pairs):
            for source, destination in pairs:
                _link_file(source, destination)
            reused["motion_conversion"] = str(source_dir)
    if not same_wav:
        return reused

    demucs_dir = source_intermediate / "demucs"
    demucs = _read_json_or_empty(demucs_dir / "summary.json")
    stems = {name: Path(path) for name, path in (demucs.get("stem_files") or {}).items()}
    if (
        demucs_settings is not None
        and demucs.get("returncode") == 0
        and all(demucs.get(key) == value for key, value in demucs_settings.items())
        and stems
        and all(path.is_file() and path.is_relative_to(demucs_dir) for path in stems.values())
    ):
        target_dir = intermediate_dir / "demucs"
        for path in stems.values():
            _link_file(path, target_dir / path.relative_to(demucs_dir))
        demucs.update(
            output_dir=str(target_dir),
            stem_files={name: str(target_dir / path.relative_to(demucs_dir)) for name, path in stems.items()},
            reused_from=str(demucs_dir),
        )
        _write_json(target_dir / "summary.json", demucs)
        reused["demucs"] = str(source_dir)

    madmom_path = source_intermediate / "madmom_downbeats.json"
    if madmom_path.is_file() and "fallback_error" not in _read_json_or_empty(madmom_path):
        _link_file(madmom_path, intermediate_dir / "madmom_downbeats.json")
        reused["beat_tracking"] = str(source_dir)
    return reused


def _find_single_file(input_dir: Path, suffix: str) -> Path:
    matches = sorted(Path(input_dir).glob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"No {suffix} file found in {input_dir}")
    return matches[0]


def _resolve_target_dump_tool(explicit: Path | None) -> Path:
    if explicit is not None:
        tool = explicit
        if tool.exists():
            return tool
        raise FileNotFoundError(f"mmd_target_dump tool not found: {tool}")
    for candidate in DEFAULT_SABA_TOOL_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "mmd_target_dump.exe was not found. Build saba_mmd or pass --target-dump-tool."
    )


def _audio_frame_count(audio_path: Path, fps: float) -> tuple[int, float, int]:
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration_sec = float(len(y) / sr)
    return max(1, int(math.ceil(duration_sec * float(fps)))), duration_sec, int(sr)


def _run_mmd_global_transform(
    *,
    tool_path: Path,
    model_path: Path,
    vmd_path: Path,
    output_path: Path,
    probe_output_path: Path,
    end_frame: int,
    force: bool,
) -> dict[str, Any]:
    if output_path.exists() and probe_output_path.exists() and not force:
        return {
            "status": "cached",
            "tool_path": str(tool_path),
            "model_path": str(model_path),
            "vmd_path": str(vmd_path),
            "output_path": str(output_path),
            "probe_output_path": str(probe_output_path),
        }

    if not model_path.exists():
        raise FileNotFoundError(f"PMX/PMD model not found: {model_path}")
    if not vmd_path.exists():
        raise FileNotFoundError(f"VMD not found: {vmd_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(tool_path),
        str(model_path),
        "-vmd",
        str(vmd_path),
        "-o",
        str(probe_output_path),
        "-global-transform",
        str(output_path),
        "-start",
        "0",
        "-end",
        str(max(0, int(end_frame))),
        "-step",
        "1",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=utf8_subprocess_env(),
    )
    summary = {
        "status": "ok" if completed.returncode == 0 else "failed",
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "tool_path": str(tool_path),
        "model_path": str(model_path),
        "vmd_path": str(vmd_path),
        "output_path": str(output_path),
    }
    if completed.returncode != 0:
        raise RuntimeError(
            "mmd_target_dump failed while generating global transforms. "
            "The tool must be rebuilt with -global-transform support, and the PMX must match the VMD.\n"
            f"Command: {' '.join(command)}\n"
            f"stderr: {completed.stderr.strip()}"
        )
    return summary


def _find_stem_files(model_output_dir: Path) -> dict[str, str]:
    stem_paths: dict[str, str] = {}
    for stem_name in STEM_NAMES:
        matches = sorted(model_output_dir.rglob(f"{stem_name}.wav"))
        if matches:
            stem_paths[stem_name] = str(matches[0])
    return stem_paths


def _resolve_demucs_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _run_demucs(
    *,
    audio_path: Path,
    song_id: str,
    output_dir: Path,
    model: str,
    python_executable: Path | None,
    device: str,
    shifts: int,
    jobs: int,
    segment: int | None,
    skip: bool,
    force: bool,
) -> dict[str, Any]:
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not force:
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        if cached.get("returncode") in (None, 0):
            return cached
    if skip:
        summary = {
            "song_id": song_id,
            "audio_path": str(audio_path),
            "output_dir": str(output_dir),
            "model": model,
            "device": device,
            "shifts": int(shifts),
            "jobs": int(jobs),
            "segment": segment,
            "command": None,
            "returncode": None,
            "stem_files": {},
            "warning": "Demucs was skipped. Downstream code must allow stem fallback.",
        }
        _write_json(summary_path, summary)
        return summary

    output_dir.mkdir(parents=True, exist_ok=True)
    python_command = str(python_executable) if python_executable is not None else sys.executable
    command = [
        python_command,
        "-m",
        "music_processing.demucs_cli",
        "-n",
        str(model),
        "-d",
        str(device),
        "--shifts",
        str(int(shifts)),
        "-o",
        str(output_dir),
    ]
    if int(jobs) > 0:
        command.extend(["-j", str(int(jobs))])
    if segment is not None:
        command.extend(["--segment", str(int(segment))])
    command.append(str(audio_path))

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=utf8_subprocess_env(),
    )
    model_output_dir = output_dir / model
    stem_paths = _find_stem_files(model_output_dir)
    summary = {
        "song_id": song_id,
        "audio_path": str(audio_path),
        "output_dir": str(output_dir),
        "model": model,
        "device": device,
        "shifts": int(shifts),
        "jobs": int(jobs),
        "segment": segment,
        "command": command,
        "returncode": completed.returncode,
        "stem_files": stem_paths,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    _write_json(summary_path, summary)
    if completed.returncode != 0:
        raise RuntimeError(f"Demucs failed: {completed.stderr.strip()}")
    return summary


def _fallback_madmom(audio_path: Path, *, fps: float, num_frames: int) -> dict[str, Any]:
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset, sr=sr, trim=False)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beats = [
        {
            "index": int(idx),
            "time_sec": round(float(time_sec), 3),
            "beat_index": int(((idx - 1) % 4) + 1),
            "label": str(int(((idx - 1) % 4) + 1)),
        }
        for idx, time_sec in enumerate(beat_times, start=1)
        if 0 <= int(round(float(time_sec) * fps)) < int(num_frames)
    ]
    downbeats = [
        {
            "index": len([b for b in beats[:idx] if int(b["beat_index"]) == 1]),
            "time_sec": row["time_sec"],
            "beat_index": 1,
            "label": "1",
        }
        for idx, row in enumerate(beats)
        if int(row["beat_index"]) == 1
    ]
    return {
        "audio_path": str(audio_path),
        "method": "librosa_fallback_beat_track",
        "tempo_bpm": float(np.asarray(tempo).reshape(-1)[0]) if np.asarray(tempo).size else 0.0,
        "beats": beats,
        "beat_count": len(beats),
        "downbeats": downbeats,
        "downbeat_count": len(downbeats),
        "warning": "madmom failed or is unavailable; bars use every 4th librosa beat.",
    }


def _load_or_compute_madmom(audio_path: Path, output_path: Path, *, fps: float, num_frames: int, force: bool) -> dict[str, Any]:
    if output_path.exists() and not force:
        return json.loads(output_path.read_text(encoding="utf-8"))
    try:
        payload = compute_madmom_downbeats(audio_path)
    except Exception as exc:  # madmom can be fragile on fresh environments.
        payload = _fallback_madmom(audio_path, fps=fps, num_frames=num_frames)
        payload["fallback_error"] = repr(exc)
    _write_json(output_path, payload)
    return payload


def _stack_feature_dict(feature_by_name: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    missing = [name for name in names if name not in feature_by_name]
    if missing:
        raise ValueError(f"Missing audio feature columns: {missing}")
    return np.stack([feature_by_name[name] for name in names], axis=1).astype(np.float32)


def _build_audio_features(
    *,
    audio_path: Path,
    demucs_summary: dict[str, Any],
    madmom_summary: dict[str, Any],
    num_frames: int,
    fps: float,
    distance_clip_frames: int,
    allow_stem_fallback: bool,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    y_mix, sr = _load_aligned_audio(audio_path, num_frames=num_frames, fps=fps)
    frame_edges = _frame_edges(num_frames=num_frames, sr=sr, fps=fps)
    mix_abs_mean, mix_rms, mix_peak = _frame_stats(y_mix, frame_edges)

    by_name: dict[str, np.ndarray] = {
        "mix_abs_mean": mix_abs_mean,
        "mix_rms": mix_rms,
        "mix_peak": mix_peak,
    }
    rms_channels: dict[str, np.ndarray] = {"mix": mix_rms}

    stem_files = dict(demucs_summary.get("stem_files") or {})
    for stem_name in STEM_NAMES:
        stem_path_text = stem_files.get(stem_name)
        if not stem_path_text:
            if not allow_stem_fallback:
                raise FileNotFoundError(f"Missing Demucs stem `{stem_name}`.")
            stem_path = audio_path
        else:
            stem_path = Path(stem_path_text)
            if not stem_path.is_absolute():
                stem_path = (REPO_ROOT / stem_path).resolve()
            if not stem_path.exists():
                if not allow_stem_fallback:
                    raise FileNotFoundError(f"Demucs stem not found: {stem_path}")
                stem_path = audio_path

        y_stem, stem_sr = _load_aligned_audio(stem_path, num_frames=num_frames, fps=fps)
        stem_edges = _frame_edges(num_frames=num_frames, sr=stem_sr, fps=fps)
        stem_abs_mean, stem_rms, stem_peak = _frame_stats(y_stem, stem_edges)
        by_name[f"{stem_name}_abs_mean"] = stem_abs_mean
        by_name[f"{stem_name}_rms"] = stem_rms
        by_name[f"{stem_name}_peak"] = stem_peak
        rms_channels[stem_name] = stem_rms

    for source_name, rms_values in rms_channels.items():
        by_name[f"{source_name}_rms_delta"] = _delta_feature(rms_values)

    beats = list(madmom_summary.get("beats") or [])
    downbeats = list(madmom_summary.get("downbeats") or [])
    bars = downbeats or [row for row in beats if int(row.get("beat_index", 0) or 0) == 1]
    beat_frames = _event_frames_from_rows(beats, fps=fps, num_frames=num_frames)
    downbeat_frames = _event_frames_from_rows(downbeats, fps=fps, num_frames=num_frames)
    bar_frames = _event_frames_from_rows(bars, fps=fps, num_frames=num_frames)

    for event_name, event_frames in (
        ("beat", beat_frames),
        ("downbeat", downbeat_frames),
        ("bar", bar_frames),
    ):
        pulse = _pulse_track(event_frames, num_frames)
        since, until = _distance_tracks(event_frames, num_frames, clip_frames=distance_clip_frames)
        by_name[f"{event_name}_pulse"] = pulse
        by_name[f"{event_name}_since"] = since
        by_name[f"{event_name}_until"] = until

    full_audio = _stack_feature_dict(by_name, FULL_AUDIO_FEATURE_NAMES)
    cut_audio = _stack_feature_dict(by_name, DEMUX_EVENTS_NO_RMS_AUDIO_FEATURES)
    return full_audio, FULL_AUDIO_FEATURE_NAMES, cut_audio, list(DEMUX_EVENTS_NO_RMS_AUDIO_FEATURES)


def _build_dance_features(
    *,
    motion_json_path: Path,
    num_frames: int,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str], dict[str, Any]]:
    frame_numbers, positions, rotations, bone_names = load_motion_transforms(motion_json_path)
    positions, position_alignment = align_motion_positions(positions, target_frames=num_frames)
    rotations, rotation_alignment = align_motion_rotations(rotations, target_frames=num_frames)
    # The current final keyframe-stage2 checkpoint was trained on the base
    # 60-joint motion descriptor: relpos, relvel, root pos/vel, joint speeds,
    # and motion energy. Rotation/vector extras would change 427 -> 460 dims.
    full_dance, full_names = build_motion_features(positions)
    compact_dance, compact_names, compact_metadata = _compact_dance_payload(full_dance, full_names)
    metadata = {
        "source_motion_json": str(motion_json_path),
        "source_frame_count": int(len(frame_numbers)),
        "source_first_frame": int(frame_numbers[0]) if len(frame_numbers) else 0,
        "source_last_frame": int(frame_numbers[-1]) if len(frame_numbers) else 0,
        "target_frame_count": int(num_frames),
        "bone_count": int(len(bone_names)),
        "bone_names": bone_names,
        "position_alignment": position_alignment,
        "rotation_alignment": rotation_alignment,
        "compact_metadata": compact_metadata,
    }
    return full_dance, full_names, compact_dance, compact_names, metadata


def _vmd_motion_frame_count(vmd_path: Path | None) -> int | None:
    """Frames a dance VMD covers: its last bone or morph key plus one (None for a VMD without motion keys)."""

    if vmd_path is None or not Path(vmd_path).is_file():
        return None
    data = Path(vmd_path).read_bytes()
    offset = 50  # 30-byte signature + 20-byte model name
    last_frame = -1
    for record_size in (111, 23):  # bone keys, then morph keys; both store the frame after a 15-byte name
        if offset + 4 > len(data):
            break
        count = int(np.frombuffer(data, dtype="<u4", count=1, offset=offset)[0])
        offset += 4
        if count:
            records = np.frombuffer(data, dtype=np.dtype([("name", "S15"), ("frame", "<u4"), ("rest", f"V{record_size - 19}")]), count=count, offset=offset)
            last_frame = max(last_frame, int(records["frame"].max()))
        offset += count * record_size
    return last_frame + 1 if last_frame >= 0 else None


def _choose_frame_count(
    *,
    audio_frames: int,
    motion_json_path: Path | None,
    frame_policy: str,
    fps: float,
    vmd_path: Path | None = None,
) -> int:
    if motion_json_path is None or str(frame_policy) == "audio":
        return int(audio_frames)
    frame_numbers, positions, _rotations, _bone_names = load_motion_transforms(motion_json_path)
    motion_frames = int(positions.shape[0])
    # The motion JSON is evaluated up to the audio length, holding the last pose, so the dance's own end comes from its VMD.
    vmd_frames = _vmd_motion_frame_count(vmd_path)
    if vmd_frames is not None:
        motion_frames = min(motion_frames, vmd_frames)
    if str(frame_policy) == "motion":
        return max(1, motion_frames)
    return max(1, min(int(audio_frames), motion_frames))


def _load_cut_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    args = checkpoint.get("args") or {}
    model = AudioDanceFiLMCutPredictor(
        audio_dim=int(checkpoint.get("audio_dim", 17)),
        dance_dim=int(checkpoint.get("dance_dim", 168)),
        hidden_dim=int(args.get("hidden_dim", 128)),
        dropout=float(args.get("dropout", 0.1)),
        film_scale=float(args.get("film_scale", 0.5)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _load_stagea_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    args = checkpoint.get("args") or {}
    dims = checkpoint.get("feature_dims") or {}
    model = build_stagea_model(
        model_type=str(args.get("model_type", "cnn")),
        audio_dim=int(dims.get("audio", 17)),
        dance_dim=int(dims.get("dance", 168)),
        cut_context_dim=int(dims.get("prev_cut", 4)),
        model_dim=int(args.get("model_dim", 192)),
        num_heads=int(args.get("num_heads", 4)),
        num_layers=int(args.get("num_layers", 3)),
        cnn_kernel_size=int(args.get("cnn_kernel_size", 5)),
        cnn_dilations=args.get("cnn_dilations", "1,2,4,8"),
        dropout=float(args.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _load_stage2_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    args = argparse.Namespace(**dict(checkpoint.get("args") or {}))
    model = build_stage2_model(
        args=args,
        audio_input_dim=len(checkpoint["audio_mean"]),
        dance_input_dim=len(checkpoint["dance_mean"]),
        condition_input_dim=len(checkpoint["cond_mean"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _select_stage2_audio_features(
    checkpoint: dict[str, Any],
    *,
    full_audio: np.ndarray,
    cut_audio: np.ndarray,
) -> tuple[np.ndarray, str]:
    expected_dim = int(len(checkpoint["audio_mean"]))
    if expected_dim == int(cut_audio.shape[1]):
        return cut_audio, "demucs_events_no_rms"
    if expected_dim == int(full_audio.shape[1]):
        return full_audio, "full_audio"
    raise ValueError(
        "Stage-B keyframe model audio dimension does not match available features: "
        f"checkpoint expects {expected_dim}, cut_audio has {cut_audio.shape[1]}, "
        f"full_audio has {full_audio.shape[1]}"
    )


def _select_stage2_dance_features(
    checkpoint: dict[str, Any],
    *,
    full_dance: np.ndarray,
    compact_dance: np.ndarray,
) -> tuple[np.ndarray, str]:
    expected_dim = int(len(checkpoint["dance_mean"]))
    if expected_dim == int(compact_dance.shape[1]):
        return compact_dance, "compact_dance"
    if expected_dim == int(full_dance.shape[1]):
        return full_dance, "full_dance"
    raise ValueError(
        "Stage-B keyframe model dance dimension does not match available features: "
        f"checkpoint expects {expected_dim}, compact_dance has {compact_dance.shape[1]}, "
        f"full_dance has {full_dance.shape[1]}"
    )


def _series_from_values(values: np.ndarray, *, fps: float) -> list[dict[str, Any]]:
    return [
        {
            "frame": int(frame),
            "time_sec": round(float(frame) / float(fps), 3),
            "value": float(value),
        }
        for frame, value in enumerate(np.asarray(values, dtype=np.float32).tolist())
    ]


def _predict_stagea_density(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    *,
    audio: np.ndarray,
    dance: np.ndarray,
    cut_frames: list[int],
    fps: float,
    window_frames: int,
    stride_frames: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    audio_mean = np.asarray(checkpoint["audio_mean"], dtype=np.float32)
    audio_std = np.maximum(np.asarray(checkpoint["audio_std"], dtype=np.float32), 1e-6)
    dance_mean = np.asarray(checkpoint["dance_mean"], dtype=np.float32)
    dance_std = np.maximum(np.asarray(checkpoint["dance_std"], dtype=np.float32), 1e-6)
    args = checkpoint.get("args") or {}
    max_prev_cut_age_sec = float(args.get("max_prev_cut_age_sec", 10.0))
    max_prev_cut_gap_sec = float(args.get("max_prev_cut_gap_sec", 10.0))

    starts = _build_window_starts(len(audio), int(window_frames), int(stride_frames))
    density_sum = np.zeros(len(audio), dtype=np.float32)
    density_count = np.zeros(len(audio), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    sorted_cuts = sorted(int(frame) for frame in cut_frames)
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for start in starts:
            end = min(len(audio), int(start) + int(window_frames))
            valid_length = int(end - start)
            previous = [frame for frame in sorted_cuts if frame < int(start)]
            last_cut = previous[-1] if previous else None
            prev_cut = previous[-2] if len(previous) >= 2 else None
            cut_context = _prev_cut_context(
                start_frame=int(start),
                window_frames=valid_length,
                fps=fps,
                last_cut_frame=last_cut,
                previous_cut_frame=prev_cut,
                max_age_sec=max_prev_cut_age_sec,
                max_gap_sec=max_prev_cut_gap_sec,
            )
            batch_audio = ((audio[start:end] - audio_mean[None, :]) / audio_std[None, :]).astype(np.float32)
            batch_dance = ((dance[start:end] - dance_mean[None, :]) / dance_std[None, :]).astype(np.float32)
            valid = np.ones((1, valid_length), dtype=np.float32)
            logits = model(
                torch.from_numpy(batch_audio[None, :, :]).to(device),
                torch.from_numpy(batch_dance[None, :, :]).to(device),
                torch.from_numpy(cut_context[None, :, :]).to(device),
                torch.from_numpy(valid).to(device),
            )
            density = float(torch.sigmoid(logits)[0].detach().cpu())
            density_sum[start:end] += density
            density_count[start:end] += 1.0
            rows.append(
                {
                    "start_frame": int(start),
                    "end_frame_exclusive": int(end),
                    "window_frames": int(valid_length),
                    "density": round(density, 6),
                    "last_cut_frame": last_cut,
                    "previous_cut_frame": prev_cut,
                }
            )

    density_count = np.maximum(density_count, 1.0)
    return density_sum / density_count, rows


def _select_cuts(
    *,
    cut_probabilities: np.ndarray,
    density_by_frame: np.ndarray,
    fps: float,
    mode: str,
    density_cut_percent: float,
    min_score: float,
    prominence_weight: float,
    min_gap_sec: float,
    window_sec: float,
    density_cut_percent_unit: str = "cut_keyframes",
    budget_scale: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pred_keyframe_mass = float(np.sum(density_by_frame))
    total_budget = max(0, int(round(pred_keyframe_mass * max(0.0, float(budget_scale)))))
    ratio = min(max(0.0, float(density_cut_percent)) / 100.0, 1.0)
    if density_cut_percent_unit == "cut_keyframes":
        # Same as export_final_cut_key_timings.density_event_budgets: the percent is the share of
        # keyframes reserved for cut pairs, and each cut pair uses two keyframes (f-1, f).
        cut_budget = max(0, int(round(total_budget * ratio))) // 2
    else:
        cut_budget = max(0, int(round(total_budget * ratio)))
    series = _series_from_values(cut_probabilities, fps=fps)
    candidates = cut_candidate_peaks(
        series,
        min_score=float(min_score),
        prominence_weight=float(prominence_weight),
    )
    duration_sec = len(cut_probabilities) / float(fps)
    if mode == "global":
        selected = select_global_cuts(candidates, total_cuts=cut_budget, min_gap_sec=min_gap_sec)
    elif mode == "uniform":
        selected = select_uniform_cuts(
            candidates,
            intervals=[(0.0, duration_sec)],
            total_cuts=cut_budget,
            min_gap_sec=min_gap_sec,
        )
    else:
        selected = select_adaptive_cuts(
            candidates,
            duration_sec=duration_sec,
            total_cuts=cut_budget,
            window_sec=window_sec,
            min_gap_sec=min_gap_sec,
        )
    selected_payload = [
        {
            **event_payload(event, index=index, fps=fps, event_type="cut"),
            "probability": round(float(event.get("probability", 0.0)), 6),
            "score": round(float(event.get("score", 0.0)), 6),
            "cut_pair_frames": cut_pair_frames(int(event["frame"])),
        }
        for index, event in enumerate(selected, start=1)
    ]
    budget = {
        "pred_keyframe_mass": round(pred_keyframe_mass, 6),
        "density_budget_scale": float(budget_scale),
        "total_keyframe_budget": int(total_budget),
        "density_cut_percent": float(density_cut_percent),
        "cut_budget_requested": int(cut_budget),
        "cut_count_selected": len(selected_payload),
        "cut_candidate_count": len(candidates),
    }
    return selected_payload, budget


def _predict_stage2_keyframes(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    *,
    audio: np.ndarray,
    dance: np.ndarray,
    cut_frames: list[int],
    fps: float,
    max_frames: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    audio_mean = np.asarray(checkpoint["audio_mean"], dtype=np.float32)
    audio_std = np.maximum(np.asarray(checkpoint["audio_std"], dtype=np.float32), 1e-6)
    dance_mean = np.asarray(checkpoint["dance_mean"], dtype=np.float32)
    dance_std = np.maximum(np.asarray(checkpoint["dance_std"], dtype=np.float32), 1e-6)
    cond_mean = np.asarray(checkpoint["cond_mean"], dtype=np.float32)
    cond_std = np.maximum(np.asarray(checkpoint["cond_std"], dtype=np.float32), 1e-6)

    num_frames = int(len(audio))
    dummy_keyframes: list[int] = sorted({frame for cut in cut_frames for frame in cut_pair_frames(cut)})
    intervals = _build_prediction_intervals(
        num_frames,
        np.asarray(cut_frames, dtype=np.int32),
        np.asarray(dummy_keyframes, dtype=np.int32),
    )
    probabilities = np.zeros(num_frames, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for interval in intervals:
            interval_start = int(interval["interval_start_frame"])
            interval_end = int(interval["interval_end_frame"])
            if interval_start > interval_end:
                continue
            chunks = []
            current = interval_start
            while current <= interval_end:
                chunk_end = min(interval_end, current + int(max_frames) - 1)
                chunks.append((current, chunk_end))
                current = chunk_end + 1

            for chunk_start, chunk_end in chunks:
                length = int(chunk_end - chunk_start + 1)
                condition = _build_condition_features(
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    interval_start=interval_start,
                    interval_end=interval_end,
                    has_left_cut=interval.get("left_cut_frame") is not None,
                    has_right_cut=interval.get("right_cut_frame") is not None,
                )
                batch_audio = ((audio[chunk_start : chunk_end + 1] - audio_mean[None, :]) / audio_std[None, :]).astype(
                    np.float32
                )
                batch_dance = ((dance[chunk_start : chunk_end + 1] - dance_mean[None, :]) / dance_std[None, :]).astype(
                    np.float32
                )
                batch_condition = ((condition - cond_mean[None, :]) / cond_std[None, :]).astype(np.float32)

                padded_audio = np.zeros((1, int(max_frames), batch_audio.shape[1]), dtype=np.float32)
                padded_dance = np.zeros((1, int(max_frames), batch_dance.shape[1]), dtype=np.float32)
                padded_condition = np.zeros((1, int(max_frames), batch_condition.shape[1]), dtype=np.float32)
                valid = np.zeros((1, int(max_frames)), dtype=np.float32)
                padded_audio[0, :length] = batch_audio
                padded_dance[0, :length] = batch_dance
                padded_condition[0, :length] = batch_condition
                valid[0, :length] = 1.0
                outputs = model(
                    torch.from_numpy(padded_audio).to(device),
                    torch.from_numpy(padded_dance).to(device),
                    torch.from_numpy(padded_condition).to(device),
                    valid_mask=torch.from_numpy(valid).to(device),
                )
                chunk_probs = torch.sigmoid(outputs["keyframe_logits"])[0, :length].detach().cpu().numpy()
                probabilities[chunk_start : chunk_end + 1] = chunk_probs.astype(np.float32)
                rows.append(
                    {
                        "chunk_start_frame": int(chunk_start),
                        "chunk_end_frame": int(chunk_end),
                        "interval_start_frame": int(interval_start),
                        "interval_end_frame": int(interval_end),
                        "interval_position": str(interval.get("interval_position")),
                    }
                )
    return probabilities, rows


def _select_keyframes(
    *,
    keyframe_probabilities: np.ndarray,
    density_by_frame: np.ndarray,
    selected_cuts: list[dict[str, Any]],
    fps: float,
    total_keyframe_budget: int,
    cut_exclusion_sec: float,
    keyframe_min_gap_frames: int,
    keyframe_min_score: float,
    prominence_weight: float,
    bin_sec: float,
    alpha: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    cut_pair_set = {
        int(frame)
        for event in selected_cuts
        for frame in event.get("cut_pair_frames", cut_pair_frames(int(event["frame"])))
    }
    exclusion_radius = int(round(float(cut_exclusion_sec) * float(fps)))
    blocked_frames = frames_near_events(selected_cuts, radius=exclusion_radius)
    blocked_frames.update(cut_pair_set)
    noncut_budget = max(0, int(total_keyframe_budget) - len(cut_pair_set))

    series = _series_from_values(keyframe_probabilities, fps=fps)
    candidates = keyframe_candidate_peaks(
        series,
        min_score=float(keyframe_min_score),
        min_gap_frames=int(keyframe_min_gap_frames),
        prominence_weight=float(prominence_weight),
        reserved_frames=blocked_frames,
    )
    density_dict = {int(frame): float(value) for frame, value in enumerate(density_by_frame)}
    selected, bin_summary = select_density_uniform_keyframes(
        candidates,
        total_keyframes=noncut_budget,
        min_gap_frames=int(keyframe_min_gap_frames),
        density_by_frame=density_dict,
        blocked_frames=blocked_frames,
        fps=fps,
        bin_sec=bin_sec,
        alpha=alpha,
    )
    payload = [
        {
            **event_payload(event, index=index, fps=fps, event_type="keyframe"),
            "probability": round(float(event.get("probability", 0.0)), 6),
            "score": round(float(event.get("score", 0.0)), 6),
            "density_value": round(float(event.get("density_value", density_dict.get(int(event["frame"]), 0.0))), 6),
        }
        for index, event in enumerate(selected, start=1)
    ]
    budget = {
        "total_keyframe_budget": int(total_keyframe_budget),
        "reserved_cut_keyframe_count": len(cut_pair_set),
        "noncut_keyframe_budget": int(noncut_budget),
        "keyframe_candidate_count": len(candidates),
        "keyframe_count_selected": len(payload),
        "cut_exclusion_sec": float(cut_exclusion_sec),
        "keyframe_min_gap_frames": int(keyframe_min_gap_frames),
    }
    return payload, budget, bin_summary


def _write_shotplan_feature_bundle(
    *,
    output_dir: Path,
    song_id: str,
    full_audio: np.ndarray,
    full_audio_names: list[str],
    full_dance: np.ndarray,
    full_dance_names: list[str],
    frame_times_sec: np.ndarray,
) -> Path:
    """Write a one-song feature bundle in the format expected by V3 shot-plan inference.

    The final V3.1 model was trained from DCM feature files with
    `audio_features` as 29-D full audio and `dance_features` as the source
    427-D motion descriptor.  Its inference loader then compacts dance to
    168-D using the nearby manifest, so unseen features are exported in the
    same shape rather than feeding the already compacted dance directly.
    """

    feature_dir = output_dir / "shotplan_features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_path = feature_dir / f"{song_id}.npz"
    np.savez_compressed(
        feature_path,
        audio_features=np.asarray(full_audio, dtype=np.float32),
        dance_features=np.asarray(full_dance, dtype=np.float32),
        frame_times_sec=np.asarray(frame_times_sec, dtype=np.float32),
    )
    _write_json(
        feature_dir / "manifest.json",
        {
            "dataset_type": "unseen_mmd_shotplan_feature_bundle",
            "song_id": song_id,
            "audio_feature_names": list(full_audio_names),
            "dance_feature_names": list(full_dance_names),
            "audio_feature_dim": int(full_audio.shape[1]),
            "dance_feature_dim": int(full_dance.shape[1]),
            "frame_count": int(full_audio.shape[0]),
        },
    )
    return feature_path


def _shotplan_inputs_from_timing_payload(payload: dict[str, Any]) -> tuple[list[int], list[dict[str, object]], list[dict[str, object]]]:
    """Convert final timing JSON into V3 shot-plan input frames and cut pairs."""

    songs = payload.get("songs") or []
    if not songs:
        return [], [], []
    song = songs[0]
    frame_count = int(song.get("frame_count", 0) or 0)
    max_frame = max(0, frame_count - 1)
    frames: set[int] = set()
    timing_debug: list[dict[str, object]] = []
    cut_pairs: list[dict[str, object]] = []

    for event in song.get("selected_cuts") or []:
        if not isinstance(event, dict):
            continue
        raw_pair = event.get("cut_pair_frames") or cut_pair_frames(int(event.get("frame", 0)))
        pair = [max(0, min(max_frame, int(frame))) for frame in raw_pair[:2]]
        pair = sorted(set(pair))
        if len(pair) < 2:
            continue
        for frame in pair:
            frames.add(int(frame))
            timing_debug.append(
                {
                    "type": "cut_pair",
                    "timing_frame": int(event.get("frame", pair[-1])),
                    "source_frame": int(frame),
                    "cut_index": int(event.get("index", len(cut_pairs) + 1)),
                }
            )
        cut_pairs.append(
            {
                "timing_frame": int(event.get("frame", pair[-1])),
                "source_frames": pair[:2],
                "duplicate_count": 2,
                "cut_index": int(event.get("index", len(cut_pairs) + 1)),
            }
        )

    for event in song.get("selected_keyframes") or []:
        if not isinstance(event, dict) or "frame" not in event:
            continue
        frame = max(0, min(max_frame, int(event["frame"])))
        frames.add(frame)
        timing_debug.append(
            {
                "type": "keyframe",
                "timing_frame": int(event["frame"]),
                "source_frame": int(frame),
                "keyframe_index": int(event.get("index", 0) or 0),
            }
        )

    if max_frame >= 0:
        frames.add(0)
        frames.add(max_frame)
    return sorted(frames), sorted(timing_debug, key=lambda row: int(row["source_frame"])), cut_pairs


def _apply_interpolation_stage_to_plan(
    *,
    plan_payload: dict[str, Any],
    interpolation_run_dir: Path,
    checkpoint_name: str,
    device: torch.device,
) -> dict[str, Any]:
    """Predict only the `interpolation` field with the separate final post-stage."""

    checkpoint_path = interpolation_run_dir / checkpoint_name
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    fields = [str(field) for field in checkpoint["fields"]]
    vocabularies = {
        str(field): [str(value) for value in values]
        for field, values in checkpoint["vocabularies"].items()
    }
    value_to_id = {
        field: {value: idx for idx, value in enumerate(values)}
        for field, values in vocabularies.items()
    }
    interpolation_vocab = [str(value) for value in checkpoint["interpolation_vocab"]]
    model = build_interpolation_stage_model(checkpoint).to(device)
    predicted = _predict_interpolations(
        [row for row in plan_payload.get("keyframes", []) if isinstance(row, dict)],
        model=model,
        fields=fields,
        value_to_id=value_to_id,
        interpolation_vocab=interpolation_vocab,
        device=device,
    )
    output = dict(plan_payload)
    output["interpolation_stage_model"] = str(checkpoint_path)
    output["inference_note"] = str(output.get("inference_note", "")) + " Interpolation is predicted by a separate post-stage model."
    output["keyframes"] = predicted
    return output


def _optimized_path_for_candidate(candidate_path: Path) -> Path:
    stem = re.sub(r"_top\d+_candidates$", "_optimized_cp_sat", candidate_path.stem)
    if stem == candidate_path.stem:
        stem = f"{candidate_path.stem}_optimized_cp_sat"
    return candidate_path.with_name(f"{stem}.json")


def _debug_path_for_optimized_plan(optimized_path: Path) -> Path:
    return optimized_path.with_name(f"{optimized_path.stem}_optimizer_debug.json")


def _append_cli_arg(command: list[str], flag: str, value: Any) -> None:
    command.extend([flag, str(value)])


def _shotplan_height_calibration_config(args: argparse.Namespace) -> CameraSolveConfigV3:
    """Build the VMD solve config used by the unseen demo pipeline.

    The model still predicts the same V3.1 discrete labels; this only nudges the
    look-at anchor at VMD export time for close face/chest shots.  Defaults are
    from the small m0g1_d70 DCA eval experiment that improved LCD without
    meaningfully changing FID_k.
    """

    if not bool(getattr(args, "shotplan_height_calibration", True)):
        return CameraSolveConfigV3()

    face_offset = float(getattr(args, "shotplan_height_calibration_face_offset", DEFAULT_SHOTPLAN_HEIGHT_CALIBRATION_OFFSETS[0][2]))
    chest_offset = float(getattr(args, "shotplan_height_calibration_chest_offset", DEFAULT_SHOTPLAN_HEIGHT_CALIBRATION_OFFSETS[2][2]))
    return CameraSolveConfigV3(
        anchor_y_calibration_offsets=(
            ("face", "extreme_close", face_offset),
            ("face", "close", face_offset),
            ("chest", "extreme_close", chest_offset),
            ("chest", "close", chest_offset),
        )
    )


def _shotplan_height_calibration_summary(args: argparse.Namespace) -> dict[str, Any]:
    enabled = bool(getattr(args, "shotplan_height_calibration", True))
    return {
        "enabled": enabled,
        "exported_rules": [
            {
                "target_anchor": "face",
                "distance": "very_close",
                "y_offset_fraction": float(
                    getattr(args, "shotplan_height_calibration_face_offset", DEFAULT_SHOTPLAN_HEIGHT_CALIBRATION_OFFSETS[0][2])
                ),
            },
            {
                "target_anchor": "face",
                "distance": "close",
                "y_offset_fraction": float(
                    getattr(args, "shotplan_height_calibration_face_offset", DEFAULT_SHOTPLAN_HEIGHT_CALIBRATION_OFFSETS[0][2])
                ),
            },
            {
                "target_anchor": "chest",
                "distance": "very_close",
                "y_offset_fraction": float(
                    getattr(args, "shotplan_height_calibration_chest_offset", DEFAULT_SHOTPLAN_HEIGHT_CALIBRATION_OFFSETS[2][2])
                ),
            },
            {
                "target_anchor": "chest",
                "distance": "close",
                "y_offset_fraction": float(
                    getattr(args, "shotplan_height_calibration_chest_offset", DEFAULT_SHOTPLAN_HEIGHT_CALIBRATION_OFFSETS[2][2])
                ),
            },
        ]
        if enabled
        else [],
        "offset_unit": "full_body_height_fraction",
        "note": "Applied only when exporting the unseen demo shot plan to VMD.",
    }


def _run_optimization_stage(
    *,
    output_dir: Path,
    plan_path: Path,
    checkpoint_path: Path,
    model: torch.nn.Module,
    manifest: dict[str, Any],
    norm_stats: dict[str, np.ndarray],
    shotplan_feature_path: Path,
    targets_path: Path | None,
    interpolation_run_dir: Path,
    interpolation_checkpoint_name: str,
    skip_interpolation: bool,
    device: torch.device,
    logit_temperature: float,
    fps: float,
    optimization_args: argparse.Namespace,
) -> dict[str, Any]:
    top_k = max(1, int(optimization_args.optimization_top_k))
    candidate_path = plan_path.with_name(f"{plan_path.stem}_top{top_k}_candidates.json")
    topk_summary = _export_plan_candidates(
        plan_path=plan_path,
        output_path=candidate_path,
        checkpoint_path=checkpoint_path,
        model=model,
        manifest=manifest,
        norm_stats=norm_stats,
        feature_root=shotplan_feature_path.parent,
        device=device,
        top_k=top_k,
        logit_temperature=float(logit_temperature),
        calibration_biases={},
    )

    manifest_name = "cp_sat_optimization_manifest.json"
    command = [
        sys.executable,
        "-m",
        "shotplan_transformer.optimize_topk_candidate_files",
        "--source-root",
        str(output_dir),
        "--candidate-glob",
        candidate_path.name,
        "--schema",
        "v3_1",
        "--top-k",
        str(top_k),
        "--max-time-sec",
        str(float(optimization_args.optimization_max_time_sec)),
        "--local-cost-scale",
        str(int(optimization_args.optimization_local_cost_scale)),
        "--allow-missing-gt",
        "--manifest-name",
        manifest_name,
    ]
    if bool(optimization_args.optimization_disable_sequence_stat_objective):
        command.append("--disable-sequence-stat-objective")

    for flag, attr in (
        ("--distance-bounce-weight", "optimization_distance_bounce_weight"),
        ("--distance-bounce-min-rank-delta", "optimization_distance_bounce_min_rank_delta"),
        ("--distance-bounce-max-frame-gap", "optimization_distance_bounce_max_frame_gap"),
        ("--composition-bounce-weight", "optimization_composition_bounce_weight"),
        ("--composition-bounce-min-rank-delta", "optimization_composition_bounce_min_rank_delta"),
        ("--composition-bounce-max-frame-gap", "optimization_composition_bounce_max_frame_gap"),
        ("--top-pitch-bounce-weight", "optimization_top_pitch_bounce_weight"),
        ("--top-pitch-bounce-max-frame-gap", "optimization_top_pitch_bounce_max_frame_gap"),
        ("--cut-identity-weight", "optimization_cut_identity_weight"),
        ("--cut-contrast-weight", "optimization_cut_contrast_weight"),
        ("--cut-contrast-min-score", "optimization_cut_contrast_min_score"),
        ("--cut-contrast-medium-rank-delta", "optimization_cut_contrast_medium_rank_delta"),
        ("--cut-contrast-strong-rank-delta", "optimization_cut_contrast_strong_rank_delta"),
        ("--cut-context-speed-weight", "optimization_cut_context_speed_weight"),
        ("--cut-context-speed-min-speed", "optimization_cut_context_speed_min_speed"),
        ("--cut-context-speed-max-frame-gap", "optimization_cut_context_speed_max_frame_gap"),
        ("--repeated-distance-bounce-weight", "optimization_repeated_distance_bounce_weight"),
        ("--repeated-distance-bounce-min-rank-delta", "optimization_repeated_distance_bounce_min_rank_delta"),
        ("--repeated-distance-bounce-max-frame-gap", "optimization_repeated_distance_bounce_max_frame_gap"),
        ("--distance-level-duration-weight", "optimization_distance_level_duration_weight"),
        ("--distance-level-duration-include-levels", "optimization_distance_level_duration_include_levels"),
        ("--distance-level-duration-upper-key", "optimization_distance_level_duration_upper_key"),
        (
            "--distance-level-duration-penalty-sec-per-unit",
            "optimization_distance_level_duration_penalty_sec_per_unit",
        ),
    ):
        _append_cli_arg(command, flag, getattr(optimization_args, attr))

    _append_cli_arg(command, "--cut-context-speed-fps", fps)
    _append_cli_arg(command, "--distance-level-duration-fps", fps)
    command.extend(["--distance-level-duration-end-mode", "midpoint"])

    stats_path = Path(optimization_args.optimization_distance_level_duration_stats).resolve()
    if int(optimization_args.optimization_distance_level_duration_weight) > 0 and stats_path.exists():
        command.extend(["--distance-level-duration-stats", str(stats_path)])
    elif int(optimization_args.optimization_distance_level_duration_weight) > 0:
        print(f"[optimization] distance-level duration stats missing, disabling that objective: {stats_path}")
        command[command.index("--distance-level-duration-weight") + 1] = "0"

    if bool(optimization_args.optimization_disallow_new_top_pitch_candidates):
        command.append("--disallow-new-top-pitch-candidates")

    print("[optimization] Exported top-k candidates:", candidate_path)
    print("[optimization] Running:", " ".join(f'"{part}"' if " " in part else part for part in command))
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=utf8_subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.returncode != 0:
        raise RuntimeError(f"CP-SAT optimization failed with code {completed.returncode}")

    cp_sat_plan_path = _optimized_path_for_candidate(candidate_path)
    cp_sat_debug_path = _debug_path_for_optimized_plan(cp_sat_plan_path)
    if not cp_sat_plan_path.exists():
        return {
            "optimization_status": "no-solution",
            "topk_candidates_json": str(candidate_path),
            "topk_summary": topk_summary,
            "optimization_cp_sat_json": str(cp_sat_plan_path),
            "optimization_debug": str(cp_sat_debug_path) if cp_sat_debug_path.exists() else None,
            "optimization_manifest": str(output_dir / manifest_name),
        }

    final_optimized_plan_path = cp_sat_plan_path
    optimized_interpolation_path: Path | None = None
    if not skip_interpolation:
        optimized_payload = json.loads(cp_sat_plan_path.read_text(encoding="utf-8"))
        optimized_interpolation_payload = _apply_interpolation_stage_to_plan(
            plan_payload=optimized_payload,
            interpolation_run_dir=interpolation_run_dir,
            checkpoint_name=interpolation_checkpoint_name,
            device=device,
        )
        optimized_interpolation_path = cp_sat_plan_path.with_name(f"{cp_sat_plan_path.stem}_interpstage.json")
        _write_json(optimized_interpolation_path, optimized_interpolation_payload)
        final_optimized_plan_path = optimized_interpolation_path

    optimized_vmd_path: Path | None = None
    optimized_vmd_debug_path: Path | None = None
    if targets_path is not None and targets_path.exists():
        optimized_vmd_path = final_optimized_plan_path.with_suffix(".vmd")
        optimized_vmd_debug_path = final_optimized_plan_path.with_name(f"{final_optimized_plan_path.stem}_debug.json")
        _write_vmd_from_plan(
            final_optimized_plan_path,
            targets_path,
            optimized_vmd_path,
            optimized_vmd_debug_path,
            solve_config=_shotplan_height_calibration_config(optimization_args),
            stabilize_forward=bool(optimization_args.actor_forward_stabilization),
        )

    return {
        "optimization_status": "ok" if optimized_vmd_path is not None else "ok-json-only",
        "topk_candidates_json": str(candidate_path),
        "topk_summary": topk_summary,
        "optimization_json": str(final_optimized_plan_path),
        "optimization_cp_sat_json": str(cp_sat_plan_path),
        "optimization_debug": str(cp_sat_debug_path) if cp_sat_debug_path.exists() else None,
        "optimization_interpolation_json": str(optimized_interpolation_path) if optimized_interpolation_path else None,
        "optimization_vmd": str(optimized_vmd_path) if optimized_vmd_path is not None else None,
        "optimization_vmd_debug": str(optimized_vmd_debug_path) if optimized_vmd_debug_path is not None else None,
        "optimization_manifest": str(output_dir / manifest_name),
        "optimization_command": command,
        "height_calibration": _shotplan_height_calibration_summary(optimization_args),
        "actor_forward_stabilization": bool(optimization_args.actor_forward_stabilization),
    }


def _run_shotplan_stage(
    *,
    output_dir: Path,
    song_id: str,
    final_timing_payload: dict[str, Any],
    shotplan_feature_path: Path,
    targets_path: Path | None,
    shotplan_run_dir: Path,
    shotplan_checkpoint_name: str,
    interpolation_run_dir: Path,
    interpolation_checkpoint_name: str,
    skip_interpolation: bool,
    device: torch.device,
    logit_temperature: float,
    decode_distance_fov_joint: bool,
    skip_optimization: bool,
    optimization_args: argparse.Namespace,
    fps: float,
) -> dict[str, Any]:
    """Run final V3.1 shot-plan inference and, when possible, export camera VMD."""

    emit_stage_event("shot_plan", "running", "Shot-plan prediction")
    checkpoint_path = shotplan_run_dir / shotplan_checkpoint_name
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    manifest = _checkpoint_manifest(checkpoint)
    model = _build_cut_context_model(checkpoint).to(device)
    norm_stats = load_normalization_stats(shotplan_run_dir / "normalization_stats.npz")
    frames, timing_debug, cut_pairs = _shotplan_inputs_from_timing_payload(final_timing_payload)
    songs = final_timing_payload.get("songs") or [{}]
    frame_count = int(songs[0].get("frame_count", 0) or 0)
    source_start = 0
    source_end = max(0, frame_count - 1)

    keyframes = predict_v3_cut_context_for_clip(
        model=model,
        manifest=manifest,
        norm_stats=norm_stats,
        feature_path=shotplan_feature_path,
        full_frames=frames,
        cut_pairs=cut_pairs,
        source_start_frame=source_start,
        source_end_frame=source_end,
        device=device,
        logit_temperature=float(logit_temperature),
        calibration_biases={},
        decode_distance_fov_joint=bool(decode_distance_fov_joint),
    )
    distance_schema = _close_dense_distance_schema_for_manifest(manifest)
    plan_payload: dict[str, Any] = {
        "schema": (
            "camera_autoshot.reverse_v3.close_dense_no_wide"
            if distance_schema is not None
            else "camera_autoshot.reverse_v3"
        ),
        "source_model": str(checkpoint_path),
        "clip_id": song_id,
        "source_song_id": song_id,
        "source_start_frame": source_start,
        "source_end_frame": source_end,
        "sample_fps": float(fps),
        "timing_source": str(output_dir / "final_cut_key_timings.json"),
        "feature_path": str(shotplan_feature_path),
        "decode_distance_fov_joint": bool(decode_distance_fov_joint),
        "inference_note": (
            "Unseen MMD two-stage ReverseV3.1 shot-plan inference: "
            "cut-transition frames first, then ordinary keyframes."
        ),
        "keyframes": keyframes,
    }
    if distance_schema is not None:
        plan_payload["distance_schema"] = distance_schema

    plan_path = output_dir / "camera_poses_v3_cutctx.json"
    timing_debug_path = output_dir / "camera_poses_v3_cutctx_timing_debug.json"
    vmd_path = output_dir / "camera_poses_v3_cutctx.vmd"
    debug_path = output_dir / "camera_poses_v3_cutctx_debug.json"
    _write_json(plan_path, plan_payload)
    _write_json(timing_debug_path, {"converted_timing": timing_debug, "cut_pairs": cut_pairs})
    emit_stage_event("shot_plan", "completed", "Shot-plan prediction")

    emit_stage_event("camera_trajectory", "running", "Camera trajectory/VMD")
    vmd_written = False
    solve_config = _shotplan_height_calibration_config(optimization_args)
    if targets_path is not None and targets_path.exists():
        _write_vmd_from_plan(
            plan_path,
            targets_path,
            vmd_path,
            debug_path,
            solve_config=solve_config,
            stabilize_forward=bool(optimization_args.actor_forward_stabilization),
        )
        vmd_written = True

    final_plan_path = plan_path
    final_vmd_path = vmd_path if vmd_written else None
    interpolation_plan_path: Path | None = None
    interpolation_vmd_path: Path | None = None
    interpolation_debug_path: Path | None = None
    if not skip_interpolation:
        interp_payload = _apply_interpolation_stage_to_plan(
            plan_payload=plan_payload,
            interpolation_run_dir=interpolation_run_dir,
            checkpoint_name=interpolation_checkpoint_name,
            device=device,
        )
        interpolation_plan_path = output_dir / "camera_poses_v3_cutctx_interpstage.json"
        interpolation_vmd_path = output_dir / "camera_poses_v3_cutctx_interpstage.vmd"
        interpolation_debug_path = output_dir / "camera_poses_v3_cutctx_interpstage_debug.json"
        _write_json(interpolation_plan_path, interp_payload)
        if targets_path is not None and targets_path.exists():
            _write_vmd_from_plan(
                interpolation_plan_path,
                targets_path,
                interpolation_vmd_path,
                interpolation_debug_path,
                solve_config=solve_config,
                stabilize_forward=bool(optimization_args.actor_forward_stabilization),
            )
            final_plan_path = interpolation_plan_path
            final_vmd_path = interpolation_vmd_path

    optimization_summary: dict[str, Any] | None = None
    if not skip_optimization:
        emit_stage_event("optimization", "running", "Shot-plan optimization")
        optimization_summary = _run_optimization_stage(
            output_dir=output_dir,
            plan_path=plan_path,
            checkpoint_path=checkpoint_path,
            model=model,
            manifest=manifest,
            norm_stats=norm_stats,
            shotplan_feature_path=shotplan_feature_path,
            targets_path=targets_path,
            interpolation_run_dir=interpolation_run_dir,
            interpolation_checkpoint_name=interpolation_checkpoint_name,
            skip_interpolation=skip_interpolation,
            device=device,
            logit_temperature=float(logit_temperature),
            fps=float(fps),
            optimization_args=optimization_args,
        )
        emit_stage_event("optimization", "completed", "Shot-plan optimization")
        if str(optimization_summary.get("optimization_status", "")).startswith("ok"):
            optimized_plan = Path(str(optimization_summary.get("optimization_json")))
            optimized_vmd = optimization_summary.get("optimization_vmd")
            final_plan_path = optimized_plan
            final_vmd_path = Path(str(optimized_vmd)) if optimized_vmd else None
    else:
        optimization_summary = {"optimization_status": "skipped"}
        emit_stage_event("optimization", "skipped", "Shot-plan optimization")

    emit_stage_event(
        "camera_trajectory",
        "completed" if final_vmd_path is not None else "missing",
        "Camera trajectory/VMD",
    )

    return {
        "shotplan_model": str(checkpoint_path),
        "shotplan_json": str(plan_path),
        "shotplan_timing_debug": str(timing_debug_path),
        "shotplan_vmd": str(vmd_path) if vmd_written else None,
        "shotplan_debug": str(debug_path) if vmd_written else None,
        "interpolation_model": str(interpolation_run_dir / interpolation_checkpoint_name) if not skip_interpolation else None,
        "interpolation_json": str(interpolation_plan_path) if interpolation_plan_path else None,
        "interpolation_vmd": str(interpolation_vmd_path) if interpolation_vmd_path and interpolation_vmd_path.exists() else None,
        "interpolation_debug": str(interpolation_debug_path) if interpolation_debug_path and interpolation_debug_path.exists() else None,
        **(optimization_summary or {}),
        "height_calibration": _shotplan_height_calibration_summary(optimization_args),
        "actor_forward_stabilization": bool(optimization_args.actor_forward_stabilization),
        "final_shotplan_json": str(final_plan_path),
        "final_camera_vmd": str(final_vmd_path) if final_vmd_path is not None else None,
        "target_probe_json": str(targets_path) if targets_path is not None else None,
        "keyframe_count": len(keyframes),
        "cut_pair_count": len(cut_pairs),
        "status": "ok" if final_vmd_path is not None else "ok-json-only",
        "warning": None if final_vmd_path is not None else "Target probe JSON was unavailable, so camera VMD was not exported.",
    }


def _load_saba_batch_renderer() -> Any:
    """Load the existing SABA batch renderer without importing from cwd."""

    module_path = REPO_ROOT / "saba_mmd" / "build" / "Release" / "batch_render_gt_range.py"
    if not module_path.exists():
        raise FileNotFoundError(f"SABA batch renderer not found: {module_path}")
    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location("batch_render_gt_range", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load SABA batch renderer: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ascii_render_input(path: Path, worker_copy: Path) -> Path:
    """Return a path saba_viewer can hand to ffmpeg.

    saba_viewer runs ffmpeg through the Windows ANSI code page, so audio (and, to be safe, motion)
    files under non-ASCII paths are copied into the ASCII worker folder first.
    """

    if str(path).isascii():
        return path
    shutil.copy2(path, worker_copy)
    return worker_copy


def _render_preview_video(
    *,
    output_dir: Path,
    song_id: str,
    model_path: Path | None,
    motion_vmd_path: Path | None,
    camera_vmd_path: Path | None,
    wav_path: Path,
    duration_sec: float,
    fps: int,
    width: int,
    height: int,
    viewer_session_size: int,
) -> dict[str, Any]:
    """Render the final camera VMD to MP4 using the shared SABA worker lease."""

    if model_path is None or motion_vmd_path is None or camera_vmd_path is None:
        return {
            "status": "skipped",
            "warning": "Preview rendering needs model, motion VMD, and camera VMD paths.",
        }
    if not model_path.exists():
        return {"status": "failed", "error": f"Model file not found: {model_path}"}
    if not motion_vmd_path.exists():
        return {"status": "failed", "error": f"Motion VMD not found: {motion_vmd_path}"}
    if not camera_vmd_path.exists():
        return {"status": "failed", "error": f"Camera VMD not found: {camera_vmd_path}"}
    if not wav_path.exists():
        return {"status": "failed", "error": f"WAV file not found: {wav_path}"}

    try:
        batch_render = _load_saba_batch_renderer()
        preview_path = output_dir / f"{camera_vmd_path.stem}.mp4"
        with batch_render.acquire_saba_worker(
            worker_id="auto",
            lease_info={"task": "unseen_mmd_preview_render", "song_id": song_id, "output_dir": str(output_dir)},
        ) as lease:
            worker_output = lease.worker_dir / f"{song_id}_{preview_path.name}"
            render_motion_path = _ascii_render_input(motion_vmd_path, lease.worker_dir / "render_motion.vmd")
            render_wav_path = _ascii_render_input(wav_path, lease.worker_dir / f"render_audio{wav_path.suffix.lower()}")
            row = {
                "status": "planned",
                "model": str(model_path),
                "motion": str(render_motion_path),
                "camera": str(camera_vmd_path),
                "wav": str(render_wav_path),
                "output": str(worker_output),
                "export_duration_sec": f"{max(0.1, float(duration_sec)):.6f}",
            }
            try:
                batch_render.render_planned_rows(
                    [row],
                    model_path,
                    viewer_session_size=max(1, int(viewer_session_size)),
                    fps=int(fps),
                    width=int(width),
                    height=int(height),
                    work_dir=lease.worker_dir,
                    worker_lease=lease,
                )
            finally:
                for copied, source in ((render_motion_path, motion_vmd_path), (render_wav_path, wav_path)):
                    if copied != source:
                        copied.unlink(missing_ok=True)
            if row.get("status") != "ok" or not worker_output.exists():
                return {
                    "status": "failed",
                    "error": "SABA renderer did not produce a non-empty MP4.",
                    "worker_output": str(worker_output),
                }
            shutil.move(str(worker_output), str(preview_path))
        return {
            "status": "ok",
            "preview_video": str(preview_path),
            "width": int(width),
            "height": int(height),
            "fps": int(fps),
            "duration_sec": round(float(duration_sec), 6),
        }
    except Exception as exc:  # Render preview should be surfaced without hiding the generated VMD.
        return {"status": "failed", "error": repr(exc)}


def main() -> None:
    configure_utf8_stdio()
    pipeline_started_at = time.perf_counter()
    stage_timings: list[dict[str, Any]] = []
    args = parse_args()
    with timed_stage(stage_timings, "input_setup", "Input discovery and frame setup"):
        input_dir = Path(args.input_dir).resolve()
        wav_path = Path(args.wav).resolve() if args.wav else _find_single_file(input_dir, ".wav")
        vmd_path = Path(args.vmd).resolve() if args.vmd else _find_single_file(input_dir, ".vmd")
        render_model_path = Path(args.model_path).resolve() if args.model_path else None
        if args.render_model_path is not None:
            render_model_path = Path(args.render_model_path).resolve()
        song_id = _slug(args.song_id or input_dir.name or wav_path.stem)
        output_dir = Path(args.output_root).resolve() / song_id
        intermediate_dir = output_dir / "intermediate"
        output_dir.mkdir(parents=True, exist_ok=True)
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        audio_frames, audio_duration_sec, sample_rate = _audio_frame_count(wav_path, args.fps)
        motion_json_path: Path | None = Path(args.motion_json).resolve() if args.motion_json else None
        target_probe_path = intermediate_dir / f"{song_id}_target_probe.json"
        run_inputs = {
            "wav": wav_path,
            "vmd": vmd_path,
            "model": Path(args.model_path).resolve() if args.model_path else None,
            "fps": float(args.fps),
            "motion_end_frame": audio_frames - 1,
        }
        reused_preprocessing: dict[str, str] = {}
        if args.reuse_from is not None and not args.force:
            reused_preprocessing = _reuse_earlier_preprocessing(
                Path(args.reuse_from).resolve(),
                song_id=song_id,
                intermediate_dir=intermediate_dir,
                inputs=run_inputs,
                demucs_settings=None
                if args.skip_demucs
                else {"model": str(args.demucs_model), "shifts": int(args.demucs_shifts), "segment": args.demucs_segment},
            )
            print(f"Reused from {args.reuse_from}: {', '.join(reused_preprocessing) or 'nothing'}", flush=True)
        _write_json(intermediate_dir / "inputs.json", run_inputs)

    motion_conversion_summary: dict[str, Any] | None = None
    if motion_json_path is None:
        with timed_stage(stage_timings, "motion_conversion", "VMD to global motion conversion"):
            if args.model_path is None:
                raise SystemExit(
                    "Raw .vmd -> dance feature conversion needs a matching PMX/PMD model. "
                    "Pass --model-path <model.pmx> or pass --motion-json <*_GlobalTransform.json>."
                )
            tool_path = _resolve_target_dump_tool(args.target_dump_tool)
            motion_json_path = intermediate_dir / f"{song_id}_GlobalTransform.json"
            motion_conversion_summary = _run_mmd_global_transform(
                tool_path=tool_path,
                model_path=Path(args.model_path).resolve(),
                vmd_path=vmd_path,
                output_path=motion_json_path,
                probe_output_path=target_probe_path,
                end_frame=audio_frames - 1,
                force=bool(args.force),
            )
    else:
        emit_stage_event("motion_conversion", "skipped", "VMD to global motion conversion")
        stage_timings.append(
            {
                "id": "motion_conversion",
                "label": "VMD to global motion conversion",
                "status": "skipped",
                "duration_sec": 0.0,
            }
        )

    with timed_stage(stage_timings, "frame_alignment", "Audio/motion frame alignment"):
        num_frames = _choose_frame_count(
            audio_frames=audio_frames,
            motion_json_path=motion_json_path,
            frame_policy=str(args.frame_policy),
            fps=float(args.fps),
            vmd_path=vmd_path,
        )

    with timed_stage(stage_timings, "demucs", "Demucs stem separation"):
        demucs_summary = _run_demucs(
            audio_path=wav_path,
            song_id=song_id,
            output_dir=intermediate_dir / "demucs",
            model=str(args.demucs_model),
            python_executable=Path(args.demucs_python).resolve() if args.demucs_python else None,
            device=_resolve_demucs_device(str(args.demucs_device)),
            shifts=int(args.demucs_shifts),
            jobs=int(args.demucs_jobs),
            segment=args.demucs_segment,
            skip=bool(args.skip_demucs),
            force=bool(args.force),
        )
    with timed_stage(stage_timings, "beat_tracking", "Beat/downbeat/bar extraction"):
        madmom_summary = _load_or_compute_madmom(
            wav_path,
            intermediate_dir / "madmom_downbeats.json",
            fps=float(args.fps),
            num_frames=int(num_frames),
            force=bool(args.force),
        )

    with timed_stage(stage_timings, "audio_features", "Audio feature extraction"):
        full_audio, full_audio_names, cut_audio, cut_audio_names = _build_audio_features(
            audio_path=wav_path,
            demucs_summary=demucs_summary,
            madmom_summary=madmom_summary,
            num_frames=int(num_frames),
            fps=float(args.fps),
            distance_clip_frames=int(args.density_distance_clip_frames),
            allow_stem_fallback=bool(args.allow_stem_fallback),
        )
    with timed_stage(stage_timings, "dance_features", "Dance feature extraction"):
        full_dance, full_dance_names, compact_dance, compact_dance_names, dance_metadata = _build_dance_features(
            motion_json_path=motion_json_path,
            num_frames=int(num_frames),
        )

    with timed_stage(stage_timings, "feature_bundle", "Feature bundle write"):
        features_path = output_dir / "features.npz"
        frame_times_sec = (np.arange(int(num_frames), dtype=np.float32) / float(args.fps)).astype(np.float32)
        cut_model_features = np.concatenate([cut_audio, compact_dance], axis=1).astype(np.float32)
        np.savez_compressed(
            features_path,
            full_audio_features=full_audio,
            full_audio_feature_names=np.asarray(full_audio_names, dtype=object),
            cut_audio_features=cut_audio,
            cut_audio_feature_names=np.asarray(cut_audio_names, dtype=object),
            full_dance_features=full_dance,
            full_dance_feature_names=np.asarray(full_dance_names, dtype=object),
            compact_dance_features=compact_dance,
            compact_dance_feature_names=np.asarray(compact_dance_names, dtype=object),
            cut_model_features=cut_model_features,
            cut_model_feature_names=np.asarray(cut_audio_names + compact_dance_names, dtype=object),
            frame_times_sec=frame_times_sec,
        )

    with timed_stage(stage_timings, "model_load", "Timing model load"):
        device = _device_from_arg(str(args.device))
        cut_model, cut_checkpoint = _load_cut_model(Path(args.cut_model), device)
        stagea_model, stagea_checkpoint = _load_stagea_model(Path(args.density_model), device)
        stage2_model, stage2_checkpoint = _load_stage2_model(Path(args.keyframe_model), device)

    with timed_stage(stage_timings, "cut_inference", "Cut probability inference"):
        cut_probabilities = _predict_song(
            cut_model,
            features=cut_model_features,
            feature_mean=np.asarray(cut_checkpoint["feature_mean"], dtype=np.float32),
            feature_std=np.maximum(np.asarray(cut_checkpoint["feature_std"], dtype=np.float32), 1e-6),
            window_frames=int((cut_checkpoint.get("args") or {}).get("window_frames", 240)),
            stride=int((cut_checkpoint.get("args") or {}).get("stride", 60)),
            device=device,
        )

    with timed_stage(stage_timings, "density_pre_cut", "Stage-A density before cut context"):
        density_pre, density_windows_pre = _predict_stagea_density(
            stagea_model,
            stagea_checkpoint,
            audio=cut_audio,
            dance=compact_dance,
            cut_frames=[],
            fps=float(args.fps),
            window_frames=int(args.density_window_frames),
            stride_frames=int(args.density_stride_frames),
        )
    with timed_stage(stage_timings, "cut_selection", "Cut timing selection"):
        selected_cuts, cut_budget = _select_cuts(
            cut_probabilities=cut_probabilities,
            density_by_frame=density_pre,
            fps=float(args.fps),
            mode=str(args.cut_selection_mode),
            density_cut_percent=float(args.density_cut_percent),
            min_score=float(args.cut_min_score),
            prominence_weight=float(args.cut_prominence_weight),
            min_gap_sec=float(args.cut_min_gap_sec),
            window_sec=float(args.cut_window_sec),
            density_cut_percent_unit=str(args.density_cut_percent_unit),
            budget_scale=float(args.density_budget_scale),
        )

    selected_cut_frames = [int(event["frame"]) for event in selected_cuts]
    with timed_stage(stage_timings, "density_after_cut", "Stage-A density after cut context"):
        density_final, density_windows_final = _predict_stagea_density(
            stagea_model,
            stagea_checkpoint,
            audio=cut_audio,
            dance=compact_dance,
            cut_frames=selected_cut_frames,
            fps=float(args.fps),
            window_frames=int(args.density_window_frames),
            stride_frames=int(args.density_stride_frames),
        )
        # Recompute the budget after adding previous-cut context. Keep the already selected cut count stable.
        final_total_budget = max(0, int(round(float(np.sum(density_final)) * max(0.0, float(args.density_budget_scale)))))
        cut_budget["pred_keyframe_mass_after_cut_context"] = round(float(np.sum(density_final)), 6)
        cut_budget["total_keyframe_budget_after_cut_context"] = int(final_total_budget)

    with timed_stage(stage_timings, "stage2_inference", "Stage-B keyframe probability inference"):
        stage2_audio, stage2_audio_feature_set = _select_stage2_audio_features(
            stage2_checkpoint,
            full_audio=full_audio,
            cut_audio=cut_audio,
        )
        stage2_dance, stage2_dance_feature_set = _select_stage2_dance_features(
            stage2_checkpoint,
            full_dance=full_dance,
            compact_dance=compact_dance,
        )
        keyframe_probabilities, stage2_chunks = _predict_stage2_keyframes(
            stage2_model,
            stage2_checkpoint,
            audio=stage2_audio,
            dance=stage2_dance,
            cut_frames=selected_cut_frames,
            fps=float(args.fps),
            max_frames=int(args.stage2_max_frames),
        )
    with timed_stage(stage_timings, "keyframe_selection", "Non-cut keyframe selection"):
        selected_keyframes, key_budget, density_bin_summary = _select_keyframes(
            keyframe_probabilities=keyframe_probabilities,
            density_by_frame=density_final,
            selected_cuts=selected_cuts,
            fps=float(args.fps),
            total_keyframe_budget=final_total_budget,
            cut_exclusion_sec=float(args.keyframe_cut_exclusion_sec),
            keyframe_min_gap_frames=int(args.keyframe_min_gap_frames),
            keyframe_min_score=float(args.keyframe_min_score),
            prominence_weight=float(args.keyframe_prominence_weight),
            bin_sec=float(args.density_allocation_bin_sec),
            alpha=float(args.density_uniform_alpha),
        )

    final_payload = {
        "dataset_type": "unseen_mmd_cut_key_timings",
        "created_by": Path(__file__).name,
        "input": {
            "input_dir": str(input_dir),
            "wav_path": str(wav_path),
            "vmd_path": str(vmd_path),
            "motion_json_path": str(motion_json_path),
            "model_path": str(args.model_path) if args.model_path else None,
            "fps": float(args.fps),
            "sample_rate": int(sample_rate),
            "audio_duration_sec": round(float(audio_duration_sec), 6),
            "num_frames": int(num_frames),
            "frame_policy": str(args.frame_policy),
        },
        "models": {
            "cut_model": str(args.cut_model),
            "density_model": str(args.density_model),
            "keyframe_model": str(args.keyframe_model),
            "keyframe_model_audio_feature_set": stage2_audio_feature_set,
            "keyframe_model_dance_feature_set": stage2_dance_feature_set,
        },
        "selection": {
            "cut_selection_mode": str(args.cut_selection_mode),
            "density_cut_percent": float(args.density_cut_percent),
            "density_budget_scale": float(args.density_budget_scale),
            "density_cut_percent_unit": str(args.density_cut_percent_unit),
            "cut_min_gap_sec": float(args.cut_min_gap_sec),
            "cut_min_score": float(args.cut_min_score),
            "keyframe_cut_exclusion_sec": float(args.keyframe_cut_exclusion_sec),
            "keyframe_min_gap_frames": int(args.keyframe_min_gap_frames),
            "density_allocation_bin_sec": float(args.density_allocation_bin_sec),
            "density_uniform_alpha": float(args.density_uniform_alpha),
        },
        "songs": [
            {
                "id": song_id,
                "split": "unseen",
                "video_duration_sec": round(float(num_frames) / float(args.fps), 6),
                "fps": float(args.fps),
                "frame_count": int(num_frames),
                "feature_path": str(features_path),
                "coverage_segments": [
                    {
                        "start_frame": 0,
                        "end_frame": int(num_frames) - 1,
                        "start_time_sec": 0.0,
                        "end_time_sec": round((int(num_frames) - 1) / float(args.fps), 3),
                    }
                ],
                "selected_cuts": selected_cuts,
                "selected_keyframes": selected_keyframes,
                "counts": {
                    "selected_cut_count": len(selected_cuts),
                    "selected_cut_keyframe_pair_count": len(
                        {frame for cut in selected_cuts for frame in cut.get("cut_pair_frames", [])}
                    ),
                    "selected_noncut_keyframe_count": len(selected_keyframes),
                    "selected_total_keyframe_count": len(selected_keyframes)
                    + len({frame for cut in selected_cuts for frame in cut.get("cut_pair_frames", [])}),
                },
                "density_budget": {
                    **cut_budget,
                    **key_budget,
                },
                "series_paths": {
                    "cut_probability_npy": str(output_dir / "cut_probabilities.npy"),
                    "stagea_density_npy": str(output_dir / "stagea_density.npy"),
                    "stage2_keyframe_probability_npy": str(output_dir / "stage2_keyframe_probabilities.npy"),
                },
            }
        ],
    }

    shotplan_summary: dict[str, Any] | None = None
    shotplan_feature_path: Path | None = None
    if not bool(args.skip_shotplan):
        with timed_stage(stage_timings, "shotplan_features", "Shot-plan feature bundle write"):
            shotplan_feature_path = _write_shotplan_feature_bundle(
                output_dir=output_dir,
                song_id=song_id,
                full_audio=full_audio,
                full_audio_names=full_audio_names,
                full_dance=full_dance,
                full_dance_names=full_dance_names,
                frame_times_sec=frame_times_sec,
            )
        with timed_stage(
            stage_timings,
            "shotplan_camera",
            "Shot-plan, optimization, and camera VMD",
            emit_events=False,
        ):
            shotplan_summary = _run_shotplan_stage(
                output_dir=output_dir,
                song_id=song_id,
                final_timing_payload=final_payload,
                shotplan_feature_path=shotplan_feature_path,
                targets_path=target_probe_path if target_probe_path.exists() else None,
                shotplan_run_dir=Path(args.shotplan_run_dir).resolve(),
                shotplan_checkpoint_name=str(args.shotplan_checkpoint_name),
                interpolation_run_dir=Path(args.interpolation_run_dir).resolve(),
                interpolation_checkpoint_name=str(args.interpolation_checkpoint_name),
                skip_interpolation=bool(args.skip_interpolation),
                device=device,
                logit_temperature=float(args.shotplan_logit_temperature),
                decode_distance_fov_joint=bool(args.decode_distance_fov_joint),
                skip_optimization=bool(args.skip_optimization),
                optimization_args=args,
                fps=float(args.fps),
            )
        final_payload["models"]["shotplan_model"] = shotplan_summary["shotplan_model"]
        final_payload["models"]["interpolation_model"] = shotplan_summary.get("interpolation_model")
        final_payload["models"]["shotplan_optimizer"] = "full_guard_weak_cut_cp_sat"
        final_payload["songs"][0]["shotplan_paths"] = {
            "feature_path": str(shotplan_feature_path),
            "shotplan_json": shotplan_summary.get("shotplan_json"),
            "shotplan_vmd": shotplan_summary.get("shotplan_vmd"),
            "interpolation_json": shotplan_summary.get("interpolation_json"),
            "interpolation_vmd": shotplan_summary.get("interpolation_vmd"),
            "topk_candidates_json": shotplan_summary.get("topk_candidates_json"),
            "optimization_json": shotplan_summary.get("optimization_json"),
            "optimization_cp_sat_json": shotplan_summary.get("optimization_cp_sat_json"),
            "optimization_debug": shotplan_summary.get("optimization_debug"),
            "optimization_vmd": shotplan_summary.get("optimization_vmd"),
            "final_shotplan_json": shotplan_summary.get("final_shotplan_json"),
            "final_camera_vmd": shotplan_summary.get("final_camera_vmd"),
        }
        if not bool(args.skip_render_preview):
            with timed_stage(stage_timings, "preview_render", "Preview render"):
                render_summary = _render_preview_video(
                    output_dir=output_dir,
                    song_id=song_id,
                    model_path=render_model_path,
                    motion_vmd_path=vmd_path,
                    camera_vmd_path=(
                        Path(str(shotplan_summary.get("final_camera_vmd")))
                        if shotplan_summary.get("final_camera_vmd")
                        else None
                    ),
                    wav_path=wav_path,
                    duration_sec=float(num_frames) / float(args.fps),
                    fps=int(args.render_preview_fps),
                    width=int(args.render_preview_width),
                    height=int(args.render_preview_height),
                    viewer_session_size=int(args.render_preview_session_size),
                )
            shotplan_summary["render_preview"] = render_summary
            if render_summary.get("preview_video"):
                shotplan_summary["preview_video"] = render_summary.get("preview_video")
                final_payload["songs"][0]["shotplan_paths"]["preview_video"] = render_summary.get("preview_video")
        else:
            emit_stage_event("preview_render", "skipped", "Preview render")
            stage_timings.append(
                {
                    "id": "preview_render",
                    "label": "Preview render",
                    "status": "skipped",
                    "duration_sec": 0.0,
                }
            )
            shotplan_summary["render_preview"] = {"status": "skipped"}
    else:
        emit_stage_event("shot_plan", "skipped", "Shot-plan prediction")
        emit_stage_event("optimization", "skipped", "Shot-plan optimization")
        emit_stage_event("camera_trajectory", "skipped", "Camera trajectory/VMD")
        emit_stage_event("preview_render", "skipped", "Preview render")
        for stage_id, label in (
            ("shotplan_features", "Shot-plan feature bundle write"),
            ("shotplan_camera", "Shot-plan, optimization, and camera VMD"),
            ("preview_render", "Preview render"),
        ):
            stage_timings.append(
                {
                    "id": stage_id,
                    "label": label,
                    "status": "skipped",
                    "duration_sec": 0.0,
                }
            )

    output_write_started_at = time.perf_counter()
    np.save(output_dir / "cut_probabilities.npy", cut_probabilities.astype(np.float32))
    np.save(output_dir / "stagea_density.npy", density_final.astype(np.float32))
    np.save(output_dir / "stage2_keyframe_probabilities.npy", keyframe_probabilities.astype(np.float32))
    _write_json(
        output_dir / "preprocess_summary.json",
        {
            "song_id": song_id,
            "motion_conversion": motion_conversion_summary,
            "demucs": demucs_summary,
            "madmom": madmom_summary,
            "dance_metadata": dance_metadata,
            "feature_path": str(features_path),
            "reused_preprocessing": reused_preprocessing,
        },
    )
    _write_json(
        output_dir / "inference_debug.json",
        {
            "density_windows_before_cut_context": density_windows_pre,
            "density_windows_after_cut_context": density_windows_final,
            "stage2_chunks": stage2_chunks,
            "density_bin_summary": density_bin_summary,
            "coverage_intervals": coverage_intervals(final_payload["songs"][0]),
        },
    )
    _write_json(output_dir / "final_cut_key_timings.json", final_payload)
    output_write_duration = round(float(time.perf_counter() - output_write_started_at), 4)
    emit_stage_event("output_write", "completed", "Output artifact write", duration_sec=output_write_duration)
    stage_timings.append(
        {
            "id": "output_write",
            "label": "Output artifact write",
            "status": "ok",
            "duration_sec": output_write_duration,
        }
    )
    run_timing = timing_payload(stage_timings, total_started_at=pipeline_started_at)
    final_payload["timing_breakdown"] = run_timing
    _write_json(output_dir / "final_cut_key_timings.json", final_payload)
    _write_json(
        output_dir / "summary.json",
        {
            "song_id": song_id,
            "output_dir": str(output_dir),
            "features": str(features_path),
            "final_cut_key_timings": str(output_dir / "final_cut_key_timings.json"),
            "shotplan_features": str(shotplan_feature_path) if shotplan_feature_path is not None else None,
            "shotplan": shotplan_summary,
            "shot_plan_json": shotplan_summary.get("final_shotplan_json") if shotplan_summary else None,
            "camera_vmd": shotplan_summary.get("final_camera_vmd") if shotplan_summary else None,
            "preview_video": shotplan_summary.get("preview_video") if shotplan_summary else None,
            "selected_cut_count": len(selected_cuts),
            "selected_noncut_keyframe_count": len(selected_keyframes),
            "selected_total_keyframe_count": final_payload["songs"][0]["counts"]["selected_total_keyframe_count"],
            "num_frames": int(num_frames),
            "duration_sec": round(float(num_frames) / float(args.fps), 6),
            "render_model_path": str(render_model_path) if render_model_path is not None else None,
            "timing_breakdown": run_timing,
        },
    )

    print(f"Song id: {song_id}")
    print(f"Frames: {num_frames} @ {args.fps:g} fps")
    print(f"Selected cuts: {len(selected_cuts)}")
    print(f"Selected non-cut keyframes: {len(selected_keyframes)}")
    print(f"Final timing JSON: {output_dir / 'final_cut_key_timings.json'}")
    if shotplan_summary is not None:
        print(f"Shot-plan JSON: {shotplan_summary.get('final_shotplan_json')}")
        print(f"Camera VMD: {shotplan_summary.get('final_camera_vmd')}")
        if shotplan_summary.get("preview_video"):
            print(f"Preview MP4: {shotplan_summary.get('preview_video')}")
    print(f"Pipeline total time: {run_timing['total_sec']:.3f}s")


if __name__ == "__main__":
    main()
