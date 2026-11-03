from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "local_assets" / "amc_raw_data"
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent / "results"
DEFAULT_OUTPUT_ROOT = DEFAULT_RESULTS_ROOT / "demucs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Separate amcN/aN.wav into Demucs stems and save outputs under music_processing/results."
    )
    parser.add_argument("--index", type=int, required=True, help="Song index N for amcN/aN.wav")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root folder where Demucs outputs will be written.",
    )
    parser.add_argument(
        "--model",
        default="htdemucs",
        help="Demucs pretrained model name. Default is htdemucs.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device passed to Demucs. Default is cpu.",
    )
    parser.add_argument(
        "--shifts",
        type=int,
        default=1,
        help="Demucs random shifts for stabilization. Higher is slower. Default is 1.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Demucs parallel jobs value. Default is 0.",
    )
    parser.add_argument(
        "--segment",
        type=int,
        default=None,
        help="Optional Demucs segment size override.",
    )
    return parser.parse_args()


def _find_stem_files(model_output_dir: Path) -> dict[str, str]:
    stem_paths: dict[str, str] = {}
    for stem_name in ("vocals", "bass", "drums", "other", "guitar", "piano"):
        matches = list(model_output_dir.rglob(f"{stem_name}.wav"))
        if matches:
            stem_paths[stem_name] = str(matches[0])
    return stem_paths


def resolve_audio_path(index: int, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    wav_path = data_root / f"amc{index}" / f"a{index}.wav"
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV not found: {wav_path}")
    return wav_path


def run_demucs(
    *,
    index: int,
    output_root: Path,
    model: str,
    device: str,
    shifts: int,
    jobs: int,
    segment: int | None,
) -> dict:
    audio_path = resolve_audio_path(index=index)
    song_output_dir = output_root / f"c{index}"
    song_output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "demucs",
        "-n",
        model,
        "-d",
        device,
        "--shifts",
        str(shifts),
        "-o",
        str(song_output_dir),
    ]
    if jobs > 0:
        command.extend(["-j", str(jobs)])
    if segment is not None:
        command.extend(["--segment", str(segment)])
    command.append(str(audio_path))

    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    model_output_dir = song_output_dir / model
    stem_paths = _find_stem_files(model_output_dir)
    summary = {
        "index": index,
        "audio_path": str(audio_path),
        "output_root": str(song_output_dir),
        "model": model,
        "device": device,
        "shifts": shifts,
        "jobs": jobs,
        "segment": segment,
        "command": command,
        "stem_files": stem_paths,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    summary_path = song_output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = run_demucs(
        index=args.index,
        output_root=args.output_root,
        model=args.model,
        device=args.device,
        shifts=args.shifts,
        jobs=args.jobs,
        segment=args.segment,
    )
    print(f"Audio: {summary['audio_path']}")
    print(f"Model: {summary['model']}")
    print(f"Output: {summary['output_root']}")
    print("Stem files:")
    for stem_name, stem_path in summary["stem_files"].items():
        print(f"- {stem_name}: {stem_path}")
    print(f"Saved summary: {Path(summary['output_root']) / 'summary.json'}")


if __name__ == "__main__":
    main()
