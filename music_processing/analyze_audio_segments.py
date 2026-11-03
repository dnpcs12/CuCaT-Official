from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from music_processing.audio_peaks import (
    DEFAULT_DATA_ROOT,
    DEFAULT_RESULTS_ROOT,
    AudioAnalysisResult,
    compute_audio_analysis,
    resolve_audio_path,
    write_peak_csv,
    write_timed_event_csv,
)


DEFAULT_OUTPUT_ROOT = DEFAULT_RESULTS_ROOT / "audio_segment_probe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe song structure from amcN/aN.wav and save plots and peak summaries."
    )
    parser.add_argument("--index", type=int, default=0, help="Song index N for amcN/aN.wav")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root folder that contains amcN subfolders",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Folder where plots and summaries will be written",
    )
    parser.add_argument("--segments", type=int, default=8, help="Coarse section count estimate")
    parser.add_argument(
        "--peak-quantile",
        type=float,
        default=0.9,
        help="Quantile threshold used for highlight peak selection",
    )
    parser.add_argument(
        "--peak-min-gap-sec",
        type=float,
        default=1.25,
        help="Minimum spacing between highlight peaks",
    )
    parser.add_argument(
        "--highlight-target-count",
        type=int,
        default=12,
        help="Target count of highlight peaks to keep for overlays",
    )
    parser.add_argument(
        "--beats-per-bar",
        type=int,
        default=4,
        help="Fixed meter assumption used to group detected beats into bars",
    )
    return parser.parse_args()


def _plot_novelty(
    result: AudioAnalysisResult,
    output_path: Path,
    *,
    title: str,
) -> None:
    summary = result.summary
    novelty = result.novelty
    novelty_times = result.novelty_times
    candidate_times = np.array([peak["time_sec"] for peak in summary["candidate_peaks"]], dtype=float)
    candidate_values = np.array([peak["novelty"] for peak in summary["candidate_peaks"]], dtype=float)
    highlight_times = np.array([peak["time_sec"] for peak in summary["highlight_peaks"]], dtype=float)
    highlight_values = np.array([peak["novelty"] for peak in summary["highlight_peaks"]], dtype=float)
    bar_times = np.array([bar["time_sec"] for bar in summary["bars"]], dtype=float)
    structure_sections = summary.get("structure_sections", [])

    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.plot(novelty_times, novelty, color="#7FB3D5", linewidth=1.2, label="novelty")
    if len(candidate_times):
        ax.scatter(
            candidate_times,
            candidate_values,
            s=12,
            color="#F5B041",
            alpha=0.5,
            label="candidate peaks",
        )
    if len(highlight_times):
        ax.scatter(
            highlight_times,
            highlight_values,
            s=38,
            color="#E74C3C",
            label="highlight peaks",
            zorder=3,
        )
    if len(bar_times):
        ax.vlines(
            bar_times,
            ymin=0,
            ymax=np.interp(bar_times, novelty_times, novelty),
            color="#5DADE2",
            alpha=0.28,
            linewidth=1.0,
            label=f"bars ({summary['meter_assumption']['beats_per_bar']} beats approx)",
        )
    for idx, section in enumerate(structure_sections):
        start_time = float(section["start_time_sec"])
        if idx > 0:
            ax.axvline(start_time, color="#C0392B", alpha=0.8, linewidth=1.5)
        center_time = (float(section["start_time_sec"]) + float(section["end_time_sec"])) * 0.5
        ax.text(
            center_time,
            novelty.max() * 1.02,
            section["label"],
            color="#922B21",
            fontsize=9,
            ha="center",
            va="bottom",
            rotation=0,
        )
    for boundary_time in summary["estimated_segment_boundaries_sec"]:
        ax.axvline(boundary_time, color="#48C9B0", alpha=0.45, linewidth=1.2, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Onset Strength")
    ax.set_ylim(top=novelty.max() * 1.12 if len(novelty) else 1.0)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_similarity(result: AudioAnalysisResult, output_path: Path, *, title: str) -> None:
    summary = result.summary
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    img = librosa.display.specshow(
        result.similarity,
        x_axis="time",
        y_axis="time",
        sr=summary["sample_rate"],
        hop_length=512,
        cmap="magma",
        ax=ax,
    )
    boundary_times = summary["estimated_segment_boundaries_sec"]
    if boundary_times:
        ax.vlines(boundary_times, ymin=0, ymax=summary["duration_sec"], color="#48C9B0", alpha=0.35, linewidth=1.0)
        ax.hlines(boundary_times, xmin=0, xmax=summary["duration_sec"], color="#48C9B0", alpha=0.35, linewidth=1.0)
    structure_times = summary.get("structure_change_times_sec", [])
    if structure_times:
        ax.vlines(structure_times, ymin=0, ymax=summary["duration_sec"], color="#C0392B", alpha=0.55, linewidth=1.3)
        ax.hlines(structure_times, xmin=0, xmax=summary["duration_sec"], color="#C0392B", alpha=0.55, linewidth=1.3)
    ax.set_title(title)
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def run_analysis(
    *,
    index: int,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    segments: int = 8,
    peak_quantile: float = 0.9,
    peak_min_gap_sec: float = 1.25,
    highlight_target_count: int = 12,
    beats_per_bar: int = 4,
) -> dict:
    wav_path = resolve_audio_path(index=index, data_root=data_root)
    output_dir = output_root / f"c{index}"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = compute_audio_analysis(
        wav_path=wav_path,
        segments=segments,
        peak_quantile=peak_quantile,
        peak_min_gap_sec=peak_min_gap_sec,
        highlight_target_count=highlight_target_count,
        beats_per_bar=beats_per_bar,
    )

    novelty_plot_path = output_dir / "novelty_plot.png"
    similarity_plot_path = output_dir / "similarity_matrix.png"
    candidate_csv_path = output_dir / "candidate_peaks.csv"
    highlight_csv_path = output_dir / "highlight_peaks.csv"
    beat_csv_path = output_dir / "beats.csv"
    bar_csv_path = output_dir / "bars.csv"
    structure_csv_path = output_dir / "structure_sections.csv"
    summary_path = output_dir / "summary.json"

    _plot_novelty(
        result,
        novelty_plot_path,
        title=f"Novelty and highlight peaks: c{index}",
    )
    _plot_similarity(
        result,
        similarity_plot_path,
        title=f"Recurrence similarity: c{index}",
    )

    write_peak_csv(candidate_csv_path, result.summary["candidate_peaks"])
    write_peak_csv(highlight_csv_path, result.summary["highlight_peaks"])
    write_timed_event_csv(beat_csv_path, result.summary["beats"])
    write_timed_event_csv(bar_csv_path, result.summary["bars"])
    write_timed_event_csv(structure_csv_path, result.summary["structure_sections"])

    summary = dict(result.summary)
    summary["index"] = index
    summary["outputs"] = {
        "novelty_plot": str(novelty_plot_path),
        "similarity_matrix": str(similarity_plot_path),
        "candidate_csv": str(candidate_csv_path),
        "highlight_csv": str(highlight_csv_path),
        "beat_csv": str(beat_csv_path),
        "bar_csv": str(bar_csv_path),
        "structure_csv": str(structure_csv_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = run_analysis(
        index=args.index,
        data_root=args.data_root,
        output_root=args.output_root,
        segments=args.segments,
        peak_quantile=args.peak_quantile,
        peak_min_gap_sec=args.peak_min_gap_sec,
        highlight_target_count=args.highlight_target_count,
        beats_per_bar=args.beats_per_bar,
    )

    print(f"WAV: {summary['wav_path']}")
    print(f"Duration: {summary['duration_sec']:.2f}s @ {summary['sample_rate']} Hz")
    print(f"Tempo: {summary['tempo_bpm']:.2f} BPM")
    print(f"Candidate peaks: {summary['candidate_peak_count']}")
    print(f"Bars ({summary['meter_assumption']['beats_per_bar']} beats approx): {summary['bar_count']}")
    print("Bar times (sec):")
    print(", ".join(str(bar["time_sec"]) for bar in summary["bars"][:20]))
    print("Structure sections:")
    print(
        " | ".join(
            f"{section['label']}[{section['start_time_sec']}-{section['end_time_sec']}]"
            for section in summary["structure_sections"]
        )
    )
    print("Highlight peaks (sec):")
    print(", ".join(str(peak["time_sec"]) for peak in summary["highlight_peaks"]))
    print(f"Saved summary: {Path(summary['outputs']['highlight_csv']).with_name('summary.json')}")


if __name__ == "__main__":
    main()
