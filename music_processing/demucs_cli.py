"""Run the Demucs command line, writing WAV stems with soundfile.

Demucs 4.0.1 saves stems with ``torchaudio.save``, which needs TorchCodec from
torchaudio 2.9 on. This entry point writes WAV stems with soundfile, the backend
older torchaudio used, so the stems do not depend on the torchaudio version.
It takes the same arguments as ``python -m demucs``.
"""
from __future__ import annotations

from pathlib import Path

import soundfile
import torch
from demucs import separate
from demucs.audio import prevent_clip

_demucs_save_audio = separate.save_audio


def save_audio(
    wav: torch.Tensor,
    path: str | Path,
    samplerate: int,
    bitrate: int = 320,
    clip: str = "rescale",
    bits_per_sample: int = 16,
    as_float: bool = False,
    preset: int = 2,
) -> None:
    if Path(path).suffix.lower() != ".wav":
        _demucs_save_audio(
            wav,
            path,
            samplerate,
            bitrate=bitrate,
            clip=clip,
            bits_per_sample=bits_per_sample,
            as_float=as_float,
            preset=preset,
        )
        return
    wav = prevent_clip(wav, mode=clip)
    subtype = "FLOAT" if as_float else f"PCM_{bits_per_sample}"
    soundfile.write(str(path), wav.t().cpu().numpy(), samplerate, subtype=subtype)


if __name__ == "__main__":
    separate.save_audio = save_audio
    separate.main()
