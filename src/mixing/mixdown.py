"""
Simple mixing placeholders for combining accompaniment and vocal stems.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import soundfile as sf


def _load_wave(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float32), sr


def mix_stems(stems: Iterable[Path], out_path: Path) -> Path:
    """
    Mix multiple mono stems into a single mono WAV.
    """
    stems = list(stems)
    if len(stems) < 2:
        raise ValueError("Provide at least two stems to mix.")

    audio_arrays: List[np.ndarray] = []
    sample_rate: int | None = None
    max_length = 0
    for stem in stems:
        data, sr = _load_wave(Path(stem))
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError("All stems must share the same sample rate.")
        audio_arrays.append(data)
        max_length = max(max_length, len(data))

    stacked = []
    for data in audio_arrays:
        if len(data) < max_length:
            padded = np.pad(data, (0, max_length - len(data)), mode="constant")
        else:
            padded = data
        stacked.append(padded)

    mix = np.sum(stacked, axis=0) / len(stacked)
    peak = np.max(np.abs(mix))
    if peak > 1.0:
        mix /= peak

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path.as_posix(), mix.astype(np.float32), sample_rate or 44100)
    return out_path
