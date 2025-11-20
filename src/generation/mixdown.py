"""
High-level mixdown utilities that orchestrate combining stems into a master track.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import soundfile as sf

from src.mixing import mix_stems


def create_final_mix(
    accompaniment: Iterable[Path],
    vocals: Path,
    out_path: Path,
) -> Path:
    """
    Combine accompaniment stems with vocals into a single track.
    """
    stems: List[Path] = [Path(stem) for stem in accompaniment]
    stems.append(Path(vocals))
    return mix_stems(stems, Path(out_path))


def mix_backing_and_vocal(
    backing_path: Path,
    vocal_path: Path,
    out_path: Path,
    vocal_gain_db: float = -3.0,
) -> Path:
    """
    Mix a backing track and vocal track into a new WAV file.
    """
    backing, sr = sf.read(backing_path)
    vocal, sr_v = sf.read(vocal_path)

    def _to_stereo(data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            return np.stack((data, data), axis=1)
        if data.shape[1] == 1:
            return np.repeat(data, 2, axis=1)
        if data.shape[1] > 2:
            return data[:, :2]
        return data

    backing = _to_stereo(backing)
    vocal = _to_stereo(vocal)

    if sr_v != sr:
        target_len = int(round(len(vocal) * sr / sr_v))
        if target_len <= 0:
            target_len = len(backing)
        old_idx = np.linspace(0, len(vocal) - 1, num=len(vocal))
        new_idx = np.linspace(0, len(vocal) - 1, num=target_len)
        resampled = []
        for ch in range(vocal.shape[1]):
            resampled.append(np.interp(new_idx, old_idx, vocal[:, ch]))
        vocal = np.stack(resampled, axis=1).astype(np.float32)

    if len(backing) != len(vocal):
        max_length = max(len(backing), len(vocal))
        pad_width_back = ((0, max_length - len(backing)), (0, 0))
        pad_width_voc = ((0, max_length - len(vocal)), (0, 0))
        backing = np.pad(backing, pad_width_back, mode="constant")
        vocal = np.pad(vocal, pad_width_voc, mode="constant")

    gain = 10 ** (vocal_gain_db / 20.0)
    mix = backing + vocal * gain

    peak = np.max(np.abs(mix))
    if peak > 1.0:
        mix = mix / peak * 0.99

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path.as_posix(), mix.astype(np.float32), sr)
    return out_path
