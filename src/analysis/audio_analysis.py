"""
Audio analysis utilities for extracting simple statistics from reference tracks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import librosa
import numpy as np

# Frequency band definitions (in Hz) for low, mid, and high energy regions.
FREQUENCY_BANDS: Dict[str, tuple[float, float | None]] = {
    "low": (20.0, 250.0),
    "mid": (250.0, 2000.0),
    "high": (2000.0, None),
}


def _mean_magnitude(magnitude: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    return float(np.mean(magnitude[mask]))


def analyze_track(path: str | Path) -> dict:
    """
    Analyze a single reference track for tempo, duration, and band energies.
    """
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    y, sr = librosa.load(audio_path.as_posix(), sr=None, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    stft = librosa.stft(y)
    magnitude = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0] * 2 - 2)

    band_energy: Dict[str, float] = {}
    for band, (low, high) in FREQUENCY_BANDS.items():
        mask = freqs >= low
        if high is not None:
            mask &= freqs < high
        band_energy[band] = _mean_magnitude(magnitude, mask)

    duration = float(len(y) / sr)

    return {
        "tempo": float(tempo),
        "duration": duration,
        "band_energy": band_energy,
    }


def _extract_band_names(feature_dicts: Iterable[dict]) -> List[str]:
    feature_iter = iter(feature_dicts)
    try:
        first = next(feature_iter)
    except StopIteration as exc:
        raise ValueError("No feature dictionaries provided") from exc
    bands = list(first.get("band_energy", {}).keys())
    if not bands:
        raise ValueError("band_energy information is missing from features")
    return bands


def aggregate_style_stats(features: List[dict]) -> dict:
    """
    Aggregate per-track features into a collection-level style profile.
    """
    if not features:
        raise ValueError("features list must not be empty")

    tempos = np.array([float(f["tempo"]) for f in features], dtype=float)
    tempo_range = [float(np.min(tempos)), float(np.max(tempos))]
    tempo_mean = float(np.mean(tempos))

    band_names = _extract_band_names(features)
    energy_profile = {}
    for band in band_names:
        values = [float(f["band_energy"][band]) for f in features]
        energy_profile[band] = float(np.mean(values))

    return {
        "tempo_range": tempo_range,
        "tempo_mean": tempo_mean,
        "energy_profile": energy_profile,
    }
