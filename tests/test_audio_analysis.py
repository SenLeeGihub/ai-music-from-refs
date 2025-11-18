import numpy as np
import pytest
import soundfile as sf

from src.analysis.audio_analysis import aggregate_style_stats, analyze_track


def test_analyze_track_on_sine_wave(tmp_path):
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    waveform = 0.5 * np.sin(2 * np.pi * 440 * t)

    audio_path = tmp_path / "sine.wav"
    sf.write(audio_path, waveform, sr)

    features = analyze_track(audio_path)

    assert set(features.keys()) == {"tempo", "duration", "band_energy"}
    assert features["tempo"] >= 0
    assert features["duration"] == pytest.approx(duration, rel=0.05)

    band_energy = features["band_energy"]
    for band in ("low", "mid", "high"):
        assert band in band_energy
        assert band_energy[band] >= 0


def test_aggregate_style_stats_computes_ranges():
    features = [
        {
            "tempo": 100.0,
            "duration": 1.0,
            "band_energy": {"low": 0.5, "mid": 0.3, "high": 0.1},
        },
        {
            "tempo": 120.0,
            "duration": 1.0,
            "band_energy": {"low": 0.7, "mid": 0.5, "high": 0.2},
        },
    ]

    style = aggregate_style_stats(features)

    assert style["tempo_range"][0] == pytest.approx(100.0)
    assert style["tempo_range"][1] == pytest.approx(120.0)
    assert style["tempo_mean"] == pytest.approx(110.0)
    assert style["energy_profile"]["low"] == pytest.approx(0.6)
    assert style["energy_profile"]["mid"] == pytest.approx(0.4)
    assert style["energy_profile"]["high"] == pytest.approx(0.15)
