from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.vocals import synthesize_vocals


def test_synthesize_vocals_defaults_to_placeholder(tmp_path, monkeypatch):
    melody_path = tmp_path / "melody.mid"
    melody_path.write_text("placeholder midi")
    lyrics = {"title": "Test", "sections": [{"name": "Verse", "lines": ["Line 1", "Line 2"]}]}
    out_path = tmp_path / "out.wav"

    result = synthesize_vocals(
        melody_midi=melody_path,
        lyrics=lyrics,
        out_wav=out_path,
        backend="placeholder",
        duration_seconds=1.5,
    )

    assert result == out_path
    assert out_path.exists()
    data, sr = sf.read(out_path)
    assert len(data) == int(sr * 1.5)
    assert np.allclose(data, 0.0)


def test_synthesize_vocals_ai_backend_falls_back(tmp_path, monkeypatch):
    melody_path = tmp_path / "melody.mid"
    melody_path.write_text("placeholder midi")
    lyrics = {"title": "AI Test"}
    out_path = tmp_path / "ai.wav"

    monkeypatch.setenv("VOCALS_BACKEND", "placeholder")

    result = synthesize_vocals(
        melody_midi=melody_path,
        lyrics=lyrics,
        out_wav=out_path,
        duration_seconds=2.0,
    )

    assert result.exists()
    data, sr = sf.read(result)
    assert len(data) == int(sr * 2.0)
    assert np.allclose(data, 0.0)


def test_synthesize_vocals_diffsinger_backend_falls_back(tmp_path, monkeypatch):
    melody_path = tmp_path / "melody.mid"
    melody_path.write_text("placeholder midi")
    lyrics = {"title": "DiffSinger Test"}
    out_path = tmp_path / "diffsinger.wav"

    monkeypatch.setenv("VOCALS_BACKEND", "diffsinger")
    monkeypatch.setenv("VOCAL_ENGINE_CMD", "python -c \"import sys; sys.exit(1)\"")

    result = synthesize_vocals(
        melody_midi=melody_path,
        lyrics=lyrics,
        out_wav=out_path,
        duration_seconds=1.0,
    )

    assert result.exists()
    data, sr = sf.read(result)
    assert len(data) == int(sr * 1.0)
    assert np.allclose(data, 0.0)
