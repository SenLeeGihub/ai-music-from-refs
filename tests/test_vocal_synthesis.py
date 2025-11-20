from pathlib import Path

import numpy as np
import soundfile as sf
from src.vocals import (
    synthesize_vocals_placeholder,
    synthesize_vocals,
    register_vocal_backend,
)


def test_synthesize_vocals_placeholder_returns_path_and_manifest(tmp_path: Path):
    melody_path = tmp_path / "melody.mid"
    melody_path.write_text("placeholder midi")
    lyrics = {"title": "Mock Song", "theme": "Dream"}
    out_path = tmp_path / "vocals" / "demo.wav"

    result = synthesize_vocals_placeholder(melody_path, lyrics, out_path)

    assert result == out_path
    assert out_path.exists()
    assert np.allclose(sf.read(out_path)[0], 0.0)

    manifest = out_path.with_suffix(".json")
    assert manifest.exists()


def test_synthesize_vocals_placeholder_custom_duration(tmp_path: Path):
    melody_path = tmp_path / "melody.mid"
    melody_path.write_text("placeholder midi")
    lyrics = {"title": "Custom Duration", "theme": "Chill"}
    out_path = tmp_path / "vocals" / "long.wav"

    synthesize_vocals_placeholder(melody_path, lyrics, out_path, duration_seconds=2.5)

    data, sr = sf.read(out_path)
    assert len(data) == int(sr * 2.5)


def test_synthesize_vocals_can_use_registered_backend(tmp_path: Path):
    melody_path = tmp_path / "melody.mid"
    melody_path.write_text("placeholder midi")
    lyrics = {"title": "Custom Backend", "theme": "Energy"}
    out_path = tmp_path / "vocals" / "custom.wav"

    called = {"count": 0}

    def fake_backend(melody_midi, lyrics_dict, wav_path, reference_voice_dir, duration_seconds):
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(wav_path.as_posix(), np.ones(100, dtype=np.float32), 8000)
        called["count"] += 1
        return wav_path

    register_vocal_backend("test-backend", fake_backend)

    result = synthesize_vocals(
        melody_midi=melody_path,
        lyrics=lyrics,
        out_wav=out_path,
        duration_seconds=1.0,
        backend="test-backend",
    )

    assert result.exists()
    assert called["count"] == 1
