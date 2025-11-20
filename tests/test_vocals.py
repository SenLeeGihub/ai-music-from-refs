from pathlib import Path
import json
import soundfile as sf
import numpy as np

from src.vocals import synthesize_vocals_placeholder


def test_synthesize_vocals_placeholder_writes_outputs(tmp_path: Path):
    melody = tmp_path / "melody.mid"
    melody.write_text("placeholder midi")
    lyrics = {"title": "Test Song", "theme": "Dream"}
    out_path = tmp_path / "vocals" / "demo.wav"

    result = synthesize_vocals_placeholder(melody, lyrics, out_path)

    assert result == out_path
    assert result.exists()
    audio, sr = sf.read(result)
    assert sr == 44100
    assert np.allclose(audio, 0.0)

    manifest = json.loads(result.with_suffix(".json").read_text(encoding="utf-8"))
    assert manifest["lyrics_title"] == "Test Song"
    assert manifest["melody_midi"] == str(melody)
