from pathlib import Path

import json
import numpy as np
import soundfile as sf

from src.singing import SingingSynthesizer, VocalSynthesisRequest


def test_placeholder_synth_writes_audio_and_manifest(tmp_path: Path):
    style_profile = {"tempo_mean": 100.0, "tempo_range": [90.0, 110.0]}
    lyrics = {
        "title": "City Lights",
        "theme": "Night drive",
        "sections": [{"type": "verse", "name": "Verse 1", "lines": ["line 1", "line 2"]}],
    }
    melody_path = tmp_path / "melody.mid"
    melody_path.write_text("placeholder midi data")

    request = VocalSynthesisRequest(
        style_profile=style_profile,
        lyrics=lyrics,
        melody_path=melody_path,
        output_dir=tmp_path / "vocals",
        duration_seconds=1.0,
        sample_rate=8000,
    )

    synth = SingingSynthesizer()
    audio_path = synth.synthesize(request)

    assert audio_path.exists()
    data, sr = sf.read(audio_path)
    assert sr == 8000
    assert np.allclose(data, 0.0)

    manifest_path = audio_path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["lyrics_title"] == "City Lights"
    assert manifest["style_snapshot"]["tempo_mean"] == 100.0
