"""
Placeholder singing synthesis module that records the expected inputs/outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import soundfile as sf


@dataclass
class VocalSynthesisRequest:
    """Container describing the inputs required for singing synthesis."""

    style_profile: Dict
    lyrics: Dict
    melody_path: Path
    output_dir: Path
    output_name: str = "vocals_placeholder.wav"
    sample_rate: int = 44100
    duration_seconds: float = 5.0
    voice: Optional[str] = None


class SingingSynthesizer:
    """
    Placeholder synthesizer.

    Future implementations can replace the synthesize method with calls to actual TTS/SVS engines.
    """

    def __init__(self, backend_name: str = "placeholder"):
        self.backend_name = backend_name

    def synthesize(self, request: VocalSynthesisRequest) -> Path:
        """
        Generate a placeholder waveform (silence) and a JSON manifest describing the request.
        """
        request.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = request.output_dir / request.output_name
        num_samples = max(1, int(request.sample_rate * request.duration_seconds))
        buffer = np.zeros(num_samples, dtype=np.float32)
        sf.write(output_path.as_posix(), buffer, request.sample_rate)

        manifest = {
            "backend": self.backend_name,
            "voice": request.voice or "unspecified",
            "melody_path": str(Path(request.melody_path)),
            "lyrics_title": request.lyrics.get("title"),
            "lyrics_theme": request.lyrics.get("theme"),
            "style_snapshot": {
                "tempo_mean": request.style_profile.get("tempo_mean"),
                "tempo_range": request.style_profile.get("tempo_range"),
            },
            "output_audio": str(output_path),
            "sample_rate": request.sample_rate,
            "duration_seconds": request.duration_seconds,
        }
        manifest_path = output_path.with_suffix(".json")
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path
