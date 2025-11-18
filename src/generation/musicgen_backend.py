"""
Wrapper utilities around AudioCraft's MusicGen model for prompt-based audio generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import soundfile as sf
import torch
from audiocraft.models import MusicGen


class MusicGenBackend:
    """
    Thin wrapper around AudioCraft's MusicGen model.
    """

    def __init__(self, model_name: str = "facebook/musicgen-small", device: str | None = None):
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = resolved_device
        self.model = MusicGen.get_pretrained(model_name, device=resolved_device)
        self.sample_rate = getattr(self.model, "sample_rate", 32000)

    def generate_clips(self, prompts: Sequence[str], duration: int = 20) -> List[np.ndarray]:
        """
        Generate audio clips for each prompt.
        """
        if not prompts:
            raise ValueError("prompts must contain at least one prompt")

        self.model.set_generation_params(duration=duration)
        with torch.no_grad():
            generated = self.model.generate(prompts)

        clips: List[np.ndarray] = []
        for clip in generated:
            tensor = clip
            if hasattr(tensor, "detach"):
                tensor = tensor.detach()
            if hasattr(tensor, "cpu"):
                tensor = tensor.cpu()
            if hasattr(tensor, "numpy"):
                wave = tensor.numpy()
            else:
                wave = np.asarray(tensor, dtype=np.float32)

            wave = np.asarray(wave, dtype=np.float32)
            if wave.ndim > 1:
                wave = np.squeeze(wave, axis=0)
            clips.append(wave)

        return clips

    @staticmethod
    def save_wav(wave: np.ndarray, path: str | Path, sample_rate: int) -> None:
        """
        Save a waveform to disk as a WAV file.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path.as_posix(), np.asarray(wave, dtype=np.float32), sample_rate)
