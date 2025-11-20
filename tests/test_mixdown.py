from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.mixing import mix_stems
from src.generation.mixdown import create_final_mix, mix_backing_and_vocal


def _write_wave(path: Path, value: float, length: int, sr: int = 16000):
    data = np.full(length, value, dtype=np.float32)
    sf.write(path.as_posix(), data, sr)


def test_mix_stems_combines_files(tmp_path: Path):
    stem1 = tmp_path / "a.wav"
    stem2 = tmp_path / "b.wav"
    _write_wave(stem1, 0.5, 1600)
    _write_wave(stem2, -0.5, 800)

    output = tmp_path / "mix.wav"
    mix_stems([stem1, stem2], output)

    assert output.exists()
    data, sr = sf.read(output)
    assert len(data) == 1600
    assert sr == 16000
    assert np.allclose(data[:800], 0.0)
    assert np.allclose(data[800:], 0.25)


def test_mix_stems_requires_multiple_inputs(tmp_path: Path):
    stem1 = tmp_path / "single.wav"
    _write_wave(stem1, 0.2, 100)

    output = tmp_path / "mix.wav"
    with pytest.raises(ValueError):
        mix_stems([stem1], output)


def test_create_final_mix_wraps_mixdown(tmp_path: Path):
    stem1 = tmp_path / "a.wav"
    stem2 = tmp_path / "b.wav"
    _write_wave(stem1, 0.4, 200)
    _write_wave(stem2, 0.4, 200)

    vocals = tmp_path / "vocals.wav"
    _write_wave(vocals, -0.4, 200)

    output = tmp_path / "final.wav"
    create_final_mix([stem1, stem2], vocals, output)

    data, sr = sf.read(output)
    assert sr == 16000
    # 0.4 + 0.4 - 0.4, divided by 3 tracks = ~0.1333
    assert np.allclose(data, 0.133333333, atol=1e-3)


def test_mix_backing_and_vocal_resamples_vocals(tmp_path: Path):
    sr_back = 48000
    sr_vocal = 44100
    length_back = sr_back // 2
    length_vocal = sr_vocal // 2
    backing = tmp_path / "back.wav"
    vocal = tmp_path / "voc.wav"

    t_back = np.linspace(0, 1, length_back, endpoint=False)
    t_voc = np.linspace(0, 1, length_vocal, endpoint=False)
    backing_data = np.sin(2 * np.pi * 100 * t_back)
    sf.write(backing, np.column_stack((backing_data, backing_data)), sr_back)
    sf.write(vocal, np.sin(2 * np.pi * 200 * t_voc), sr_vocal)

    out_path = tmp_path / "mix.wav"
    mix_backing_and_vocal(backing, vocal, out_path, vocal_gain_db=-6.0)

    data, sr_out = sf.read(out_path)
    assert sr_out == sr_back
    assert data.shape == (length_back, 2)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
