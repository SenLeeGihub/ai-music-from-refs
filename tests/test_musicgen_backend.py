import importlib
import sys
import types
from typing import Any, Dict, List

import numpy as np
import pytest


class _DummyTensor:
    def __init__(self, data: np.ndarray):
        self._data = np.asarray(data, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._data


class _DummyMusicGenModel:
    def __init__(self):
        self.sample_rate = 44100
        self.generation_params: Dict[str, Any] = {}
        self.generated_prompts: List[str] = []

    def set_generation_params(self, **kwargs):
        self.generation_params = kwargs

    def generate(self, prompts):
        self.generated_prompts = list(prompts)
        data = np.linspace(0.0, 1.0, num=8, dtype=np.float32)
        return [_DummyTensor(data) for _ in prompts]


def _install_musicgen_stub(monkeypatch):
    state: Dict[str, Any] = {}

    audiocraft_pkg = types.ModuleType("audiocraft")
    models_mod = types.ModuleType("audiocraft.models")

    class _MusicGen:
        @staticmethod
        def get_pretrained(model_name: str, device: str):
            state["requested_model_name"] = model_name
            state["requested_device"] = device
            model = _DummyMusicGenModel()
            state["model"] = model
            return model

    models_mod.MusicGen = _MusicGen
    audiocraft_pkg.models = models_mod

    monkeypatch.setitem(sys.modules, "audiocraft", audiocraft_pkg)
    monkeypatch.setitem(sys.modules, "audiocraft.models", models_mod)

    class _DummyTorch(types.ModuleType):
        class _NoGrad:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        def __init__(self):
            super().__init__("torch")
            self.cuda = types.SimpleNamespace(is_available=lambda: False)

        def no_grad(self):
            return self._NoGrad()

    monkeypatch.setitem(sys.modules, "torch", _DummyTorch())

    return state


def _load_backend(monkeypatch):
    state = _install_musicgen_stub(monkeypatch)
    module = importlib.import_module("src.generation.musicgen_backend")
    importlib.reload(module)
    return module, state


def test_musicgen_backend_initialization(monkeypatch):
    backend_module, state = _load_backend(monkeypatch)

    backend = backend_module.MusicGenBackend(device="cpu")
    assert backend.device == "cpu"
    assert state["requested_model_name"] == "facebook/musicgen-small"
    assert backend.sample_rate == 44100


def test_generate_clips_returns_numpy_arrays(monkeypatch):
    backend_module, state = _load_backend(monkeypatch)
    backend = backend_module.MusicGenBackend(device="cpu")

    prompts = ["Calm piano", "Energetic synthwave"]
    clips = backend.generate_clips(prompts, duration=12)

    assert len(clips) == len(prompts)
    assert state["model"].generation_params["duration"] == 12
    assert state["model"].generated_prompts == prompts
    for clip in clips:
        assert isinstance(clip, np.ndarray)
        assert clip.dtype == np.float32


def test_generate_clips_requires_prompts(monkeypatch):
    backend_module, _ = _load_backend(monkeypatch)
    backend = backend_module.MusicGenBackend(device="cpu")

    with pytest.raises(ValueError):
        backend.generate_clips([])


def test_save_wav_writes_file(monkeypatch, tmp_path):
    backend_module, _ = _load_backend(monkeypatch)
    written = {}

    def fake_write(path, data, sample_rate):
        written["path"] = path
        written["data"] = data
        written["sr"] = sample_rate

    monkeypatch.setattr(backend_module.sf, "write", fake_write)

    wave = np.ones(10, dtype=np.float32)
    output_path = tmp_path / "clips" / "demo.wav"
    backend_module.MusicGenBackend.save_wav(wave, output_path, 16000)

    assert "clips/demo.wav" in written["path"].replace("\\", "/")
    np.testing.assert_allclose(written["data"], wave)
    assert written["sr"] == 16000
