import pytest

from src.style import style_profile
from src.style.style_profile import build_style_profile, describe_style_with_llm


def test_build_style_profile_returns_expected_keys():
    features = [
        {
            "tempo": 100.0,
            "duration": 20.0,
            "band_energy": {"low": 0.5, "mid": 0.4, "high": 0.3},
        },
        {
            "tempo": 120.0,
            "duration": 30.0,
            "band_energy": {"low": 0.7, "mid": 0.6, "high": 0.2},
        },
    ]

    profile = build_style_profile(features)

    assert profile["tempo_range"] == [100.0, 120.0]
    assert profile["tempo_mean"] == pytest.approx(110.0)
    assert profile["energy_profile"]["low"] == pytest.approx(0.6)
    assert set(profile.keys()) == {"tempo_range", "tempo_mean", "energy_profile"}


def test_build_style_profile_requires_non_empty_features():
    with pytest.raises(ValueError):
        build_style_profile([])


class _DummyResponse:
    def __init__(self, text: str):
        self.output_text = text

        class _Content:
            def __init__(self, text: str):
                self.text = text

        class _Output:
            def __init__(self, text: str):
                self.content = [_Content(text)]

        self.output = [_Output(text)]


class _DummyResponsesAPI:
    def __init__(self, response_text: str, state: dict):
        self._response_text = response_text
        self._state = state

    def create(self, **kwargs):
        self._state["create_kwargs"] = kwargs
        return _DummyResponse(self._response_text)


class _DummyClient:
    def __init__(self, response_text: str = "Smooth and upbeat."):
        self._state = {}
        self.responses = _DummyResponsesAPI(response_text, self._state)

    @property
    def call_kwargs(self):
        return self._state.get("create_kwargs", {})


def test_describe_style_uses_openai_client():
    profile = {
        "tempo_range": [90.0, 110.0],
        "tempo_mean": 100.0,
        "energy_profile": {"low": 0.5, "mid": 0.3, "high": 0.2},
    }

    client = _DummyClient("Energetic and warm.")
    result = describe_style_with_llm(profile, client=client, model="test-model")

    assert "Energetic" in result
    kwargs = client.call_kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["input"][0]["role"] == "system"
    assert "Tempo range" in kwargs["input"][1]["content"]


def test_describe_style_without_client_requires_api_key(monkeypatch):
    profile = {
        "tempo_range": [90.0, 110.0],
        "tempo_mean": 100.0,
        "energy_profile": {"low": 0.5, "mid": 0.3, "high": 0.2},
    }

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        style_profile.describe_style_with_llm(profile)


def test_describe_style_without_client_uses_internal_client(monkeypatch):
    profile = {
        "tempo_range": [100.0, 105.0],
        "tempo_mean": 102.0,
        "energy_profile": {"low": 0.4, "mid": 0.3, "high": 0.2},
    }

    client = _DummyClient("Dreamy yet steady.")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(style_profile, "_create_openai_client", lambda: client)

    output = style_profile.describe_style_with_llm(profile)
    assert "Dreamy" in output
