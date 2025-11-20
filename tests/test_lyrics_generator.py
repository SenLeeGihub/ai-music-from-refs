import json

import pytest

from src.lyrics import lyrics_generator


class _DummyResponse:
    def __init__(self, text: str):
        self.output_text = text
        self.output = []


class _DummyClient:
    def __init__(self, text: str):
        self._text = text
        self.responses = self
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _DummyResponse(self._text)


def test_generate_lyrics_returns_dict(monkeypatch):
    style_profile = {
        "tempo_range": [90.0, 110.0],
        "tempo_mean": 100.0,
        "energy_profile": {"low": 0.4, "mid": 0.35, "high": 0.25},
    }
    lyrics_payload = {
        "title": "晨光",
        "language": "zh",
        "theme": "希望",
        "sections": [
            {"type": "verse", "name": "Verse 1", "lines": ["第一行", "第二行"]},
            {"type": "chorus", "name": "Chorus", "lines": ["副歌A", "副歌B"]},
        ],
    }
    client = _DummyClient(json.dumps(lyrics_payload))

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(lyrics_generator, "OpenAI", lambda api_key=None: client)

    result = lyrics_generator.generate_lyrics(style_profile, "温暖流行风格", theme="希望")

    assert isinstance(result, dict)
    assert result["title"] == "晨光"
    assert result["language"] == "zh"
    assert "sections" in result
    assert isinstance(result["sections"], list)
    assert result["sections"][0]["type"] == "verse"
    kwargs = client.calls[0]
    assert kwargs["model"] == "gpt-5.1"
    assert "Tempo" not in kwargs["input"][1]["content"]  # ensure Chinese prompt
    assert "标题可以使用简短英文" in kwargs["input"][1]["content"]
    assert "3-5 个段落" in kwargs["input"][1]["content"]


def test_generate_lyrics_falls_back_on_plain_text(monkeypatch):
    style_profile = {"tempo_range": [80.0, 90.0], "energy_profile": {}}
    client = _DummyClient("纯文本歌词")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(lyrics_generator, "OpenAI", lambda api_key=None: client)

    result = lyrics_generator.generate_lyrics(style_profile, "清新民谣")
    assert result["sections"][0]["lines"][0].startswith("纯文本")


def test_lyrics_to_text_formats_sections():
    lyrics = {
        "title": "雨后的街",
        "theme": "回忆",
        "language": "zh",
        "sections": [
            {"type": "verse", "name": "Verse 1", "lines": ["风吹过旧街", "雨滴在指尖"]},
            {"type": "chorus", "name": "Chorus", "lines": ["我们在灯火阑珊", "轻唱未完的歌"]},
        ],
    }

    text = lyrics_generator.lyrics_to_text(lyrics)

    assert "雨后的街" in text
    assert "[Verse 1]" in text
    assert "我们在灯火阑珊" in text


def test_prompt_includes_style_inputs(monkeypatch):
    style_profile = {"tempo_range": [88.0, 104.0], "energy_profile": {"low": 0.5}}
    style_text = "Mock style summary"
    lyrics_payload = {
        "title": "Echoes",
        "language": "zh",
        "theme": None,
        "sections": [
            {"type": "verse", "name": "Verse 1", "lines": ["line 1", "line 2"]},
            {"type": "chorus", "name": "Chorus", "lines": ["line a", "line b"]},
        ],
    }
    client = _DummyClient(json.dumps(lyrics_payload))

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(lyrics_generator, "OpenAI", lambda api_key=None: client)

    result = lyrics_generator.generate_lyrics(style_profile, style_text)

    kwargs = client.calls[0]
    prompt_text = kwargs["input"][1]["content"]
    assert style_text in prompt_text
    assert "88" in prompt_text and "104" in prompt_text
    assert result["title"] == "Echoes"
