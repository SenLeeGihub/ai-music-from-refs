from src.generation.composition import build_musicgen_prompt


def test_build_musicgen_prompt_includes_core_elements():
    style_profile = {
        "tempo_range": [95.0, 110.0],
        "tempo_mean": 102.0,
        "energy_profile": {"low": 0.6, "mid": 0.3, "high": 0.1},
    }
    style_text = "Warm indie-electronic mood with shimmering guitars."

    prompt = build_musicgen_prompt(style_profile, style_text)

    assert "Warm indie-electronic mood" in prompt
    assert "95-110 BPM" in prompt
    assert "Focus on" in prompt
    assert "frequencies" in prompt
