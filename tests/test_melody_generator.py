from pathlib import Path

import pytest

from src.melody import (
    generate_placeholder_melody,
    melody_to_midi,
    generate_simple_melody_midi,
)


def test_generate_placeholder_melody_creates_sequence():
    style_profile = {"tempo_mean": 120, "energy_profile": {"low": 0.2, "high": 0.6}}
    notes = generate_placeholder_melody(style_profile, bars=2)

    assert isinstance(notes, list)
    assert len(notes) > 0
    assert all(notes[i].start_beat <= notes[i + 1].start_beat for i in range(len(notes) - 1))


def test_generate_placeholder_melody_limits_interval():
    style_profile = {"tempo_mean": 110, "energy_profile": {"low": 0.3, "high": 0.1}}
    notes = generate_placeholder_melody(style_profile, bars=2)
    max_jump = max(abs(notes[i].pitch - notes[i - 1].pitch) for i in range(1, len(notes)))

    assert max_jump <= 7


def test_melody_to_midi_writes_file(tmp_path: Path):
    style_profile = {"tempo_mean": 96, "energy_profile": {"low": 0.5, "high": 0.2}}
    notes = generate_placeholder_melody(style_profile, bars=1)

    midi_path = tmp_path / "demo.mid"
    melody_to_midi(notes, midi_path, tempo_bpm=style_profile["tempo_mean"])

    assert midi_path.exists()
    assert midi_path.stat().st_size > 0


def test_generate_simple_melody_midi(tmp_path: Path):
    style_profile = {"tempo_mean": 100, "energy_profile": {"low": 0.3, "high": 0.4}}
    lyrics = {
        "sections": [
            {"type": "verse", "name": "Verse 1", "lines": ["line 1", "line 2"]},
            {"type": "chorus", "name": "Chorus", "lines": ["line a", "line b"]},
        ]
    }
    out_path = tmp_path / "simple.mid"
    result_path = generate_simple_melody_midi(style_profile, lyrics, out_path)

    assert result_path.exists()
    assert result_path.stat().st_size > 0
    # ensure bpm fallback works when not provided
    result_path_2 = generate_simple_melody_midi({"energy_profile": {}}, lyrics, tmp_path / "simple2.mid", bpm=None)
    assert result_path_2.exists()
    assert result_path_2.stat().st_size > 0
