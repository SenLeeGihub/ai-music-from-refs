"""
Placeholder melody generation utilities producing simple MIDI-ready data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Optional
import random

from mido import Message, MetaMessage, MidiFile, MidiTrack


@dataclass(frozen=True)
class MelodyNote:
    """Represents a single note event in beats."""

    pitch: int
    start_beat: float
    duration: float
    velocity: int = 90


_KEY_TO_ROOT = {
    "C": 60,
    "D": 62,
    "E": 64,
    "F": 65,
    "G": 67,
    "A": 69,
    "B": 71,
}

_MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
_MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]


def _build_scale_pitches(root: int, intervals: Sequence[int], octaves: int = 2) -> List[int]:
    values: List[int] = []
    for octave in range(-1, octaves + 1):
        offset = 12 * octave
        for interval in intervals:
            values.append(root + interval + offset)
    return sorted(set(values))


def _choose_scale(style_profile: Dict | None) -> tuple[int, Sequence[int]]:
    if not style_profile:
        return _KEY_TO_ROOT["C"], _MAJOR_INTERVALS

    tempo = style_profile.get("tempo_mean", 110)
    energy = style_profile.get("energy_profile", {})
    energy_bias = energy.get("high", 0.0) - energy.get("low", 0.0)

    if tempo >= 112 or energy_bias >= 0:
        return _KEY_TO_ROOT["C"], _MAJOR_INTERVALS
    return _KEY_TO_ROOT["A"], _MINOR_INTERVALS


def generate_placeholder_melody(
    style_profile: Optional[Dict] = None,
    *,
    bars: int = 4,
    beats_per_bar: int = 4,
    lines: Optional[List[str]] = None,
    max_interval: int = 7,
) -> List[MelodyNote]:
    """
    Generate a simple melodic line using constrained random movement.
    """
    if lines:
        bars = max(len(lines), bars)
    if bars <= 0:
        raise ValueError("bars must be positive")

    root, scale = _choose_scale(style_profile)
    scale_pool = _build_scale_pitches(root, scale, octaves=2)
    note_duration = 1.0  # quarter-note granularity
    total_steps = int(bars * beats_per_bar)

    seed_value = int(style_profile.get("tempo_mean", 0)) if style_profile else 0
    rng = random.Random(seed_value)

    notes: List[MelodyNote] = []
    last_pitch = root + scale[0]
    for step in range(total_steps):
        if not notes:
            pitch = last_pitch
        else:
            candidates = [p for p in scale_pool if abs(p - last_pitch) <= max_interval]
            if not candidates:
                candidates = [last_pitch]
            pitch = rng.choice(candidates)

        start = step * note_duration
        notes.append(MelodyNote(pitch=pitch, start_beat=start, duration=note_duration))
        last_pitch = pitch

        if beats_per_bar > 0 and (step + 1) % beats_per_bar == 0:
            last_pitch = root + scale[0]

    return notes


def melody_to_midi(
    notes: Sequence[MelodyNote],
    path: str | Path,
    *,
    tempo_bpm: float = 110,
) -> None:
    """
    Persist a melody as a single-track MIDI file.
    """
    if not notes:
        raise ValueError("notes must not be empty")

    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)

    microseconds_per_beat = int(60_000_000 / max(1.0, tempo_bpm))
    track.append(MetaMessage("set_tempo", tempo=microseconds_per_beat, time=0))

    sorted_notes = sorted(notes, key=lambda n: n.start_beat)
    ticks_per_beat = midi.ticks_per_beat
    current_tick = 0
    for note in sorted_notes:
        start_tick = int(note.start_beat * ticks_per_beat)
        delta = max(0, start_tick - current_tick)
        track.append(Message("note_on", note=note.pitch, velocity=note.velocity, time=delta))
        current_tick = start_tick

        end_tick = start_tick + max(1, int(note.duration * ticks_per_beat))
        track.append(Message("note_off", note=note.pitch, velocity=0, time=end_tick - current_tick))
        current_tick = end_tick

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output_path.as_posix())


def generate_simple_melody_midi(
    style_profile: Dict,
    lyrics: Dict,
    out_path: Path,
    bpm: float | None = None,
) -> Path:
    """
    Generate a placeholder melody from style + lyrics and save it to MIDI.
    """
    tempo = bpm or style_profile.get("tempo_mean") or 90

    sections = lyrics.get("sections") or []
    lines: List[str] = []
    for section in sections:
        section_lines = section.get("lines")
        if isinstance(section_lines, list):
            lines.extend([line for line in section_lines if isinstance(line, str)])
        elif isinstance(section_lines, str):
            lines.append(section_lines)

    bars = len(lines) or max(4, len(sections) * 2)
    notes = generate_placeholder_melody(style_profile, bars=bars, lines=lines)
    melody_to_midi(notes, out_path, tempo_bpm=tempo)
    return out_path
