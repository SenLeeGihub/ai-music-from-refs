"""
End-to-end utility that analyzes references, generates demos, and builds vocals.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.audio_analysis import analyze_track
from src.generation.composition import build_musicgen_prompt
from src.generation.musicgen_backend import MusicGenBackend
from src.generation.mixdown import mix_backing_and_vocal
from src.style.style_profile import build_style_profile, describe_style_with_llm
from src.lyrics import generate_lyrics, lyrics_to_text
from src.melody import generate_simple_melody_midi
from src.vocals.vocal_synthesis import (
    synthesize_vocals_external_engine,
    synthesize_vocals_placeholder,
)
from src.vocals.recorded_vocal_loader import find_latest_recorded_vocal


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _list_audio_files(root: Path) -> List[Path]:
    return sorted(
        [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]
    )


def _build_prompt_variations(base_prompt: str) -> List[str]:
    adjectives = ["cinematic", "driving", "dreamy"]
    prompts = [f"{base_prompt} Emphasize a {adj} vibe." for adj in adjectives]
    return prompts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MusicGen demos from reference songs.")
    parser.add_argument(
        "--vocal-source",
        choices=["placeholder", "recorded", "engine"],
        default="placeholder",
        help="Select which vocal source to use for mixes.",
    )
    parser.add_argument(
        "--voice-ref-dir",
        type=Path,
        default=Path("data/voice_refs"),
        help="Directory of reference voices for external engines (overridden by VOICE_REF_DIR).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    refs_dir = Path("data/refs")
    if not refs_dir.exists():
        raise FileNotFoundError("Reference directory data/refs was not found.")

    reference_paths = _list_audio_files(refs_dir)
    if not reference_paths:
        raise RuntimeError("No reference audio files found in data/refs.")

    print(f"Found {len(reference_paths)} reference tracks.")

    features = []
    for path in reference_paths:
        print(f"Analyzing {path.name}...")
        features.append(analyze_track(str(path)))

    style_profile = build_style_profile(features)
    style_text = describe_style_with_llm(style_profile)
    print("Style description:")
    print(style_text)

    lyrics = generate_lyrics(style_profile, style_text)
    lyrics_text = lyrics_to_text(lyrics)
    lyrics_dir = Path("outputs/lyrics")
    lyrics_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lyrics_path = lyrics_dir / f"{timestamp}_lyrics.txt"
    lyrics_path.write_text(lyrics_text, encoding="utf-8")
    print(f"Lyrics saved to {lyrics_path}")

    melody_dir = Path("outputs/melody")
    melody_dir.mkdir(parents=True, exist_ok=True)
    melody_path = melody_dir / f"{timestamp}_melody.mid"
    generate_simple_melody_midi(style_profile, lyrics, melody_path)
    print(f"Melody MIDI saved to {melody_path}")

    base_prompt = build_musicgen_prompt(style_profile, style_text)
    prompts = _build_prompt_variations(base_prompt)

    print("Generating music with prompts:")
    for idx, prompt in enumerate(prompts, start=1):
        print(f"Prompt {idx}: {prompt}")

    backend = MusicGenBackend()
    clips = backend.generate_clips(prompts, duration=20)

    output_dir = Path("outputs/demos")
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir = Path("outputs/final_mix")
    final_dir.mkdir(parents=True, exist_ok=True)
    final_mix_paths: List[Path] = []
    voice_ref_dir_env = os.getenv("VOICE_REF_DIR")
    voice_ref_dir = Path(voice_ref_dir_env) if voice_ref_dir_env else Path(args.voice_ref_dir)
    vocal_source = args.vocal_source

    for idx, clip in enumerate(clips, start=1):
        backing_path = output_dir / f"demo_{idx:02d}.wav"
        backend.save_wav(clip, backing_path, backend.sample_rate)
        print(f"Saved {backing_path}")

        data, sr = sf.read(backing_path)
        duration_seconds = len(data) / sr if len(data) else 20.0

        vocal_out_path = output_dir / f"demo_{idx:02d}_vocal.wav"
        if vocal_source == "placeholder":
            vocal_path = synthesize_vocals_placeholder(
                melody_midi=melody_path,
                lyrics=lyrics,
                out_wav=vocal_out_path,
                reference_voice_dir=None,
                duration_seconds=duration_seconds,
            )
            print(f"Placeholder vocals saved to {vocal_path}")
        elif vocal_source == "recorded":
            recorded_vocal = find_latest_recorded_vocal(Path("data/recorded"))
            if recorded_vocal is None:
                raise SystemExit(
                    "No recorded vocals found in data/recorded. Add a WAV/MP3 first or switch --vocal-source."
                )
            vocal_path = recorded_vocal
            print(f"Using recorded vocal from {vocal_path}")
        elif vocal_source == "engine":
            vocal_path = synthesize_vocals_external_engine(
                melody_midi=melody_path,
                lyrics_txt=lyrics_path,
                out_wav=vocal_out_path,
                voice_ref_dir=voice_ref_dir,
            )
            print(f"External engine vocals saved to {vocal_path}")
        else:
            raise ValueError(f"Unsupported vocal source '{vocal_source}'")

        final_mix_path = final_dir / f"demo_{idx:02d}_mix.wav"
        mix_backing_and_vocal(
            backing_path=backing_path,
            vocal_path=vocal_path,
            out_path=final_mix_path,
            vocal_gain_db=-3.0,
        )
        final_mix_paths.append(final_mix_path)
        print(f"Final mix saved to {final_mix_path}")

    if final_mix_paths:
        print("Generated mixes:")
        for path in final_mix_paths:
            print(f" - {path}")


if __name__ == "__main__":
    main()
