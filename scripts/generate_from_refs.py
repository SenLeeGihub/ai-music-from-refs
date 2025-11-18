"""
End-to-end utility that analyzes reference tracks and generates MusicGen demos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from src.analysis.audio_analysis import analyze_track
from src.generation.composition import build_musicgen_prompt
from src.generation.musicgen_backend import MusicGenBackend
from src.style.style_profile import build_style_profile, describe_style_with_llm


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _list_audio_files(root: Path) -> List[Path]:
    return sorted(
        [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]
    )


def _build_prompt_variations(base_prompt: str) -> List[str]:
    adjectives = ["cinematic", "driving", "dreamy"]
    prompts = [f"{base_prompt} Emphasize a {adj} vibe." for adj in adjectives]
    return prompts


def main():
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

    base_prompt = build_musicgen_prompt(style_profile, style_text)
    prompts = _build_prompt_variations(base_prompt)

    print("Generating music with prompts:")
    for idx, prompt in enumerate(prompts, start=1):
        print(f"Prompt {idx}: {prompt}")

    backend = MusicGenBackend()
    clips = backend.generate_clips(prompts, duration=20)

    output_dir = Path("outputs/demos")
    for idx, clip in enumerate(clips, start=1):
        output_path = output_dir / f"demo_{idx:02d}.wav"
        backend.save_wav(clip, output_path, backend.sample_rate)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
