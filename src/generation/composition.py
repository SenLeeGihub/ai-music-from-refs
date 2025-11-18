"""
Composition utilities for bridging analyzed style data and MusicGen prompts.
"""

from __future__ import annotations

from typing import Dict, Iterable, List


def _format_energy_profile(energy_profile: Dict[str, float]) -> str:
    if not energy_profile:
        return "Balanced spectrum"

    total = sum(energy_profile.values()) or 1.0
    pieces: List[str] = []
    for band in ("low", "mid", "high"):
        if band in energy_profile:
            percent = energy_profile[band] / total * 100.0
            descriptor = "rich" if percent > 38 else "present" if percent > 25 else "subtle"
            pieces.append(f"{descriptor} {band} frequencies (~{percent:.0f}%)")
    if not pieces:
        return "Even frequency distribution"
    return ", ".join(pieces)


def build_musicgen_prompt(style_profile: Dict[str, float], style_text: str) -> str:
    """
    Combine aggregate stats with an LLM-generated blurb into a MusicGen prompt.
    """
    tempo_range = style_profile.get("tempo_range", [0, 0])
    tempo_mean = style_profile.get("tempo_mean", 0)
    energy_profile = style_profile.get("energy_profile", {})

    energy_text = _format_energy_profile(energy_profile)
    descriptive_text = style_text.strip()

    prompt_parts = [
        descriptive_text,
        f"Keep the groove around {tempo_range[0]:.0f}-{tempo_range[1]:.0f} BPM (avg {tempo_mean:.0f}).",
        f"Focus on {energy_text}.",
        "High fidelity mix ready for creative sampling.",
    ]
    prompt = " ".join(part for part in prompt_parts if part)
    return prompt.strip()
