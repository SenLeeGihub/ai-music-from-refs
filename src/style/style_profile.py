"""
Utilities for building aggregated style profiles and producing natural-language
descriptions of a reference collection using the OpenAI Responses API.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from openai import OpenAI, OpenAIError

from src.analysis.audio_analysis import aggregate_style_stats


def build_style_profile(track_features: List[dict]) -> Dict[str, Any]:
    """
    Build a style_profile dictionary from analyze_track outputs.
    """
    if not track_features:
        raise ValueError("track_features must not be empty")

    stats = aggregate_style_stats(track_features)
    return {
        "tempo_range": stats["tempo_range"],
        "tempo_mean": stats["tempo_mean"],
        "energy_profile": stats["energy_profile"],
    }


def _format_energy_profile(energy_profile: Dict[str, float]) -> str:
    parts = []
    for band, value in energy_profile.items():
        parts.append(f"{band.capitalize()} energy: {value:.3f}")
    return ", ".join(parts)


def _build_prompt(style_profile: Dict[str, Any]) -> List[Dict[str, str]]:
    energy_text = _format_energy_profile(style_profile["energy_profile"])
    tempo_range = style_profile["tempo_range"]
    prompt = (
        "Given the following aggregate statistics from reference tracks, "
        "write 2-3 sentences in English describing the musical style for a MusicGen prompt. "
        "Explicitly mention mood/energy, likely instrumentation, and fitting usage scenarios "
        "(e.g., background scoring, workout mix). "
        "Interpret the numbers qualitatively rather than listing them verbatim.\n\n"
        f"Tempo range: {tempo_range[0]:.1f} - {tempo_range[1]:.1f} BPM\n"
        f"Average tempo: {style_profile['tempo_mean']:.1f} BPM\n"
        f"Energy profile: {energy_text}\n"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise music analyst that summarizes sonic character "
                "for AI music generation prompts."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    return messages


def _extract_response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    output = getattr(response, "output", None)
    if output:
        chunks: List[str] = []
        for item in output:
            for content in getattr(item, "content", []):
                content_text = getattr(content, "text", None)
                if content_text:
                    chunks.append(content_text)
        if chunks:
            return "\n".join(chunk.strip() for chunk in chunks if chunk.strip())
    raise RuntimeError("Failed to extract text from OpenAI response")


def _create_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required")
    return OpenAI(api_key=api_key)


def describe_style_with_llm(
    style_profile: Dict[str, Any],
    *,
    client: Optional[OpenAI] = None,
    model: str = "gpt-5.1",
) -> str:
    """
    Generate a natural-language description for the provided style profile using an LLM.
    """
    if client is None:
        client = _create_openai_client()

    messages = _build_prompt(style_profile)
    try:
        response = client.responses.create(model=model, input=messages)
    except OpenAIError as exc:
        raise RuntimeError("Failed to generate style description") from exc

    description = _extract_response_text(response)
    return description.strip()
