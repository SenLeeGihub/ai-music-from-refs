"""
Vocals helper utilities.
"""

from .vocal_synthesis import (
    synthesize_vocals,
    synthesize_vocals_placeholder,
    synthesize_vocals_external_engine,
    register_vocal_backend,
)

__all__ = [
    "synthesize_vocals_placeholder",
    "synthesize_vocals_external_engine",
    "synthesize_vocals",
    "register_vocal_backend",
]
