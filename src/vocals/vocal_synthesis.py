"""
Configurable vocal synthesis utilities with a placeholder backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Literal
import os
import shlex
import subprocess
import tempfile

import json
import numpy as np
import soundfile as sf


VocalBackend = Literal["placeholder", "ai", "voice_clone", "diffsinger"]
VocalBackendImpl = Callable[[Path, Dict, Path, Optional[Path], Optional[float]], Path]

_VOCAL_BACKENDS: Dict[VocalBackend, VocalBackendImpl] = {}


def register_vocal_backend(name: VocalBackend, backend: VocalBackendImpl) -> None:
    """
    Register a vocal synthesis backend implementation.
    """
    _VOCAL_BACKENDS[name] = backend


def synthesize_vocals_external_engine(
    melody_midi: Path,
    lyrics_txt: Path,
    out_wav: Path,
    voice_ref_dir: Optional[Path] = None,
) -> Path:
    """
    Invoke an external vocal engine defined by VOCAL_ENGINE_CMD.
    """
    cmd_template = os.getenv("VOCAL_ENGINE_CMD")
    if not cmd_template:
        raise ValueError("VOCAL_ENGINE_CMD environment variable is not set.")

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    resolved_paths = {
        "midi": str(melody_midi.resolve()),
        "lyrics": str(lyrics_txt.resolve()),
        "out": str(out_wav.resolve()),
        "ref_dir": str(voice_ref_dir.resolve()) if voice_ref_dir else "",
    }
    try:
        command_str = cmd_template.format(**resolved_paths)
    except KeyError as exc:
        raise ValueError(f"Missing placeholder in VOCAL_ENGINE_CMD template: {exc}") from exc

    subprocess.run(command_str, shell=True, check=True)

    if not out_wav.exists() or out_wav.stat().st_size == 0:
        raise RuntimeError(f"External vocal engine did not create a valid output at {out_wav}.")

    return out_wav


def synthesize_vocals_placeholder(
    melody_midi: Path,
    lyrics: Dict,
    out_wav: Path,
    reference_voice_dir: Optional[Path] = None,
    duration_seconds: Optional[float] = None,
) -> Path:
    """
    Generate a silent vocal track and manifest for downstream integration.
    """
    print(
        "[Vocals Placeholder] No singing synthesis backend configured yet. "
        "This function currently writes a silent track and manifest only."
    )
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 44100
    duration = max(0.1, float(duration_seconds) if duration_seconds is not None else 5.0)
    num_samples = int(sample_rate * duration)
    data = np.zeros(num_samples, dtype=np.float32)
    sf.write(out_wav.as_posix(), data, sample_rate)

    manifest = {
        "melody_midi": str(melody_midi),
        "lyrics_title": lyrics.get("title"),
        "lyrics_theme": lyrics.get("theme"),
        "output_wav": str(out_wav),
        "reference_voice_dir": str(reference_voice_dir) if reference_voice_dir else None,
        "sample_rate": sample_rate,
        "duration": duration,
    }
    manifest_path = out_wav.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_wav


def _run_external_engine(
    melody_midi: Path,
    lyrics: Dict,
    out_wav: Path,
    mode: VocalBackend,
    reference_voice_dir: Optional[Path] = None,
    duration_seconds: Optional[float] = None,
    tmp_dir: Optional[Path] = None,
) -> bool:
    """
    Prepare temp assets and invoke an external singing engine via subprocess.
    Returns True on success (output written), False otherwise.
    """
    cmd_template = os.getenv("VOCAL_ENGINE_CMD")
    if not cmd_template:
        raise NotImplementedError(
            "VOCAL_ENGINE_CMD environment variable is not set. "
            "Please configure the command template for external vocal synthesis."
        )

    duration = duration_seconds or 5.0
    tmp_context = (
        tempfile.TemporaryDirectory(dir=str(tmp_dir))
        if tmp_dir is not None
        else tempfile.TemporaryDirectory()
    )
    with tmp_context as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        if tmp_dir is not None:
            tmp_dir.mkdir(parents=True, exist_ok=True)
        lyrics_json_path = tmpdir / "lyrics.json"
        lyrics_txt_path = tmpdir / "lyrics.txt"
        lyrics_json_path.write_text(json.dumps(lyrics, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_lyrics_txt(lyrics, lyrics_txt_path)

        format_map = {
            "melody_midi": melody_midi.as_posix(),
            "midi": melody_midi.as_posix(),
            "melody": melody_midi.as_posix(),
            "lyrics_json": lyrics_json_path.as_posix(),
            "lyrics_txt": lyrics_txt_path.as_posix(),
            "output_wav": out_wav.as_posix(),
            "out_wav": out_wav.as_posix(),
            "reference_voice_dir": reference_voice_dir.as_posix() if reference_voice_dir else "",
            "ref_dir": reference_voice_dir.as_posix() if reference_voice_dir else "",
            "duration_seconds": duration,
            "mode": mode,
        }
        try:
            cmd_str = cmd_template.format(**format_map)
        except KeyError as exc:
            raise ValueError(f"Missing placeholder in VOCAL_ENGINE_CMD template: {exc}") from exc

        command = shlex.split(cmd_str)

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[Vocals Backend:{mode}] command failed: {exc}.")
            return False

    if not out_wav.exists() or out_wav.stat().st_size == 0:
        print(f"[Vocals Backend:{mode}] Output file missing or empty.")
        return False

    return True


def _synthesize_with_diffsinger(
    melody_midi: Path,
    lyrics: Dict,
    out_wav: Path,
    reference_voice_dir: Optional[Path],
    duration_seconds: Optional[float],
) -> Path:
    """
    Dedicated hook for DiffSinger-style engines.
    """
    tmp_dir_env = os.getenv("VOCAL_ENGINE_TMP")
    tmp_dir = Path(tmp_dir_env) if tmp_dir_env else Path("outputs/vocals/tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    success = _run_external_engine(
        melody_midi=melody_midi,
        lyrics=lyrics,
        out_wav=out_wav,
        mode="diffsinger",
        reference_voice_dir=reference_voice_dir,
        duration_seconds=duration_seconds,
        tmp_dir=tmp_dir,
    )
    if success:
        return out_wav
    duration = duration_seconds or 5.0
    return synthesize_vocals_placeholder(
        melody_midi,
        lyrics,
        out_wav,
        reference_voice_dir,
        duration,
    )


def synthesize_vocals(
    melody_midi: Path,
    lyrics: Dict,
    out_wav: Path,
    backend: Optional[VocalBackend] = None,
    reference_voice_dir: Optional[Path] = None,
    duration_seconds: Optional[float] = None,
) -> Path:
    """
    Synthesize vocals using a configured backend (default placeholder).
    """
    backend_name_env = os.getenv("VOCALS_BACKEND")
    backend_name: VocalBackend = backend or backend_name_env or "placeholder"  # type: ignore[assignment]
    if backend_name == "placeholder" and duration_seconds is None:
        raise ValueError("duration_seconds is required for the placeholder vocal backend.")
    impl = _VOCAL_BACKENDS.get(backend_name)
    if impl is None:
        if backend_name == "diffsinger":
            return _synthesize_with_diffsinger(
                melody_midi,
                lyrics,
                out_wav,
                reference_voice_dir,
                duration_seconds,
            )
        if backend_name in ("ai", "voice_clone"):
            success = _run_external_engine(
                melody_midi=melody_midi,
                lyrics=lyrics,
                out_wav=out_wav,
                mode=backend_name,
                reference_voice_dir=reference_voice_dir,
                duration_seconds=duration_seconds,
            )
            if not success:
                duration = duration_seconds or 5.0
                return synthesize_vocals_placeholder(
                    melody_midi,
                    lyrics,
                    out_wav,
                    reference_voice_dir,
                    duration,
                )
            return out_wav
        raise ValueError(f"Unknown vocal backend '{backend_name}'")
    return impl(melody_midi, lyrics, out_wav, reference_voice_dir, duration_seconds)


# Register default backend
register_vocal_backend("placeholder", synthesize_vocals_placeholder)


def _write_lyrics_txt(lyrics: Dict, path: Path) -> None:
    """
    Serialize structured lyrics into a simple line-by-line text format.
    """
    path.write_text(_lyrics_to_text(lyrics), encoding="utf-8")


def _lyrics_to_text(lyrics: Dict) -> str:
    lines = []
    sections = lyrics.get("sections") or []
    for section in sections:
        sec_lines = section.get("lines") or []
        lines.extend(str(line).strip() for line in sec_lines if str(line).strip())
    return "\n".join(lines).strip() or lyrics.get("theme", "")
