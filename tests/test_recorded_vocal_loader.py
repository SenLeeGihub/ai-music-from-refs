from pathlib import Path
import os

from src.vocals.recorded_vocal_loader import find_latest_recorded_vocal


def test_find_latest_recorded_vocal_returns_latest(tmp_path: Path):
    older = tmp_path / "take1.wav"
    older.write_bytes(b"old")
    newer_dir = tmp_path / "sub"
    newer_dir.mkdir()
    newer = newer_dir / "take2.wav"
    newer.write_bytes(b"new")

    os.utime(older, (older.stat().st_atime, older.stat().st_mtime - 100))
    os.utime(newer, (newer.stat().st_atime, newer.stat().st_mtime + 100))

    result = find_latest_recorded_vocal(tmp_path)
    assert result == newer


def test_find_latest_recorded_vocal_handles_missing_dir(tmp_path: Path):
    missing = tmp_path / "missing"
    assert find_latest_recorded_vocal(missing) is None
