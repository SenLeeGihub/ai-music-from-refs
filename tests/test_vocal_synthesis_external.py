from pathlib import Path
import subprocess

from src.vocals import synthesize_vocals_external_engine


def test_synthesize_vocals_external_engine_runs_with_env(monkeypatch, tmp_path: Path):
    midi_path = tmp_path / "melody.mid"
    midi_path.write_text("midi-data")
    lyrics_path = tmp_path / "lyrics.txt"
    lyrics_path.write_text("la la la")
    out_path = tmp_path / "output.wav"
    out_path.write_bytes(b"stub")  # ensure non-empty output for validation
    voice_ref_dir = tmp_path / "voice"
    voice_ref_dir.mkdir()

    captured = {}

    def fake_run(cmd, shell, check):
        captured["cmd"] = cmd
        captured["shell"] = shell
        captured["check"] = check

    monkeypatch.setenv(
        "VOCAL_ENGINE_CMD",
        "echo midi={midi} lyrics={lyrics} out={out} ref={ref_dir}",
    )
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = synthesize_vocals_external_engine(
        melody_midi=midi_path,
        lyrics_txt=lyrics_path,
        out_wav=out_path,
        voice_ref_dir=voice_ref_dir,
    )

    assert result == out_path
    assert captured["shell"] is True
    assert captured["check"] is True
    expected_command = (
        f"echo midi={midi_path.resolve()} "
        f"lyrics={lyrics_path.resolve()} "
        f"out={out_path.resolve()} "
        f"ref={voice_ref_dir.resolve()}"
    )
    assert captured["cmd"] == expected_command
