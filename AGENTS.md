# AGENTS for ai-music-from-refs

## Project goal

- Build a Python project that:
  - Takes several reference songs (audio files) in `data/refs/`.
  - Analyzes their style (tempo, energy, rough key, etc.).
  - Builds a `style_profile` JSON and a natural-language style description.
  - Uses AI music models (e.g. AudioCraft / MusicGen) to generate new songs
    in a similar style, and saves audio to `outputs/demos/`.

## Tech stack

- Python 3.10+
- Libraries:
  - `librosa` for audio analysis
  - `numpy`, `soundfile`
  - `audiocraft` (MusicGen) for music generation
  - `pytest` for tests
  - `openai` for LLM calls (Responses API, model gpt-5.1)

## Environment / commands (must work on Linux)

- Create venv:
  - `python3 -m venv .venv && source .venv/bin/activate`
- Install deps:
  - `pip install -r requirements.txt`
- Run tests:
  - `pytest`
- Run demo:
  - `python scripts/generate_from_refs.py`
- Do NOT hardcode Windows paths; use `pathlib.Path`.

## Code layout

- `src/analysis/audio_analysis.py`:
  - Functions to analyze reference tracks with `librosa`.
- `src/style/style_profile.py`:
  - Build aggregated style_profile JSON.
  - Call OpenAI Responses API to describe style in English.
- `src/generation/musicgen_backend.py`:
  - Wrap AudioCraft / MusicGen, with automatic CPU/GPU selection.
- `src/generation/composition.py`:
  - Build prompts and tie style + generation together.
- `scripts/generate_from_refs.py`:
  - Top-level script that:
    - reads `data/refs`,
    - analyzes tracks,
    - builds style,
    - generates several 20–30s demos.

## Cross-platform & GPU

- Code must run on:
  - Windows + WSL (dev)
  - Linux with GPU (prod)
- In MusicGen code, automatically detect GPU:
  - use `torch.cuda.is_available()` and `device="cuda"` when possible.
- Do not use OS-specific shell commands; if needed, guard them.

## First milestone

- From a few `.wav`/`.mp3` files in `data/refs/`, generate 3 short
  (e.g. 20s) MusicGen clips in similar style into `outputs/demos/`.

