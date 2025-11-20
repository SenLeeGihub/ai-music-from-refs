## AI Music From Reference Tracks

This project analyzes a handful of reference songs, summarizes their shared style, and uses AudioCraft's MusicGen model to synthesize fresh demos that match the captured vibe.

### Prerequisites

1. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   # Linux/macOS
   source .venv/bin/activate
   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Provide reference tracks**
   - Place `.wav`, `.mp3`, or other supported audio files inside `data/refs/`.
   - Multiple tracks (2–5) work best so the aggregated style is meaningful.
4. **Set your OpenAI API key**
   ```bash
   # Linux/macOS
   export OPENAI_API_KEY="sk-..."
   # Windows PowerShell
   $env:OPENAI_API_KEY="sk-..."
   ```

### Generate MusicGen Demos

Run the orchestration script. It will:
1. Analyze every file inside `data/refs/`.
2. Build a consolidated `style_profile`.
3. Ask GPT-5.1 via the OpenAI Responses API for an English style summary.
4. Generate matching Mandarin lyrics + a simple placeholder melody (MIDI) and store them under `outputs/lyrics/` and `outputs/melody/`.
5. Craft three lightly varied MusicGen prompts.
6. Generate ~20 second clips and save them in `outputs/demos/`.
7. For each demo: estimate the duration, create vocals according to `--vocal-source` (`placeholder`, `recorded`, or `engine`), and mix the vocal with the backing track into `outputs/final_mix/demo_XX_mix.wav` (also printing the generated mix paths).

```bash
python scripts/generate_from_refs.py \
  --vocal-source placeholder \
  --voice-ref-dir data/voice_refs
```

The script automatically selects GPU acceleration if available via `torch.cuda.is_available()`. At the end of a run it prints the saved lyrics path, melody MIDI path, and every demo file path so you can quickly locate the outputs. Review the printed prompts and generated WAV files to iterate on your reference set. Use different adjectives or edit the prompts in code if you want more variations.

### Vocal Sources and External Engines

You can pick how vocals are sourced by passing `--vocal-source`:

| Mode | Description |
| --- | --- |
| `placeholder` (default) | Calls the built-in silent track generator for debugging and layout checks. The placeholder matches the length of each backing track. |
| `recorded` | Grabs the most recently modified audio file inside `data/recorded/` (WAV/MP3/FLAC/OGG/M4A) via `find_latest_recorded_vocal` and mixes it with each MusicGen clip. Handy when you already have a singer take. |
| `engine` | Invokes any external vocal synthesis engine defined by the `VOCAL_ENGINE_CMD` environment variable. The script formats the command template with `{midi}`, `{lyrics}`, `{out}`, and `{ref_dir}` placeholders (absolute paths). |

When `--vocal-source engine` is used, set:

```bash
$env:VOCAL_ENGINE_CMD = "python tools/diffsinger_cli.py --melody {midi} --lyrics {lyrics} --voice {ref_dir} --out {out}"
$env:VOICE_REF_DIR = "data/voice_refs/jay-style"  # optional override; falls back to --voice-ref-dir
```

The template receives:

- `{midi}`: generated melody MIDI path
- `{lyrics}`: plain-text lyrics file saved in `outputs/lyrics/`
- `{out}`: desired vocal WAV path (must be created by the engine)
- `{ref_dir}`: voice reference directory (either `VOICE_REF_DIR` or the `--voice-ref-dir` argument)

If the engine exits successfully and the output file is non-empty, it is mixed with the backing demo; otherwise the run stops with an error so you can inspect the command.

### Optional: Generate Lyrics

After you obtain a `style_profile` and the LLM-generated `style_text`, you can request matching Mandarin lyrics (with an optional short English title) using utilities in `src/lyrics/lyrics_generator.py`.

```python
from src.lyrics import generate_lyrics, lyrics_to_text

style_profile = {"tempo_range": [95, 110], "tempo_mean": 102, "energy_profile": {"low": 0.5, "mid": 0.3, "high": 0.2}}
style_text = "Uplifting synth-pop with shimmering guitars."

lyrics = generate_lyrics(style_profile, style_text, theme="夜行城市")
print(lyrics_to_text(lyrics))
```

The lyrics helper uses the OpenAI Responses API (`gpt-5.1`), so ensure `OPENAI_API_KEY` is set before running this snippet. The function returns a structured dict you can store as JSON or convert to plain text with `lyrics_to_text`.
