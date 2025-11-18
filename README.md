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
4. Craft three lightly varied MusicGen prompts.
5. Generate ~20 second clips and save them in `outputs/demos/`.

```bash
python scripts/generate_from_refs.py
```

The script automatically selects GPU acceleration if available via `torch.cuda.is_available()`. Review the printed prompts and generated WAV files to iterate on your reference set. Use different adjectives or edit the prompts in code if you want more variations.*** End Patch
