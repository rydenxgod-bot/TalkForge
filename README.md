# TalkForge

> Upload a portrait. Add audio. Get a talking-head video with accurate lip sync — in one click.

TalkForge is an open-source, single-feature V1 tool for generating lip-synced talking-head videos from a static portrait image and an audio file. It runs as a local Gradio web app or entirely inside Google Colab — no account, no subscription, no cloud upload required.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-yellow.svg)](https://www.python.org)
[![Powered by SadTalker](https://img.shields.io/badge/Model-SadTalker-purple.svg)](https://github.com/OpenTalker/SadTalker)

---

## Features

- **One-click generation** — upload an image, upload audio, press Generate
- **Live status updates** — watch each pipeline stage in real time
- **In-browser video preview** — the finished MP4 plays directly in the page
- **Download / Discard / Generate New** — full workflow in a single screen
- **Swappable model backend** — SadTalker is the default; replacing it is a single-file change (see [Swapping the Model](#swapping-the-model))
- **Google Colab ready** — `TalkForge.ipynb` sets up everything automatically, no configuration needed
- **MIT licensed** — use it, fork it, build on it

---

## How It Works

```
Portrait image  ─┐
                  ├──► SadTalker (lip sync + head motion) ──► GFPGAN (enhance) ──► MP4
Audio file      ─┘
```

SadTalker reconstructs a 3-D face from the portrait, drives it with the audio's mel-spectrogram, and renders frames with natural head pose and blink motion — not just mouth movement.

---

## Project Structure

```
TalkForge/
├── app/
│   ├── interface.py      ← Gradio UI (layout, events, CSS)
│   ├── pipeline.py       ← Generation pipeline (validate → pre-process → model → output)
│   └── utils.py          ← File helpers, audio/image utilities
├── models/
│   ├── base.py           ← Abstract BaseLipSyncModel (defines the swap interface)
│   ├── sadtalker.py      ← SadTalker backend wrapper
│   └── SadTalker/        ← Cloned at runtime (git-ignored)
├── weights/              ← Downloaded model checkpoints (git-ignored)
│   ├── checkpoints/      ← SadTalker .safetensors / .pth.tar + BFM_Fitting/
│   └── gfpgan/weights/   ← GFPGANv1.4.pth
├── outputs/              ← Generated MP4 files (git-ignored)
├── examples/             ← Drop test portrait + audio here
├── main.py               ← CLI entry point
├── setup_weights.py      ← Pre-download all model weights
├── requirements.txt      ← Python dependencies
├── TalkForge.ipynb       ← Google Colab one-click notebook
└── LICENSE               ← MIT
```

---

## Hardware Requirements

| | Minimum | Recommended |
|---|---------|-------------|
| GPU VRAM | 4 GB | 8 GB+ |
| RAM | 8 GB | 16 GB+ |
| Storage | 8 GB free | 12 GB free |
| GPU | NVIDIA (CUDA 11.8+) | NVIDIA RTX 3060+ |

> **CPU is supported** but generation takes several minutes per clip instead of ~30 seconds on GPU.

---

## Running on Google Colab  *(Easiest — recommended for beginners)*

Google Colab provides a free NVIDIA T4 GPU — ideal for TalkForge.

### Step 1 — Open the notebook

Go to [colab.research.google.com](https://colab.research.google.com) → **File → Open notebook → Upload** → select `TalkForge.ipynb`.

### Step 2 — Enable GPU runtime

```
Runtime → Change runtime type → T4 GPU → Save
```

### Step 3 — Run all cells

Click **Runtime → Run all** (or press `Ctrl+F9`) and wait.

The notebook will automatically:
- Install FFmpeg, cmake, and build tools
- Clone the TalkForge repository
- Clone SadTalker source code
- Install all Python packages (including dlib, compiled from source)
- Download all model weights (~3.2 GB total)
- Verify the setup
- Launch the Gradio app

### Step 4 — Open the public URL

At the end of the last cell, look for a line like:

```
Running on public URL: https://xxxxxxxxxxxx.gradio.live
```

Click it — TalkForge opens in your browser instantly.

### Step 5 — Generate your first video

1. **Upload a portrait** — JPG, PNG, or WEBP. Use a clear, front-facing photo.
2. **Upload audio** — WAV, MP3, M4A, or OGG.
3. Click **✦ Generate Video**.
4. Watch the live status: *Preparing model → Loading weights → Processing audio → Generating frames → Rendering video → Finalizing output*.
5. When done, the video plays automatically in the page.
6. Use **⬇ Download Video** to save it, **✕ Discard** to delete it, or **↺ Generate New** to start over.

> **Important:** Free Colab sessions are temporary. Download your generated videos before the session ends. They are in `outputs/` in the Colab file browser.

---

## Running Locally

### Prerequisites

- Python 3.9 – 3.11
- Git
- FFmpeg

**Install FFmpeg:**
```bash
# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg -y

# macOS (Homebrew)
brew install ffmpeg

# Windows (winget)
winget install -e --id Gyan.FFmpeg
```

### Setup

```bash
# 1. Clone TalkForge
git clone https://github.com/your-username/TalkForge.git
cd TalkForge

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Clone SadTalker source
git clone --depth 1 https://github.com/OpenTalker/SadTalker.git models/SadTalker

# 5. Install SadTalker's own requirements
pip install -r models/SadTalker/requirements.txt

# 6. Download all model weights (~3.2 GB)
python setup_weights.py

# 7. Launch TalkForge
python main.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

### CLI flags

```bash
# Create a public shareable Gradio URL
python main.py --share

# Run on a different port
python main.py --port 8080

# Use a custom output directory
python main.py --output-dir /data/outputs

# Use a custom weights directory
python main.py --weights-dir /data/weights

# All options
python main.py --help
```

---

## Swapping the Model

The model backend is a plug-in. To replace SadTalker with MuseTalk, Wav2Lip, or any other model:

1. Create `models/my_model.py` and subclass `BaseLipSyncModel` from `models/base.py`
2. Implement the three required methods:
   - `is_ready() -> bool`
   - `download_weights(progress_cb=None) -> None`
   - `generate(image_path, audio_path, output_path, progress_cb=None) -> str`
3. Change one import in `app/pipeline.py` (line ~16):
   ```python
   # Before:
   from models.sadtalker import SadTalkerModel as _LipSyncModel
   # After:
   from models.my_model import MyModel as _LipSyncModel
   ```

No frontend changes required — the UI and pipeline are model-agnostic.

---

## Troubleshooting

### "dlib fails to install"

```bash
sudo apt install cmake build-essential libopenblas-dev liblapack-dev
pip install dlib
```

### "CUDA out of memory"

- Restart your runtime (Colab) or Python process
- Close other GPU applications
- The model runs on CPU automatically if CUDA is unavailable (slower but functional)

### "No module named 'basicsr' / 'gfpgan'"

```bash
pip install basicsr realesrgan gfpgan --upgrade
```

### "FFmpeg not found"

```bash
# Confirm FFmpeg is in PATH:
which ffmpeg      # Linux / macOS
where ffmpeg      # Windows

# If missing, install it (see Prerequisites above)
```

### "SadTalker weights download is very slow"

```bash
# Use the HuggingFace mirror:
export HF_ENDPOINT=https://hf-mirror.com
python setup_weights.py
```

### "Face not detected in image"

- Use a clear, front-facing portrait
- Ensure the face is well-lit and unobstructed
- Minimum recommended resolution: 256 × 256 px
- Avoid heavy occlusion (sunglasses, masks, hats covering the face)

### "Generated video has no audio"

FFmpeg is required for audio muxing. Verify it is installed and on your PATH:
```bash
ffprobe -version
```

### "Weight download fails halfway"

Re-run `python setup_weights.py` — it skips files already downloaded (checks file size).

---

## Roadmap

| Version | Features |
|---------|----------|
| **V1** *(current)* | Portrait + audio → lip-synced MP4 via SadTalker |
| **V1.1** | Batch processing, configurable output resolution, video-to-video driving |
| **V1.2** | MuseTalk backend option, faster inference, Docker image |
| **V2** | Emotion control, background replacement, multi-speaker |
| **V2+** | REST API, CLI tool, hosted demo |

---

## Contributing

Pull requests are welcome. For major changes, open an issue first to discuss.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit: `git commit -m 'Add my feature'`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## Acknowledgements

- [SadTalker](https://github.com/OpenTalker/SadTalker) — lip sync and 3-D head motion model
- [GFPGAN](https://github.com/TencentARC/GFPGAN) — face quality enhancement
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — background enhancement
- [Gradio](https://github.com/gradio-app/gradio) — web UI framework

---

## License

MIT — see [LICENSE](LICENSE) for full text.
