# ⚡ TalkForge

**Upload a portrait. Upload audio. Get a talking video.**

TalkForge is a free, open-source, single-feature AI tool that turns any front-facing portrait image and audio clip into a lip-synced talking-head MP4. Powered by [SadTalker](https://github.com/OpenTalker/SadTalker) and served through a clean Gradio web interface.

No accounts. No payments. No hidden APIs. One button.

---

## ✨ Features

- **Upload & Generate** — Drop in a JPG/PNG portrait and a WAV/MP3 audio file, click one button
- **Live progress messages** — Preparing Model → Loading Weights → Processing Audio → Generating Frames → Rendering Video → Finalising Output
- **Auto-preview** — Generated MP4 plays inside the page immediately
- **Download / Discard / Generate New** — Three action buttons for post-generation workflow
- **Full reset** — "Generate New" clears everything so you can start fresh instantly
- **Google Colab ready** — Open notebook, run all, get a public URL. No setup required

---

## 🗂 Project Structure

```
TalkForge/
├── app/
│   ├── __init__.py
│   ├── inference.py       # Model wrapper (swap model here — no frontend changes needed)
│   └── ui.py              # Gradio UI (all visual logic lives here)
├── SadTalker/             # Auto-cloned by setup.py / Colab notebook
├── weights/               # Auto-downloaded model checkpoints
├── outputs/               # Generated MP4 files saved here
├── examples/              # Sample portrait + audio for testing
├── static/                # Reserved for future static assets
├── main.py                # Entry point: python main.py --share
├── setup.py               # One-shot automated setup script
├── requirements.txt
├── TalkForge.ipynb        # Google Colab notebook (run all → done)
└── README.md
```

---

## ⚙️ Hardware Requirements

| Environment | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA 6 GB VRAM | NVIDIA 8 GB+ VRAM (T4 / A100) |
| RAM | 8 GB | 16 GB |
| Disk | 8 GB free | 12 GB free |
| Python | 3.9+ | 3.10 |

> CPU-only is technically possible but will be **very slow** (10–30 min per video). A GPU is strongly recommended.

---

## 🚀 Google Colab (Beginner-Friendly — Recommended)

This is the easiest way to run TalkForge with zero local setup.

### Step-by-step

**1. Open the notebook**

Go to [Google Colab](https://colab.research.google.com) and upload `TalkForge.ipynb`, or open it directly from GitHub.

**2. Select a GPU runtime**

```
Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save
```

**3. Run all cells**

```
Runtime → Run all
```

**4. Wait (~5–10 minutes on first run)**

Colab will automatically:
- Install FFmpeg and system libraries
- Clone TalkForge and SadTalker
- Install all Python packages
- Download all AI model weights from HuggingFace
- Launch the Gradio web interface

**5. Click the public URL**

When the last cell finishes, you will see output like:

```
Running on public URL: https://xxxxxxxxxxxx.gradio.live
```

Click it. TalkForge is live and ready to use.

> **Subsequent runs are faster.** Colab caches downloaded weights in the session. If you close and reopen, weights download again (about 2–3 minutes).

---

## 💻 Local Setup

### Prerequisites

- Python 3.9 or 3.10
- NVIDIA GPU with CUDA (strongly recommended)
- Git
- FFmpeg installed and on your PATH

**Install FFmpeg:**
```bash
# Ubuntu / Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add to PATH
```

### Install

```bash
# 1. Clone TalkForge
git clone https://github.com/rydenxgod-bot/TalkForge.git
cd TalkForge

# 2. (Optional but recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Run automated setup
#    This clones SadTalker, installs all dependencies, and downloads weights
python setup.py
```

### Launch

```bash
# With a public share link (accessible from any browser)
python main.py --share

# Local only
python main.py

# Custom port
python main.py --port 7861
```

Open `http://localhost:7860` (or the share URL printed in the terminal).

---

## 🎬 How to Use

1. **Upload a portrait image** — JPG or PNG, front-facing, clear face. One person in frame works best.
2. **Upload an audio file** — WAV or MP3. Clear speech. Any length.
3. **Click Generate Video** — Progress messages update as the model processes.
4. **Preview your video** — It plays automatically when ready.
5. **Download, Discard, or Generate New** — Use the three buttons below the video.

### Tips for best results

- Use a well-lit, front-facing portrait photo
- A plain or simple background gives cleaner output
- Higher quality audio = better lip sync
- MP3 is fine; WAV gives marginally better audio extraction
- Avoid very dark or heavily filtered images

---

## 🔄 Swapping the Model

TalkForge is designed so the AI model can be replaced without touching the frontend.

All inference logic lives in `app/inference.py`. The UI calls one function:

```python
generate_talking_head(image_path, audio_path, progress_callback) -> str
```

To swap SadTalker for MuseTalk, Wav2Lip, or any other model:
1. Edit `app/inference.py`
2. Keep the same function signature
3. Return the path to the generated MP4

The frontend (`app/ui.py`) does not need to change.

---

## 🛠 Troubleshooting

**"CUDA not available" or very slow generation**
→ Make sure your runtime is set to T4 GPU (Colab: Runtime → Change runtime type).

**"SadTalker not found" error**
→ Re-run Cell 3 of the notebook, or run `python setup.py` locally.

**"Weights not found" error**
→ Re-run Cells 5 and 6 of the notebook, or run `python setup.py` locally.

**FFmpeg not found**
→ Install FFmpeg: `apt-get install ffmpeg` (Linux/Colab) or `brew install ffmpeg` (macOS).

**No face detected in image**
→ Use a clearer, front-facing portrait. The model needs to see a face clearly.

**Generation hangs or times out**
→ This can happen with very long audio. Try a shorter clip (under 60 seconds) first.

**Gradio share link not working**
→ Gradio share links expire after 72 hours. Re-run Cell 8 to get a fresh link.

**Import errors after install**
→ Restart the Colab runtime (Runtime → Restart runtime) and re-run from Cell 4.

---

## 🗺 Roadmap

**V1 (current)**
- Single image + audio → lip-synced MP4
- Gradio web UI with progress states
- Google Colab notebook, zero-config

**V2 (planned)**
- Video input support (animate a reference video)
- Resolution selector (256px / 512px)
- Head motion intensity slider
- Pose style presets (still / subtle / expressive)
- Batch processing for multiple files

**V3 (future)**
- MuseTalk integration as alternative model
- Real-time streaming preview
- Docker image for one-command local deploy
- Optional voice cloning pipeline

---

## 📄 License

MIT License — free to use, modify, and distribute. See [LICENSE](LICENSE).

---

## 🙏 Acknowledgements

- [SadTalker](https://github.com/OpenTalker/SadTalker) — Core talking-head synthesis model
- [GFPGAN](https://github.com/TencentARC/GFPGAN) — Face enhancement
- [Gradio](https://gradio.app) — Web UI framework
- [HuggingFace](https://huggingface.co) — Model hosting

---

Made with ⚡ by [Vaexor](https://vaexor.netlify.app) · Open Source · MIT
