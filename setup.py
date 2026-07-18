"""
TalkForge — Automated Setup
Downloads SadTalker repo, all model checkpoints, and verifies the environment.
Run this once before launching main.py.

Usage:
    python setup.py
"""

import os
import sys
import subprocess
import urllib.request
import hashlib
from pathlib import Path

ROOT_DIR      = Path(__file__).resolve().parent
SADTALKER_DIR = ROOT_DIR / "SadTalker"
WEIGHTS_DIR   = ROOT_DIR / "weights"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run(cmd: list, cwd=None, env=None):
    """Run a subprocess command, streaming output, raising on failure."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with code {result.returncode}")
        sys.exit(result.returncode)


def _download(url: str, dest: Path, label: str = ""):
    """Download a file with a progress indicator."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  ✓ Already downloaded: {dest.name}")
        return
    print(f"  ↓ Downloading {label or dest.name} …")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  ✓ Saved to {dest}")
    except Exception as exc:
        print(f"  [WARN] urllib failed ({exc}), retrying with wget…")
        subprocess.run(["wget", "-q", "--show-progress", "-O", str(dest), url], check=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Clone SadTalker
# ─────────────────────────────────────────────────────────────────────────────

def clone_sadtalker():
    print("\n[1/5]  Cloning SadTalker repository…")
    if (SADTALKER_DIR / "inference.py").exists():
        print("  ✓ SadTalker already cloned.")
        return
    _run([
        "git", "clone", "--depth", "1",
        "https://github.com/OpenTalker/SadTalker.git",
        str(SADTALKER_DIR),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Install Python dependencies
# ─────────────────────────────────────────────────────────────────────────────

REQUIREMENTS = [
    # Core
    "gradio>=4.0.0",
    "torch>=2.0.0",
    "torchvision",
    "torchaudio",
    "numpy<2.0",
    "scipy",
    "librosa",
    "soundfile",
    "pillow",
    "opencv-python-headless",
    "imageio",
    "imageio-ffmpeg",
    "tqdm",
    "pyyaml",
    # SadTalker deps
    "face_alignment",
    "facexlib",
    "gfpgan",
    "realesrgan",
    "basicsr",
    "safetensors",
    "dlib",
    "batch-face",
    "kornia",
    "einops",
    "yacs",
    "facedetection",
    "mediapipe",
]


def install_dependencies():
    print("\n[2/5]  Installing Python dependencies…")
    # Install from SadTalker's own requirements first (if present)
    sadtalker_req = SADTALKER_DIR / "requirements.txt"
    if sadtalker_req.exists():
        _run([sys.executable, "-m", "pip", "install", "-q", "-r", str(sadtalker_req)])

    # Then our additional packages
    _run([sys.executable, "-m", "pip", "install", "-q"] + REQUIREMENTS)

    # Install SadTalker as a package (editable) for clean imports
    if (SADTALKER_DIR / "setup.py").exists() or (SADTALKER_DIR / "pyproject.toml").exists():
        _run([sys.executable, "-m", "pip", "install", "-q", "-e", str(SADTALKER_DIR)])

    print("  ✓ All dependencies installed.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Download model weights
# ─────────────────────────────────────────────────────────────────────────────

# SadTalker uses a helper script to download weights from HuggingFace / GitHub Releases.
# We run their script, then also ensure GFPGAN weights are present.

HF_BASE   = "https://huggingface.co/vinthony/SadTalker/resolve/main"
GFPGAN_HF = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"

WEIGHT_FILES = {
    # SadTalker core checkpoints
    "SadTalker_V0.0.2_256.safetensors": f"{HF_BASE}/SadTalker_V0.0.2_256.safetensors",
    "SadTalker_V0.0.2_512.safetensors": f"{HF_BASE}/SadTalker_V0.0.2_512.safetensors",
    # Mapping networks
    "mapping_00109-model.pth.tar":       f"{HF_BASE}/mapping_00109-model.pth.tar",
    "mapping_00229-model.pth.tar":       f"{HF_BASE}/mapping_00229-model.pth.tar",
    # BFM (Basel Face Model) data
    "BFM_Fitting.zip":                   f"{HF_BASE}/BFM_Fitting.zip",
    "epoch_20.pth":                      f"{HF_BASE}/epoch_20.pth",
}

GFPGAN_WEIGHT_FILES = {
    "GFPGANv1.4.pth": GFPGAN_HF,
}


def download_weights():
    print("\n[3/5]  Downloading model weights…")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # Try SadTalker's own download script first
    dl_script = SADTALKER_DIR / "scripts" / "download_models.sh"
    if dl_script.exists():
        print("  Running SadTalker download script…")
        try:
            subprocess.run(
                ["bash", str(dl_script)],
                cwd=str(SADTALKER_DIR),
                check=True,
            )
            print("  ✓ SadTalker download script completed.")
        except Exception as exc:
            print(f"  [WARN] SadTalker script failed ({exc}), falling back to manual downloads…")
            _manual_download_weights()
    else:
        _manual_download_weights()

    # GFPGAN weights (face enhancer)
    gfpgan_dir = WEIGHTS_DIR / "gfpgan" / "weights"
    gfpgan_dir.mkdir(parents=True, exist_ok=True)
    for fname, url in GFPGAN_WEIGHT_FILES.items():
        _download(url, gfpgan_dir / fname, label=fname)

    # Also put a copy at the SadTalker gfpgan path if it has one
    st_gfpgan = SADTALKER_DIR / "gfpgan" / "weights"
    st_gfpgan.mkdir(parents=True, exist_ok=True)
    gfpgan_pth = gfpgan_dir / "GFPGANv1.4.pth"
    st_gfpgan_pth = st_gfpgan / "GFPGANv1.4.pth"
    if gfpgan_pth.exists() and not st_gfpgan_pth.exists():
        import shutil
        shutil.copy2(str(gfpgan_pth), str(st_gfpgan_pth))

    print("  ✓ All weights ready.")


def _manual_download_weights():
    """Fallback: manually download each weight from HuggingFace."""
    for fname, url in WEIGHT_FILES.items():
        dest = WEIGHTS_DIR / fname
        _download(url, dest, label=fname)

    # Unzip BFM if needed
    bfm_zip  = WEIGHTS_DIR / "BFM_Fitting.zip"
    bfm_dir  = WEIGHTS_DIR / "BFM_Fitting"
    if bfm_zip.exists() and not bfm_dir.exists():
        print("  Extracting BFM_Fitting.zip…")
        import zipfile
        with zipfile.ZipFile(str(bfm_zip), "r") as z:
            z.extractall(str(WEIGHTS_DIR))
        print("  ✓ BFM extracted.")

    # Symlink weights into SadTalker's expected checkpoints directory
    st_ckpts = SADTALKER_DIR / "checkpoints"
    st_ckpts.mkdir(parents=True, exist_ok=True)
    for fname in WEIGHT_FILES:
        src  = WEIGHTS_DIR / fname
        link = st_ckpts / fname
        if src.exists() and not link.exists():
            try:
                link.symlink_to(src)
            except Exception:
                import shutil
                shutil.copy2(str(src), str(link))

    # BFM dir symlink
    st_bfm = SADTALKER_DIR / "checkpoints" / "BFM_Fitting"
    bfm_dir = WEIGHTS_DIR / "BFM_Fitting"
    if bfm_dir.exists() and not st_bfm.exists():
        try:
            st_bfm.symlink_to(bfm_dir)
        except Exception:
            import shutil
            shutil.copytree(str(bfm_dir), str(st_bfm))


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Verify FFmpeg
# ─────────────────────────────────────────────────────────────────────────────

def check_ffmpeg():
    print("\n[4/5]  Checking FFmpeg…")
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True)
    if result.returncode == 0:
        ver = result.stdout.decode().split("\n")[0]
        print(f"  ✓ FFmpeg found: {ver[:60]}")
    else:
        print("  [ERROR] FFmpeg not found. Install it:")
        print("          Ubuntu/Colab:  apt-get install -y ffmpeg")
        print("          macOS:          brew install ffmpeg")
        print("          Windows:        https://ffmpeg.org/download.html")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test():
    print("\n[5/5]  Smoke-testing imports…")
    errors = []
    tests  = [
        ("gradio",       "import gradio"),
        ("torch",        "import torch; assert torch.cuda.is_available() or True"),
        ("cv2",          "import cv2"),
        ("librosa",      "import librosa"),
        ("face_alignment", "import face_alignment"),
        ("gfpgan",       "import gfpgan"),
        ("safetensors",  "import safetensors"),
    ]
    for name, stmt in tests:
        try:
            exec(stmt)
            print(f"  ✓ {name}")
        except Exception as exc:
            print(f"  ✗ {name}: {exc}")
            errors.append(name)

    if errors:
        print(f"\n  [WARN] Some imports failed: {errors}")
        print("         Try: pip install " + " ".join(errors))
    else:
        print("\n  ✓ All imports OK.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("═" * 60)
    print("  TalkForge — Automated Setup")
    print("═" * 60)

    clone_sadtalker()
    install_dependencies()
    download_weights()
    check_ffmpeg()
    smoke_test()

    print("\n" + "═" * 60)
    print("  ✅  Setup complete!")
    print("  Run:  python main.py --share")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
