"""
TalkForge - Inference Pipeline
Wraps SadTalker to generate lip-synced talking-head video from image + audio.
Swapping the model later only requires editing this file.
"""

import os
import sys
import uuid
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parent.parent
SADTALKER_DIR = ROOT_DIR / "SadTalker"
WEIGHTS_DIR   = ROOT_DIR / "weights"
OUTPUTS_DIR   = ROOT_DIR / "outputs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _check_sadtalker():
    """Ensure SadTalker repo and weights are present."""
    if not SADTALKER_DIR.exists():
        raise RuntimeError(
            "SadTalker not found. Please run the Colab notebook (or setup.sh) first."
        )
    checkpoint = WEIGHTS_DIR / "SadTalker_V0.0.2_256.safetensors"
    if not checkpoint.exists():
        raise RuntimeError(
            "SadTalker weights not found. Please run the Colab notebook (or setup.sh) first."
        )


def generate_talking_head(
    image_path: str,
    audio_path: str,
    progress_callback=None,
) -> str:
    """
    Run SadTalker inference.

    Parameters
    ----------
    image_path      : Path to the portrait image (jpg/png).
    audio_path      : Path to the audio file (wav/mp3).
    progress_callback: Optional callable(message: str) for status updates.

    Returns
    -------
    str : Absolute path to the generated MP4 file.
    """

    def _progress(msg: str):
        if progress_callback:
            progress_callback(msg)
        print(f"[TalkForge] {msg}")

    _progress("Checking model weights…")
    _check_sadtalker()

    # Unique output directory per run
    run_id     = uuid.uuid4().hex[:8]
    output_dir = OUTPUTS_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    _progress("Preparing input files…")

    # Copy inputs to a temp location to avoid path issues
    img_ext  = Path(image_path).suffix.lower() or ".png"
    aud_ext  = Path(audio_path).suffix.lower() or ".wav"
    tmp_img  = output_dir / f"input{img_ext}"
    tmp_aud  = output_dir / f"input{aud_ext}"
    shutil.copy2(image_path, tmp_img)
    shutil.copy2(audio_path, tmp_aud)

    _progress("Loading model & weights…")

    # ── SadTalker CLI command ──────────────────────────────────────────────
    cmd = [
        sys.executable,
        str(SADTALKER_DIR / "inference.py"),
        "--driven_audio",   str(tmp_aud),
        "--source_image",   str(tmp_img),
        "--result_dir",     str(output_dir),
        "--checkpoint_dir", str(WEIGHTS_DIR),
        "--still",               # minimal head motion (stable for portraits)
        "--preprocess",  "full", # process the full image (not just face crop)
        "--enhancer",    "gfpgan", # face enhancement for sharper output
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SADTALKER_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    _progress("Processing audio & generating lip sync frames…")

    proc = subprocess.Popen(
        cmd,
        cwd=str(SADTALKER_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Stream output so we can emit progress messages
    stage_map = {
        "3dmm":       "Extracting 3D face model…",
        "audio":      "Analysing audio coefficients…",
        "face":       "Detecting & aligning face…",
        "render":     "Rendering animated frames…",
        "enhancer":   "Enhancing face quality…",
        "ffmpeg":     "Compositing final video…",
        "saving":     "Saving output…",
        "result":     "Finalising output…",
    }

    for line in proc.stdout:
        line_lower = line.lower()
        for keyword, message in stage_map.items():
            if keyword in line_lower:
                _progress(message)
                break

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"SadTalker exited with code {proc.returncode}. "
            "Check that all weights are present and FFmpeg is installed."
        )

    _progress("Finalising output…")

    # SadTalker writes the MP4 into output_dir; find it
    mp4_files = sorted(output_dir.glob("*.mp4"))
    if not mp4_files:
        raise RuntimeError(
            "SadTalker finished but no MP4 was found in the output directory."
        )

    result_path = mp4_files[-1]
    final_path  = OUTPUTS_DIR / f"talkforge_{run_id}.mp4"
    shutil.move(str(result_path), str(final_path))

    # Clean up the per-run temp directory
    shutil.rmtree(output_dir, ignore_errors=True)

    _progress("Done!")
    return str(final_path)
