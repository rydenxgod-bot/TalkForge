"""
TalkForge — Inference Pipeline
Wraps SadTalker to produce a lip-synced talking-head MP4 from image + audio.
To swap the model later, only this file needs to change.
"""

import os
import sys
import uuid
import shutil
import subprocess
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
    """Raise a clear error if SadTalker or weights are missing."""
    if not (SADTALKER_DIR / "inference.py").exists():
        raise RuntimeError(
            "SadTalker not found at: " + str(SADTALKER_DIR) +
            "\nPlease run the Colab notebook from the top."
        )
    checkpoint = WEIGHTS_DIR / "SadTalker_V0.0.2_256.safetensors"
    if not checkpoint.exists():
        # Also check SadTalker's own checkpoints dir
        alt = SADTALKER_DIR / "checkpoints" / "SadTalker_V0.0.2_256.safetensors"
        if not alt.exists():
            raise RuntimeError(
                "SadTalker weights not found.\n"
                "Please run Cells 6 and 7 of the Colab notebook."
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
    image_path       : Path to the portrait image (jpg/png).
    audio_path       : Path to the audio file (wav/mp3).
    progress_callback: Optional callable(message: str) for status updates.

    Returns
    -------
    str : Absolute path to the generated MP4.
    """

    def _p(msg: str):
        if progress_callback:
            progress_callback(msg)
        print(f"[TalkForge] {msg}")

    _p("Checking model weights…")
    _check_sadtalker()

    # Unique output dir per run
    run_id     = uuid.uuid4().hex[:8]
    output_dir = OUTPUTS_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    _p("Preparing input files…")

    img_ext = Path(image_path).suffix.lower() or ".png"
    aud_ext = Path(audio_path).suffix.lower() or ".wav"
    tmp_img = output_dir / f"input{img_ext}"
    tmp_aud = output_dir / f"input{aud_ext}"
    shutil.copy2(image_path, tmp_img)
    shutil.copy2(audio_path, tmp_aud)

    _p("Loading model & weights…")

    # Resolve checkpoint dir — prefer our weights/ dir, fall back to SadTalker/checkpoints/
    ckpt_dir = str(WEIGHTS_DIR)
    if not (WEIGHTS_DIR / "SadTalker_V0.0.2_256.safetensors").exists():
        ckpt_dir = str(SADTALKER_DIR / "checkpoints")

    # Build the SadTalker command
    cmd = [
        sys.executable,
        str(SADTALKER_DIR / "inference.py"),
        "--driven_audio",   str(tmp_aud),
        "--source_image",   str(tmp_img),
        "--result_dir",     str(output_dir),
        "--checkpoint_dir", ckpt_dir,
        "--still",                   # minimal head motion for portraits
        "--preprocess",  "full",     # use full image, not just face crop
        "--enhancer",    "gfpgan",   # face enhancement (sharpness)
    ]

    env = os.environ.copy()
    # Make sure SadTalker's own modules are importable
    sadtalker_str = str(SADTALKER_DIR)
    env["PYTHONPATH"] = sadtalker_str + os.pathsep + env.get("PYTHONPATH", "")

    _p("Processing audio & generating lip sync frames…")

    # Stage keywords SadTalker prints → friendly progress messages
    STAGE_MAP = {
        "3dmm":      "Extracting 3D face coefficients…",
        "audio":     "Analysing audio…",
        "face mesh": "Building face mesh…",
        "render":    "Rendering animated frames…",
        "gfpgan":    "Enhancing face quality…",
        "ffmpeg":    "Compositing final video…",
        "saving":    "Saving output…",
        "result":    "Finalising output…",
    }
    last_stage = ""

    proc = subprocess.Popen(
        cmd,
        cwd=str(SADTALKER_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in proc.stdout:
        line_lower = line.lower().strip()
        for keyword, message in STAGE_MAP.items():
            if keyword in line_lower and message != last_stage:
                _p(message)
                last_stage = message
                break

    proc.wait()

    if proc.returncode != 0:
        # Try without enhancer as a fallback (sometimes gfpgan weights cause issues)
        _p("Retrying without face enhancer…")
        cmd_noenhancer = [c for c in cmd if c != "gfpgan" and c != "--enhancer"]
        proc2 = subprocess.Popen(
            cmd_noenhancer,
            cwd=str(SADTALKER_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc2.stdout:
            pass  # drain output
        proc2.wait()
        if proc2.returncode != 0:
            raise RuntimeError(
                f"SadTalker failed (exit code {proc.returncode}). "
                "Check that all weights are present and the image has a clear front-facing face."
            )

    _p("Finalising output…")

    # SadTalker writes the MP4 inside output_dir — find it
    mp4_files = sorted(output_dir.glob("**/*.mp4"))
    if not mp4_files:
        raise RuntimeError(
            "SadTalker finished but no MP4 was produced. "
            "Make sure the portrait has a clear, front-facing face."
        )

    final_path = OUTPUTS_DIR / f"talkforge_{run_id}.mp4"
    shutil.move(str(mp4_files[-1]), str(final_path))
    shutil.rmtree(output_dir, ignore_errors=True)

    _p("Done!")
    return str(final_path)
