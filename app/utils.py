"""
TalkForge — Shared utilities.
"""

import os
import uuid
import shutil
import subprocess
import mimetypes
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_AUDIO_EXT = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"}


def validate_image(path: str) -> None:
    """Raise ValueError if *path* is not a supported image file."""
    ext = Path(path).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXT:
        raise ValueError(
            f"Unsupported image format '{ext}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_IMAGE_EXT))}"
        )
    if not os.path.isfile(path):
        raise ValueError(f"Image file not found: {path}")


def validate_audio(path: str) -> None:
    """Raise ValueError if *path* is not a supported audio file."""
    ext = Path(path).suffix.lower()
    if ext not in ALLOWED_AUDIO_EXT:
        raise ValueError(
            f"Unsupported audio format '{ext}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXT))}"
        )
    if not os.path.isfile(path):
        raise ValueError(f"Audio file not found: {path}")


def unique_output_path(output_dir: str, prefix: str = "talkforge", ext: str = ".mp4") -> str:
    """Return a unique file path inside *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    name = f"{prefix}_{uuid.uuid4().hex[:8]}{ext}"
    return str(Path(output_dir) / name)


def safe_copy(src: str, dst_dir: str, new_name: Optional[str] = None) -> str:
    """
    Copy *src* into *dst_dir*.  If *new_name* is None, a UUID-based name is
    used to avoid collisions.  Returns the destination path.
    """
    os.makedirs(dst_dir, exist_ok=True)
    ext = Path(src).suffix
    name = new_name or f"{uuid.uuid4().hex}{ext}"
    dst = os.path.join(dst_dir, name)
    shutil.copy2(src, dst)
    return dst


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def convert_audio_to_wav(src: str, dst_dir: str) -> str:
    """
    Convert any supported audio to a 16-bit 16 kHz mono WAV using FFmpeg.
    Returns the path to the new WAV file.
    """
    dst = unique_output_path(dst_dir, prefix="audio", ext=".wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", src,
        "-ar", "16000",
        "-ac", "1",
        "-sample_fmt", "s16",
        dst,
    ]
    _run(cmd, "Audio conversion failed")
    return dst


def get_audio_duration(path: str) -> float:
    """Return duration of audio file in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def resize_image_if_needed(src: str, dst_dir: str, max_side: int = 512) -> str:
    """
    If the longest side of the image exceeds *max_side*, resize with FFmpeg
    while preserving aspect ratio.  Returns the (possibly new) image path.
    """
    try:
        from PIL import Image
        img = Image.open(src)
        w, h = img.size
        if max(w, h) <= max_side:
            return src  # no resize needed
        img.close()
    except Exception:
        return src  # PIL unavailable — pass through unchanged

    dst = unique_output_path(dst_dir, prefix="resized", ext=Path(src).suffix or ".png")
    scale = f"iw*min(1\\,{max_side}/max(iw\\,ih)):-2"
    cmd = ["ffmpeg", "-y", "-i", src, "-vf", f"scale={scale}", dst]
    _run(cmd, "Image resize failed")
    return dst


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _run(cmd: list, error_prefix: str = "Command failed") -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{error_prefix}.\nCommand: {' '.join(cmd)}\n"
            f"stderr: {result.stderr[-2000:]}"
        )
