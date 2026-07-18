"""
TalkForge — SadTalker backend wrapper.

SadTalker (https://github.com/OpenTalker/SadTalker) is cloned automatically by
the Colab notebook into models/SadTalker/.  This wrapper drives inference via
SadTalker's CLI (inference.py), which is the most stable integration surface.

Directory layout expected under <project_root>/weights/:
    weights/
      checkpoints/                   ← SadTalker model files
        SadTalker_V0.0.2_256.safetensors
        mapping_00229-model.pth.tar
        mapping_00109-model.pth.tar
        BFM_Fitting/                 ← 3-D morphable model data
          BFM_model_front.mat
          BFM_model_back.mat
          similarity_Lm3D_all.mat
          std_exp.txt
      gfpgan/
        weights/
          GFPGANv1.4.pth            ← face enhancer

To swap this model for another (MuseTalk, Wav2Lip, …):
  1. Create models/my_model.py that subclasses BaseLipSyncModel.
  2. Implement is_ready(), download_weights(), generate().
  3. Change the import in app/pipeline.py.  No frontend changes needed.
"""

import os
import sys
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Optional

from models.base import BaseLipSyncModel

# ---------------------------------------------------------------------------
# Expected weight layout (relative to project root)
# ---------------------------------------------------------------------------

_CKPT_FILES = [
    "weights/checkpoints/SadTalker_V0.0.2_256.safetensors",
    "weights/checkpoints/mapping_00229-model.pth.tar",
    "weights/checkpoints/mapping_00109-model.pth.tar",
]
_BFM_FILES = [
    "weights/checkpoints/BFM_Fitting/BFM_model_front.mat",
    "weights/checkpoints/BFM_Fitting/BFM_model_back.mat",
    "weights/checkpoints/BFM_Fitting/similarity_Lm3D_all.mat",
    "weights/checkpoints/BFM_Fitting/std_exp.txt",
]
_GFPGAN_FILES = [
    "weights/gfpgan/weights/GFPGANv1.4.pth",
]

REQUIRED_FILES = _CKPT_FILES + _BFM_FILES + _GFPGAN_FILES

# Public download URLs (HuggingFace / GitHub Releases)
_HF = "https://huggingface.co/vinthony/SadTalker-V0.0.2/resolve/main"
WEIGHT_URLS: dict[str, str] = {
    "weights/checkpoints/SadTalker_V0.0.2_256.safetensors": f"{_HF}/SadTalker_V0.0.2_256.safetensors",
    "weights/checkpoints/mapping_00229-model.pth.tar":       f"{_HF}/mapping_00229-model.pth.tar",
    "weights/checkpoints/mapping_00109-model.pth.tar":       f"{_HF}/mapping_00109-model.pth.tar",
    "weights/checkpoints/BFM_Fitting/BFM_model_front.mat":   f"{_HF}/BFM_Fitting/BFM_model_front.mat",
    "weights/checkpoints/BFM_Fitting/BFM_model_back.mat":    f"{_HF}/BFM_Fitting/BFM_model_back.mat",
    "weights/checkpoints/BFM_Fitting/similarity_Lm3D_all.mat": f"{_HF}/BFM_Fitting/similarity_Lm3D_all.mat",
    "weights/checkpoints/BFM_Fitting/std_exp.txt":            f"{_HF}/BFM_Fitting/std_exp.txt",
    "weights/gfpgan/weights/GFPGANv1.4.pth": (
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    ),
}

# SadTalker source location (relative to project root)
SADTALKER_SRC = Path("models/SadTalker")


class SadTalkerModel(BaseLipSyncModel):
    """Drives SadTalker inference from Python via subprocess."""

    def __init__(self, weights_dir: str = "weights", device: str = "auto"):
        super().__init__(weights_dir=weights_dir, device=device)
        # Project root = parent of the models/ package
        self._root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._sadtalker_src = self._root / SADTALKER_SRC

    # ------------------------------------------------------------------
    # BaseLipSyncModel interface
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Return True when SadTalker source AND all weight files are present."""
        if not self._sadtalker_src.exists():
            return False
        for rel in REQUIRED_FILES:
            if not (self._root / rel).is_file():
                return False
        return True

    def download_weights(self, progress_cb: Optional[Callable[[str], None]] = None) -> None:
        """Clone SadTalker and download any missing weight files."""

        def _log(msg: str) -> None:
            if progress_cb:
                progress_cb(msg)
            print(msg)

        # 1. Clone SadTalker source if missing
        if not self._sadtalker_src.exists():
            _log("Cloning SadTalker repository…")
            self._sadtalker_src.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/OpenTalker/SadTalker.git",
                 str(self._sadtalker_src)],
                check=True,
            )
            _log("✓ SadTalker source cloned")

        # 2. Download missing weight files
        for rel_path, url in WEIGHT_URLS.items():
            full = self._root / rel_path
            if full.is_file() and full.stat().st_size > 1024:
                continue
            _log(f"Downloading {Path(rel_path).name}…")
            full.parent.mkdir(parents=True, exist_ok=True)
            _wget(url, str(full))

        _log("✓ All weights ready")

    def generate(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Run SadTalker inference and return the path to the generated MP4."""

        def _cb(msg: str) -> None:
            if progress_cb:
                progress_cb(msg)

        # Auto-download if not ready
        if not self.is_ready():
            _cb("Downloading missing model weights…")
            self.download_weights(progress_cb=_cb)

        # ── Resolve absolute paths ──────────────────────────────────────
        image_path  = str(Path(image_path).resolve())
        audio_path  = str(Path(audio_path).resolve())
        output_path = str(Path(output_path).resolve())
        sadtalker_dir = str(self._sadtalker_src.resolve())

        # SadTalker CLI flags
        checkpoint_dir = str((self._root / "weights" / "checkpoints").resolve())
        bfm_folder     = str((self._root / "weights" / "checkpoints" / "BFM_Fitting").resolve())
        gfpgan_dir     = str((self._root / "weights" / "gfpgan" / "weights").resolve())

        _cb("Preparing model…")

        # Build environment — add SadTalker src + GFPGAN weights path
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{sadtalker_dir}:{existing}" if existing else sadtalker_dir
        env["GFPGAN_MODEL_PATH"] = gfpgan_dir

        _cb("Loading weights…")

        with tempfile.TemporaryDirectory() as tmp_out:
            # SadTalker writes outputs into --result_dir; we use a temp dir
            # and then move the file to the requested output_path.
            cmd = [
                sys.executable,
                os.path.join(sadtalker_dir, "inference.py"),
                "--driven_audio",  audio_path,
                "--source_image",  image_path,
                "--result_dir",    tmp_out,
                "--checkpoint_dir", checkpoint_dir,
                "--bfm_folder",    bfm_folder,
                "--gfpgan_path",   os.path.join(gfpgan_dir, "GFPGANv1.4.pth"),
                "--enhancer",      "gfpgan",
                "--device",        self.device,
                "--preprocess",    "crop",
                "--size",          "256",
                "--still",         # reduces excessive head motion for portrait shots
            ]

            _cb("Processing audio…")

            proc = subprocess.Popen(
                cmd,
                cwd=sadtalker_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Stream stdout so we can forward key lines as progress updates
            stage_keywords = {
                "3DMM":          "Extracting 3D face parameters…",
                "audio2":        "Processing audio features…",
                "coeff":         "Generating expression coefficients…",
                "rendering":     "Rendering frames…",
                "enhancing":     "Enhancing faces (GFPGAN)…",
                "background":    "Enhancing background…",
                "concatenat":    "Finalizing video…",
                "generate":      "Generating frames…",
                "Loading":       "Loading weights…",
            }
            last_stage = ""
            stdout_lines: list[str] = []

            assert proc.stdout is not None
            for line in proc.stdout:
                stdout_lines.append(line)
                low = line.lower()
                for kw, label in stage_keywords.items():
                    if kw.lower() in low and label != last_stage:
                        _cb(label)
                        last_stage = label
                        break

            proc.wait()

            if proc.returncode != 0:
                tail = "".join(stdout_lines[-60:])
                raise RuntimeError(
                    f"SadTalker exited with code {proc.returncode}.\n"
                    f"Last output:\n{tail}"
                )

            _cb("Rendering video…")

            # Locate generated MP4 (SadTalker writes to tmp_out/<name>/<date>.mp4)
            mp4s = sorted(Path(tmp_out).rglob("*.mp4"))
            if not mp4s:
                tail = "".join(stdout_lines[-40:])
                raise RuntimeError(
                    f"SadTalker finished but no MP4 was found in {tmp_out}.\n"
                    f"Output:\n{tail}"
                )

            # Take the most recently modified mp4
            best = max(mp4s, key=lambda p: p.stat().st_mtime)

            _cb("Finalizing output…")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(best), output_path)

        return output_path


# ---------------------------------------------------------------------------
# Internal download helper
# ---------------------------------------------------------------------------

def _wget(url: str, dst: str) -> None:
    """Download *url* → *dst* using wget, curl, or Python urllib (in that order)."""
    if shutil.which("wget"):
        subprocess.run(["wget", "-q", "--show-progress", "-O", dst, url], check=True)
    elif shutil.which("curl"):
        subprocess.run(["curl", "-L", "--progress-bar", "-o", dst, url], check=True)
    else:
        import urllib.request
        urllib.request.urlretrieve(url, dst)
