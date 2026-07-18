"""
TalkForge — SadTalker backend wrapper.

SadTalker (https://github.com/OpenTalker/SadTalker) is cloned automatically by
the Colab notebook into models/SadTalker/.  This wrapper drives inference via
SadTalker's CLI (inference.py).

Weight layout under <project_root>/weights/:
    weights/
      checkpoints/
        SadTalker_V0.0.2_256.safetensors   ← main model
        mapping_00229-model.pth.tar
        mapping_00109-model.pth.tar
        BFM_Fitting/                        ← 3-D morphable model sources
          01_MorphableModel.mat             ← used to auto-generate BFM_model_front.mat
          Exp_Pca.bin
          std_exp.txt
          similarity_Lm3D_all.mat
          BFM_exp_idx.mat
          BFM_front_idx.mat
          BFM09_model_info.mat
          facemodel_info.mat
          select_vertex_id.mat
      gfpgan/
        weights/
          GFPGANv1.4.pth
          alignment_WFLW_4HG.pth
          detection_Resnet50_Final.pth
          parsing_parsenet.pth

NOTE: BFM_model_front.mat is NOT downloaded — SadTalker generates it on first run
from 01_MorphableModel.mat and Exp_Pca.bin via transferBFM09().

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
# Verified download URLs (all HTTP 200 confirmed)
# ---------------------------------------------------------------------------

_GH  = "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc"
_HF  = "https://huggingface.co/vinthony/SadTalker/resolve/main"
_FX  = "https://github.com/xinntao/facexlib/releases/download"
_GFP = "https://github.com/TencentARC/GFPGAN/releases/download"

WEIGHT_URLS: dict = {
    # ── SadTalker main models (GitHub Releases) ──────────────────────────
    "weights/checkpoints/SadTalker_V0.0.2_256.safetensors": f"{_GH}/SadTalker_V0.0.2_256.safetensors",
    "weights/checkpoints/mapping_00229-model.pth.tar":       f"{_GH}/mapping_00229-model.pth.tar",
    "weights/checkpoints/mapping_00109-model.pth.tar":       f"{_GH}/mapping_00109-model.pth.tar",
    # ── BFM Fitting source files (HuggingFace) ───────────────────────────
    # BFM_model_front.mat is NOT here; SadTalker generates it from these:
    "weights/checkpoints/BFM_Fitting/01_MorphableModel.mat":   f"{_HF}/BFM_Fitting/01_MorphableModel.mat",
    "weights/checkpoints/BFM_Fitting/Exp_Pca.bin":             f"{_HF}/BFM_Fitting/Exp_Pca.bin",
    "weights/checkpoints/BFM_Fitting/std_exp.txt":             f"{_HF}/BFM_Fitting/std_exp.txt",
    "weights/checkpoints/BFM_Fitting/similarity_Lm3D_all.mat": f"{_HF}/BFM_Fitting/similarity_Lm3D_all.mat",
    "weights/checkpoints/BFM_Fitting/BFM_exp_idx.mat":         f"{_HF}/BFM_Fitting/BFM_exp_idx.mat",
    "weights/checkpoints/BFM_Fitting/BFM_front_idx.mat":       f"{_HF}/BFM_Fitting/BFM_front_idx.mat",
    "weights/checkpoints/BFM_Fitting/BFM09_model_info.mat":    f"{_HF}/BFM_Fitting/BFM09_model_info.mat",
    "weights/checkpoints/BFM_Fitting/facemodel_info.mat":      f"{_HF}/BFM_Fitting/facemodel_info.mat",
    "weights/checkpoints/BFM_Fitting/select_vertex_id.mat":    f"{_HF}/BFM_Fitting/select_vertex_id.mat",
    # ── GFPGAN / facexlib enhancement models (GitHub Releases) ───────────
    "weights/gfpgan/weights/GFPGANv1.4.pth":             f"{_GFP}/v1.3.0/GFPGANv1.4.pth",
    "weights/gfpgan/weights/alignment_WFLW_4HG.pth":     f"{_FX}/v0.1.0/alignment_WFLW_4HG.pth",
    "weights/gfpgan/weights/detection_Resnet50_Final.pth":f"{_FX}/v0.1.0/detection_Resnet50_Final.pth",
    "weights/gfpgan/weights/parsing_parsenet.pth":        f"{_FX}/v0.2.2/parsing_parsenet.pth",
}

# Minimum required subset that must be present for is_ready() → True
REQUIRED_FILES = list(WEIGHT_URLS.keys())

# SadTalker source clone location (relative to project root)
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
        """Clone SadTalker source and download all missing weight files."""

        def _log(msg: str) -> None:
            if progress_cb:
                progress_cb(msg)
            print(msg, flush=True)

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

        # 2. Download each missing weight file
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
        image_path    = str(Path(image_path).resolve())
        audio_path    = str(Path(audio_path).resolve())
        output_path   = str(Path(output_path).resolve())
        sadtalker_dir = str(self._sadtalker_src.resolve())

        # SadTalker expects:
        #   --checkpoint_dir  → directory containing .safetensors and .pth.tar files
        #   --bfm_folder      → directory containing BFM_Fitting source files
        checkpoint_dir = str((self._root / "weights" / "checkpoints").resolve())
        bfm_folder     = str((self._root / "weights" / "checkpoints" / "BFM_Fitting").resolve())

        # SadTalker searches for gfpgan enhancement models relative to CWD:
        #   {cwd}/gfpgan/weights/GFPGANv1.4.pth  etc.
        # We symlink (or copy) our pre-downloaded weights into the SadTalker
        # working directory so the enhancer finds them without network access.
        self._link_gfpgan_weights(sadtalker_dir)

        _cb("Preparing model…")

        # ── Build environment ───────────────────────────────────────────
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{sadtalker_dir}:{existing}" if existing else sadtalker_dir
        # Tell facexlib / GFPGAN where to find pretrained models
        gfpgan_dir = str((self._root / "weights" / "gfpgan" / "weights").resolve())
        env["FACEXLIB_MODEL_PATH"] = gfpgan_dir
        env["GFPGAN_MODEL_PATH"]   = gfpgan_dir

        _cb("Loading weights…")

        with tempfile.TemporaryDirectory() as tmp_out:
            cmd = [
                sys.executable,
                os.path.join(sadtalker_dir, "inference.py"),
                "--driven_audio",   audio_path,
                "--source_image",   image_path,
                "--result_dir",     tmp_out,
                "--checkpoint_dir", checkpoint_dir,
                "--bfm_folder",     bfm_folder,
                "--enhancer",       "gfpgan",
                "--device",         self.device if self.device != "cpu" else "cpu",
                "--preprocess",     "crop",
                "--size",           "256",
                "--still",
            ]
            # SadTalker uses --cpu flag (not --device cpu)
            if self.device == "cpu":
                cmd = [a for a in cmd if a not in ("--device", "cpu")]
                cmd.append("--cpu")

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

            # Stream stdout and forward key stage lines as progress updates
            _STAGE_KEYWORDS = {
                "3dmm":         "Extracting 3D face parameters…",
                "audio2":       "Processing audio features…",
                "coeff":        "Generating expression coefficients…",
                "render":       "Rendering frames…",
                "enhanc":       "Enhancing faces with GFPGAN…",
                "generat":      "Generating frames…",
                "loading":      "Loading weights…",
                "transferbfm":  "Building BFM face model (first run only)…",
                "transfer bfm": "Building BFM face model (first run only)…",
            }
            last_stage = ""
            stdout_lines: list = []

            assert proc.stdout is not None
            for line in proc.stdout:
                stdout_lines.append(line)
                low = line.lower()
                for kw, label in _STAGE_KEYWORDS.items():
                    if kw in low and label != last_stage:
                        _cb(label)
                        last_stage = label
                        break

            proc.wait()

            if proc.returncode != 0:
                tail = "".join(stdout_lines[-80:])
                raise RuntimeError(
                    f"SadTalker exited with code {proc.returncode}.\n"
                    f"Last output:\n{tail}"
                )

            _cb("Rendering video…")

            mp4s = sorted(Path(tmp_out).rglob("*.mp4"))
            if not mp4s:
                tail = "".join(stdout_lines[-40:])
                raise RuntimeError(
                    f"SadTalker finished but produced no MP4 in {tmp_out}.\n{tail}"
                )

            best = max(mp4s, key=lambda p: p.stat().st_mtime)

            _cb("Finalizing output…")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(best), output_path)

        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _link_gfpgan_weights(self, sadtalker_dir: str) -> None:
        """
        Ensure SadTalker's working directory contains gfpgan/weights/ pointing
        to our pre-downloaded files.  Uses symlinks where possible, falls back
        to hard copies.
        """
        src_dir = self._root / "weights" / "gfpgan" / "weights"
        dst_dir = Path(sadtalker_dir) / "gfpgan" / "weights"
        dst_dir.mkdir(parents=True, exist_ok=True)

        for f in src_dir.iterdir():
            dst = dst_dir / f.name
            if dst.exists() or dst.is_symlink():
                continue
            try:
                dst.symlink_to(f.resolve())
            except (OSError, NotImplementedError):
                shutil.copy2(str(f), str(dst))


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def _wget(url: str, dst: str) -> None:
    """Download *url* → *dst* using wget, curl, or Python urllib."""
    if shutil.which("wget"):
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", dst, url],
            check=True,
        )
    elif shutil.which("curl"):
        subprocess.run(
            ["curl", "-L", "--progress-bar", "-o", dst, url],
            check=True,
        )
    else:
        import urllib.request
        urllib.request.urlretrieve(url, dst)
