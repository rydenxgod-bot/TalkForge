"""
TalkForge — Standalone weight downloader.

Pre-downloads all required model checkpoints before launching the app.
Safe to re-run — already-downloaded files with correct size are skipped.

Weight sources (all verified HTTP 200):
  SadTalker models  → GitHub Releases OpenTalker/SadTalker@v0.0.2-rc
  BFM Fitting       → HuggingFace vinthony/SadTalker
  GFPGAN / facexlib → GitHub Releases TencentARC/GFPGAN, xinntao/facexlib

NOTE: BFM_model_front.mat is NOT downloaded here.
SadTalker generates it automatically on first inference run from
01_MorphableModel.mat + Exp_Pca.bin via its transferBFM09() function.

Usage:
    python setup_weights.py
    python setup_weights.py --weights-dir /path/to/weights
"""

import sys
import os
import argparse

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.sadtalker import SadTalkerModel, WEIGHT_URLS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download TalkForge model weights",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights-dir", default="weights",
                        help="Directory to store downloaded weights")
    parser.add_argument("--device", default="auto",
                        help="Device hint: auto | cuda | cpu")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  TalkForge — Model Weight Setup")
    print("=" * 60)
    print(f"  Weights dir : {os.path.abspath(args.weights_dir)}")
    print(f"  Device      : {args.device}")
    print("=" * 60)
    print()
    print("  Weight sources:")
    print("  • SadTalker models  → GitHub Releases (OpenTalker/SadTalker@v0.0.2-rc)")
    print("  • BFM Fitting       → HuggingFace (vinthony/SadTalker)")
    print("  • GFPGAN / facexlib → GitHub Releases")
    print()

    model = SadTalkerModel(weights_dir=args.weights_dir, device=args.device)

    if model.is_ready():
        print("✓  All weights already present. Nothing to download.\n")
        return

    print("Downloading required model weights…\n")

    def progress(msg: str) -> None:
        print(f"  ▸  {msg}")

    model.download_weights(progress_cb=progress)

    if model.is_ready():
        print("\n✓  All weights downloaded successfully!\n")
        print("You can now launch TalkForge:\n")
        print("    python main.py\n")
        print("Or with a public share URL:\n")
        print("    python main.py --share\n")
    else:
        # Report exactly which files are still missing
        from pathlib import Path
        root = Path(os.path.dirname(os.path.abspath(__file__)))
        print("\n✗  Some weights are still missing:\n")
        for rel in WEIGHT_URLS:
            full = root / rel
            if not full.is_file() or full.stat().st_size <= 1024:
                print(f"    MISSING: {rel}")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
