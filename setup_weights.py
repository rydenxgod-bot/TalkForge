"""
TalkForge — Standalone weight downloader.

Pre-downloads all required model checkpoints before launching the app.
Safe to re-run — already-downloaded files are skipped.

Usage:
    python setup_weights.py
    python setup_weights.py --weights-dir /path/to/weights
"""

import sys
import os
import argparse

# Make sure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.sadtalker import SadTalkerModel


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

    print("\n" + "=" * 58)
    print("  TalkForge — Model Weight Setup")
    print("=" * 58)
    print(f"  Weights dir : {os.path.abspath(args.weights_dir)}")
    print(f"  Device      : {args.device}")
    print("=" * 58 + "\n")

    model = SadTalkerModel(weights_dir=args.weights_dir, device=args.device)

    if model.is_ready():
        print("✓  All weights already present. Nothing to download.\n")
        return

    print("Downloading required model weights …\n")

    def progress(msg: str) -> None:
        print(f"  ▸  {msg}")

    model.download_weights(progress_cb=progress)

    if model.is_ready():
        print("\n✓  All weights downloaded successfully!\n")
        print("You can now launch TalkForge:\n")
        print("    python main.py\n")
        print("Or with a public share URL (useful for remote machines):\n")
        print("    python main.py --share\n")
    else:
        print("\n✗  Some weights are still missing. Check the output above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
