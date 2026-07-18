"""
TalkForge — Entry point
Run:  python main.py
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.ui import launch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TalkForge — AI Lip-Sync Video Generator")
    parser.add_argument("--share",   action="store_true", default=False,
                        help="Create a public Gradio share link (required for Colab)")
    parser.add_argument("--port",    type=int, default=7860,
                        help="Port to serve the web UI on (default: 7860)")
    parser.add_argument("--no-share", dest="share", action="store_false")
    args = parser.parse_args()

    print("\n" + "═" * 56)
    print("  ⚡  TalkForge  —  AI Talking-Head Video Generator")
    print("═" * 56 + "\n")

    launch(share=args.share, port=args.port)
