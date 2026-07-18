"""
TalkForge — Entry point
Launches the Gradio web interface for lip-sync video generation.
"""

import argparse
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.interface import build_interface


def parse_args():
    parser = argparse.ArgumentParser(description="TalkForge — AI Lip Sync Video Generator")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a public Gradio share link (required for Colab)")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode")
    parser.add_argument("--weights-dir", type=str, default="weights",
                        help="Directory containing model weight files")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory for generated video output")
    return parser.parse_args()


def main():
    args = parse_args()

    # Create required directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.weights_dir, exist_ok=True)
    os.makedirs("uploads", exist_ok=True)

    print("\n" + "=" * 60)
    print("  TalkForge — AI Lip Sync Video Generator")
    print("=" * 60)
    print(f"  Output directory : {os.path.abspath(args.output_dir)}")
    print(f"  Weights directory: {os.path.abspath(args.weights_dir)}")
    print(f"  Share link       : {'enabled' if args.share else 'disabled'}")
    print("=" * 60 + "\n")

    demo = build_interface(
        output_dir=args.output_dir,
        weights_dir=args.weights_dir,
    )

    demo.queue(max_size=5).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True,
        favicon_path=None,
    )


if __name__ == "__main__":
    main()
