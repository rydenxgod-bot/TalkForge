"""
TalkForge — Gradio UI
Single-page: upload image + audio → Generate → preview MP4 → download.
"""

import time
import threading
import gradio as gr
from pathlib import Path
from app.inference import generate_talking_head

ROOT_DIR = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────
# Custom CSS — dark cyber aesthetic
# ─────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #080c14 !important;
    font-family: 'Space Grotesk', system-ui, sans-serif !important;
    color: #e2e8f0 !important;
}

/* Header */
#tf-header {
    text-align: center;
    padding: 2.5rem 1rem 1.8rem;
    border-bottom: 1px solid #1a2540;
    margin-bottom: 1.8rem;
}
#tf-header h1 {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: clamp(2rem, 5vw, 3.2rem) !important;
    font-weight: 700 !important;
    background: linear-gradient(125deg, #00e5ff 0%, #a855f7 60%, #00e5ff 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    margin: 0 0 0.5rem !important;
    animation: shimmer 4s linear infinite;
}
@keyframes shimmer { to { background-position: 200% center; } }
#tf-header p { color: #4a6080; font-size: 0.95rem; margin: 0; }

/* Upload panels */
.tf-panel {
    background: #0d1526 !important;
    border: 1px solid #1a2540 !important;
    border-radius: 14px !important;
    padding: 1.4rem !important;
    transition: border-color 0.2s;
}
.tf-panel:hover { border-color: #00e5ff50 !important; }

.tf-panel label span {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: #00e5ff !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

/* Status box */
#tf-status textarea {
    background: #060a12 !important;
    border: 1px solid #1a2540 !important;
    border-radius: 10px !important;
    color: #00e5ff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 0.9rem 1.2rem !important;
    min-height: 48px !important;
    resize: none !important;
}

/* Generate button */
#tf-generate {
    background: linear-gradient(125deg, #00e5ff, #7c3aed) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    padding: 0.85rem 2rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
    width: 100%;
}
#tf-generate:hover  { opacity: 0.85 !important; transform: translateY(-1px); }
#tf-generate:active { transform: translateY(0); }

/* Action buttons */
#tf-download, #tf-discard, #tf-new {
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    transition: opacity 0.2s !important;
}
#tf-download:hover, #tf-discard:hover, #tf-new:hover { opacity: 0.8 !important; }
#tf-download { background: #059669 !important; color: #fff !important; border: none !important; }
#tf-discard  { background: #1a2540 !important; color: #64748b !important; border: 1px solid #253050 !important; }
#tf-new      { background: linear-gradient(125deg, #00e5ff, #7c3aed) !important; color: #fff !important; border: none !important; }

/* Video output */
#tf-video video {
    border-radius: 14px;
    width: 100%;
    max-height: 480px;
    background: #000;
    margin-top: 0.5rem;
}

/* Divider label */
.tf-divider {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #1a2540;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    border-top: 1px solid #1a2540;
    padding-top: 0.8rem;
    margin: 0.6rem 0 1rem;
}

footer { display: none !important; }
"""


# ─────────────────────────────────────────────
# Build UI
# ─────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    with gr.Blocks(
        css=CSS,
        title="TalkForge",
        theme=gr.themes.Base(
            primary_hue="cyan",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Space Grotesk"),
        ),
    ) as demo:

        # Header
        gr.HTML("""
        <div id="tf-header">
            <h1>⚡ TalkForge</h1>
            <p>Upload a portrait &amp; audio · Click Generate · Get a lip-synced talking video</p>
        </div>
        """)

        # Upload row
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, elem_classes="tf-panel"):
                image_input = gr.Image(
                    type="filepath",
                    label="Portrait Image (JPG / PNG)",
                    image_mode="RGB",
                    sources=["upload", "clipboard"],
                    height=260,
                    show_download_button=False,
                )
                gr.HTML('<p style="color:#2d4060;font-size:0.76rem;margin-top:0.4rem">Front-facing portrait · single person · good lighting</p>')

            with gr.Column(scale=1, elem_classes="tf-panel"):
                audio_input = gr.Audio(
                    type="filepath",
                    label="Audio File (WAV / MP3)",
                    sources=["upload", "microphone"],
                )
                gr.HTML('<p style="color:#2d4060;font-size:0.76rem;margin-top:0.4rem">Clear speech · WAV or MP3 · any length</p>')

        # Generate button
        gr.HTML('<div style="height:1.2rem"></div>')
        with gr.Row():
            with gr.Column(scale=1, min_width=0):
                pass
            with gr.Column(scale=2, min_width=0):
                generate_btn = gr.Button(
                    "⚡  Generate Video",
                    elem_id="tf-generate",
                    variant="primary",
                )
            with gr.Column(scale=1, min_width=0):
                pass

        # Status
        gr.HTML('<div style="height:0.6rem"></div>')
        status_box = gr.Textbox(
            value="Ready — upload a portrait and audio to begin.",
            label="",
            interactive=False,
            elem_id="tf-status",
            show_label=False,
            show_copy_button=False,
            lines=1,
        )

        # Output
        gr.HTML('<div class="tf-divider">Output</div>')

        video_output = gr.Video(
            label="",
            visible=False,
            elem_id="tf-video",
            show_label=False,
            autoplay=True,
        )

        # Action buttons (hidden until video is ready)
        with gr.Row(visible=False) as action_row:
            download_file = gr.File(
                label="",
                visible=False,
                elem_id="tf-download-file",
            )
            dl_btn   = gr.Button("⬇  Download", elem_id="tf-download", variant="secondary")
            disc_btn = gr.Button("🗑  Discard",  elem_id="tf-discard",  variant="secondary")
            new_btn  = gr.Button("↩  Generate New", elem_id="tf-new", variant="secondary")

        # State: holds the path of the current generated MP4
        current_path = gr.State(value=None)

        # ── GENERATE ──────────────────────────────────────
        def on_generate(image, audio):
            if image is None:
                raise gr.Error("Please upload a portrait image first.")
            if audio is None:
                raise gr.Error("Please upload an audio file first.")

            # Initial UI state
            yield (
                gr.update(value="⚙️  Starting…"),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(interactive=False),
                None,
                gr.update(visible=False),
            )

            result_holder = [None]
            error_holder  = [None]
            status_holder = ["⚙️  Starting…"]

            def _cb(msg):
                status_holder[0] = msg

            def _worker():
                try:
                    result_holder[0] = generate_talking_head(
                        image_path=image,
                        audio_path=audio,
                        progress_callback=_cb,
                    )
                except Exception as exc:
                    error_holder[0] = str(exc)
                    _cb(f"❌  {exc}")

            t = threading.Thread(target=_worker, daemon=True)
            t.start()

            while t.is_alive():
                yield (
                    gr.update(value=status_holder[0]),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(interactive=False),
                    None,
                    gr.update(visible=False),
                )
                time.sleep(0.7)

            t.join()

            if error_holder[0]:
                raise gr.Error(error_holder[0])

            out = result_holder[0]
            if out:
                yield (
                    gr.update(value="✅  Done! Your video is ready below."),
                    gr.update(value=out, visible=True),
                    gr.update(visible=True),
                    gr.update(interactive=True),
                    out,
                    gr.update(visible=False),
                )
            else:
                yield (
                    gr.update(value="❌  Generation failed. Check the Colab logs."),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(interactive=True),
                    None,
                    gr.update(visible=False),
                )

        generate_btn.click(
            fn=on_generate,
            inputs=[image_input, audio_input],
            outputs=[status_box, video_output, action_row, generate_btn, current_path, download_file],
        )

        # ── DOWNLOAD ──────────────────────────────────────
        def on_download(path):
            if path and Path(path).exists():
                return gr.update(value=path, visible=True)
            raise gr.Error("No video to download.")

        dl_btn.click(
            fn=on_download,
            inputs=[current_path],
            outputs=[download_file],
        )

        # ── DISCARD ───────────────────────────────────────
        def on_discard():
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value="Ready — upload files to try again."),
                None,
            )

        disc_btn.click(
            fn=on_discard,
            outputs=[video_output, action_row, download_file, status_box, current_path],
        )

        # ── GENERATE NEW (full reset) ──────────────────────
        def on_new():
            return (
                gr.update(value=None),
                gr.update(value=None),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value="Ready — upload an image and audio to begin."),
                None,
                gr.update(visible=False),
            )

        new_btn.click(
            fn=on_new,
            outputs=[
                image_input, audio_input,
                video_output, action_row,
                status_box, current_path, download_file,
            ],
        )

        # Footer
        gr.HTML("""
        <div style="text-align:center;padding:2rem 0 0.5rem;
             color:#1a2540;font-family:'JetBrains Mono',monospace;
             font-size:0.68rem;letter-spacing:0.08em;">
            TalkForge · MIT License · Open Source · by Vaexor
        </div>
        """)

    return demo


def launch(share: bool = True, port: int = 7860):
    demo = build_ui()
    demo.queue(max_size=4)
    demo.launch(
        share=share,
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
        quiet=False,
    )


if __name__ == "__main__":
    launch()
