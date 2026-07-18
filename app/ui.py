"""
TalkForge - Gradio UI
Single-page interface: upload image + audio → Generate → preview MP4.
"""

import time
import gradio as gr
from pathlib import Path
from app.inference import generate_talking_head

ROOT_DIR = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────
# Custom CSS  (dark terminal / cyber aesthetic)
# ─────────────────────────────────────────────
CUSTOM_CSS = """
/* ── Global ───────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #0a0a0f !important;
    font-family: 'Space Grotesk', system-ui, sans-serif !important;
    color: #e2e8f0 !important;
}

/* ── Header ───────────────────────────────── */
#talkforge-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 2rem;
}

#talkforge-header h1 {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: clamp(2rem, 5vw, 3.4rem) !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #00e5ff 0%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    margin: 0 0 0.5rem !important;
}

#talkforge-header p {
    color: #64748b;
    font-size: 1rem;
    margin: 0;
    letter-spacing: 0.04em;
}

/* ── Upload cards ─────────────────────────── */
.upload-card {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    transition: border-color 0.2s ease;
}
.upload-card:hover { border-color: #00e5ff !important; }

.upload-card label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    color: #00e5ff !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem !important;
    display: block;
}

/* Gradio file/image component inner boxes */
.upload-card .wrap { border-color: #1e293b !important; background: #070b14 !important; }
.upload-card .wrap:hover { border-color: #00e5ff !important; }

/* ── Generate button ──────────────────────── */
#generate-btn {
    background: linear-gradient(135deg, #00e5ff 0%, #7c3aed 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    padding: 0.9rem 2rem !important;
    cursor: pointer;
    transition: opacity 0.2s ease, transform 0.15s ease;
    width: 100%;
    max-width: 360px;
    margin: 0 auto;
    display: block;
}
#generate-btn:hover  { opacity: 0.88; transform: translateY(-1px); }
#generate-btn:active { transform: translateY(0); }

/* ── Status / progress ────────────────────── */
#status-box {
    background: #070b14 !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
    padding: 1.25rem 1.5rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    color: #00e5ff !important;
    min-height: 52px;
    display: flex;
    align-items: center;
}

/* ── Video output ─────────────────────────── */
#video-output video {
    border-radius: 14px;
    width: 100%;
    max-height: 480px;
    background: #000;
}

/* ── Action buttons row ───────────────────── */
.action-btn {
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    padding: 0.7rem 1.2rem !important;
    border: none !important;
    cursor: pointer;
    transition: opacity 0.2s ease;
}
.action-btn:hover { opacity: 0.82; }

#btn-download {
    background: #10b981 !important;
    color: #fff !important;
}
#btn-discard {
    background: #1e293b !important;
    color: #94a3b8 !important;
    border: 1px solid #334155 !important;
}
#btn-new {
    background: linear-gradient(135deg, #00e5ff 0%, #7c3aed 100%) !important;
    color: #fff !important;
}

/* ── Dividers ─────────────────────────────── */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin: 0.5rem 0 1rem;
    border-top: 1px solid #1e293b;
    padding-top: 0.75rem;
}

/* Hide default Gradio footer */
footer { display: none !important; }
"""

# ─────────────────────────────────────────────
# Progress messages shown during generation
# ─────────────────────────────────────────────
PROGRESS_STAGES = [
    "⚙️  Preparing model…",
    "🧠  Loading weights…",
    "🎵  Processing audio…",
    "👁️  Detecting face…",
    "🎬  Generating frames…",
    "✨  Enhancing quality…",
    "🎞️  Rendering video…",
    "📦  Finalising output…",
]


# ─────────────────────────────────────────────
# Core generation function (called by Gradio)
# ─────────────────────────────────────────────
def run_generation(image, audio, progress=gr.Progress(track_tqdm=False)):
    """
    Gradio-compatible generation function.
    Yields status strings during processing, then yields the output path.
    """
    if image is None:
        raise gr.Error("Please upload a portrait image first.")
    if audio is None:
        raise gr.Error("Please upload an audio file first.")

    # We use a mutable list so the callback can update it
    status_holder = ["⚙️  Starting…"]

    def _cb(msg: str):
        status_holder[0] = msg

    import threading
    result_holder = [None]
    error_holder  = [None]

    def _worker():
        try:
            result_holder[0] = generate_talking_head(
                image_path=image,
                audio_path=audio,
                progress_callback=_cb,
            )
        except Exception as exc:
            import traceback
            error_holder[0] = str(exc)
            _cb(f"❌ Error: {exc}")

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    # Poll while thread runs, yielding status updates to Gradio
    stage_idx = 0
    while thread.is_alive():
        current = status_holder[0]
        yield current, None
        time.sleep(0.8)
        stage_idx = min(stage_idx + 1, len(PROGRESS_STAGES) - 1)

    thread.join()

    if error_holder[0]:
        raise gr.Error(error_holder[0])

    output_path = result_holder[0]
    yield "✅  Generation complete!", output_path


# ─────────────────────────────────────────────
# Build Gradio interface
# ─────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="TalkForge",
        theme=gr.themes.Base(
            primary_hue="cyan",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Space Grotesk"),
        ),
    ) as demo:

        # ── Header ──────────────────────────────────────
        gr.HTML("""
        <div id="talkforge-header">
            <h1>⚡ TalkForge</h1>
            <p>Upload a portrait &amp; audio · Click Generate · Get a lip-synced talking video</p>
        </div>
        """)

        # ── Upload row ───────────────────────────────────
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, elem_classes="upload-card"):
                gr.HTML('<label>Portrait Image</label>')
                image_input = gr.Image(
                    type="filepath",
                    label="",
                    image_mode="RGB",
                    sources=["upload", "clipboard"],
                    height=280,
                    show_label=False,
                    show_download_button=False,
                )
                gr.HTML('<p style="color:#475569;font-size:0.78rem;margin:0.4rem 0 0;">JPG or PNG · front-facing portrait works best</p>')

            with gr.Column(scale=1, elem_classes="upload-card"):
                gr.HTML('<label>Audio File</label>')
                audio_input = gr.Audio(
                    type="filepath",
                    label="",
                    sources=["upload", "microphone"],
                    show_label=False,
                )
                gr.HTML('<p style="color:#475569;font-size:0.78rem;margin:0.4rem 0 0;">WAV or MP3 · clear speech recommended</p>')

        # ── Generate button ──────────────────────────────
        gr.HTML('<div style="height:1.5rem"></div>')
        generate_btn = gr.Button(
            "⚡ Generate Video",
            elem_id="generate-btn",
            variant="primary",
        )

        # ── Status box ───────────────────────────────────
        gr.HTML('<div style="height:1rem"></div>')
        status_text = gr.Textbox(
            value="Ready — upload an image and audio to begin.",
            label="",
            interactive=False,
            elem_id="status-box",
            show_label=False,
            show_copy_button=False,
        )

        # ── Video output + action buttons ────────────────
        gr.HTML('<div class="section-label">Output</div>')

        video_output = gr.Video(
            label="",
            visible=False,
            elem_id="video-output",
            show_label=False,
            autoplay=True,
        )

        with gr.Row(visible=False) as action_row:
            download_btn = gr.DownloadButton(
                "⬇  Download",
                elem_id="btn-download",
                elem_classes="action-btn",
                variant="secondary",
            )
            discard_btn = gr.Button(
                "🗑  Discard",
                elem_id="btn-discard",
                elem_classes="action-btn",
                variant="secondary",
            )
            new_btn = gr.Button(
                "↩  Generate New",
                elem_id="btn-new",
                elem_classes="action-btn",
                variant="secondary",
            )

        # Keep a state variable for the current output path
        current_output = gr.State(value=None)

        # ── Event: Generate ──────────────────────────────
        def on_generate(image, audio):
            # Immediately hide old output & disable button
            yield (
                gr.update(value="⚙️  Starting…"),          # status
                gr.update(visible=False),                   # video
                gr.update(visible=False),                   # action_row
                gr.update(interactive=False),               # generate_btn
                None,                                       # current_output
            )
            output_path = None
            try:
                for status, path in run_generation(image, audio):
                    output_path = path
                    yield (
                        gr.update(value=status),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(interactive=False),
                        None,
                    )
            except gr.Error as exc:
                yield (
                    gr.update(value=f"❌  {exc}"),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(interactive=True),
                    None,
                )
                return

            if output_path:
                yield (
                    gr.update(value="✅  Done! Your video is ready below."),
                    gr.update(value=output_path, visible=True),
                    gr.update(visible=True),
                    gr.update(interactive=True),
                    output_path,
                )
            else:
                yield (
                    gr.update(value="❌  Generation failed. Check logs."),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(interactive=True),
                    None,
                )

        generate_btn.click(
            fn=on_generate,
            inputs=[image_input, audio_input],
            outputs=[status_text, video_output, action_row, generate_btn, current_output],
        )

        # ── Event: Download ──────────────────────────────
        def on_download(path):
            return path

        download_btn.click(fn=on_download, inputs=[current_output], outputs=[download_btn])

        # ── Event: Discard ───────────────────────────────
        def on_discard():
            return (
                gr.update(value=None, visible=False),          # video
                gr.update(visible=False),                      # action_row
                gr.update(value="Ready — upload files to try again."),
            )

        discard_btn.click(
            fn=on_discard,
            outputs=[video_output, action_row, status_text],
        )

        # ── Event: Generate New (full reset) ─────────────
        def on_new():
            return (
                gr.update(value=None),                           # image
                gr.update(value=None),                           # audio
                gr.update(value=None, visible=False),            # video
                gr.update(visible=False),                        # action_row
                gr.update(value="Ready — upload an image and audio to begin."),
                None,                                            # current_output
            )

        new_btn.click(
            fn=on_new,
            outputs=[image_input, audio_input, video_output, action_row, status_text, current_output],
        )

        # ── Footer ───────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center;padding:2rem 0 1rem;color:#1e293b;font-family:'JetBrains Mono',monospace;font-size:0.72rem;letter-spacing:0.08em;">
            TalkForge · MIT License · Open Source
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
