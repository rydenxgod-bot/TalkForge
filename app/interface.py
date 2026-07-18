"""
TalkForge — Gradio web interface.

Single-page layout:
  • Upload portrait image
  • Upload audio file
  • Click "Generate Video"
  • Watch live status messages while the model runs
  • Preview the finished MP4
  • Download  /  Discard  /  Generate New
"""

import os
import gradio as gr

from app.pipeline import run_pipeline, init_model


# ---------------------------------------------------------------------------
# CSS — modern, clean look
# ---------------------------------------------------------------------------

CSS = """
/* ── Base reset ── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
  background: #0d0f14 !important;
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  color: #e8eaf0;
}

/* ── Header ── */
#talkforge-header {
  text-align: center;
  padding: 2.5rem 1rem 1.5rem;
}
#talkforge-header h1 {
  font-size: clamp(2rem, 5vw, 3.2rem);
  font-weight: 800;
  letter-spacing: -0.03em;
  background: linear-gradient(135deg, #7c6fef 0%, #e06fd8 60%, #f5a960 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 0.4rem;
}
#talkforge-header p {
  font-size: 1.05rem;
  color: #8a8fa8;
  margin: 0;
}

/* ── Upload columns ── */
.upload-col {
  background: #181b24;
  border: 1.5px dashed #2d3144;
  border-radius: 16px;
  padding: 1rem;
  transition: border-color 0.2s;
}
.upload-col:hover {
  border-color: #7c6fef;
}
.upload-col label {
  font-size: 0.78rem !important;
  font-weight: 600 !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #7c6fef !important;
  margin-bottom: 0.5rem;
}

/* ── Generate button ── */
#generate-btn {
  background: linear-gradient(135deg, #7c6fef, #e06fd8) !important;
  border: none !important;
  border-radius: 12px !important;
  font-size: 1.05rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.02em;
  color: #fff !important;
  padding: 0.85rem 2rem !important;
  cursor: pointer;
  transition: opacity 0.2s, transform 0.15s;
  box-shadow: 0 4px 24px rgba(124, 111, 239, 0.35);
  width: 100%;
}
#generate-btn:hover { opacity: 0.88; transform: translateY(-1px); }
#generate-btn:active { transform: translateY(0); }
#generate-btn:disabled { opacity: 0.45; cursor: not-allowed; transform: none; }

/* ── Status panel ── */
#status-box {
  background: #181b24;
  border: 1.5px solid #2d3144;
  border-radius: 14px;
  padding: 1.1rem 1.4rem;
  min-height: 56px;
  display: flex;
  align-items: center;
  gap: 0.8rem;
  font-size: 0.97rem;
  color: #c4c9e0;
}

/* ── Video preview ── */
#video-preview video {
  border-radius: 14px;
  width: 100%;
  max-height: 480px;
  background: #0d0f14;
}

/* ── Action buttons row ── */
.action-btn {
  border-radius: 10px !important;
  font-weight: 600 !important;
  font-size: 0.92rem !important;
  padding: 0.65rem 1.1rem !important;
  cursor: pointer;
  transition: opacity 0.18s;
}
.action-btn:hover { opacity: 0.82; }

#download-btn {
  background: #1e6e45 !important;
  color: #fff !important;
  border: none !important;
}
#discard-btn {
  background: #6b1a1a !important;
  color: #fff !important;
  border: none !important;
}
#new-btn {
  background: #2a2d3e !important;
  color: #e8eaf0 !important;
  border: 1.5px solid #3a3f58 !important;
}

/* ── Spinner animation ── */
@keyframes spin {
  to { transform: rotate(360deg); }
}
.spinner {
  display: inline-block;
  width: 16px;
  height: 16px;
  border: 2px solid #3a3f58;
  border-top-color: #7c6fef;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
  flex-shrink: 0;
}

/* ── Hide Gradio branding footer ── */
footer { display: none !important; }

/* ── Responsive ── */
@media (max-width: 640px) {
  #talkforge-header { padding: 1.5rem 0.5rem 1rem; }
}
"""

# ---------------------------------------------------------------------------
# JS helpers
# ---------------------------------------------------------------------------

JS_SPINNER = """
function addSpinner(status) {
  var box = document.querySelector('#status-box');
  if (!box) return status;
  if (status && status !== '' && !box.querySelector('.spinner')) {
    var sp = document.createElement('span');
    sp.className = 'spinner';
    box.prepend(sp);
  }
  return status;
}
"""


# ---------------------------------------------------------------------------
# Interface builder
# ---------------------------------------------------------------------------

def build_interface(output_dir: str = "outputs", weights_dir: str = "weights") -> gr.Blocks:
    """Construct and return the Gradio Blocks app."""

    # Initialise the model singleton (non-blocking; weights download happens at first generate)
    init_model(weights_dir=weights_dir)

    with gr.Blocks(css=CSS, title="TalkForge — AI Lip Sync") as demo:

        # ── Header ──────────────────────────────────────────────────────
        gr.HTML("""
        <div id="talkforge-header">
          <h1>TalkForge</h1>
          <p>Upload a portrait &amp; audio — get a talking-head video with accurate lip sync</p>
        </div>
        """)

        # ── Upload section ───────────────────────────────────────────────
        with gr.Row(equal_height=True):
            with gr.Column(elem_classes="upload-col"):
                image_input = gr.Image(
                    label="Portrait Image",
                    type="filepath",
                    sources=["upload"],
                    image_mode="RGB",
                    elem_id="image-upload",
                    show_label=True,
                    height=260,
                )

            with gr.Column(elem_classes="upload-col"):
                audio_input = gr.Audio(
                    label="Audio File",
                    type="filepath",
                    sources=["upload"],
                    elem_id="audio-upload",
                    show_label=True,
                )

        gr.HTML("<div style='height:1rem'></div>")

        # ── Generate button ──────────────────────────────────────────────
        generate_btn = gr.Button(
            "✦  Generate Video",
            elem_id="generate-btn",
            variant="primary",
            interactive=True,
        )

        # ── Status display ───────────────────────────────────────────────
        status_display = gr.HTML(
            value="",
            elem_id="status-box",
            visible=False,
        )

        # ── Video preview ────────────────────────────────────────────────
        video_output = gr.Video(
            label="Generated Video",
            elem_id="video-preview",
            visible=False,
            autoplay=True,
            show_download_button=False,
        )

        # ── Action buttons (shown after generation) ──────────────────────
        with gr.Row(visible=False) as action_row:
            download_btn = gr.DownloadButton(
                label="⬇  Download Video",
                elem_id="download-btn",
                elem_classes="action-btn",
                value=None,
            )
            discard_btn = gr.Button(
                "✕  Discard",
                elem_id="discard-btn",
                elem_classes="action-btn",
            )
            new_btn = gr.Button(
                "↺  Generate New",
                elem_id="new-btn",
                elem_classes="action-btn",
            )

        # ── Error display ────────────────────────────────────────────────
        error_display = gr.HTML(value="", visible=False)

        # ── Hidden state ─────────────────────────────────────────────────
        video_path_state = gr.State(value=None)

        # ====================================================================
        # Event handlers
        # ====================================================================

        def on_generate(image_path, audio_path):
            """
            Generator that drives the pipeline and yields UI updates.
            Each yield is a tuple of component updates in the order:
              status_display, video_output, action_row, error_display,
              video_path_state, download_btn, generate_btn
            """
            if not image_path:
                yield (
                    _status_html("Please upload a portrait image.", icon="⚠"),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(value="", visible=False),
                    None,
                    gr.update(value=None),
                    gr.update(interactive=True),
                )
                return

            if not audio_path:
                yield (
                    _status_html("Please upload an audio file.", icon="⚠"),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(value="", visible=False),
                    None,
                    gr.update(value=None),
                    gr.update(interactive=True),
                )
                return

            # Show status, hide video, disable button
            yield (
                _status_html("Starting…", spinner=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value="", visible=False),
                None,
                gr.update(value=None),
                gr.update(interactive=False),
            )

            try:
                video_path = None
                for status_msg, maybe_video in run_pipeline(
                    image_path=image_path,
                    audio_path=audio_path,
                    output_dir=output_dir,
                ):
                    if maybe_video is not None:
                        video_path = maybe_video
                    yield (
                        _status_html(status_msg, spinner=(maybe_video is None)),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(value="", visible=False),
                        video_path,
                        gr.update(value=None),
                        gr.update(interactive=False),
                    )

                # Final yield — show video + action buttons
                if video_path:
                    yield (
                        _status_html("✓ Done! Your video is ready.", spinner=False),
                        gr.update(value=video_path, visible=True),
                        gr.update(visible=True),
                        gr.update(value="", visible=False),
                        video_path,
                        gr.update(value=video_path),
                        gr.update(interactive=True),
                    )
                else:
                    raise RuntimeError("Pipeline finished without producing a video file.")

            except Exception as exc:
                error_html = (
                    f'<div style="background:#2d1515;border:1.5px solid #8b2020;'
                    f'border-radius:12px;padding:1rem 1.2rem;color:#e07070;'
                    f'font-size:0.92rem;margin-top:0.5rem">'
                    f'<strong>Error:</strong> {_escape(str(exc))}</div>'
                )
                yield (
                    _status_html("Generation failed.", icon="✕"),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(value=error_html, visible=True),
                    None,
                    gr.update(value=None),
                    gr.update(interactive=True),
                )

        generate_btn.click(
            fn=on_generate,
            inputs=[image_input, audio_input],
            outputs=[
                status_display,
                video_output,
                action_row,
                error_display,
                video_path_state,
                download_btn,
                generate_btn,
            ],
        )

        # ── Discard ──────────────────────────────────────────────────────
        def on_discard(video_path):
            """Delete the generated file and hide the preview."""
            if video_path:
                try:
                    os.remove(video_path)
                except OSError:
                    pass
            return (
                gr.update(visible=False),   # video_output
                gr.update(visible=False),   # action_row
                gr.update(value="", visible=False),   # status_display
                None,                        # video_path_state
                gr.update(value=None),       # download_btn
            )

        discard_btn.click(
            fn=on_discard,
            inputs=[video_path_state],
            outputs=[video_output, action_row, status_display, video_path_state, download_btn],
        )

        # ── Generate New (full page reset) ───────────────────────────────
        def on_new():
            return (
                None,                                # image_input
                None,                                # audio_input
                gr.update(visible=False),            # video_output
                gr.update(visible=False),            # action_row
                gr.update(value="", visible=False),  # status_display
                gr.update(value="", visible=False),  # error_display
                None,                                # video_path_state
                gr.update(value=None),               # download_btn
                gr.update(interactive=True),         # generate_btn
            )

        new_btn.click(
            fn=on_new,
            inputs=[],
            outputs=[
                image_input,
                audio_input,
                video_output,
                action_row,
                status_display,
                error_display,
                video_path_state,
                download_btn,
                generate_btn,
            ],
        )

    return demo


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _status_html(message: str, spinner: bool = False, icon: str = "") -> gr.update:
    """Return an HTML update for the status display component."""
    inner = ""
    if spinner:
        inner = '<span class="spinner"></span>'
    elif icon:
        inner = f'<span style="font-size:1.1rem">{icon}</span>'
    html = (
        f'<div id="status-box" style="display:flex;align-items:center;gap:0.8rem;'
        f'background:#181b24;border:1.5px solid #2d3144;border-radius:14px;'
        f'padding:1.1rem 1.4rem;min-height:56px;font-size:0.97rem;color:#c4c9e0">'
        f'{inner}<span>{_escape(message)}</span></div>'
    )
    return gr.update(value=html, visible=True)


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )
