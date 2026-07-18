"""
TalkForge — Generation pipeline.

Coordinates input validation, audio conversion, model inference, and output
management.  The frontend calls `run_pipeline()` and receives a generator that
yields (status_message, output_video_path_or_None) tuples so the UI can show
live progress while the model runs.
"""

import os
import uuid
import traceback
from pathlib import Path
from typing import Generator, Optional, Tuple

from app.utils import (
    validate_image,
    validate_audio,
    unique_output_path,
    safe_copy,
    convert_audio_to_wav,
    resize_image_if_needed,
)

# ---------------------------------------------------------------------------
# Model selection — swap this import to change the backend
# ---------------------------------------------------------------------------
from models.sadtalker import SadTalkerModel as _LipSyncModel

# Singleton model instance (created once per process)
_model: Optional[_LipSyncModel] = None
_weights_dir: str = "weights"


def init_model(weights_dir: str = "weights", device: str = "auto") -> None:
    """
    Call once at startup to initialise the singleton model.
    Subsequent calls are no-ops.
    """
    global _model, _weights_dir
    _weights_dir = weights_dir
    if _model is None:
        _model = _LipSyncModel(weights_dir=weights_dir, device=device)


def _get_model() -> _LipSyncModel:
    global _model
    if _model is None:
        _model = _LipSyncModel(weights_dir=_weights_dir)
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    image_path: str,
    audio_path: str,
    output_dir: str = "outputs",
) -> Generator[Tuple[str, Optional[str]], None, None]:
    """
    Run the full lip-sync pipeline.

    Yields ``(status_message, video_path_or_None)`` tuples.
    The final yield has the video path set; all earlier yields have ``None``.

    Raises
    ------
    ValueError
        If inputs are invalid (wrong format, missing file).
    RuntimeError
        If model inference fails.
    """
    tmp_dir = os.path.join(output_dir, f"tmp_{uuid.uuid4().hex[:8]}")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        # ── 1. Validate inputs ─────────────────────────────────────────
        yield ("Validating inputs…", None)
        validate_image(image_path)
        validate_audio(audio_path)

        # ── 2. Pre-process image ────────────────────────────────────────
        yield ("Preparing image…", None)
        processed_image = resize_image_if_needed(image_path, tmp_dir, max_side=512)

        # ── 3. Pre-process audio → WAV ──────────────────────────────────
        yield ("Preparing audio…", None)
        ext = Path(audio_path).suffix.lower()
        if ext != ".wav":
            processed_audio = convert_audio_to_wav(audio_path, tmp_dir)
        else:
            processed_audio = safe_copy(audio_path, tmp_dir)

        # ── 4. Resolve output path ──────────────────────────────────────
        final_output = unique_output_path(output_dir, prefix="talkforge")

        # ── 5. Model inference (yields live status via callback) ────────
        model = _get_model()
        progress_log: list[str] = []

        def _progress(msg: str) -> None:
            progress_log.append(msg)

        # We collect the model's internal status messages via the callback;
        # Gradio receives them as a separate stream of yield calls below.
        # Because the model is synchronous, we run it in a thread and poll.
        import threading

        result_holder: dict = {"path": None, "error": None}
        status_holder: dict = {"msg": "Preparing model…"}

        def _inference_thread() -> None:
            try:
                def _cb(msg: str) -> None:
                    status_holder["msg"] = msg

                result_holder["path"] = model.generate(
                    image_path=processed_image,
                    audio_path=processed_audio,
                    output_path=final_output,
                    progress_cb=_cb,
                )
            except Exception as exc:
                result_holder["error"] = exc

        thread = threading.Thread(target=_inference_thread, daemon=True)
        thread.start()

        import time
        last_msg = ""
        while thread.is_alive():
            current = status_holder["msg"]
            if current != last_msg:
                yield (current, None)
                last_msg = current
            time.sleep(0.4)
        thread.join()

        # Yield final status from thread one more time if changed
        if status_holder["msg"] != last_msg:
            yield (status_holder["msg"], None)

        if result_holder["error"] is not None:
            raise result_holder["error"]

        # ── 6. Return result ────────────────────────────────────────────
        video_path = result_holder["path"]
        if not video_path or not os.path.isfile(video_path):
            raise RuntimeError("Model finished but output file was not found.")

        yield ("Done! Your video is ready.", video_path)

    except Exception:
        # Clean up temp dir on failure
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    else:
        # Clean up temp dir on success too
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
