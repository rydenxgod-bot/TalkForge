"""
TalkForge — Base model interface.

All lip-sync models must subclass BaseLipSyncModel and implement `generate`.
This makes swapping models (SadTalker → MuseTalk → Wav2Lip, etc.) a single-file
change without touching the frontend or pipeline.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional


class BaseLipSyncModel(ABC):
    """Abstract base for every lip-sync backend."""

    def __init__(self, weights_dir: str = "weights", device: str = "auto"):
        self.weights_dir = Path(weights_dir)
        self.device = self._resolve_device(device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True when all weights are present and the model can run."""

    @abstractmethod
    def download_weights(self, progress_cb: Optional[Callable[[str], None]] = None) -> None:
        """Download any missing model checkpoints to ``self.weights_dir``."""

    @abstractmethod
    def generate(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Run inference and write the result to ``output_path``.

        Parameters
        ----------
        image_path : str
            Path to the portrait image (JPEG / PNG).
        audio_path : str
            Path to the driving audio (WAV / MP3).
        output_path : str
            Desired path for the generated MP4.
        progress_cb : callable, optional
            Called with a human-readable status string at each pipeline stage.

        Returns
        -------
        str
            Absolute path to the finished MP4.
        """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(device={self.device!r})"
