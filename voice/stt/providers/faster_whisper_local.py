import torch
import numpy as np
from faster_whisper import WhisperModel


class FasterWhisper:
    def __init__(self, model_size="medium", device=None, beam_size=5):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.beam_size = beam_size

        try:
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type="int8" if device == "cpu" else "float16"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Faster-Whisper: {e}")

    def transcribe(self, audio_bytes):
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        samples = samples.astype(np.float32) / 32768.0

        segments, info = self.model.transcribe(
            samples,
            beam_size=self.beam_size
        )

        return " ".join(
            segment.text.strip()
            for segment in segments
            if segment.text.strip()
        )