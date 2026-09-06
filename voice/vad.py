import array

import torch
from silero_vad import load_silero_vad


class VAD:
    def __init__(self, samplerate=16000, threshold=0.5):
        self.samplerate = samplerate
        self.threshold = threshold
        self.chunk_size = 512

        # Accumulates microphone samples until we have 512 samples
        self.buffer = array.array("h")

        self.model = load_silero_vad()

    def process_audio(self, audio_chunk):
        # Convert bytes to int16 samples
        samples = array.array("h")
        samples.frombytes(audio_chunk)

        # Append to analysis buffer
        self.buffer.extend(samples)

        if len(self.buffer) < self.chunk_size:
            return None

        # Take exactly 512 samples
        chunk = self.buffer[:self.chunk_size]
        del self.buffer[:self.chunk_size]

        # int16 to float32 [-1, 1]
        input_tensor = torch.tensor(chunk, dtype=torch.float32) / 32768.0

        with torch.no_grad(): speech_prob = self.model(input_tensor, self.samplerate).item()

        return speech_prob > self.threshold