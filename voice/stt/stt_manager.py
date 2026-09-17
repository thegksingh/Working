from config import (
    STT_MODE,
    STT_HYBRID_THRESHOLD_SEC,
    LOCAL_WHISPER_MODEL,
    SAMPLE_RATE,
    BYTES_PER_SAMPLE,
)
from .providers.faster_whisper_local import FasterWhisper
from .providers.deepgram_cloud import DeepgramCloud

class STTManager:
    def __init__(self, mode=None):
        self.mode = mode or STT_MODE
        self.local_provider = None
        
        # Store providers in a list 
        self.cloud_providers = []
        self.current_provider_idx = 0

        self.bytes_per_second = SAMPLE_RATE * BYTES_PER_SAMPLE
        self.threshold_bytes = STT_HYBRID_THRESHOLD_SEC * self.bytes_per_second

        if self.mode in ["local", "hybrid"]:
            print(f"STT: Faster-Whisper ({LOCAL_WHISPER_MODEL})")
            self._init_local_provider()

        if self.mode in ["cloud", "hybrid"]:
            print("STT: Cloud STT Providers")
            # All providers should implement a common interface with a transcribe method
            self.cloud_providers = [
                DeepgramCloud(),
                # providerCloud(), 
                # providerCloud2(), 
            ]

    def _init_local_provider(self):
        if self.local_provider is None:
            self.local_provider = FasterWhisper(model_size=LOCAL_WHISPER_MODEL)

    def transcribe(self, audio_bytes: bytes) -> str:
        if not audio_bytes:
            return ""

        if self.mode == "hybrid":
            if len(audio_bytes) < self.threshold_bytes:
                return self._transcribe_cloud(audio_bytes)
            else:
                return self._transcribe_local_chunked(audio_bytes)

        elif self.mode == "local":
            return self._transcribe_local_chunked(audio_bytes)

        elif self.mode == "cloud":
            return self._transcribe_cloud(audio_bytes)

        raise ValueError(f"Unknown STT mode: {self.mode}")

    def _transcribe_cloud(self, audio_bytes: bytes) -> str:
        # Cascading Failover: Tries cloud providers in order. Falls back to local if ALL fail.
        if not self.cloud_providers:
            print("No cloud providers configured. Falling back to local.")
            self._init_local_provider()
            return self._transcribe_local_chunked(audio_bytes)

        attempts = 0
        while attempts < len(self.cloud_providers):
            current_provider = self.cloud_providers[self.current_provider_idx]
            provider_name = current_provider.__class__.__name__
            
            try:
                return current_provider.transcribe(audio_bytes)
            except Exception as e:
                print(f"\n[Provider Failover] {provider_name} failed entirely: {e}")
                # 3. Shift to the next provider, looping back to 0 if we hit the end
                self.current_provider_idx = (self.current_provider_idx + 1) % len(self.cloud_providers)
                attempts += 1
                
        # 4. If the while loop finishes, every single cloud provider is dead.
        print("\n ALL Cloud Providers exhausted. Falling back to Local AI")
        self._init_local_provider()
        return self._transcribe_local_chunked(audio_bytes)

    def _merge_overlapping_text(self, text1: str, text2: str) -> str:
        if not text1: return text2
        if not text2: return text1

        words1 = text1.split()
        words2 = text2.split()

        max_overlap = min(len(words1), len(words2), 5)

        for i in range(max_overlap, 0, -1):
            w1_clean = [w.lower().strip(".,!?") for w in words1[-i:]]
            w2_clean = [w.lower().strip(".,!?") for w in words2[:i]]

            if w1_clean == w2_clean:
                return " ".join(words1 + words2[i:])

        return text1 + " " + text2

    def _transcribe_local_chunked(self, audio_bytes: bytes) -> str:
        full_text = ""
        overlap_bytes = int(0.5 * self.bytes_per_second)
        step_bytes = self.threshold_bytes - overlap_bytes

        for i in range(0, len(audio_bytes), step_bytes):
            chunk = audio_bytes[i : i + self.threshold_bytes]

            if len(chunk) <= overlap_bytes and full_text:
                continue

            text_chunk = self.local_provider.transcribe(chunk)

            if text_chunk:
                full_text = self._merge_overlapping_text(full_text, text_chunk)

        return full_text.strip()