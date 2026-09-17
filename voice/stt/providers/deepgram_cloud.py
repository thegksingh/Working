import io
import wave
import threading
from deepgram import DeepgramClient
from deepgram.core.events import EventType
from config import DEEPGRAM_API_KEYS

class DeepgramCloud:
    
    def __init__(self, on_partial=None, on_final=None):
        if not DEEPGRAM_API_KEYS:
            raise ValueError("No Deepgram API keys found in config.py")

        self.keys = DEEPGRAM_API_KEYS
        self.current_key_idx = 0
        
        # WebSocket State Variables
        self.client = None
        self._connect_ctx = None
        self.connection = None
        self._listen_thread = None
        
        self.on_partial = on_partial
        self.on_final = on_final

    # MODE 1: REST API (For Hybrid Batching < 15s)
    def transcribe(self, audio_bytes: bytes) -> str:
        """Synchronous REST API call for pre-recorded payloads."""
        
        # 1. Wrap raw bytes into a valid WAV file in memory
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)       # Mono
            wav_file.setsampwidth(2)       # 16-bit PCM (2 bytes)
            wav_file.setframerate(16000)   # 16kHz
            wav_file.writeframes(audio_bytes)
            
        wav_data = wav_io.getvalue()

        # 2. Key Rotation Loop for REST API
        attempts = 0
        while attempts < len(self.keys):
            try:
                current_key = self.keys[self.current_key_idx]
                client = DeepgramClient(api_key=current_key)
                
                # Use v5 media endpoint
                response = client.listen.v1.media.transcribe_file(
                    request=wav_data,
                    model="nova-2",
                    smart_format=True,
                )
                
                text = response.results.channels[0].alternatives[0].transcript
                return text.strip()

            except Exception as e:
                print(f"\n[REST] Deepgram key {self.current_key_idx} failed: {e}. Rotating...")
                self.current_key_idx = (self.current_key_idx + 1) % len(self.keys)
                attempts += 1

        raise RuntimeError("All Deepgram API keys failed for REST transcription.")

    # MODE 2: LIVE WEBSOCKET (For Real-Time Cloud)
    def start(self):
        options = {
            "model": "nova-2",
            "language": "en-US",
            "encoding": "linear16",
            "sample_rate": 16000,
            "channels": 1,
            "interim_results": True,
        }

        attempts = 0
        while attempts < len(self.keys):
            try:
                current_key = self.keys[self.current_key_idx]
                self.client = DeepgramClient(api_key=current_key)
                
                self._connect_ctx = self.client.listen.v1.connect(**options)
                self.connection = self._connect_ctx.__enter__()

                self.connection.on(EventType.MESSAGE, self._handle_message)
                self.connection.on(EventType.ERROR, self._handle_error)

                self._listen_thread = threading.Thread(
                    target=self.connection.start_listening, daemon=True
                )
                self._listen_thread.start()
                return 

            except Exception as e:
                print(f"\n[WebSocket] Deepgram key {self.current_key_idx} failed: {e}. Rotating...")
                self._cleanup_connection()
                self.current_key_idx = (self.current_key_idx + 1) % len(self.keys)
                attempts += 1

        raise RuntimeError("All Deepgram API keys have been exhausted or failed.")

    def send_frame(self, frame: bytes):
        if self.connection is not None:
            self.connection.send_media(frame)

    def finish(self):
        if self.connection is not None:
            self.connection.send_finalize()
        self._cleanup_connection()

    def _cleanup_connection(self):
        if self._connect_ctx is not None:
            try:
                self._connect_ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._connect_ctx = None
        self.connection = None
        
        if self._listen_thread is not None:
            self._listen_thread.join(timeout=2)
            self._listen_thread = None

    def _handle_message(self, message):
        if getattr(message, "type", None) != "Results": return
        if not message.channel.alternatives: return
        
        text = message.channel.alternatives[0].transcript
        if not text: return

        if message.is_final and self.on_final:
            self.on_final(text)
        elif not message.is_final and self.on_partial:
            self.on_partial(text)

    def _handle_error(self, error):
        print(f"\nDeepgram SDK Error: {error}")