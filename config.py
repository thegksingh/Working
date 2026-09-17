import os
from dotenv import load_dotenv

load_dotenv()

# Grab keys from .env (e.g., DEEPGRAM_KEYS="key1,key2,key3")
# If only one key, it just makes a list of one.
raw_keys = os.getenv("DEEPGRAM_KEYS", "")
DEEPGRAM_API_KEYS = [k.strip() for k in raw_keys.split(",")] if raw_keys else []





# STT Configuration 
STT_MODE = "hybrid"               # "hybrid", "cloud", or "local"
STT_HYBRID_THRESHOLD_SEC = 15     # Payloads < 15s go to Cloud REST, >= 15s go to Local
LOCAL_WHISPER_MODEL = "medium"    # "tiny", "base", "small", "medium", "large"
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2              # 16-bit mono PCM = 2 bytes per sample