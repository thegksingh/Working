import os
from dotenv import load_dotenv

load_dotenv()

# Grab keys from .env (e.g., DEEPGRAM_KEYS="key1,key2,key3")
# If only one key, it just makes a list of one.
raw_keys = os.getenv("DEEPGRAM_KEYS", "")
DEEPGRAM_API_KEYS = [k.strip() for k in raw_keys.split(",")] if raw_keys else []