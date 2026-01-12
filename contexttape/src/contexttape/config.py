import os

# Backward compatibility: support old TS_DIR names
WIKI_IS_DIR = os.getenv("WIKI_IS_DIR") or os.getenv("WIKI_TS_DIR", "data/wiki")
CHAT_IS_DIR = os.getenv("CHAT_IS_DIR") or os.getenv("CHAT_TS_DIR", "data/chat")
TOP_K_DEFAULT = int(os.getenv("TOP_K", "5"))
DEBUG_DIR = "debug"
