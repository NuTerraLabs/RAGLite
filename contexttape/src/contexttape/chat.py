from __future__ import annotations
from typing import Tuple
from .storage import ISStore
from .embed import get_client, embed_text_1536

def store_chat_turn(store: ISStore, role: str, text: str, client=None, verbose: bool = False) -> Tuple[int, int]:
    client = client or get_client()
    tagged = f"{role}: {text}"
    emb = embed_text_1536(client, tagged, verbose=verbose)
    # Prefer int8 on chat turns for footprint
    return store.append_text_with_embedding(tagged, emb, quantize=True)
