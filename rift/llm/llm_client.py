from __future__ import annotations
from typing import List, Dict


class LLMClient:
    """
    Minimal interface used by the entire RIFT engine.
    """

    def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError

    def embed_text(self, text: str) -> List[float]:
        raise NotImplementedError
