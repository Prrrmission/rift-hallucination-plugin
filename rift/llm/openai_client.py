from __future__ import annotations
import time
from typing import List, Dict
from openai import OpenAI
from rift.llm.llm_client import LLMClient

"""
OpenAI-backed implementation of LLMClient.
RIFT is provider-agnostic; this is just one adapter.
"""

class OpenAIClient(LLMClient):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-large",
        max_retries: int = 2,
        retry_delay: float = 0.25,
    ):
        self.client = OpenAI()
        self.model = model
        self.embedding_model = embedding_model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        for attempt in range(self.max_retries + 1):
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
                # NEW: use attribute access, not dict-style
                content = r.choices[0].message.content
                # content can be str or list[ChatCompletionMessageContentPart]
                if isinstance(content, str):
                    return content.strip()
                else:
                    # fallback: join parts if it's a list of parts
                    combined = "".join(
                        part.text for part in content if getattr(part, "text", None)
                    )
                    return combined.strip()
            except Exception:
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_delay)

    def embed_text(self, text: str) -> List[float]:
        for attempt in range(self.max_retries + 1):
            try:
                r = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text,
                )
                return r.data[0].embedding
            except Exception:
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_delay)
