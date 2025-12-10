from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class AnswerEntry:
    user_text: str
    text: str
    embedding: List[float]
    weight: float
    turn_index: int


class ConversationMemory:
    """
    ConversationMemory now stores *two distinct weights*:

    - USER INPUT  → extremely low weight (0.01)
    - ASSISTANT OUTPUT → normal weight (1.0)

    This prevents user statements (especially fictional or coerced ones)
    from ever poisoning anchor formation, while preserving complete
    backward compatibility with your existing engine.
    """

    def __init__(self, max_turns: int = 200, user_weight: float = 0.01, assistant_weight: float = 1.0):
        self.max_turns = max_turns
        self.user_weight = user_weight
        self.assistant_weight = assistant_weight

        self.entries: List[AnswerEntry] = []
        self.turn_index: int = 0

    def reset(self) -> None:
        """Clear conversation memory and reset turn counter."""
        self.entries.clear()
        self.turn_index = 0

    def add_turn(self, user_text: str, answer_text: str, embedding: List[float]) -> None:
        """
        Store the turn with separate weights for user vs assistant.

        USER ENTRY:
            - user_text is stored as-is
            - embedding reused
            - weight = self.user_weight  (very low)

        ASSISTANT ENTRY:
            - assistant's answer stored as usual
            - embedding reused
            - weight = self.assistant_weight (normal)
        """

        # ---- USER ENTRY (low energy influence) ----
        user_entry = AnswerEntry(
            user_text=user_text,
            text=user_text,                 # User’s message as the content
            embedding=embedding,
            weight=self.user_weight,        # near-zero influence
            turn_index=self.turn_index,
        )
        self.entries.append(user_entry)

        # ---- ASSISTANT ENTRY (normal energy influence) ----
        assistant_entry = AnswerEntry(
            user_text=user_text,            # question that triggered the answer
            text=answer_text,
            embedding=embedding,
            weight=self.assistant_weight,   # strong influence
            turn_index=self.turn_index,
        )
        self.entries.append(assistant_entry)

        self.turn_index += 1

        # Enforce max buffer
        if len(self.entries) > self.max_turns:
            overflow = len(self.entries) - self.max_turns
            self.entries = self.entries[overflow:]

    def recent_entries(self, n: int) -> List[AnswerEntry]:
        """Return up to n most recent answer entries."""
        if n <= 0:
            return []
        return self.entries[-n:]

    def message_list_with_user(self, system_prompt: str, user_msg: str) -> List[Dict[str, str]]:
        """
        Reconstruct the chat format for the LLM, preserving turns and order.
        """
        msgs: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        for e in self.entries:
            # user_text always contains the question that triggered this entry
            msgs.append({"role": "user", "content": e.user_text})
            msgs.append({"role": "assistant", "content": e.text})

        msgs.append({"role": "user", "content": user_msg})
        return msgs
