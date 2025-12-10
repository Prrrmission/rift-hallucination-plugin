from __future__ import annotations
import math
from typing import List, Tuple
from rift.memory.history import AnswerEntry


def compute_age_decayed_weights(
    entries: List[AnswerEntry],
    current_turn: int,
    half_life: int,
) -> List[Tuple[AnswerEntry, float]]:
    if not entries:
        return []

    if half_life <= 0:
        return [(e, e.weight) for e in entries]

    lam = math.log(2) / half_life
    out = []
    for e in entries:
        age = max(current_turn - e.turn_index, 0)
        w = e.weight * math.exp(-lam * age)
        out.append((e, w))
    return out
