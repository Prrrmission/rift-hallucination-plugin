from __future__ import annotations
from typing import List, Tuple
from config import RIFTConfig

from rift.core.utils import simple_negation_flag, cosine_distance
from rift.memory.history import AnswerEntry


def detect_contradictions(
    candidate_text: str,
    candidate_emb: List[float],
    weighted_entries: List[Tuple[AnswerEntry, float]],
    config: RIFTConfig,
) -> Tuple[int, float]:
    """
    Pure, domain-agnostic contradiction detector.

    Idea:
    - If candidate has opposite negation polarity from a strongly similar
      past answer, penalize as a contradiction.
    - No hand-built slots, no special myths, no regex heuristics.
    """
    if not weighted_entries:
        return 0, 0.0

    contradictions = 0
    penalty = 0.0

    cand_neg = simple_negation_flag(candidate_text)

    for entry, w in weighted_entries:
        entry_neg = simple_negation_flag(entry.text)

        # Only care about "yes/no" flips
        if cand_neg == entry_neg:
            continue

        sim = 1.0 - cosine_distance(candidate_emb, entry.embedding)
        if sim >= config.contradiction_similarity_threshold:
            contradictions += 1
            penalty += config.contradiction_penalty

    penalty = min(penalty, config.max_contradiction_penalty)
    return contradictions, penalty
