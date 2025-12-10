from __future__ import annotations
from typing import Optional, Dict, List

from rift.core.utils import split_into_sentences, cosine_distance, cosine
from rift.core.truth_field import detect_contradictions
from rift.memory.decay import compute_age_decayed_weights
from rift.memory.history import ConversationMemory
from config import RIFTConfig


def compute_answer_energy(
    draft: str,
    llm,
    memory: ConversationMemory,
    config: RIFTConfig,
    question_emb: Optional[List[float]] = None,
):
    sentences = split_into_sentences(draft, config.max_sentences_for_scoring)
    full_emb = llm.embed_text(draft)

    entries = memory.recent_entries(config.max_history)

    # No history → only QA relevance
    if not entries:
        qa = 0.0
        if question_emb is not None:
            sim = cosine(question_emb, full_emb)
            qa = max(0.0, 1 - max(sim, 0.0)) * config.qa_relevance_weight

        return qa, {
            "base": 0.0,
            "qa": qa,
            "contra": 0.0,
            "sentences": len(sentences),
            "contradictions": 0,
            "embedding": full_emb,
        }

    weighted = compute_age_decayed_weights(entries, memory.turn_index, config.decay_half_life_turns)

    def stability(emb):
        support = 0.0
        wsum = 0.0
        for e, w in weighted:
            sim = max(0.0, min(1.0, 1 - cosine_distance(emb, e.embedding)))
            support += w * (sim ** config.kernel_power)
            wsum += w
        if wsum == 0:
            return 0.0
        return min(1.0, support / wsum)

    if sentences:
        base = sum(1 - stability(llm.embed_text(s)) for s in sentences) / len(sentences)
    else:
        base = 1 - stability(full_emb)

    contradictions, contra_penalty = detect_contradictions(draft, full_emb, weighted, config)

    qa_penalty = 0.0
    if question_emb is not None:
        sim = cosine(question_emb, full_emb)
        qa_penalty = max(0.0, 1 - max(sim, 0.0)) * config.qa_relevance_weight

    total = base + contra_penalty + qa_penalty

    return total, {
        "base": base,
        "qa": qa_penalty,
        "contra": contra_penalty,
        "sentences": len(sentences),
        "contradictions": contradictions,
        "embedding": full_emb,
    }
