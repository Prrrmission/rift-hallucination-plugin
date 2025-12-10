from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
import math

from .semantic_entailment import entailment_score, contradiction_score


@dataclass
class Anchor:
    value_text: str
    embedding: List[float]
    confidence: float = 1.0


# Map: question_key -> list of anchors for that question
ANCHORS: Dict[str, List[Anchor]] = {}


def _cosine(u: List[float], v: List[float]) -> float:
    if not u or not v or len(u) != len(v):
        return 0.0
    num = sum(a * b for a, b in zip(u, v))
    du = math.sqrt(sum(a * a for a in u))
    dv = math.sqrt(sum(b * b for b in v))
    if du == 0.0 or dv == 0.0:
        return 0.0
    return num / (du * dv)


def question_key_from_prompt(prompt: str) -> Optional[str]:
    """
    Non-heuristic, per-question key:
    - strip whitespace
    - lowercase

    This keeps anchors local to a specific literal question form.
    """
    key = prompt.strip().lower()
    return key or None


def score_against_anchors(
    answer_text: str,
    anchors: Dict[str, List[Anchor]],
    llm_or_embed,
) -> tuple[float, float]:
    """
    Compute semantic agreement / contradiction against all known anchors.

    Returns:
        A_align  = max semantic agreement strength (0–1)
        A_contra = max semantic contradiction strength (0–1)

    Only anchors that are semantically close in embedding space
    are allowed to exert any force.
    """

    # Resolve client that has both chat + embeddings
    client = None
    if hasattr(llm_or_embed, "chat_completion") and hasattr(llm_or_embed, "embed_text"):
        client = llm_or_embed
    elif (
        hasattr(llm_or_embed, "__self__")
        and hasattr(llm_or_embed.__self__, "chat_completion")
        and hasattr(llm_or_embed.__self__, "embed_text")
    ):
        client = llm_or_embed.__self__

    if client is None or not anchors:
        return 0.0, 0.0

    # Embed the answer once
    ans_emb = client.embed_text(answer_text)

    # Locality threshold: only nearby anchors count
    relevance_threshold = 0.50

    best_align = 0.0
    best_contra = 0.0

    # Consider all anchors across all question keys,
    # but only if they are embedding-close to the answer.
    for anchor_list in anchors.values():
        for anchor in anchor_list:
            sim = _cosine(ans_emb, anchor.embedding)
            if sim < relevance_threshold:
                # Too far in meaning → no influence
                continue

            truth = anchor.value_text

            align = entailment_score(client, answer_text, truth)
            contra = contradiction_score(client, answer_text, truth)

            if align > best_align:
                best_align = align
            if contra > best_contra:
                best_contra = contra

    return best_align, best_contra


def maybe_update_anchor(
    prompt: str,
    answer_text: str,
    energy: float,
    embed_fn,
    config,
) -> None:
    """
    If the answer is low-energy (trusted), register it as an anchor
    for its literal question key, and slightly reinforce existing
    anchors under that key.
    """
    key = question_key_from_prompt(prompt)

    # No question key → don't anchor
    if key is None:
        return

    # Only anchor if we are below the verification threshold
    if energy > getattr(config, "verification_energy_threshold", 0.5):
        return

    ans_emb = embed_fn(answer_text)

    if key not in ANCHORS:
        ANCHORS[key] = []

    # Slightly reinforce existing anchors for this question
    for anchor in ANCHORS[key]:
        anchor.confidence = min(anchor.confidence + 0.05, 1.0)

    # Add new anchor for this question
    ANCHORS[key].append(
        Anchor(
            value_text=answer_text,
            embedding=ans_emb,
            confidence=0.5,
        )
    )
