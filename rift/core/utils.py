from __future__ import annotations
import math
import re
from typing import Sequence, List


def cosine(u: Sequence[float], v: Sequence[float]) -> float:
    dot = 0.0
    nu = 0.0
    nv = 0.0
    for a, b in zip(u, v):
        dot += a * b
        nu += a * a
        nv += b * b
    if nu == 0 or nv == 0:
        return 0.0
    return dot / (math.sqrt(nu) * math.sqrt(nv))


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - (dot / (math.sqrt(na) * math.sqrt(nb)))


def split_into_sentences(text: str, max_sentences: int) -> List[str]:
    if not text.strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts[:max_sentences]


def normalize_literal(s: str) -> str:
    s = s.strip().lower()
    s = s.strip('\'"“”‘’')
    s = re.sub(r'[.!?]+$', '', s)
    return s


def simple_negation_flag(text: str) -> bool:
    t = " " + text.lower() + " "
    return any(n in t for n in [" not ", " no ", " never ", " none ", "n't "])
