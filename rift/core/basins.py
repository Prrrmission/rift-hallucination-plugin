from __future__ import annotations

import json
import os
from typing import Dict, Tuple, Any, List

import numpy as np


class BasinStore:
    """
    Simple embedding basin store:

      key -> center vector (np.ndarray)

    - score(emb) returns (similarity, details)
    - reinforce(key, emb) updates the basin center
    - save/load to JSON as key -> list[float]

    This version is robust to old / corrupted JSON files:
    if the loaded structure is not a dict, we reset to {}.
    """

    def __init__(self, path: str):
        self.path = path
        self.basins: Dict[str, np.ndarray] = {}
        self.load()

    # ---------------------------
    # Persistence
    # ---------------------------

    def load(self) -> None:
        """Load basins from JSON, if present. Reset to {} on any mismatch."""
        if not os.path.exists(self.path):
            self.basins = {}
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                self.basins = {
                    k: np.array(v, dtype=float) for k, v in data.items()
                }
            else:
                # Old / unexpected format -> drop it
                self.basins = {}
        except Exception:
            # Any error -> start fresh
            self.basins = {}

    def save(self) -> None:
        """Save basins as key -> list[float]."""
        # If something went wrong and basins isn't a dict, reset it
        if not isinstance(self.basins, dict):
            self.basins = {}

        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        pack = {k: v.tolist() for k, v in self.basins.items()}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(pack, f)

    # ---------------------------
    # Core operations
    # ---------------------------

    def score(self, emb: List[float]) -> Tuple[float, Dict[str, Any]]:
        """
        Return (similarity, details) where similarity is the best cosine sim
        between `emb` and any stored basin center.
        """
        if not self.basins:
            return 0.0, {"best_key": None}

        arr = np.array(emb, dtype=float)

        best_key = None
        best_sim = -1.0

        for k, center in self.basins.items():
            num = float(np.dot(arr, center))
            denom = float(np.linalg.norm(arr) * np.linalg.norm(center))
            sim = 0.0 if denom == 0.0 else num / denom

            if sim > best_sim:
                best_sim = sim
                best_key = k

        return best_sim, {"best_key": best_key}

    def reinforce(self, key: str, emb: List[float], alpha: float = 0.3) -> None:
        """
        Move the basin center for `key` toward `emb` by fraction alpha.
        If key doesn't exist yet, we create it.
        """
        if not isinstance(self.basins, dict):
            self.basins = {}

        arr = np.array(emb, dtype=float)

        if key in self.basins:
            self.basins[key] = (1.0 - alpha) * self.basins[key] + alpha * arr
        else:
            self.basins[key] = arr

        self.save()
