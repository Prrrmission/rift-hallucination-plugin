from __future__ import annotations
from typing import Optional, List

from config import RIFTConfig
from rift.core.energy import compute_answer_energy
from rift.core.anchors import (
    ANCHORS,
    score_against_anchors,
    maybe_update_anchor,
    question_key_from_prompt,
)
from rift.core.basins import BasinStore
from rift.memory.history import ConversationMemory


BASINS = BasinStore("data/rift_basins.json")


class RIFTEngine:
    def __init__(self, llm, memory: ConversationMemory, config: RIFTConfig):
        self.llm = llm
        self.memory = memory
        self.config = config

    def _refusal_text(self, user_msg: str) -> str:
        return (
            "I cannot assert that. It does not align with established facts or "
            "the semantic truth structure of this domain."
        )

    def generate(self, user_msg: str) -> str:
        # Single global key (anchors.py returns "global")
        key = question_key_from_prompt(user_msg)

        # Question embedding for QA relevance
        q_emb = self.llm.embed_text(user_msg)

        best_text: Optional[str] = None
        best_emb: Optional[List[float]] = None
        best_energy: Optional[float] = None
        best_A_align: float = 0.0
        best_A_contra: float = 0.0
        best_basin_pull: float = 0.0

        accepted_as_truth: bool = True  # will flip to False on refusal

        # =============================================================
        # MULTI-ATTEMPT SAMPLING LOOP
        # =============================================================
        for attempt in range(1, self.config.max_attempts + 1):
            # No system prompt: pure RIFT physics
            msgs = self.memory.message_list_with_user("", user_msg)

            if attempt > 1:
                msgs.append(
                    {
                        "role": "system",
                        "content": (
                            "Re-answer using only established, widely-verified facts. "
                            "Avoid speculation or fabricated scenarios."
                        ),
                    }
                )

            draft = self.llm.chat_completion(msgs).strip()

            # Core energy (stability + QA + internal contradictions)
            raw_E, info = compute_answer_energy(
                draft,
                self.llm,
                self.memory,
                self.config,
                question_emb=q_emb,
            )

            # Semantic anchor alignment / contradiction (global field)
            A_align, A_contra = score_against_anchors(
                draft,
                ANCHORS,
                self.llm,
            )

            # Basin pull (only if not contradicting anchors)
            basin_pull = 0.0
            if self.config.enable_basins and A_contra == 0.0:
                basin_pull, _ = BASINS.score(info["embedding"])

            E = (
                raw_E
                + self.config.anchor_contra_weight * A_contra
                - self.config.anchor_align_weight * A_align
                - self.config.basin_weight * basin_pull
            )

            if self.config.enable_logging:
                print(
                    f"[RIFT] Attempt {attempt}: energy={E:.3f} "
                    f"(base={info['base']:.3f}, "
                    f"qa={info['qa']:.3f}, "
                    f"contra={A_contra:.3f}, "
                    f"align={A_align:.3f}, "
                    f"basin={basin_pull:.3f})"
                )

            # Track best candidate
            if best_energy is None or E < best_energy:
                best_energy = E
                best_text = draft
                best_emb = info["embedding"]
                best_A_align = A_align
                best_A_contra = A_contra
                best_basin_pull = basin_pull

            if E <= self.config.energy_threshold:
                break

        # Safety
        assert best_text is not None
        assert best_emb is not None
        assert best_energy is not None

        # Are there any anchors yet for this key?
        has_anchors = bool(ANCHORS.get(key))

        # =============================================================
        # 1) HIGH-ENERGY OVERRIDE → USE ANCHOR IF AVAILABLE
        # =============================================================
        used_anchor = False
        if key and has_anchors and best_energy > self.config.collapse_energy_threshold:
            anchors_for_key = ANCHORS.get(key) or []
            if anchors_for_key:
                strongest = max(anchors_for_key, key=lambda a: a.confidence)
                best_text = strongest.value_text
                best_emb = strongest.embedding
                best_energy = self.config.verification_energy_threshold
                used_anchor = True

                if self.config.enable_logging:
                    print(f"[RIFT] Override → using anchor truth for key '{key}'")

        # =============================================================
        # 2) IF STILL UNSTABLE → SNAP TO ANCHOR IF POSSIBLE, ELSE REFUSE
        # =============================================================
        if best_energy > self.config.collapse_energy_threshold and not used_anchor:
            anchors_for_key = ANCHORS.get(key) or []
            if anchors_for_key:
                strongest = max(anchors_for_key, key=lambda a: a.confidence)
                if self.config.enable_logging:
                    print(
                        f"[RIFT] High-energy candidate (E={best_energy:.3f}) "
                        f"→ snapping to anchor truth for key '{key}'."
                    )
                best_text = strongest.value_text
                best_emb = strongest.embedding
                best_energy = self.config.verification_energy_threshold
                accepted_as_truth = True
            else:
                if self.config.enable_logging:
                    print(
                        f"[RIFT] High-energy candidate (E={best_energy:.3f}) "
                        f"with no anchors → refusal."
                    )
                best_text = self._refusal_text(user_msg)
                best_energy = self.config.verification_energy_threshold
                accepted_as_truth = False

        # =============================================================
        # 3) TRUTH-GATE (SEMANTIC ONLY, ONLY WHEN ANCHORS EXIST)
        # =============================================================
        if self.config.require_truth_support and has_anchors:
            truth_support = best_A_align + best_basin_pull

            if self.config.enable_logging:
                print(
                    f"[RIFT] Truth-gate: align={best_A_align:.3f}, "
                    f"contradiction={best_A_contra:.3f}, "
                    f"basin={best_basin_pull:.3f}, "
                    f"support={truth_support:.3f}, "
                    f"threshold={self.config.truth_support_min:.3f}"
                )

            if best_A_contra > 0.0 or truth_support < self.config.truth_support_min:
                anchors_for_key = ANCHORS.get(key) or []
                if anchors_for_key:
                    strongest = max(anchors_for_key, key=lambda a: a.confidence)
                    if self.config.enable_logging:
                        print(
                            "[RIFT] Truth-gate → snapping to anchor truth instead "
                            "of refusing."
                        )
                    best_text = strongest.value_text
                    best_emb = strongest.embedding
                    best_energy = self.config.verification_energy_threshold
                    accepted_as_truth = True
                else:
                    if self.config.enable_logging:
                        print("[RIFT] Truth-gate → refusing answer (no anchors).")
                    best_text = self._refusal_text(user_msg)
                    best_energy = self.config.verification_energy_threshold
                    accepted_as_truth = False
        else:
            if self.config.enable_logging:
                print(
                    "[RIFT] Truth-gate skipped "
                    "(no anchors yet or not required)."
                )

        # =============================================================
        # MEMORY + BASIN + ANCHORS UPDATE
        # =============================================================
        # We still record the conversational turn (even refusals)
        self.memory.add_turn(
            user_text=user_msg,
            answer_text=best_text,
            embedding=best_emb,
        )

        # Only reinforce basins / anchors if we actually accepted it as truth
        if (
            accepted_as_truth
            and self.config.enable_basins
            and key
            and best_A_contra == 0.0
            and best_A_align > 0.0
        ):
            BASINS.reinforce(key, best_emb)

        if accepted_as_truth:
            maybe_update_anchor(
                prompt=user_msg,
                answer_text=best_text,
                energy=best_energy,
                embed_fn=self.llm.embed_text,
                config=self.config,
            )

        return best_text
