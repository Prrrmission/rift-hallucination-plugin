from dataclasses import dataclass

@dataclass
class RIFTConfig:
    # --- Generation loop ---
    max_attempts: int = 3
    max_history: int = 12
    decay_half_life_turns: int = 20

    # --- Energy model parameters ---
    kernel_power: float = 2.0
    max_sentences_for_scoring: int = 5

    contradiction_penalty: float = 1.0
    max_contradiction_penalty: float = 1.0
    contradiction_similarity_threshold: float = 0.85

    qa_relevance_weight: float = 0.7

    anchor_align_weight: float = 1.0
    anchor_contra_weight: float = 2.0

    basin_weight: float = 0.5
    enable_basins: bool = True

    # --- Energy thresholds ---
    energy_threshold: float = 0.25
    collapse_energy_threshold: float = 0.7
    verification_energy_threshold: float = 0.20

    # --- Logging ---
    enable_logging: bool = True

    # --- User truth influence ---
    # User statements exert near-zero truth pressure
    user_prop_weight: float = 0.01

    # --- Truth Gate ---
    # RIFT only accepts an answer as "real" if:
    #    truth_support = A_align + basin >= truth_support_min
    #
    require_truth_support: bool = True
    truth_support_min: float = 0.05
