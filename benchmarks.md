# Benchmarks

This repository includes a small, transparent benchmark suite designed to
evaluate **hallucination resistance under adversarial prompting**.

The goal is not to measure factual knowledge or raw accuracy, but to test whether
a system can **refuse or stabilize** when pressured to produce known falsehoods.

---

## The Gauntlet (Side-by-Side)

The primary benchmark is a **side-by-side gauntlet** comparing:

- **Baseline LLM**  
  A plain language model queried conversationally, without RIFT.

- **RIFT-wrapped LLM**  
  The same model wrapped with the RIFT engine (energy scoring, memory,
  contradiction detection, anchors, and basins).

Both systems:
- Use the **same underlying model**
- Receive the **same prompts and follow-ups**
- Are evaluated under **identical conversational pressure**

---

## Scenarios

Each scenario consists of:
- An initial factual question
- A sequence of follow-up prompts designed to:
  - coerce,
  - confuse,
  - fictionalize,
  - or explicitly instruct the model to output a false statement

Domains include:
- Geography
- Law
- Programming
- Medicine
- History
- Biology
- Meta / logical contradiction traps

Each scenario defines a **forbidden phrase** — a concrete string representing
a known false claim.

---

## Violation Metric

A **violation** occurs when the model’s output contains the forbidden phrase
(after simple normalization).

⚠️ **Important note on scoring**

The current violation check is intentionally simple and conservative.
It is a literal string-based test and may **over-count** cases where the model
mentions the forbidden phrase in order to explicitly refute or negate it.

For this reason:
- Numeric violation counts should be treated as a **rough indicator**, not a definitive score.
- The **raw JSON outputs** are the primary artifact of interest and are provided
  so readers can apply their own interpretation or scoring logic.

Reported per scenario:
- `num_violations` – number of turns where the forbidden phrase appeared
- `num_turns` – total turns in the scenario
- `violation_rate` – `violations / turns`

Lower violation rates generally indicate stronger resistance to hallucination
under pressure, but final judgment should be based on inspection of outputs.

---

## State Reset Policy

To ensure fair comparison:

- **RIFT state is fully reset per scenario**, including:
  - conversation memory,
  - in-memory anchors,
  - semantic basins,
  - and any persisted basin data.

- The baseline model maintains conversational context per scenario but does not
  retain state across scenarios.

Each scenario therefore begins with a **clean state** for both systems.

---

## Output Artifacts

Each benchmark run produces:
- Console logs showing each turn
- A JSON file containing full structured results:
  - per engine,
  - per scenario,
  - per turn

These artifacts are intended to be reproducible, inspectable, and easy to reanalyze.

---

## Known Limitations

- The violation metric does not distinguish between **asserting** a false claim
  and **explicitly refuting** it.
- The benchmark favors truth preservation over cooperative fictionalization;
  some models may refuse even explicitly fictional prompts when they collide
  with anchored facts.
- Results are sensitive to model choice and prompting style; absolute numbers
  may vary between runs.

These limitations are acknowledged by design. The benchmark prioritizes
transparency and behavioral clarity over exhaustive scoring.

---

## What This Benchmark Does *Not* Claim

This benchmark does **not** claim to measure:
- general intelligence,
- factual recall,
- reasoning depth,
- or downstream task performance.

It specifically measures:
> **Whether a system resists confidently asserting known falsehoods when pressured.**

---

## Reproducibility Notes

- Different models may produce different absolute results.
- Relative behavior (baseline vs RIFT) is the primary signal of interest.
- Benchmark scripts reset all relevant state automatically per scenario.

---

## Summary

This benchmark is intentionally small, explicit, and conservative.

Its purpose is to demonstrate **behavioral differences**, not to win a leaderboard.

Questions or critiques are welcome via issues or direct contact.
