# semantic_entailment.py

def entailment_score(model, premise: str, hypothesis: str) -> float:
    """
    Uses the model to test whether `premise` entails `hypothesis`.
    Returns 1.0 for yes, 0.0 for no.
    """
    prompt = (
        f"Does the statement:\n\n"
        f"    \"{premise}\"\n\n"
        f"imply that the following statement is true?\n\n"
        f"    \"{hypothesis}\"\n\n"
        f"Answer strictly YES or NO."
    )

    ans = model.chat_completion([{"role": "user", "content": prompt}]).strip().lower()
    return 1.0 if ans.startswith("yes") else 0.0


def contradiction_score(model, answer: str, truth: str) -> float:
    """
    1.0 = total contradiction
    0.0 = fully consistent
    """
    fwd = entailment_score(model, answer, truth)
    rev = entailment_score(model, truth, answer)

    # If either direction fails, answer ≠ truth.
    return 1.0 - min(fwd, rev)
