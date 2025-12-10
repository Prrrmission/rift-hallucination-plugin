"""
Run HaluEval Dialogue locally with:
  - baseline OpenAI LLM (no RIFT)
  - RIFT-wrapped LLM (RIFTEngine)

Place this file in: HaluEval/evaluation/halueval_dialogue_rift.py
and run from the HaluEval/evaluation directory:

  python halueval_dialogue_rift.py --engine baseline
  python halueval_dialogue_rift.py --engine rift

Results are written to: dialogue/dialogue_<engine>_results.json
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import List

# ---------- IMPORT YOUR RIFT STACK ----------
# Adjust these imports if your package layout is slightly different.
from config import RIFTConfig
from rift.llm.openai_client import OpenAIClient
from rift.core.engine import RIFTEngine
from rift.memory.history import ConversationMemory
# -------------------------------------------


# ============================================================
#  Engine wrappers
# ============================================================

class BaseJudgeEngine:
    """Protocol-ish base: subclasses implement name/reset/judge()."""

    def name(self) -> str:
        raise NotImplementedError

    def reset(self) -> None:
        """Reset any internal state between samples, if needed."""
        pass

    def judge(self, system_prompt: str, user_prompt: str) -> str:
        """
        Given (system, user) chat content, return a string answer.

        MUST ultimately answer "Yes" or "No" (possibly with extra text)
        for the benchmark to parse.
        """
        raise NotImplementedError


class BaselineJudge(BaseJudgeEngine):
    """
    Plain OpenAIClient, no RIFT.
    Uses your OpenAIClient.chat_completion directly.
    """

    def __init__(self):
        self._llm = OpenAIClient()

    def name(self) -> str:
        return "baseline_openai"

    def judge(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resp = self._llm.chat_completion(messages)
        return resp.strip()


class RIFTJudge(BaseJudgeEngine):
    """
    RIFTEngine used as a hallucination *detector*.

    Since RIFTEngine.generate(user_msg: str) assumes a blank system prompt,
    we just pack the system + user into a single user_msg string.
    """

    def __init__(self):
        cfg = RIFTConfig()
        llm = OpenAIClient()
        memory = ConversationMemory()
        self._engine = RIFTEngine(llm=llm, memory=memory, config=cfg)
        self._memory = memory

    def name(self) -> str:
        return "rift_v3"

    def reset(self) -> None:
        # Fresh field / anchors per run if you want.
        # If you want cross-sample memory, comment this out.
        self._memory.reset()

    def judge(self, system_prompt: str, user_prompt: str) -> str:
        combined = system_prompt + "\n\n" + user_prompt
        resp = self._engine.generate(combined)
        return resp.strip()


# ============================================================
#  HaluEval dialogue evaluation (adapted from official code)
# ============================================================

SYSTEM_PROMPT = (
    "You are a response judge. "
    "You MUST determine if the provided response contains non-factual "
    "or hallucinated information.\n"
    "The answer you give MUST be \"Yes\" or \"No\"."
)


def get_dialogue_response(
    engine: BaseJudgeEngine,
    dialog: str,
    response: str,
    instruction: str,
) -> str:
    """
    Build the same prompt structure as HaluEval's official evaluation,
    but route it through our engine abstraction.
    """
    user_content = (
        instruction
        + "\n\n#Dialogue History#: " + dialog
        + "\n#Response#: " + response
        + "\n#Your Judgement#: "
    )
    return engine.judge(SYSTEM_PROMPT, user_content)


@dataclass
class EvalStats:
    correct: int = 0
    incorrect: int = 0

    @property
    def total(self) -> int:
        return self.correct + self.incorrect

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def dump_jsonl(data, output_path: str, append: bool = False) -> None:
    mode = "a+" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        json_record = json.dumps(data, ensure_ascii=False)
        f.write(json_record + "\n")


def evaluation_dialogue_dataset(
    engine: BaseJudgeEngine,
    file: str,
    instruction: str,
    output_path: str,
) -> EvalStats:
    # Load dataset (one JSON object per line)
    with open(file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    stats = EvalStats()

    # Evaluate each sample
    for i, sample in enumerate(data):
        knowledge = sample["knowledge"]
        dialog = sample["dialogue_history"]
        hallucinated_response = sample["hallucinated_response"]
        right_response = sample["right_response"]

        # Randomly choose hallucinated or correct, as in the official script
        if random.random() > 0.5:
            response = hallucinated_response
            ground_truth = "Yes"  # "Yes = contains hallucination"
        else:
            response = right_response
            ground_truth = "No"

        # Reset engine state per sample if desired
        engine.reset()

        ans = get_dialogue_response(engine, dialog, response, instruction)
        ans = ans.replace(".", "")

        # Official HaluEval parsing logic
        if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
            # Model failed to commit to Yes/No
            gen = {
                "knowledge": knowledge,
                "dialogue_history": dialog,
                "response": response,
                "ground_truth": ground_truth,
                "judgement": "failed!",
                "raw_answer": ans,
            }
            dump_jsonl(gen, output_path, append=True)
            stats.incorrect += 1
            print(f"sample {i} fails (no clean Yes/No)......")
            continue
        elif "Yes" in ans:
            if ans != "Yes":
                ans = "Yes"
            gen = {
                "knowledge": knowledge,
                "dialogue_history": dialog,
                "response": response,
                "ground_truth": ground_truth,
                "judgement": ans,
            }
        elif "No" in ans:
            if ans != "No":
                ans = "No"
            gen = {
                "knowledge": knowledge,
                "dialogue_history": dialog,
                "response": response,
                "ground_truth": ground_truth,
                "judgement": ans,
            }
        else:
            # Should not reach here due to earlier checks
            gen = None

        assert gen is not None

        if ground_truth == ans:
            stats.correct += 1
        else:
            stats.incorrect += 1

        print(f"sample {i} success......")
        dump_jsonl(gen, output_path, append=True)

    print(
        f"{stats.correct} correct samples, {stats.incorrect} incorrect samples, "
        f"Accuracy: {stats.accuracy:.4f}"
    )
    return stats


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="HaluEval Dialogue (RIFT / baseline)")
    parser.add_argument(
        "--engine",
        choices=["baseline", "rift"],
        default="baseline",
        help="Which engine to use: baseline or rift",
    )
    args = parser.parse_args()

    if args.engine == "baseline":
        engine: BaseJudgeEngine = BaselineJudge()
    else:
        engine = RIFTJudge()

    # Paths matching official evaluate.py layout
    instruction_file = "HaluEval-main/evaluation/dialogue/dialogue_evaluation_instruction.txt"
    data_path = "HaluEval-main/data/dialogue_data.json"
    output_path = f"HaluEval-main/evaluation/dialogue/dialogue_{engine.name()}_results.json"

    with open(instruction_file, "r", encoding="utf-8") as f:
        instruction = f.read()

    print(f"Running HaluEval Dialogue with engine={engine.name()}")
    print(f"Data: {data_path}")
    print(f"Instruction file: {instruction_file}")
    print(f"Output -> {output_path}")

    stats = evaluation_dialogue_dataset(engine, data_path, instruction, output_path)

    print(
        f"\nFinal ({engine.name()}): "
        f"{stats.correct}/{stats.total} correct "
        f"({stats.accuracy * 100:.2f}% accuracy)"
    )


if __name__ == "__main__":
    main()
