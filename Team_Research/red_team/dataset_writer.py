"""
dataset_writer.py
=================
Writes the evaluation dataset as JSONL only (one line per turn).
Format is compatible with OpenAI Evals and standard LLM eval frameworks.

Output: output/rq1/eval_dataset.jsonl  (RQ1)  |  output/rq2/eval_dataset.jsonl  (RQ2)
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger("DatasetWriter")


class DatasetWriter:
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(self, records: list[dict]) -> Path:
        """
        Write one JSON line per turn to eval_dataset.jsonl.

        Schema per line:
          conversation_id   — uuid grouping all turns in a conversation
          category          — threat category probed
          seed_prompt       — the opening adversarial message
          seed_intent       — what the attacker was trying to achieve
          seed_technique    — adversarial technique used
          turn              — 0-indexed turn number within the conversation
          input             — attacker's message this turn
          output            — VIOLETS's response this turn
          label             — PASS | WARN | FAIL
          safety_score      — 0.0 (unsafe) to 1.0 (safe)
          violation         — which category was violated, or "none"
          reasoning         — one-sentence judge explanation
          escalation_note   — whether attacker made progress
          flagged           — true if label == FAIL
          conversation_flagged — true if ANY turn in this conversation was FAIL
          timestamp         — ISO 8601 UTC
        """
        path = self.output_dir / "eval_dataset.jsonl"
        written = 0

        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                conv_flagged = any(
                    t["verdict"]["label"] == "FAIL" for t in rec["turns"]
                )
                for turn in rec["turns"]:
                    v = turn["verdict"]
                    line = {
                        # ── Conversation metadata ──────────────────────────
                        "conversation_id": rec["conversation_id"],
                        "model_id": rec.get("model_id", "violets"),  # "violets" | baseline model name
                        "category": rec["category"],
                        "seed_prompt": rec["seed"]["prompt"],
                        "seed_intent": rec["seed"].get("intent", ""),
                        "seed_technique": rec["seed"].get("technique", ""),
                        # ── Turn data ──────────────────────────────────────
                        "turn": turn["turn"],
                        "input": turn["attacker_message"],
                        "output": turn["agent_response"],
                        # ── Judge output ───────────────────────────────────
                        "label": v["label"],
                        "safety_score": v["safety_score"],
                        "overall_safety_score": rec["overall_safety_score"],
                        "violation": v.get("violation", "none"),
                        "reasoning": v.get("reasoning", ""),
                        "escalation_note": v.get("escalation_note", ""),
                        # ── Flags ──────────────────────────────────────────
                        "flagged": v["label"] == "FAIL",
                        "conversation_flagged": conv_flagged,
                        # ── Meta ───────────────────────────────────────────
                        "timestamp": rec["timestamp"],
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                    written += 1

        logger.info(f"JSONL dataset written → {path} ({written} lines)")
        return path

    def write_accuracy_jsonl(self, records: list[dict]) -> Path:
        """
        Write one JSON line per turn to eval_dataset.jsonl (accuracy evaluation).

        Schema per line:
          conversation_id        — uuid grouping all turns in a conversation
          model_id               — "violets" | baseline model name
          category               — FAQ question type
          seed_prompt            — the opening query
          seed_intent            — what information the participant sought
          seed_question_type     — question type category
          turn                   — 0-indexed turn number within the conversation
          input                  — participant's message this turn
          output                 — model's response this turn
          veracity_score         — 0 (completely false) to 100 (completely accurate)
          overall_veracity_score — avg veracity score across all turns
          reasoning              — one-sentence judge explanation
          timestamp              — ISO 8601 UTC
        """
        path = self.output_dir / "eval_dataset.jsonl"
        written = 0

        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                for turn in rec["turns"]:
                    v = turn["verdict"]
                    line = {
                        # ── Conversation metadata ──────────────────────────
                        "conversation_id": rec["conversation_id"],
                        "model_id": rec.get("model_id", "violets"),
                        "category": rec["category"],
                        "seed_prompt": rec["seed"]["prompt"],
                        "seed_intent": rec["seed"].get("intent", ""),
                        "seed_question_type": rec["seed"].get("question_type", ""),
                        # ── Turn data ──────────────────────────────────────
                        "turn": turn["turn"],
                        "input": turn["participant_message"],
                        "output": turn["agent_response"],
                        # ── Judge output ───────────────────────────────────
                        "veracity_score": v["veracity_score"],
                        "overall_veracity_score": rec["overall_veracity_score"],
                        "reasoning": v.get("reasoning", ""),
                        # ── Meta ───────────────────────────────────────────
                        "timestamp": rec["timestamp"],
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                    written += 1

        logger.info(f"Accuracy JSONL written → {path} ({written} lines)")
        return path

    def write_accuracy_stats(self, records: list[dict]) -> None:
        """Print a brief accuracy evaluation summary to stdout."""
        total_convs = len(set(r["conversation_id"] for r in records))
        total_turns = sum(len(r["turns"]) for r in records)

        by_model: dict[str, list[float]] = {}
        for r in records:
            model = r.get("model_id", "violets")
            valid = [
                t["verdict"]["veracity_score"]
                for t in r["turns"]
                if t["verdict"]["veracity_score"] is not None
            ]
            if valid:
                by_model.setdefault(model, []).append(sum(valid) / len(valid))

        print("\n── Accuracy Evaluation Complete ───────────────────────")
        print(f"  Conversations  : {total_convs}")
        print(f"  Total turns    : {total_turns}")
        for model, avgs in by_model.items():
            avg = sum(avgs) / len(avgs) if avgs else 0
            print(f"  {model} avg veracity : {avg:.1f}/100")
        print(f"  Output         : {self.output_dir}/eval_dataset.jsonl")
        print("───────────────────────────────────────────────────────\n")

    def write_stats(self, records: list[dict]) -> None:
        """Print a brief summary to stdout after writing."""
        total_convs = len(records)
        all_turns = [t for r in records for t in r["turns"]]
        total_turns = len(all_turns)
        fail_turns = sum(1 for t in all_turns if t["verdict"]["label"] == "FAIL")
        warn_turns = sum(1 for t in all_turns if t["verdict"]["label"] == "WARN")
        flagged_convs = sum(1 for r in records if r["flagged"])

        print("\n── Red-Team Run Complete ──────────────────────────────")
        print(f"  Conversations  : {total_convs}")
        print(f"  Total turns    : {total_turns}")
        print(f"  PASS           : {total_turns - fail_turns - warn_turns}")
        print(f"  WARN           : {warn_turns}")
        print(f"  FAIL           : {fail_turns}")
        print(f"  Flagged convs  : {flagged_convs}")
        print(f"  Output         : {self.output_dir}/eval_dataset.jsonl")
        print("───────────────────────────────────────────────────────\n")
