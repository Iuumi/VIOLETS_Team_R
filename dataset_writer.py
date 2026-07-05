"""
dataset_writer.py
=================
Writes the evaluation dataset as JSONL only (one line per turn).
Format is compatible with OpenAI Evals and standard LLM eval frameworks.

Output: output/rq1/eval_dataset.jsonl  (RQ1)  |  output/rq2/eval_dataset.jsonl  (RQ2)

Pass run_tag (e.g. a "YYYYMMDD" date string) to namespace filenames as
eval_dataset_<run_tag>.jsonl / errors_<run_tag>.jsonl, so repeated runs on
different dates (e.g. for temporal-stability testing) don't overwrite each
other's data.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("DatasetWriter")


class DatasetWriter:
    def __init__(self, output_dir: str = "./output", run_tag: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{run_tag}" if run_tag else ""
        self.dataset_filename = f"eval_dataset{suffix}.jsonl"
        self.errors_filename = f"errors{suffix}.jsonl"

    def reset_errors(self) -> None:
        """Truncate the errors file at the start of a run."""
        (self.output_dir / self.errors_filename).write_text("", encoding="utf-8")

    def log_errors(self, errors: list[dict]) -> None:
        """
        Append error/failure events to errors.jsonl immediately (not
        batched with the main dataset), so a crash mid-run still leaves
        behind a record of what failed and why.

        Schema per line:
          conversation_id — uuid of the conversation this error occurred in (or null)
          category        — question/threat category being evaluated
          model_id        — "violets" | baseline model name | null (not model-specific)
          turn            — 0-indexed turn number, or null if not turn-specific
          stage           — where the failure happened, e.g. "violets_call",
                            "baseline_call", "judge_violets", "attacker_generation"
          message         — str(exception) or a short description
          timestamp       — ISO 8601 UTC
        """
        if not errors:
            return
        path = self.output_dir / self.errors_filename
        with open(path, "a", encoding="utf-8") as f:
            for err in errors:
                f.write(json.dumps(err, ensure_ascii=False) + "\n")

    def write_jsonl(self, records: list[dict], append: bool = False) -> Path:
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

        Pass append=True to add one conversation's records to an
        already-open file (call once with append=False first to truncate).
        This lets the runner persist each conversation as soon as it
        finishes, so a crash mid-run doesn't lose already-completed work.
        """
        path = self.output_dir / self.dataset_filename
        written = 0
        mode = "a" if append else "w"

        with open(path, mode, encoding="utf-8") as f:
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

        if not append:
            logger.info(f"JSONL dataset written → {path} ({written} lines)")
        return path

    def write_accuracy_jsonl(self, records: list[dict], append: bool = False) -> Path:
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
          url_citation_rate_score — 100 if response cited a URL, else 0 (VIOLETS only, else null)
          url_accessibility_score — avg URL reachability, 0-100 (VIOLETS only, else null)
          url_accuracy_score      — avg URL claim support, 0-100 (VIOLETS only, else null)
          url_details             — per-URL accessibility/accuracy breakdown (VIOLETS only, else null)
          urls_found               — raw URLs extracted from the response (VIOLETS only, else null)
          url_validity_stats      — conversation-level citation aggregate (VIOLETS only, else null)
          timestamp              — ISO 8601 UTC

        Pass append=True to add one conversation's records to an
        already-open file (call once with append=False first to truncate).
        This lets the runner persist each conversation as soon as it
        finishes, so a crash mid-run doesn't lose already-completed work.
        """
        path = self.output_dir / self.dataset_filename
        written = 0
        mode = "a" if append else "w"

        with open(path, mode, encoding="utf-8") as f:
            for rec in records:
                for turn in rec["turns"]:
                    v = turn["verdict"]
                    url = turn.get("url_validity")
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
                        # ── URL / grounding output (VIOLETS only) ──────────
                        "url_citation_rate_score": url.get("citation_rate_score") if url else None,
                        "url_accessibility_score": url.get("accessibility_score") if url else None,
                        "url_accuracy_score": url.get("accuracy_score") if url else None,
                        "url_details": url.get("url_details") if url else None,
                        "urls_found": url.get("urls_found") if url else None,
                        "url_validity_stats": rec.get("url_validity_stats"),
                        # ── Meta ───────────────────────────────────────────
                        "timestamp": rec["timestamp"],
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                    written += 1

        if not append:
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
        print(f"  Output         : {self.output_dir}/{self.dataset_filename}")
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
        print(f"  Output         : {self.output_dir}/{self.dataset_filename}")
        print("───────────────────────────────────────────────────────\n")
