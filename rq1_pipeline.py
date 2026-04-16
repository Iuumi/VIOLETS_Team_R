"""
RQ1 Evaluation Pipeline 

Implements the full evaluation protocol for RQ1.1 Response Veracity:
  1. Generate synthetic voter dialogues via the Participant LLM
  2. Score each VIOLETS response with the web search-enabled Evaluator LLM
  3. Flag responses below VERACITY_THRESHOLD for human review
  4. Write per-turn results to JSONL and a summary report to JSON

Reporting (per research design):
  Results are reported as M, SD, and distribution of veracity scores,
  with error-type classification for below-threshold responses.

CLI usage:
    python -m evaluation.rq1_pipeline --dialogues 10 --output results/rq1.jsonl
    python -m evaluation.rq1_pipeline --stub         # test with stub VIOLETS backend

Programmatic usage:
    from evaluation.violets_interface import set_backend
    from evaluation.rq1_pipeline import run_rq1_evaluation

    set_backend(MyVioletsBackend())
    results = run_rq1_evaluation(n_per_type=10)
    print(results.summary())
"""
import argparse
import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import (
    DIALOGUES_PER_TYPE,
    OUTPUT_DIR,
    QUESTION_CATEGORIZATION,
    VERACITY_THRESHOLD,
)
from .dialogue_generator import Dialogue, generate_dialogue
from .evaluator_llm import VeracityEvaluation, evaluate_response
from .violets_interface import StubVioletsBackend, VioletsBackend



# Data models


@dataclass
class TurnEvaluation:
    """Evaluation result for a single VIOLETS response turn."""
    dialogue_id: str
    turn_index: int
    question_type: str
    voter_query: str
    violets_response: str
    evaluation: VeracityEvaluation

    def to_dict(self) -> dict:
        return {
            "dialogue_id": self.dialogue_id,
            "turn_index": self.turn_index,
            "question_type": self.question_type,
            "voter_query": self.voter_query,
            "violets_response": self.violets_response,
            **self.evaluation.to_dict(),
        }


@dataclass
class RQ1Results:
    """Aggregated results from a full RQ1 evaluation run."""
    evaluations: list[TurnEvaluation] = field(default_factory=list)
    dialogues: list[Dialogue] = field(default_factory=list)
    run_timestamp: str = ""

    def __post_init__(self):
        if not self.run_timestamp:
            self.run_timestamp = datetime.now(timezone.utc).isoformat()

    def scored_evaluations(self) -> list[TurnEvaluation]:
        """Evaluations with a valid score (not skipped)."""
        return [e for e in self.evaluations if not e.evaluation.skipped]

    def flagged_evaluations(self) -> list[TurnEvaluation]:
        """Evaluations flagged for human review (score < VERACITY_THRESHOLD)."""
        return [e for e in self.scored_evaluations() if e.evaluation.flagged_for_review]

    def summary(self) -> dict:
        """
        Produce the summary statistics reported in the paper:
        M, SD, distribution of veracity scores, error-type breakdown.
        """
        scored = self.scored_evaluations()
        if not scored:
            return {"error": "No scored evaluations to summarise."}

        scores = [e.evaluation.veracity_score for e in scored]

        # Per question-type breakdown
        by_type: dict[str, dict] = {}
        for qtype in QUESTION_CATEGORIZATION:
            type_scores = [
                e.evaluation.veracity_score
                for e in scored
                if e.question_type == qtype
            ]
            if type_scores:
                by_type[qtype] = {
                    "n": len(type_scores),
                    "mean": round(statistics.mean(type_scores), 2),
                    "sd": round(statistics.stdev(type_scores), 2) if len(type_scores) > 1 else 0.0,
                    "min": min(type_scores),
                    "max": max(type_scores),
                    "flagged": sum(1 for s in type_scores if s < VERACITY_THRESHOLD),
                }

        # Error-type frequency across all flagged responses
        error_counts: dict[str, int] = {}
        for e in scored:
            for et in e.evaluation.error_types:
                error_counts[et] = error_counts.get(et, 0) + 1

        return {
            "run_timestamp": self.run_timestamp,
            "total_dialogues": len(self.dialogues),
            "total_evaluated": len(scored),
            "total_skipped": len(self.evaluations) - len(scored),
            "total_flagged": len(self.flagged_evaluations()),
            "flag_rate": round(len(self.flagged_evaluations()) / len(scored), 3),
            "veracity_threshold": VERACITY_THRESHOLD,
            "overall": {
                "mean": round(statistics.mean(scores), 2),
                "sd": round(statistics.stdev(scores), 2) if len(scores) > 1 else 0.0,
                "median": round(statistics.median(scores), 2),
                "min": min(scores),
                "max": max(scores),
            },
            "by_question_type": by_type,
            "error_type_distribution": dict(
                sorted(error_counts.items(), key=lambda x: -x[1])
            ),
        }



# Pipeline


def run_rq1_evaluation(
    n_per_type: int = DIALOGUES_PER_TYPE,
    violets_backend: Optional[VioletsBackend] = None,
    question_types: Optional[list[str]] = None,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> RQ1Results:
    """
    Running the full RQ1.1 evaluation pipeline.

    Args:
        n_per_type: Dialogues to generate per question type.
        violets_backend: VIOLETS query backend (stub used if None).
        question_types: Subset of question types to run (all five if None).
        output_path: Path for JSONL output; also writes <name>_summary.json.
        verbose: Print progress to stdout.

    Returns:
        RQ1Results containing all evaluations and helper methods.
    """
    if violets_backend is None:
        if verbose:
            print("[WARNING] No VIOLETS backend provided — using stub for testing.")
        violets_backend = StubVioletsBackend()

    types_to_run = question_types or list(QUESTION_CATEGORIZATION.keys())
    results = RQ1Results()

    # Initialise clients once and reuse across all calls
    from .participant_llm import _make_client as _make_participant_client
    from .evaluator_llm import _make_client as _make_evaluator_client
    participant_client = _make_participant_client()
    evaluator_client = _make_evaluator_client()

    for qtype in types_to_run:
        if verbose:
            print(f"\n[RQ1] {qtype}  ({n_per_type} dialogues)")

        for i in range(n_per_type):
            if verbose:
                print(f"  [{i + 1}/{n_per_type}] generating dialogue...", end=" ", flush=True)

            dialogue = generate_dialogue(
                question_type=qtype,
                violets_backend=violets_backend,
                participant_client=participant_client,
            )
            results.dialogues.append(dialogue)

            # Evaluate every VIOLETS turn in the dialogue
            turns = dialogue.turns
            for j, turn in enumerate(turns):
                if turn.role != "violets":
                    continue

                voter_query = turns[j - 1].content if j > 0 else ""
                evaluation = evaluate_response(
                    voter_query=voter_query,
                    violets_response=turn.content,
                    client=evaluator_client,
                )
                results.evaluations.append(
                    TurnEvaluation(
                        dialogue_id=dialogue.id,
                        turn_index=j,
                        question_type=qtype,
                        voter_query=voter_query,
                        violets_response=turn.content,
                        evaluation=evaluation,
                    )
                )

            if verbose:
                last = results.evaluations[-1]
                if last.evaluation.skipped:
                    print("skipped (too short)")
                else:
                    flag = " [FLAGGED]" if last.evaluation.flagged_for_review else ""
                    print(f"score={last.evaluation.veracity_score}{flag}")

    # Write results to disk
    if output_path:
        _write_results(results, output_path, verbose)

    if verbose:
        _print_summary(results.summary())

    return results



# Output helpers


def _write_results(results: RQ1Results, output_path: str, verbose: bool) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for e in results.evaluations:
            f.write(json.dumps(e.to_dict()) + "\n")

    summary_path = path.with_suffix("").with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(results.summary(), f, indent=2)

    if verbose:
        print(f"\n[RQ1] Evaluations → {path}")
        print(f"[RQ1] Summary     → {summary_path}")


def _print_summary(summary: dict) -> None:
    if "error" in summary:
        print(f"[RQ1] {summary['error']}")
        return

    overall = summary["overall"]
    sep = "=" * 62
    print(f"\n{sep}")
    print("  RQ1.1 Response Veracity — Summary Report")
    print(sep)
    print(f"  Dialogues generated : {summary['total_dialogues']}")
    print(f"  Responses evaluated : {summary['total_evaluated']}")
    print(f"  Responses skipped   : {summary['total_skipped']}  (under word threshold)")
    print(f"  Flagged for review  : {summary['total_flagged']}  ({summary['flag_rate']:.1%})")
    print(f"\n  Overall veracity    : M={overall['mean']}  SD={overall['sd']}  "
          f"median={overall['median']}")
    print(f"  Range               : [{overall['min']}, {overall['max']}]")

    print("\n  By question type:")
    for qtype, st in summary["by_question_type"].items():
        print(f"    {qtype:<22s}  M={st['mean']:5.1f}  SD={st['sd']:4.1f}  "
              f"flagged={st['flagged']}/{st['n']}")

    print("\n  Error-type distribution:")
    for et, count in summary["error_type_distribution"].items():
        print(f"    {et:<26s}  {count}")
    print(sep)



# CLI entry point


def main():
    parser = argparse.ArgumentParser(
        description="Run RQ1 Accuracy & Hallucination Evaluation"
    )
    parser.add_argument(
        "--dialogues", type=int, default=DIALOGUES_PER_TYPE,
        help=f"Dialogues per question type (default: {DIALOGUES_PER_TYPE})",
    )
    parser.add_argument(
        "--types", nargs="+", choices=list(QUESTION_CATEGORIZATION.keys()),
        help="Question types to evaluate (default: all five)",
    )
    parser.add_argument(
        "--output", default=f"{OUTPUT_DIR}/rq1_results.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--stub", action="store_true",
        help="Use stub VIOLETS backend (smoke-test without a live instance)",
    )
    args = parser.parse_args()

    backend = StubVioletsBackend() if args.stub else None
    run_rq1_evaluation(
        n_per_type=args.dialogues,
        violets_backend=backend,
        question_types=args.types,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
