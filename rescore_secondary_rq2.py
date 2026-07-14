"""
rescore_secondary_rq2.py
=========================
Adds a second (or third+) independent judge opinion to already-collected
RQ2 turns, without regenerating any conversations. Defaults to
deepseek/deepseek-v4-flash (the "2nd" judge, output suffix "_dual_judge");
pass --model to point at a different model family entirely (e.g.
"meta-llama/llama-3.3-70b-instruct" as a "3rd" judge) to check whether
findings are DeepSeek-specific or general across models.

Writes the primary judge's existing columns unchanged, plus new
`safety_score_<suffix>` / `label_<suffix>` / `reasoning_<suffix>` /
`escalation_note_<suffix>` columns from the given judge model, to a new
output file (does not overwrite the primary eval_dataset).

Usage:
  python rescore_secondary_rq2.py --input output/rq2/eval_dataset_20260710.jsonl
  python rescore_secondary_rq2.py --input output/rq2/eval_dataset_20260710.jsonl --model meta-llama/llama-3.3-70b-instruct --score_suffix 3rd
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from config import RedTeamConfig
from secondary_judge import SECONDARY_MODEL, SecondarySafetyJudge, make_secondary_client

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RescoreSecondaryRQ2")

CONCURRENCY = 5


def _slug(model: str) -> str:
    return model.split("/")[-1].replace(".", "").replace("_", "-").replace(":", "-")


async def rescore_row(
    judge: SecondarySafetyJudge, sem: asyncio.Semaphore, row: dict,
    score_col: str, label_col: str, reasoning_col: str, escalation_col: str,
    out_path: Path, write_lock: asyncio.Lock,
) -> dict:
    async with sem:
        verdict = await judge.evaluate_turn(row["category"], row["input"], row["output"])
    result = {
        **row,
        score_col: verdict["safety_score"],
        label_col: verdict["label"],
        reasoning_col: verdict["reasoning"],
        escalation_col: verdict["escalation_note"],
    }
    # Append as each row finishes so a crash/kill mid-run only loses the
    # in-flight rows, not everything completed so far.
    async with write_lock:
        with out_path.open("a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


async def main(
    input_path: Path, limit: int | None,
    model: str, score_suffix: str,
):
    cfg = RedTeamConfig.from_env()
    client = make_secondary_client()
    judge = SecondarySafetyJudge(client, cfg.fail_threshold, cfg.warn_threshold, model=model)
    sem = asyncio.Semaphore(CONCURRENCY)
    write_lock = asyncio.Lock()

    score_col = f"safety_score_{score_suffix}"
    label_col = f"label_{score_suffix}"
    reasoning_col = f"reasoning_{score_suffix}"
    escalation_col = f"escalation_note_{score_suffix}"

    rows = [json.loads(l) for l in input_path.open()]
    if limit:
        rows = rows[:limit]
    logger.info(f"Re-scoring {len(rows)} row(s) with judge ({model}) -> {score_col}...")

    out_suffix = "_dual_judge" if model == SECONDARY_MODEL else f"_dual_judge_{_slug(model)}"
    out_path = input_path.with_name(input_path.stem + out_suffix + ".jsonl")
    out_path.write_text("")  # truncate once up front, then append per row

    tasks = [
        rescore_row(judge, sem, row, score_col, label_col, reasoning_col, escalation_col, out_path, write_lock)
        for row in rows
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    ok = [r for r in results if not isinstance(r, Exception)]
    failed = len(results) - len(ok)
    if failed:
        logger.warning(f"{failed} row(s) raised an exception and were not written.")

    agree = sum(1 for r in ok if r["label"] == r[label_col])
    if ok:
        logger.info(f"Done. Label agreement: {agree}/{len(ok)} ({100*agree/len(ok):.1f}%)")
    logger.info(f"Output -> {out_path} ({len(ok)}/{len(rows)} rows written)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Only rescore the first N rows (for testing).")
    parser.add_argument("--model", type=str, default=SECONDARY_MODEL, help="OpenRouter chat model to judge with (default: deepseek-v4-flash).")
    parser.add_argument("--score_suffix", type=str, default="2nd", help="Column suffix, e.g. 'safety_score_<suffix>' (default: 2nd).")
    args = parser.parse_args()
    asyncio.run(main(Path(args.input), args.limit, args.model, args.score_suffix))
