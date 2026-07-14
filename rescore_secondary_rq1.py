"""
rescore_secondary_rq1.py
=========================
Adds a second, independent judge (deepseek/deepseek-v4-flash via OpenRouter,
web search via Tavily) to already-collected RQ1 turns, without regenerating
any conversations.

Writes the primary judge's existing columns unchanged, plus new
`veracity_score_2nd` / `reasoning_2nd` columns from the secondary judge, to a
new output file (does not overwrite the primary eval_dataset).

Usage:
  python rescore_secondary_rq1.py --input output/rq1/eval_dataset_20260710.jsonl
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from secondary_judge import SecondaryAccuracyJudge, make_secondary_client

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RescoreSecondaryRQ1")

# Web-search tool calls make each row slower and Tavily has its own rate
# limits, so keep concurrency lower than the RQ2 (no-tool) rescorer.
CONCURRENCY = 3


async def rescore_row(
    judge: SecondaryAccuracyJudge, sem: asyncio.Semaphore, row: dict,
    out_path: Path, write_lock: asyncio.Lock,
) -> dict:
    async with sem:
        verdict = await judge.score(row["input"], row["output"], row.get("category", ""))
    result = {
        **row,
        "veracity_score_2nd": verdict["veracity_score"],
        "reasoning_2nd": verdict["reasoning"],
    }
    # Append as each row finishes — a crash (or a killed run, e.g. hitting a
    # Tavily credit limit) loses only the in-flight rows, not everything
    # completed so far, unlike gathering all results and writing once at the end.
    async with write_lock:
        with out_path.open("a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


async def main(input_path: Path, limit: int | None, model_id_filter: str | None):
    client = make_secondary_client()
    judge = SecondaryAccuracyJudge(client)
    sem = asyncio.Semaphore(CONCURRENCY)
    write_lock = asyncio.Lock()

    rows = [json.loads(l) for l in input_path.open()]
    if model_id_filter:
        rows = [r for r in rows if r.get("model_id") == model_id_filter]
    if limit:
        rows = rows[:limit]
    logger.info(f"Re-scoring {len(rows)} row(s) with secondary judge (SecondaryAccuracyJudge)...")

    out_path = input_path.with_name(input_path.stem + "_dual_judge.jsonl")
    out_path.write_text("")  # truncate once up front, then append per row

    tasks = [rescore_row(judge, sem, row, out_path, write_lock) for row in rows]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    ok = [r for r in results if not isinstance(r, Exception)]
    failed = len(results) - len(ok)
    if failed:
        logger.warning(f"{failed} row(s) raised an exception and were not written.")

    diffs = [
        abs(r["veracity_score"] - r["veracity_score_2nd"])
        for r in ok
        if r.get("veracity_score") is not None and r.get("veracity_score_2nd") is not None
    ]
    if diffs:
        logger.info(f"Done. Mean |primary - secondary| = {sum(diffs)/len(diffs):.1f} points (n={len(diffs)})")
    logger.info(f"Output -> {out_path} ({len(ok)}/{len(rows)} rows written)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Only rescore the first N rows (for testing).")
    parser.add_argument("--model_id", type=str, default=None, help="Only rescore rows for this model_id (e.g. 'violets').")
    args = parser.parse_args()
    asyncio.run(main(Path(args.input), args.limit, args.model_id))
