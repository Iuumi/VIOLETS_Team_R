"""
rescore_secondary_rq1_notool.py
=================================
Adds a no-search RQ1 judge opinion (any OpenRouter chat model, NO web
search — pure parametric knowledge) to already-collected RQ1 turns, without
regenerating any conversations. Defaults to deepseek/deepseek-v4-flash (the
"3rd" judge); pass --model to point at a different model family entirely
(e.g. "meta-llama/llama-3.3-70b-instruct:free" as a "4th" judge) to check
whether no-search judging behavior is model-specific or general.

This isolates whether the primary-vs-secondary judge disagreement (see
rescore_secondary_rq1.py) comes from *models* interpreting "veracity"
differently, or from *search backends* (OpenAI web_search vs. Tavily)
returning different facts. Running a second, unrelated model family with no
search further checks whether a no-search finding is DeepSeek-specific.

Writes the primary judge's existing columns unchanged, plus new
`veracity_score_<suffix>` / `reasoning_<suffix>` columns, to a new output
file (does not touch the primary eval_dataset or the existing
*_dual_judge.jsonl).

Usage:
  python rescore_secondary_rq1_notool.py --input output/rq1/eval_dataset_20260710.jsonl
  python rescore_secondary_rq1_notool.py --input output/rq1/eval_dataset_20260710.jsonl --model meta-llama/llama-3.3-70b-instruct:free --score_suffix 4th
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from secondary_judge import SECONDARY_MODEL, SecondaryAccuracyJudgeNoSearch, make_secondary_client

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RescoreSecondaryRQ1NoTool")

# No tool calls / no Tavily rate limit here, so this can run at the same
# concurrency as the RQ2 (also no-tool) rescorer.
CONCURRENCY = 5


def _slug(model: str) -> str:
    return model.split("/")[-1].replace(".", "").replace("_", "-").replace(":", "-")


async def rescore_row(
    judge: SecondaryAccuracyJudgeNoSearch, sem: asyncio.Semaphore, row: dict,
    score_col: str, reasoning_col: str, out_path: Path, write_lock: asyncio.Lock,
) -> dict:
    async with sem:
        verdict = await judge.score(row["input"], row["output"], row.get("category", ""))
    result = {
        **row,
        score_col: verdict["veracity_score"],
        reasoning_col: verdict["reasoning"],
    }
    async with write_lock:
        with out_path.open("a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


async def main(
    input_path: Path, limit: int | None, model_id_filter: str | None,
    model: str, score_suffix: str,
):
    client = make_secondary_client()
    judge = SecondaryAccuracyJudgeNoSearch(client, model=model)
    sem = asyncio.Semaphore(CONCURRENCY)
    write_lock = asyncio.Lock()

    score_col = f"veracity_score_{score_suffix}"
    reasoning_col = f"reasoning_{score_suffix}"

    rows = [json.loads(l) for l in input_path.open()]
    if model_id_filter:
        rows = [r for r in rows if r.get("model_id") == model_id_filter]
    if limit:
        rows = rows[:limit]
    logger.info(f"Re-scoring {len(rows)} row(s) with no-search judge ({model}) -> {score_col}...")

    out_suffix = "_notool_judge" if model == SECONDARY_MODEL else f"_notool_judge_{_slug(model)}"
    out_path = input_path.with_name(input_path.stem + out_suffix + ".jsonl")
    out_path.write_text("")  # truncate once up front, then append per row

    tasks = [
        rescore_row(judge, sem, row, score_col, reasoning_col, out_path, write_lock)
        for row in rows
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    ok = [r for r in results if not isinstance(r, Exception)]
    failed = len(results) - len(ok)
    if failed:
        logger.warning(f"{failed} row(s) raised an exception and were not written.")

    diffs = [
        abs(r["veracity_score"] - r[score_col])
        for r in ok
        if r.get("veracity_score") is not None and r.get(score_col) is not None
    ]
    if diffs:
        logger.info(f"Done. Mean |primary - no-search {model}| = {sum(diffs)/len(diffs):.1f} points (n={len(diffs)})")
    logger.info(f"Output -> {out_path} ({len(ok)}/{len(rows)} rows written)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Only rescore the first N rows (for testing).")
    parser.add_argument("--model_id", type=str, default=None, help="Only rescore rows for this model_id (e.g. 'violets').")
    parser.add_argument("--model", type=str, default=SECONDARY_MODEL, help="OpenRouter chat model to judge with (default: deepseek-v4-flash).")
    parser.add_argument("--score_suffix", type=str, default="3rd", help="Column suffix, e.g. 'veracity_score_<suffix>' (default: 3rd).")
    args = parser.parse_args()
    asyncio.run(main(Path(args.input), args.limit, args.model_id, args.model, args.score_suffix))
