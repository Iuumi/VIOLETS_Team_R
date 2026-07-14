"""
rescore_secondary_rq3.py
=========================
Adds a second, independent embedding model (qwen/qwen3-embedding-8b via
OpenRouter) to already-collected RQ3 turns, without regenerating any
responses.

RQ3 has no LLM judge — similarity_score is deterministic cosine similarity
between the response embedding and the official FAQ answer embedding
(openai/text-embedding-3-small). The robustness check here is re-embedding
both texts with an independent embedding model and re-computing cosine
similarity, to see whether the VIOLETS-vs-baseline finding holds up outside
one specific embedding space.

Writes the primary embedding's existing `similarity_score` column
unchanged, plus a new `similarity_score_2nd` column, to a new output file
(does not overwrite the primary eval_dataset).

Usage:
  python rescore_secondary_rq3.py --input output/rq3/eval_dataset_20260710.jsonl
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RescoreSecondaryRQ3")

SECONDARY_EMBED_MODEL = "qwen/qwen3-embedding-8b"
CONCURRENCY = 5


def make_embed_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")


def _cosine(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


async def rescore_row(
    client: AsyncOpenAI, sem: asyncio.Semaphore, row: dict, cache: dict,
    out_path: Path, write_lock: asyncio.Lock,
) -> dict:
    texts_needed = [t for t in (row["model_response"], row["official_answer"]) if t not in cache]
    if texts_needed:
        async with sem:
            resp = await client.embeddings.create(model=SECONDARY_EMBED_MODEL, input=texts_needed)
        for text, item in zip(texts_needed, resp.data):
            cache[text] = item.embedding

    similarity_2nd = _cosine(cache[row["model_response"]], cache[row["official_answer"]])
    result = {**row, "similarity_score_2nd": round(similarity_2nd, 6)}

    async with write_lock:
        with out_path.open("a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


async def main(input_path: Path, limit: int | None):
    client = make_embed_client()
    sem = asyncio.Semaphore(CONCURRENCY)
    write_lock = asyncio.Lock()
    cache: dict[str, list[float]] = {}

    rows = [json.loads(l) for l in input_path.open()]
    if limit:
        rows = rows[:limit]
    logger.info(f"Re-embedding {len(rows)} row(s) with secondary embedding model ({SECONDARY_EMBED_MODEL})...")

    out_path = input_path.with_name(input_path.stem + "_dual_judge.jsonl")
    out_path.write_text("")

    # Sequential, not gathered concurrently: rows sharing the same faq_id
    # often share the same official_answer text, so processing in order
    # lets the cache dict save real API calls (concurrent tasks would race
    # on a cold cache and all pay for the same embedding). The Semaphore
    # still caps in-flight embedding calls when the cache does miss.
    results = []
    for row in rows:
        try:
            results.append(await rescore_row(client, sem, row, cache, out_path, write_lock))
        except Exception as e:
            logger.warning(f"Row failed: {e}")

    diffs = [
        abs(r["similarity_score"] - r["similarity_score_2nd"])
        for r in results
        if r.get("similarity_score") is not None and r.get("similarity_score_2nd") is not None
    ]
    if diffs:
        logger.info(f"Done. Mean |primary - secondary| = {sum(diffs)/len(diffs):.3f} (n={len(diffs)})")
    logger.info(f"Cache size (unique texts embedded): {len(cache)} for {len(rows)} rows")
    logger.info(f"Output -> {out_path} ({len(results)}/{len(rows)} rows written)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Only rescore the first N rows (for testing).")
    args = parser.parse_args()
    asyncio.run(main(Path(args.input), args.limit))
