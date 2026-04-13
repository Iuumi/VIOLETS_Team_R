"""
accuracy_runner.py — VIOLETS vs Baseline LLM Accuracy Comparison
=================================================================
Sends the same seed queries to both VIOLETS (RAG) and a plain baseline LLM,
then has a JudgeLLM score each response on accuracy/helpfulness dimensions.

Output: output/accuracy_comparison.jsonl

Usage:
  python accuracy_runner.py

New env vars (in addition to existing ones):
  BASELINE_MODEL   — model name for the baseline LLM (default: gpt-5-nano)
  QUERY_CATEGORIES — comma-separated categories to test (default: all 5)
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from config import RedTeamConfig
from seed_generator import SeedGenerator
from violets_client import VIOLETSClient
from baseline_client import BaselineLLMClient
from accuracy_judge import AccuracyJudge

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("AccuracyRunner")


async def run_comparison(
    category: str,
    seed: dict,
    violets: VIOLETSClient,
    baseline: BaselineLLMClient,
    judge: AccuracyJudge,
    cfg: RedTeamConfig,
) -> dict:
    query = seed["prompt"]
    conv_id = str(uuid.uuid4())
    short_id = conv_id[:8]

    logger.info(f"[{short_id}] Query: {query[:80]}")

    # Send the same query to both systems simultaneously
    messages = [{"role": "user", "content": query}]
    violets_resp, baseline_resp = await asyncio.gather(
        violets.chat(messages),
        baseline.chat(messages),
    )

    logger.debug(f"[{short_id}] VIOLETS: {violets_resp[:80]}")
    logger.debug(f"[{short_id}] Baseline: {baseline_resp[:80]}")

    # Judge scores both responses
    result = await judge.compare(
        query=query,
        violets_response=violets_resp,
        baseline_response=baseline_resp,
        category=category,
    )

    logger.info(
        f"[{short_id}] Winner={result['winner']}  "
        f"VIOLETS={result['violets_avg']:.2f}  "
        f"Baseline={result['baseline_avg']:.2f}"
    )

    return {
        "comparison_id": conv_id,
        "category": category,
        "seed_intent": seed.get("intent", ""),
        "query": query,
        "violets_response": violets_resp,
        "baseline_response": baseline_resp,
        **result,
        "timestamp": datetime.utcnow().isoformat(),
    }


def write_results(records: list[dict], output_dir: str = "./output") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "accuracy_comparison.jsonl"

    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Print summary
    total = len(records)
    violets_wins = sum(1 for r in records if r["winner"] == "violets")
    baseline_wins = sum(1 for r in records if r["winner"] == "baseline")
    ties = sum(1 for r in records if r["winner"] == "tie")
    avg_v = sum(r["violets_avg"] for r in records) / total if total else 0
    avg_b = sum(r["baseline_avg"] for r in records) / total if total else 0

    print("\n── Accuracy Comparison Complete ──────────────────────")
    print(f"  Total queries  : {total}")
    print(f"  VIOLETS wins   : {violets_wins} ({violets_wins/total*100:.0f}%)")
    print(f"  Baseline wins  : {baseline_wins} ({baseline_wins/total*100:.0f}%)")
    print(f"  Ties           : {ties}")
    print(f"  Avg VIOLETS    : {avg_v:.3f}")
    print(f"  Avg Baseline   : {avg_b:.3f}")
    print(f"  Output         : {path}")
    print("──────────────────────────────────────────────────────\n")


async def main():
    cfg = RedTeamConfig.from_env()
    client = AsyncOpenAI(
        api_key=cfg.openai_api_key,
        base_url=cfg.openai_base_url,
    )

    seed_gen = SeedGenerator(client, cfg)
    violets = VIOLETSClient(cfg)
    baseline = BaselineLLMClient(client, cfg)
    judge = AccuracyJudge(client, cfg)

    # Generate seeds (reuses existing SeedGenerator — no changes needed)
    all_seeds: dict[str, list[dict]] = {}
    for category in cfg.categories:
        seeds = await seed_gen.generate(category)
        all_seeds[category] = seeds
        logger.info(f"Seeds ready [{category}]: {len(seeds)}")

    semaphore = asyncio.Semaphore(cfg.concurrency)

    async def bounded(cat, seed):
        async with semaphore:
            try:
                return await run_comparison(cat, seed, violets, baseline, judge, cfg)
            except Exception as e:
                logger.error(f"Comparison failed [{cat}]: {e}")
                return None

    tasks = [
        bounded(cat, seed)
        for cat, seeds in all_seeds.items()
        for seed in seeds
    ]

    results = await asyncio.gather(*tasks)
    records = [r for r in results if r is not None]

    write_results(records, cfg.output_dir)


if __name__ == "__main__":
    asyncio.run(main())

add to config.py:
# In the RedTeamConfig dataclass, add:
baseline_model: str = "gpt-5-nano"   # The plain LLM to compare against VIOLETS

# In from_env(), add:
baseline_model=os.environ.get("BASELINE_MODEL", "gpt-5-nano"),
