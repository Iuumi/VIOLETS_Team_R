"""
accuracy_runner.py — VIOLETS vs Baseline Accuracy Evaluation (RQ1.1)
=====================================================================
Sends the same FAQ queries to both VIOLETS (RAG) and a baseline LLM across
multi-turn conversations, then scores each response on a 0–100 veracity scale.

Mirrors redteam_runner.py in structure:
  - ParticipantGenerator replaces SeedGenerator (FAQ queries instead of adversarial seeds)
  - ParticipantLLM replaces AttackerLLM (natural follow-ups instead of escalation)
  - AccuracyJudge replaces JudgeLLM (0–100 veracity scale instead of PASS/WARN/FAIL)
  - No early-stop — conversations run for max_turns
  - Output: ./output/rq1/eval_dataset.jsonl

Usage:
  python accuracy_runner.py

Environment variables (shared with redteam_runner.py via .env):
  OPENAI_API_KEY, OPENAI_BASE_URL, VIOLETS_ENDPOINT, VIOLETS_API_KEY,
  BASELINE_MODEL, BASELINE_SYSTEM_PROMPT, RUN_BASELINE,
  SEEDS_PER_CATEGORY, MAX_TURNS, CONCURRENCY
"""

import asyncio
import logging
import uuid
from datetime import datetime

from dotenv import load_dotenv
from openai import AsyncOpenAI

from config import RedTeamConfig
from participant_generator import ParticipantGenerator
from participant import ParticipantLLM
from accuracy_judge import AccuracyJudge
from dataset_writer import DatasetWriter
from violets_client import VIOLETSClient
from baseline_client import BaselineClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("AccuracyRunner")


async def run_conversation(
    category: str,
    seed: dict,
    cfg: RedTeamConfig,
    participant: ParticipantLLM,
    judge: AccuracyJudge,
    violets: VIOLETSClient,
    baseline: BaselineClient | None,
) -> list[dict]:
    """
    Run one full accuracy evaluation conversation.
    The participant drives the turns; both VIOLETS and the baseline receive
    the same participant message each turn independently.

    Returns a list of conversation records — one per agent evaluated.
    """
    conv_id = str(uuid.uuid4())
    short_id = conv_id[:8]
    logger.info(f"[{short_id}] START  category={category}  intent={seed.get('intent', '?')}")

    # One fresh session per agent per conversation
    violets_session = violets.new_session()
    baseline_session = baseline.new_session() if baseline else None

    # Participant sees VIOLETS responses (primary agent) to drive follow-up questions
    participant_history: list[dict] = []

    violets_turns: list[dict] = []
    baseline_turns: list[dict] = []

    for turn_idx in range(cfg.max_turns):
        # ── 1. Participant generates next question ─────────────────────────
        try:
            participant_msg = await participant.next_message(
                category=category,
                seed=seed,
                conversation_history=participant_history,
                turn_index=turn_idx,
            )
        except Exception as e:
            logger.error(f"[{short_id}] Participant failed turn {turn_idx}: {e}")
            break

        logger.debug(f"[{short_id}] Participant T{turn_idx}: {participant_msg[:80]}")

        # ── 2a. Send to VIOLETS ────────────────────────────────────────────
        try:
            violets_response = await violets_session.chat(participant_msg)
        except Exception as e:
            logger.error(f"[{short_id}] VIOLETS failed turn {turn_idx}: {e}")
            break

        # ── 2b. Send identical message to baseline ─────────────────────────
        baseline_response = None
        if baseline_session:
            try:
                baseline_response = await baseline_session.chat(participant_msg)
            except Exception as e:
                logger.error(f"[{short_id}] Baseline failed turn {turn_idx}: {e}")
                break

        # ── 3. Update participant history using VIOLETS as primary ─────────
        participant_history.append({"role": "participant", "content": participant_msg})
        participant_history.append({"role": "agent",       "content": violets_response})

        # ── 4. Score both responses independently ──────────────────────────
        violets_verdict = await judge.score(participant_msg, violets_response, category)
        violets_turns.append({
            "turn": turn_idx,
            "participant_message": participant_msg,
            "agent_response": violets_response,
            "verdict": violets_verdict,
        })

        if baseline_response is not None:
            baseline_verdict = await judge.score(participant_msg, baseline_response, category)
            baseline_turns.append({
                "turn": turn_idx,
                "participant_message": participant_msg,
                "agent_response": baseline_response,
                "verdict": baseline_verdict,
            })

        logger.info(
            f"[{short_id}] T{turn_idx}  "
            f"VIOLETS={violets_verdict['veracity_score']}  "
            + (f"baseline={baseline_verdict['veracity_score']}"
               if baseline_response else "baseline=skipped")
        )

    # ── Build output records ───────────────────────────────────────────────
    records = []
    timestamp = datetime.utcnow().isoformat()

    for model_id, turns in [
        ("violets", violets_turns),
        (cfg.baseline_model, baseline_turns),
    ]:
        if not turns:
            continue
        valid_scores = [
            t["verdict"]["veracity_score"]
            for t in turns
            if t["verdict"]["veracity_score"] is not None
        ]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
        records.append({
            "conversation_id": conv_id,
            "model_id": model_id,
            "category": category,
            "seed": seed,
            "turns": turns,
            "overall_veracity_score": round(avg_score, 2) if avg_score is not None else None,
            "timestamp": timestamp,
        })

    logger.info(
        f"[{short_id}] END  turns={len(violets_turns)}  "
        f"violets_avg={records[0]['overall_veracity_score'] if records else '?'}"
    )
    return records


async def main():
    cfg = RedTeamConfig.from_env()
    oai_client = AsyncOpenAI(
        api_key=cfg.openai_api_key,
        base_url=cfg.openai_base_url,
    )

    participant_gen = ParticipantGenerator(oai_client, cfg)
    participant     = ParticipantLLM(oai_client, cfg)
    judge           = AccuracyJudge(oai_client, cfg)
    violets         = VIOLETSClient(cfg)
    baseline        = BaselineClient(cfg, oai_client) if cfg.run_baseline else None
    writer          = DatasetWriter("./output/rq1")

    logger.info(
        f"Accuracy run starting | categories={cfg.accuracy_categories} | "
        f"seeds_per_category={cfg.seeds_per_category} | max_turns={cfg.max_turns} | "
        f"concurrency={cfg.concurrency} | baseline={cfg.baseline_model if baseline else 'disabled'}"
    )

    # ── Generate seeds ─────────────────────────────────────────────────────
    all_seeds: dict[str, list[dict]] = {}
    for category in cfg.accuracy_categories:
        seeds = await participant_gen.generate(category)
        all_seeds[category] = seeds
        logger.info(f"Seeds ready [{category}]: {len(seeds)}")

    # ── Run all conversations with bounded concurrency ─────────────────────
    semaphore = asyncio.Semaphore(cfg.concurrency)

    async def bounded(cat, seed):
        async with semaphore:
            try:
                return await run_conversation(cat, seed, cfg, participant, judge, violets, baseline)
            except Exception as e:
                logger.error(f"Conversation failed [{cat}]: {e}")
                return []

    tasks = [
        bounded(cat, seed)
        for cat, seeds in all_seeds.items()
        for seed in seeds
    ]

    nested = await asyncio.gather(*tasks)
    records = [rec for conv_records in nested for rec in conv_records]

    # ── Write outputs ──────────────────────────────────────────────────────
    writer.write_accuracy_jsonl(records)
    writer.write_accuracy_stats(records)


if __name__ == "__main__":
    asyncio.run(main())
