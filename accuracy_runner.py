"""
accuracy_runner.py — VIOLETS vs Baseline Accuracy Evaluation (RQ1.1)
=====================================================================
Sends the same FAQ queries to both VIOLETS (RAG) and a baseline LLM across
multi-turn conversations, then scores each response on a 0–100 veracity scale.

Mirrors redteam_runner.py in structure:
  - ParticipantGenerator replaces SeedGenerator (FAQ queries instead of adversarial seeds)
  - ParticipantLLM replaces AttackerLLM (natural follow-ups instead of escalation)
  - AccuracyJudge replaces JudgeLLM (0–100 veracity scale instead of PASS/WARN/FAIL)
  - URLValidityJudge scores VIOLETS citations only (not baseline)
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
from url_validity_judge import URLValidityJudge, compute_url_aggregate_stats
from dataset_writer import DatasetWriter
from violets_client import VIOLETSClient
from baseline_client import BaselineClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("AccuracyRunner")


def _err(conv_id, category, stage, message, model_id=None, turn=None) -> dict:
    """Build one structured error record for errors.jsonl."""
    return {
        "conversation_id": conv_id,
        "category": category,
        "model_id": model_id,
        "turn": turn,
        "stage": stage,
        "message": str(message),
        "timestamp": datetime.utcnow().isoformat(),
    }


async def run_conversation(
    category: str,
    seed: dict,
    cfg: RedTeamConfig,
    participant: ParticipantLLM,
    judge: AccuracyJudge,
    url_judge: URLValidityJudge,
    violets: VIOLETSClient,
    baseline: BaselineClient | None,
) -> tuple[list[dict], list[dict]]:
    """
    Run one full accuracy evaluation conversation.
    The participant drives the turns; both VIOLETS and the baseline receive
    the same participant message each turn independently.

    URL citation quality is scored for VIOLETS only — baseline is excluded
    because it does not produce RAG-style citations.

    Returns (records, errors) — one record per agent evaluated, and any
    failure events (dropped turns, failed calls) encountered along the way.
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
    errors: list[dict] = []

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
            errors.append(_err(conv_id, category, "participant_generation", e, turn=turn_idx))
            break

        logger.debug(f"[{short_id}] Participant T{turn_idx}: {participant_msg[:80]}")

        # ── 2. Send to VIOLETS and baseline in parallel ────────────────────
        async def _call_violets():
            return await violets_session.chat(participant_msg)

        async def _call_baseline():
            if baseline_session:
                return await baseline_session.chat(participant_msg)
            return None

        results = await asyncio.gather(
            _call_violets(), _call_baseline(), return_exceptions=True
        )

        if isinstance(results[0], Exception):
            logger.error(f"[{short_id}] VIOLETS failed turn {turn_idx}: {results[0]}")
            errors.append(_err(conv_id, category, "violets_call", results[0], model_id="violets", turn=turn_idx))
            # The conversation can't continue without a VIOLETS response to
            # drive the participant's next question, but the baseline call
            # this turn may have already succeeded — score and keep it
            # rather than discarding a valid, already-paid-for baseline turn.
            # (URL validity is skipped here: it's VIOLETS-only by design.)
            if not isinstance(results[1], Exception) and results[1] is not None:
                salvage_response = results[1]
                try:
                    salvage_verdict = await judge.score(participant_msg, salvage_response, category)
                    baseline_turns.append({
                        "turn": turn_idx,
                        "participant_message": participant_msg,
                        "agent_response": salvage_response,
                        "verdict": salvage_verdict,
                    })
                except Exception as judge_err:
                    logger.error(f"[{short_id}] Veracity judge failed (baseline) turn {turn_idx}: {judge_err}")
                    errors.append(_err(conv_id, category, "judge_baseline_veracity", judge_err,
                                        model_id=cfg.baseline_model, turn=turn_idx))
            break
        violets_response = results[0]

        baseline_response = None
        if not isinstance(results[1], Exception):
            baseline_response = results[1]
        elif results[1] is not None:
            logger.error(f"[{short_id}] Baseline failed turn {turn_idx}: {results[1]}")

        if baseline_response is None and baseline_session is not None:
            # baseline_client.chat() swallows its own API/parsing errors and
            # returns None rather than raising, so a None here (while the
            # baseline is enabled) means that call failed this turn.
            errors.append(_err(conv_id, category, "baseline_call", "baseline call returned no response",
                                model_id=cfg.baseline_model, turn=turn_idx))

        # ── 3. Update participant history using VIOLETS as primary ─────────
        participant_history.append({"role": "participant", "content": participant_msg})
        participant_history.append({"role": "agent",       "content": violets_response})

        # ── 4. Score responses — veracity for both, URL validity for VIOLETS only ──
        judge_tasks = [
            judge.score(participant_msg, violets_response, category),
            url_judge.score(participant_msg, violets_response),
        ]
        if baseline_response is not None:
            judge_tasks.append(judge.score(participant_msg, baseline_response, category))

        judge_results = await asyncio.gather(*judge_tasks, return_exceptions=True)

        veracity_result, url_result = judge_results[0], judge_results[1]
        baseline_judge_result = judge_results[2] if baseline_response is not None else None

        # ── 4a. Record VIOLETS turn ────────────────────────────────────────
        if isinstance(veracity_result, Exception):
            logger.error(f"[{short_id}] Veracity judge failed (VIOLETS) turn {turn_idx}: {veracity_result}")
            errors.append(_err(conv_id, category, "judge_violets_veracity", veracity_result,
                                model_id="violets", turn=turn_idx))
            break
        if veracity_result.get("veracity_score") is None:
            errors.append(_err(conv_id, category, "judge_violets_veracity", veracity_result.get("reasoning", ""),
                                model_id="violets", turn=turn_idx))

        if isinstance(url_result, Exception):
            logger.warning(f"[{short_id}] URL judge failed (VIOLETS) turn {turn_idx}: {url_result}")
            errors.append(_err(conv_id, category, "judge_violets_url", url_result, model_id="violets", turn=turn_idx))
            url_result = {"citation_rate_score": None, "accessibility_score": None,
                          "accuracy_score": None, "url_details": [], "reasoning": str(url_result),
                          "urls_found": []}
        elif url_result.get("citation_rate_score") is None:
            errors.append(_err(conv_id, category, "judge_violets_url", url_result.get("reasoning", ""),
                                model_id="violets", turn=turn_idx))

        violets_turns.append({
            "turn": turn_idx,
            "participant_message": participant_msg,
            "agent_response": violets_response,
            "verdict": veracity_result,
            "url_validity": url_result,
        })

        # ── 4b. Record baseline turn (veracity only) ───────────────────────
        baseline_verdict = None
        if baseline_response is not None:
            if isinstance(baseline_judge_result, Exception):
                logger.error(f"[{short_id}] Veracity judge failed (baseline) turn {turn_idx}: {baseline_judge_result}")
                errors.append(_err(conv_id, category, "judge_baseline_veracity", baseline_judge_result,
                                    model_id=cfg.baseline_model, turn=turn_idx))
            else:
                baseline_verdict = baseline_judge_result
                if baseline_verdict.get("veracity_score") is None:
                    errors.append(_err(conv_id, category, "judge_baseline_veracity",
                                        baseline_verdict.get("reasoning", ""),
                                        model_id=cfg.baseline_model, turn=turn_idx))
                baseline_turns.append({
                    "turn": turn_idx,
                    "participant_message": participant_msg,
                    "agent_response": baseline_response,
                    "verdict": baseline_verdict,
                    # url_validity intentionally omitted for baseline
                })

        logger.info(
            f"[{short_id}] T{turn_idx}  "
            f"VIOLETS={veracity_result['veracity_score']}  "
            f"url_cited={url_result['citation_rate_score']}  "
            f"url_access={url_result['accessibility_score']}  "
            f"url_acc={url_result['accuracy_score']}  "
            + (f"baseline={baseline_verdict['veracity_score']}"
               if baseline_verdict else "baseline=skipped")
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

        # Veracity aggregate (both models)
        valid_scores = [
            t["verdict"]["veracity_score"]
            for t in turns
            if t["verdict"]["veracity_score"] is not None
        ]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

        record = {
            "conversation_id": conv_id,
            "model_id": model_id,
            "category": category,
            "seed": seed,
            "turns": turns,
            "overall_veracity_score": round(avg_score, 2) if avg_score is not None else None,
            "timestamp": timestamp,
        }

        # URL validity aggregate (VIOLETS only)
        if model_id == "violets":
            url_turn_results = [t["url_validity"] for t in turns]
            url_stats = compute_url_aggregate_stats(url_turn_results)
            record["url_validity_stats"] = {
                "pct_cited":       url_stats["pct_cited"],
                "pct_accessible":  url_stats["pct_accessible"],
                "pct_accurate":    url_stats["pct_accurate"],
                "n_turns_cited":   url_stats["n_turns_cited"],
                "n_urls_total":    url_stats["n_urls_total"],
                "n_urls_accessible": url_stats["n_urls_accessible"],
                "n_urls_accurate": url_stats["n_urls_accurate"],
            }

        records.append(record)

    logger.info(
        f"[{short_id}] END  turns={len(violets_turns)}  "
        f"violets_avg={records[0]['overall_veracity_score'] if records else '?'}  "
        f"url_pct_cited={records[0].get('url_validity_stats', {}).get('pct_cited', '?') if records else '?'}"
    )
    return records, errors


async def main():
    cfg = RedTeamConfig.from_env()
    oai_client = AsyncOpenAI(
        api_key=cfg.openai_api_key,
        base_url=cfg.openai_base_url,
    )

    participant_gen = ParticipantGenerator(oai_client, cfg)
    participant     = ParticipantLLM(oai_client, cfg)
    judge           = AccuracyJudge(oai_client, cfg)
    url_judge       = URLValidityJudge(oai_client, cfg)
    violets         = VIOLETSClient(cfg)
    baseline        = BaselineClient(cfg, oai_client) if cfg.run_baseline else None
    run_tag         = datetime.utcnow().strftime("%Y%m%d")
    writer          = DatasetWriter("./output/rq1", run_tag=run_tag)

    logger.info(
        f"Accuracy run starting | run_tag={run_tag} | categories={cfg.accuracy_categories} | "
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

    # Truncate/create the output files once up front; each conversation is
    # appended as soon as it finishes so a crash mid-run doesn't lose
    # already-completed work.
    writer.write_accuracy_jsonl([])
    writer.reset_errors()

    async def bounded(cat, seed):
        async with semaphore:
            try:
                recs, errs = await run_conversation(
                    cat, seed, cfg, participant, judge, url_judge, violets, baseline
                )
            except Exception as e:
                logger.error(f"Conversation failed [{cat}]: {e}")
                recs, errs = [], [_err(None, cat, "conversation", e)]
        if recs:
            writer.write_accuracy_jsonl(recs, append=True)
        if errs:
            writer.log_errors(errs)
        return recs, errs

    tasks = [
        bounded(cat, seed)
        for cat, seeds in all_seeds.items()
        for seed in seeds
    ]

    nested = await asyncio.gather(*tasks)
    records = [rec for conv_records, _ in nested for rec in conv_records]
    all_errors = [err for _, conv_errors in nested for err in conv_errors]

    # ── Final summary (data itself was already persisted incrementally) ────
    writer.write_accuracy_stats(records)
    if all_errors:
        logger.warning(f"{len(all_errors)} error event(s) recorded → output/rq1/errors.jsonl")


if __name__ == "__main__":
    asyncio.run(main())