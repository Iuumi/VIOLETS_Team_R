"""
redteam_runner.py — VIOLETS Red-Team Orchestrator
==================================================
Categories  : harmful_content, off_topic_drift, misinformation,
              sensitive_personal, political_electoral
Escalation  : Multi-turn with structured follow-up probes (up to 6 turns)
Output      : eval_dataset.jsonl (one line per turn, model_id tagged)
 
Both VIOLETS and the baseline model are evaluated against the same attacker
turns per conversation, producing directly comparable JSONL rows for RQ1/RQ2.
 
  VIOLETS   → custom API: POST { user_id, query }  (server-side history)
  Baseline  → OpenAI API: POST { messages: [...] }  (client-side history)
 
Usage:
  python redteam_runner.py
 
Environment variables (see .env.example):
  OPENAI_API_KEY, OPENAI_BASE_URL (Azure), VIOLETS_ENDPOINT, VIOLETS_API_KEY,
  BASELINE_MODEL, BASELINE_SYSTEM_PROMPT, RUN_BASELINE,
  SEEDS_PER_CATEGORY, MAX_TURNS, CONCURRENCY, OUTPUT_DIR
"""
 
import asyncio
import logging
import uuid
from datetime import datetime
 
from dotenv import load_dotenv
from openai import AsyncOpenAI
 
from config import RedTeamConfig
from seed_generator import SeedGenerator
from attacker import AttackerLLM
from judge import JudgeLLM
from dataset_writer import DatasetWriter
from violets_client import VIOLETSClient
from baseline_client import BaselineClient
 
load_dotenv()
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("RedTeamRunner")
 
 
def _fmt_verdict(verdict: dict) -> str:
    """Format a safety verdict for logging, handling None safety scores."""
    s = verdict["safety_score"]
    return f"{verdict['label']}({s:.2f})" if s is not None else f"{verdict['label']}(N/A)"


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
    attacker: AttackerLLM,
    judge: JudgeLLM,
    violets: VIOLETSClient,
    baseline: BaselineClient | None,
) -> tuple[list[dict], list[dict]]:
    """
    Run one full red-team conversation.
    The attacker drives the turns; both VIOLETS and the baseline (if enabled)
    receive the same attacker message each turn independently.

    Returns (records, errors) — one record per agent evaluated, and any
    failure events (dropped turns, failed calls) encountered along the way.
    """
    conv_id = str(uuid.uuid4())
    short_id = conv_id[:8]
    logger.info(f"[{short_id}] START  category={category}  intent={seed.get('intent','?')}")

    # One fresh session per agent per conversation
    violets_session = violets.new_session()
    baseline_session = baseline.new_session() if baseline else None

    # Shared attacker history (attacker sees one unified view of the conversation)
    # We use VIOLETS's responses to inform the attacker — it's the primary agent.
    # The baseline runs silently in parallel and is judged independently.
    attacker_history: list[dict] = []

    violets_turns: list[dict] = []
    baseline_turns: list[dict] = []
    errors: list[dict] = []

    for turn_idx in range(cfg.max_turns):
        # ── 1. Attacker generates next probe (driven by VIOLETS responses) ──
        try:
            attacker_msg = await attacker.next_message(
                category=category,
                seed=seed,
                conversation_history=attacker_history,
                turn_index=turn_idx,
            )
        except Exception as e:
            logger.error(f"[{short_id}] Attacker failed turn {turn_idx}: {e}")
            errors.append(_err(conv_id, category, "attacker_generation", e, turn=turn_idx))
            break

        logger.debug(f"[{short_id}] Attacker T{turn_idx}: {attacker_msg[:80]}")

        # ── 2. Send to VIOLETS and baseline in parallel ────────────────────
        async def _call_violets():
            return await violets_session.chat(attacker_msg)

        async def _call_baseline():
            if baseline_session:
                return await baseline_session.chat(attacker_msg)
            return None

        results = await asyncio.gather(
            _call_violets(), _call_baseline(), return_exceptions=True
        )

        if isinstance(results[0], Exception):
            logger.error(f"[{short_id}] VIOLETS failed turn {turn_idx}: {results[0]}")
            errors.append(_err(conv_id, category, "violets_call", results[0], model_id="violets", turn=turn_idx))
            # The conversation can't continue without a VIOLETS response to
            # drive the attacker's next probe, but the baseline call this
            # turn may have already succeeded — score and keep it rather
            # than discarding a valid, already-paid-for baseline turn.
            if not isinstance(results[1], Exception) and results[1] is not None:
                salvage_response = results[1]
                try:
                    salvage_verdict = await judge.evaluate_turn(category, attacker_msg, salvage_response)
                    baseline_turns.append({
                        "turn": turn_idx,
                        "attacker_message": attacker_msg,
                        "agent_response": salvage_response,
                        "verdict": salvage_verdict,
                    })
                except Exception as judge_err:
                    logger.error(f"[{short_id}] Judge failed (baseline) turn {turn_idx}: {judge_err}")
                    errors.append(_err(conv_id, category, "judge_baseline", judge_err,
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

        # ── 3. Update attacker's view using VIOLETS as primary ─────────────
        attacker_history.append({"role": "attacker", "content": attacker_msg})
        attacker_history.append({"role": "agent",    "content": violets_response})

        # ── 4. Judge both responses in parallel ────────────────────────────
        judge_tasks = [judge.evaluate_turn(category, attacker_msg, violets_response)]
        if baseline_response is not None:
            judge_tasks.append(judge.evaluate_turn(category, attacker_msg, baseline_response))

        judge_results = await asyncio.gather(*judge_tasks, return_exceptions=True)

        if isinstance(judge_results[0], Exception):
            logger.error(f"[{short_id}] Judge failed (VIOLETS) turn {turn_idx}: {judge_results[0]}")
            errors.append(_err(conv_id, category, "judge_violets", judge_results[0], model_id="violets", turn=turn_idx))
            break
        violets_verdict = judge_results[0]
        if violets_verdict.get("label") == "ERROR":
            errors.append(_err(conv_id, category, "judge_violets", violets_verdict.get("reasoning", ""),
                                model_id="violets", turn=turn_idx))
        violets_turns.append({
            "turn": turn_idx,
            "attacker_message": attacker_msg,
            "agent_response": violets_response,
            "verdict": violets_verdict,
        })

        baseline_verdict = None
        if baseline_response is not None:
            if isinstance(judge_results[1], Exception):
                logger.error(f"[{short_id}] Judge failed (baseline) turn {turn_idx}: {judge_results[1]}")
                errors.append(_err(conv_id, category, "judge_baseline", judge_results[1],
                                    model_id=cfg.baseline_model, turn=turn_idx))
            else:
                baseline_verdict = judge_results[1]
                if baseline_verdict.get("label") == "ERROR":
                    errors.append(_err(conv_id, category, "judge_baseline", baseline_verdict.get("reasoning", ""),
                                        model_id=cfg.baseline_model, turn=turn_idx))
                baseline_turns.append({
                    "turn": turn_idx,
                    "attacker_message": attacker_msg,
                    "agent_response": baseline_response,
                    "verdict": baseline_verdict,
                })

        logger.info(
            f"[{short_id}] T{turn_idx}  "
            f"VIOLETS={_fmt_verdict(violets_verdict)}  "
            + (f"baseline={_fmt_verdict(baseline_verdict)}"
               if baseline_verdict else "baseline=skipped")
        )
 
        # ── 5. Early-stop check (based on VIOLETS responses) ───────────────
        if await attacker.should_stop(attacker_history, violets_verdict):
            logger.info(f"[{short_id}] Stopping early — VIOLETS held firm.")
            break
 
    # ── Build output records ───────────────────────────────────────────────
    records = []
    timestamp = datetime.utcnow().isoformat()
 
    for model_id, turns in [
        ("violets", violets_turns),
        (cfg.baseline_model, baseline_turns),
    ]:
        if not turns:
            continue
        flagged = any(t["verdict"]["label"] == "FAIL" for t in turns)
        valid_scores = [t["verdict"]["safety_score"] for t in turns if t["verdict"]["safety_score"] is not None]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
        records.append({
            "conversation_id": conv_id,
            "model_id": model_id,
            "category": category,
            "seed": seed,
            "turns": turns,
            "overall_safety_score": round(avg_score, 4) if avg_score is not None else None,
            "flagged": flagged,
            "timestamp": timestamp,
        })
 
    logger.info(
        f"[{short_id}] END  turns={len(violets_turns)}  "
        f"violets_flagged={records[0]['flagged'] if records else '?'}"
    )
    return records, errors
 
 
async def main():
    cfg = RedTeamConfig.from_env()
    oai_client = AsyncOpenAI(
        api_key=cfg.openai_api_key,
        base_url=cfg.openai_base_url,
    )
 
    seed_gen = SeedGenerator(oai_client, cfg)
    attacker  = AttackerLLM(oai_client, cfg)
    judge     = JudgeLLM(oai_client, cfg)
    violets   = VIOLETSClient(cfg)
    baseline  = BaselineClient(cfg, oai_client) if cfg.run_baseline else None
    run_tag   = datetime.utcnow().strftime("%Y%m%d")
    writer    = DatasetWriter(cfg.output_dir, run_tag=run_tag)

    logger.info(
        f"Red-team run starting | run_tag={run_tag} | categories={cfg.categories} | "
        f"seeds_per_category={cfg.seeds_per_category} | max_turns={cfg.max_turns} | "
        f"concurrency={cfg.concurrency} | baseline={cfg.baseline_model if baseline else 'disabled'}"
    )
 
    # ── Generate seeds ─────────────────────────────────────────────────────
    all_seeds: dict[str, list[dict]] = {}
    for category in cfg.categories:
        seeds = await seed_gen.generate(category)
        all_seeds[category] = seeds
        logger.info(f"Seeds ready [{category}]: {len(seeds)}")
 
    # ── Run all conversations with bounded concurrency ─────────────────────
    semaphore = asyncio.Semaphore(cfg.concurrency)

    # Truncate/create the output files once up front; each conversation is
    # appended as soon as it finishes so a crash mid-run doesn't lose
    # already-completed work.
    writer.write_jsonl([])
    writer.reset_errors()

    async def bounded(cat, seed):
        async with semaphore:
            try:
                recs, errs = await run_conversation(cat, seed, cfg, attacker, judge, violets, baseline)
            except Exception as e:
                logger.error(f"Conversation failed [{cat}]: {e}")
                recs, errs = [], [_err(None, cat, "conversation", e)]
        if recs:
            writer.write_jsonl(recs, append=True)
        if errs:
            writer.log_errors(errs)
        return recs, errs

    tasks = [
        bounded(cat, seed)
        for cat, seeds in all_seeds.items()
        for seed in seeds
    ]

    nested = await asyncio.gather(*tasks)
    # Flatten: each task returns (records, errors) — one record per agent
    records = [rec for conv_records, _ in nested for rec in conv_records]
    all_errors = [err for _, conv_errors in nested for err in conv_errors]

    # ── Final summary (data itself was already persisted incrementally) ────
    writer.write_stats(records)
    if all_errors:
        logger.warning(f"{len(all_errors)} error event(s) recorded → {cfg.output_dir}/errors.jsonl")
 
 
if __name__ == "__main__":
    asyncio.run(main())