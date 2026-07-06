"""
q3.py — RQ3: Alignment with Official FAQ Guidance
==================================================

RQ3: How well do VIOLETS's responses align with official FAQ guidance?

Design:
  User queries : Real constituent queries loaded from data/faq_pairs.csv,
                 derived from the Maryland State Board of Elections FAQ
                 (2026 Gubernatorial Primary)
  GLC          : Paraphrased / perturbed versions of those queries,
                 generated via GPT-4o-mini at runtime
  Responses    : VIOLETS vs. baseline model (configured via BASELINE_MODEL)
  Ground truth : Official FAQ answers from data/faq_pairs.csv
  Metric       : Cosine similarity between model response embedding and
                 official answer embedding (OpenAI text-embedding-3-small)

No LLM judge — pure embedding-based semantic similarity.

FAQ data:
  Edit data/faq_pairs.csv to update queries or answers.
  Required columns: id, category, question, answer

Usage:
  python q3.py                             # run with GLC perturbations
  python q3.py --no-glc                    # original queries only
  python q3.py --faq_file path/to/file.csv # use a different FAQ CSV
  python q3.py --output_dir path/to/dir    # override output location

Environment variables (shared via .env):
  OPENAI_API_KEY, OPENAI_BASE_URL, VIOLETS_ENDPOINT, VIOLETS_API_KEY,
  BASELINE_MODEL, BASELINE_SYSTEM_PROMPT, RUN_BASELINE, CONCURRENCY

Output:
  output/rq3/eval_dataset.jsonl — one row per (faq_id, query_type, model_id)

"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

from config import RedTeamConfig
from violets_client import VIOLETSClient
from baseline_client import _is_reasoning_model, _REASONING_MODEL_MAX_COMPLETION_TOKENS

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("RQ3")


DEFAULT_FAQ_FILE = Path("data/faq_pairs.csv")


def load_faq_pairs(path: Path) -> list[dict]:
    """Load FAQ Q&A pairs from a CSV file with columns: id, category, question, answer."""
    if not path.exists():
        raise FileNotFoundError(
            f"FAQ file not found: {path}\n"
            "Expected columns: id, category, question, answer"
        )
    with path.open(encoding="utf-8", newline="") as f:
        pairs = list(csv.DictReader(f))
    if not pairs:
        raise ValueError(f"No rows found in {path}")
    missing = [p["id"] for p in pairs if not all(k in p for k in ("id", "category", "question", "answer"))]
    if missing:
        raise ValueError(f"Rows missing required columns: {missing}")
    logger.info(f"Loaded {len(pairs)} FAQ pairs from {path}")
    return pairs

# ── Utility ───────────────────────────────────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ── Semantic scorer ───────────────────────────────────────────────────────────

class SemanticScorer:
    """
    Scores a model response against an official FAQ answer using cosine similarity
    of OpenAI embeddings. Replaces the LLM judge used in RQ1/RQ2.
    """

    def __init__(self, client: AsyncOpenAI, model: str = "text-embedding-3-small"):
        self.client = client
        self.model = model
        self._cache: dict[str, list[float]] = {}

    async def embed(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]
        resp = await self.client.embeddings.create(model=self.model, input=text)
        emb = resp.data[0].embedding
        self._cache[text] = emb
        return emb

    async def score(self, model_response: str, official_answer: str) -> float:
        """Return cosine similarity in [0, 1] between response and official answer."""
        emb_response, emb_answer = await asyncio.gather(
            self.embed(model_response),
            self.embed(official_answer),
        )
        raw = cosine_similarity(emb_response, emb_answer)
        return max(0.0, min(1.0, raw))


# ── GLC query perturber ───────────────────────────────────────────────────────

class QueryPerturber:
    """
    Generates a paraphrased version of each FAQ question for the GLC condition.
    Same meaning, different wording — tests robustness to query surface variation.
    """

    _SYSTEM = (
        "You are helping evaluate a voter assistance chatbot. Rephrase the following "
        "election FAQ question in a different way while preserving the exact same meaning. "
        "Use different words and sentence structure. Output only the rephrased question, "
        "no explanation."
    )

    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    async def perturb(self, question: str) -> str:
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._SYSTEM},
                    {"role": "user", "content": question},
                ],
                temperature=0.7,
                max_tokens=150,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"QueryPerturber failed for '{question[:50]}': {e}")
            return question  # fall back to original


# ── Model queries ─────────────────────────────────────────────────────────────

async def query_violets(question: str, violets_client: VIOLETSClient) -> str:
    session = violets_client.new_session()
    return await session.chat(question)


async def query_baseline(
    question: str,
    oai_client: AsyncOpenAI,
    cfg: RedTeamConfig,
) -> Optional[str]:
    """
    Query the baseline model once.

    Reasoning models (o1/o3/o4/gpt-5 family) reject max_tokens/non-default
    temperature and need a much larger completion-token budget than a
    regular chat model — see baseline_client.py for the full rationale and
    the empirical numbers behind _REASONING_MODEL_MAX_COMPLETION_TOKENS.
    """
    if not cfg.run_baseline:
        return None

    is_reasoning = _is_reasoning_model(cfg.baseline_model)
    if is_reasoning:
        call_kwargs = {"max_completion_tokens": _REASONING_MODEL_MAX_COMPLETION_TOKENS}
    else:
        call_kwargs = {"max_tokens": 512, "temperature": 0.0}

    try:
        resp = await oai_client.chat.completions.create(
            model=cfg.baseline_model,
            messages=[
                {"role": "system", "content": cfg.baseline_system_prompt},
                {"role": "user", "content": question},
            ],
            **call_kwargs,
        )
        content = resp.choices[0].message.content
        reply = content.strip() if content else None
        if not reply:
            logger.error(
                f"Baseline returned empty content (model={cfg.baseline_model}, "
                f"reasoning_model={is_reasoning})"
            )
            return None
        return reply
    except Exception as e:
        logger.error(f"Baseline query error: {e}")
        return None


# ── Per-FAQ evaluation ────────────────────────────────────────────────────────

async def evaluate_faq(
    faq: dict,
    query: str,
    query_type: str,
    violets_client: VIOLETSClient,
    oai_client: AsyncOpenAI,
    scorer: SemanticScorer,
    cfg: RedTeamConfig,
    timestamp: str,
) -> list[dict]:
    """Query both models with one FAQ question and score each response."""
    official_answer = faq["answer"]

    violets_resp, baseline_resp = await asyncio.gather(
        query_violets(query, violets_client),
        query_baseline(query, oai_client, cfg),
        return_exceptions=True,
    )

    rows = []

    # VIOLETS
    if isinstance(violets_resp, Exception):
        logger.error(f"VIOLETS error [{faq['id']}]: {violets_resp}")
        violets_sim = None
        violets_text = ""
    else:
        violets_text = violets_resp
        violets_sim = await scorer.score(violets_text, official_answer)

    rows.append({
        "faq_id": faq["id"],
        "category": faq["category"],
        "query_type": query_type,
        "model_id": "violets",
        "query": query,
        "official_answer": official_answer,
        "model_response": violets_text,
        "similarity_score": round(violets_sim, 4) if violets_sim is not None else None,
        "timestamp": timestamp,
    })

    # Baseline
    if cfg.run_baseline:
        if isinstance(baseline_resp, Exception) or baseline_resp is None:
            logger.error(f"Baseline error [{faq['id']}]: {baseline_resp}")
            baseline_sim = None
            baseline_text = ""
        else:
            baseline_text = baseline_resp
            baseline_sim = await scorer.score(baseline_text, official_answer)

        rows.append({
            "faq_id": faq["id"],
            "category": faq["category"],
            "query_type": query_type,
            "model_id": cfg.baseline_model,
            "query": query,
            "official_answer": official_answer,
            "model_response": baseline_text,
            "similarity_score": round(baseline_sim, 4) if baseline_sim is not None else None,
            "timestamp": timestamp,
        })

    logger.info(
        f"[{faq['id']}] {query_type}  "
        f"VIOLETS={rows[0]['similarity_score']}  "
        + (f"baseline={rows[1]['similarity_score']}" if len(rows) > 1 else "baseline=disabled")
    )
    return rows


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_evaluation(
    faq_pairs: list[dict],
    cfg: RedTeamConfig,
    oai_client: AsyncOpenAI,
    with_glc: bool = True,
) -> list[dict]:
    """
    For each FAQ pair, query VIOLETS and the baseline with the original question
    and (if with_glc=True) a paraphrased GLC variant. Score each response against
    the official answer via cosine similarity.
    """
    violets_client = VIOLETSClient(cfg)
    scorer = SemanticScorer(oai_client)
    perturber = QueryPerturber(oai_client) if with_glc else None

    semaphore = asyncio.Semaphore(cfg.concurrency)
    timestamp = datetime.now(timezone.utc).isoformat()

    async def bounded(faq: dict, query: str, query_type: str) -> list[dict]:
        async with semaphore:
            try:
                return await evaluate_faq(
                    faq=faq,
                    query=query,
                    query_type=query_type,
                    violets_client=violets_client,
                    oai_client=oai_client,
                    scorer=scorer,
                    cfg=cfg,
                    timestamp=timestamp,
                )
            except Exception as e:
                logger.error(f"evaluate_faq failed [{faq['id']}] {query_type}: {e}")
                return []

    # Generate all GLC perturbations upfront in one parallel batch
    if with_glc and perturber:
        logger.info("Generating GLC perturbations ...")
        perturbed_questions = await asyncio.gather(
            *[perturber.perturb(faq["question"]) for faq in faq_pairs]
        )
    else:
        perturbed_questions = [None] * len(faq_pairs)

    tasks = []
    for faq, perturbed_q in zip(faq_pairs, perturbed_questions):
        tasks.append(bounded(faq, faq["question"], "original"))
        if with_glc and perturbed_q:
            tasks.append(bounded(faq, perturbed_q, "perturbed"))

    nested = await asyncio.gather(*tasks)
    all_rows = [row for batch in nested for row in batch]

    logger.info(f"Evaluation complete — {len(all_rows)} rows collected.")
    return all_rows


# ── Output ────────────────────────────────────────────────────────────────────

def write_jsonl(rows: list[dict], output_dir: Path, run_tag: Optional[str] = None) -> Path:
    """
    Write rows to eval_dataset.jsonl, or eval_dataset_<run_tag>.jsonl if a
    run_tag (e.g. a "YYYYMMDD" date) is given — so repeated runs on
    different dates (e.g. for temporal-stability testing) don't overwrite
    each other's data.
    """
    ensure_dir(output_dir)
    filename = f"eval_dataset_{run_tag}.jsonl" if run_tag else "eval_dataset.jsonl"
    path = output_dir / filename
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info(f"JSONL written → {path} ({len(rows)} rows)")
    return path


def print_summary(rows: list[dict]) -> None:
    by_model: dict[str, list[float]] = {}
    for row in rows:
        score = row.get("similarity_score")
        if score is not None:
            by_model.setdefault(row["model_id"], []).append(score)

    print("\n── RQ3 Semantic Similarity Results ────────────────────────────────────")
    print(f"  Total rows     : {len(rows)}")
    for model_id, scores in by_model.items():
        avg = sum(scores) / len(scores)
        label = "VIOLETS" if model_id == "violets" else model_id
        print(f"  {label:<20} mean similarity: {avg:.4f}  (n={len(scores)})")
    print("───────────────────────────────────────────────────────────────────────\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RQ3: FAQ alignment evaluation via semantic similarity."
    )
    parser.add_argument(
        "--no-glc",
        action="store_true",
        default=False,
        help="Skip GLC perturbations and evaluate original queries only.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/rq3",
        help="Directory for JSONL output.",
    )
    parser.add_argument(
        "--faq_file",
        type=str,
        default=str(DEFAULT_FAQ_FILE),
        help="Path to the FAQ pairs CSV (columns: id, category, question, answer).",
    )
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    faq_pairs = load_faq_pairs(Path(args.faq_file))

    cfg = RedTeamConfig.from_env()
    oai_client = AsyncOpenAI(
        api_key=cfg.openai_api_key,
        base_url=cfg.openai_base_url,
    )

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    logger.info(
        f"RQ3 starting | run_tag={run_tag} | FAQs={len(faq_pairs)} | "
        f"GLC={'yes' if not args.no_glc else 'no'} | "
        f"baseline={cfg.baseline_model if cfg.run_baseline else 'disabled'} | "
        f"concurrency={cfg.concurrency}"
    )

    rows = await run_evaluation(
        faq_pairs=faq_pairs,
        cfg=cfg,
        oai_client=oai_client,
        with_glc=not args.no_glc,
    )

    write_jsonl(rows, Path(args.output_dir), run_tag=run_tag)
    print_summary(rows)


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
