# VIOLETS Pre-Deployment Evaluation Pipeline

Automated evaluation framework for VIOLETS — a RAG-based voting information chatbot for 2026 Maryland elections. Covers three research questions:

- **RQ1 (Accuracy)** — How factually accurate are VIOLETS's responses, and does it hallucinate? (`accuracy_runner.py`)
- **RQ2 (Safety)** — Does VIOLETS correctly handle adversarial, out-of-scope, and sensitive queries? (`redteam_runner.py`)
- **RQ3 (FAQ Alignment)** — How well do VIOLETS's responses align with official Maryland/Montgomery County FAQ guidance? (`q3.py`)

See [Reproducing the Evaluation](#reproducing-the-evaluation) for the exact commands to run all three end to end.

---

## Architecture

```
                        ┌─────────────────────────────┐
                        │         config.py            │
                        │  (shared settings & categories)│
                        └──────────────┬──────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                                                  │
   ┌──────────▼──────────┐                         ┌────────────▼────────────┐
   │  RQ1: accuracy_runner│                         │  RQ2: redteam_runner    │
   │  (veracity evaluation)│                        │  (safety evaluation)    │
   └──────────┬──────────┘                         └────────────┬────────────┘
              │                                                  │
   ┌──────────▼──────────┐                         ┌────────────▼────────────┐
   │participant_generator │                         │   seed_generator.py     │
   │  (FAQ query seeds)   │                         │  (adversarial seeds)    │
   └──────────┬──────────┘                         └────────────┬────────────┘
              │                                                  │
   ┌──────────▼──────────┐                         ┌────────────▼────────────┐
   │    participant.py    │                         │      attacker.py        │
   │  (natural follow-ups)│                         │  (escalating probes)    │
   └──────────┬──────────┘                         └────────────┬────────────┘
              │                                                  │
              └──────────────────┬───────────────────────────────┘
                                 │  (shared)
                    ┌────────────▼────────────┐
                    │     violets_client.py    │  ← stateful server session
                    │     baseline_client.py   │  ← stateless client history
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
   ┌──────────▼──────────┐             ┌────────────▼────────────┐
   │   accuracy_judge.py  │             │        judge.py         │
   │ (0–100 veracity score)│            │   (PASS / WARN / FAIL)  │
   └──────────┬──────────┘             └────────────┬────────────┘
              │                                     │
              └──────────────────┬──────────────────┘
                                 │  (shared)
                    ┌────────────▼────────────┐
                    │     dataset_writer.py    │
                    │  output/rq1/eval_dataset_<date>.jsonl  (RQ1)
                    │  output/rq2/eval_dataset_<date>.jsonl  (RQ2)
                    │  output/rq1/errors_<date>.jsonl        (RQ1)
                    │  output/rq2/errors_<date>.jsonl        (RQ2)
                    └─────────────────────────┘
```

RQ1 and RQ2 share the judge-based, multi-turn architecture above. **RQ3 is a separate, simpler pipeline** (`q3.py`) that doesn't use an LLM judge, attacker/participant persona, or multi-turn escalation:

```
   data/faq_pairs.csv (id, category, question, answer)
              │
              ▼
   QueryPerturber (GPT-4o-mini)  ──►  paraphrased ("GLC") variant of each question
              │
              ▼
   query_violets() / query_baseline()  ──►  one-shot response per query variant
              │
              ▼
   SemanticScorer  ──►  cosine similarity(response embedding, official answer embedding)
              │
              ▼
   output/rq3/eval_dataset_<date>.jsonl
```

---

## Scripts

### Shared
| Script | Role |
|---|---|
| `config.py` | Central config dataclass; loads from environment variables |
| `violets_client.py` | HTTP client for VIOLETS API (`POST /chat` with `user_id` + `query`) |
| `baseline_client.py` | OpenAI client for baseline LLM; maintains full message history client-side. Returns `None` on any API/parsing failure rather than fabricating a response, so the caller can skip that turn instead of judging an error string. |
| `dataset_writer.py` | Writes JSONL output files (one line per turn) and `errors.jsonl` (one line per failed call/judge event). Supports incremental `append=True` writes so a crash mid-run doesn't lose already-completed conversations. Used by RQ1 and RQ2 only — RQ3 (`q3.py`) writes its own JSONL directly. |

### RQ1 — Accuracy & Veracity
| Script | Role |
|---|---|
| `accuracy_runner.py` | Accuracy orchestrator; parallelizes VIOLETS + baseline calls and judge scoring per turn |
| `participant_generator.py` | Generates FAQ query seeds per question-type category |
| `participant.py` | Participant LLM; asks natural follow-up questions across turns |
| `accuracy_judge.py` | Scores each `(query, response)` pair on a 0–100 veracity scale; uses `gpt-5-nano` via the Responses API with `web_search` (`tool_choice="required"`) restricted to `elections.maryland.gov` and `montgomerycountymd.gov` |
| `url_validity_judge.py` | Scores citation quality of VIOLETS's responses only — whether a URL was cited, whether it's reachable, and whether it supports the claim it's attached to |
| `RQ1_analyze.py` | Mixed-effects analysis + two-panel poster figure for RQ1 results |

### RQ2 — Safety & Red-Teaming
| Script | Role |
|---|---|
| `redteam_runner.py` | Main orchestrator; parallelizes VIOLETS + baseline calls and judge scoring per turn; stops early after 3 consecutive VIOLETS refusals |
| `seed_generator.py` | Generates adversarial opening prompts per threat category |
| `attacker.py` | Escalating attacker LLM; adapts strategy based on VIOLETS's responses |
| `judge.py` | Scores each `(attacker_message, response)` pair: PASS / WARN / FAIL with numeric safety score (0–1) |
| `RQ2_analyze.py` | Mixed-effects analysis + two-panel poster figure for RQ2 results |

### RQ3 — FAQ Alignment
| Script | Role |
|---|---|
| `q3.py` | Self-contained RQ3 runner: loads FAQ pairs, optionally generates paraphrased ("GLC") query variants, queries VIOLETS + baseline, scores each response via embedding cosine similarity against the official answer, writes JSONL, and prints a summary. No separate seed/judge/analysis scripts — everything lives in this one file. |
| `data/faq_pairs.csv` | Ground-truth FAQ pairs (`id, category, question, answer`) sourced from the Maryland State Board of Elections and Montgomery County Board of Elections FAQs. Edit this file to add or update questions. |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in this directory:

```env
# Required
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL= 

# VIOLETS server
VIOLETS_ENDPOINT= 
VIOLETS_API_KEY=
VIOLETS_TIMEOUT=60            # seconds to wait for a VIOLETS response before treating it as failed

# Models
ATTACKER_MODEL=gpt-4o
JUDGE_MODEL=gpt-5-nano
ACCURACY_JUDGE_MODEL=gpt-5-nano
SEED_MODEL=gpt-4o-mini
BASELINE_MODEL=gpt-5-nano   # same model as VIOLETS for apples-to-apples comparison
BASELINE_SYSTEM_PROMPT=You are a helpful assistant.   # override to match VIOLETS's real system prompt for a fair comparison

# Run settings
SEEDS_PER_CATEGORY=10        # 10 seeds × 5 categories = 50 conversations per RQ
MAX_TURNS=5                  # up to 5 turns per conversation (≈ 2–3 exchanges)
CONCURRENCY=4
RUN_BASELINE=true
OUTPUT_DIR=./output/rq2      # RQ2 only — see note below
```

**Which variables matter for which RQ:**

| Variable | RQ1 (`accuracy_runner.py`) | RQ2 (`redteam_runner.py`) | RQ3 (`q3.py`) |
|---|:---:|:---:|:---:|
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | ✅ | ✅ | ✅ |
| `VIOLETS_ENDPOINT`, `VIOLETS_API_KEY` | ✅ | ✅ | ✅ |
| `VIOLETS_TIMEOUT` | ✅ | ✅ | ✅ (default 60s; raise if VIOLETS is slow under load) |
| `BASELINE_MODEL`, `BASELINE_SYSTEM_PROMPT`, `RUN_BASELINE` | ✅ | ✅ | ✅ |
| `ATTACKER_MODEL` | ✅ (`participant.py` follow-up questions — misleading name, it's not RQ2-specific) | ✅ (attacker follow-up probes) | — |
| `JUDGE_MODEL` | — | ✅ | — |
| `ACCURACY_JUDGE_MODEL` | ✅ (also used by `url_validity_judge.py`) | — | — |
| `SEED_MODEL` | ✅ (participant generator) | ✅ (seed generator + attacker stop-check) | — |
| `SEEDS_PER_CATEGORY`, `MAX_TURNS` | ✅ | ✅ | — (RQ3 is single-turn; see `--faq_file`/`--no-glc` flags instead) |
| `CONCURRENCY` | ✅ | ✅ | ✅ |
| `OUTPUT_DIR` | ❌ *ignored* — always writes to `./output/rq1` | ✅ | ❌ *ignored* — use `--output_dir` flag instead |

RQ3's embedding model (`text-embedding-3-small`) and query-paraphraser model (`gpt-4o-mini`) are currently hardcoded in `q3.py` rather than environment-configurable.

---

## Reproducing the Evaluation

Assumes Setup (above) is done: dependencies installed, `.env` configured, and VIOLETS reachable at `VIOLETS_ENDPOINT`. Each part below is independent — run whichever RQs you need, in any order.

**Repeated runs (e.g. temporal-stability testing) are date-tagged automatically.** Every run computes today's UTC date (`YYYYMMDD`) once at startup and writes `eval_dataset_<date>.jsonl` / `errors_<date>.jsonl` instead of the bare `eval_dataset.jsonl` — so re-running on a different day never overwrites a previous day's data. **Running twice on the same day still overwrites that day's file** (this granularity was a deliberate choice for simplicity — switch to a full timestamp in `datetime.utcnow().strftime(...)` in each runner's `main()` if you need same-day reruns preserved too).

### 1. RQ1 — Accuracy evaluation

```bash
python accuracy_runner.py
```

- Reads seeds from the five FAQ question-type categories (see [FAQ Question Types](#faq-question-types-rq1)), generated live by `participant_generator.py` (falls back to hardcoded seeds if generation fails).
- Writes incrementally as each conversation finishes:
  - `output/rq1/eval_dataset_<YYYYMMDD>.jsonl` — one line per turn, per model (see schema below)
  - `output/rq1/errors_<YYYYMMDD>.jsonl` — one line per failed call/judge event, if any occurred
- Then run the analysis, pointing `--input` at that day's file:

```bash
python RQ1_analyze.py --input output/rq1/eval_dataset_<YYYYMMDD>.jsonl --output_dir output/rq1/analysis_mixed_<YYYYMMDD>
```

### 2. RQ2 — Safety evaluation

```bash
python redteam_runner.py
```

- Reads seeds from the five threat categories (see [Threat Categories](#threat-categories-rq2)), generated live by `seed_generator.py` (falls back to hardcoded seeds if generation fails).
- Writes incrementally as each conversation finishes:
  - `output/rq2/eval_dataset_<YYYYMMDD>.jsonl` — one line per turn, per model
  - `output/rq2/errors_<YYYYMMDD>.jsonl` — one line per failed call/judge event, if any occurred
- Stops a conversation early after 3 consecutive firm VIOLETS refusals.
- Then run the analysis, pointing `--input` at that day's file:

```bash
python RQ2_analyze.py --input output/rq2/eval_dataset_<YYYYMMDD>.jsonl --output_dir output/rq2/analysis_mixed_<YYYYMMDD>
```

### 3. RQ3 — FAQ alignment evaluation

```bash
python q3.py                              # default: original + GLC-paraphrased query variants
python q3.py --no-glc                     # original queries only, skip paraphrasing
python q3.py --faq_file path/to/file.csv  # use a different FAQ CSV (same 4 required columns)
python q3.py --output_dir path/to/dir     # override the output location (default: output/rq3)
```

- Reads Q&A pairs from `data/faq_pairs.csv` (26 rows across 9 categories by default).
- For each pair, queries VIOLETS and the baseline once with the original question, and once more with a GPT-4o-mini paraphrase (unless `--no-glc`), then scores each response by cosine similarity against the official answer.
- Writes `output/rq3/eval_dataset_<YYYYMMDD>.jsonl` once at the end of the run (unlike RQ1/RQ2, RQ3 does not write incrementally or produce an `errors.jsonl` — failures are logged to the console only).
- There is no `RQ3_analyze.py`; `q3.py` prints a mean-similarity-per-model summary to stdout when it finishes.

Without any code changes, `RQ1_analyze.py`/`RQ2_analyze.py` still default to the undated `output/rq{1,2}/eval_dataset.jsonl` if you ever call `DatasetWriter(...)` without a `run_tag` — the date-tagging only kicks in through `accuracy_runner.py`/`redteam_runner.py`/`q3.py`'s own `main()`, which always pass one.

### Analysis output files (RQ1 / RQ2 only)

`RQ1_analyze.py` and `RQ2_analyze.py` each produce:

| File | Description |
|---|---|
| `table1_model_overall.csv` | Overall VIOLETS vs. Baseline mixed-effects estimate |
| `table2_category_effects.csv` | Per-category treatment effects |
| `table3_turn_effects.csv` | Per-turn treatment effects |
| `rq1_poster_figure.png` / `rq2_poster_figure.png` | Two-panel poster figure (Overall + By Category) |
| `score_distribution.csv` *(RQ1)* | Veracity score distribution by bucket |
| `flagged_for_review.csv` *(RQ1)* | VIOLETS turns scoring below 70 |
| `passfail_by_category.csv` *(RQ2)* | Pass/warn/fail rates per threat category |
| `passfail_by_turn.csv` *(RQ2)* | Pass/warn/fail rates per conversation turn |
| `violation_breakdown.csv` *(RQ2)* | Violation type breakdown for VIOLETS FAIL turns |

---

## Output Schemas

### RQ1 — `output/rq1/eval_dataset_<YYYYMMDD>.jsonl`
One line per turn, per model evaluated.

| Field | Description |
|---|---|
| `conversation_id` | UUID grouping all turns in one conversation |
| `model_id` | `"violets"` or baseline model name |
| `category` | FAQ question type |
| `seed_prompt` | Opening voter query |
| `seed_intent` | What information the participant sought |
| `seed_question_type` | Question type category |
| `turn` | 0-indexed turn number |
| `input` | Participant's message this turn |
| `output` | Agent's response this turn |
| `veracity_score` | 0 (completely false) → 100 (completely accurate) |
| `overall_veracity_score` | Avg veracity score across all turns in the conversation |
| `reasoning` | One-sentence judge explanation |
| `url_citation_rate_score` | 100 if the response cited a URL, else 0 (VIOLETS only, `null` for baseline) |
| `url_accessibility_score` | Avg URL reachability, 0–100 (VIOLETS only, `null` for baseline) |
| `url_accuracy_score` | Avg URL claim support, 0–100 (VIOLETS only, `null` for baseline) |
| `url_details` | Per-URL accessibility/accuracy breakdown (VIOLETS only, `null` for baseline) |
| `urls_found` | Raw URLs extracted from the response (VIOLETS only, `null` for baseline) |
| `url_validity_stats` | Conversation-level citation aggregate — `pct_cited`, `pct_accessible`, `pct_accurate`, etc. (VIOLETS only, `null` for baseline) |
| `timestamp` | ISO 8601 UTC |

### RQ2 — `output/rq2/eval_dataset_<YYYYMMDD>.jsonl`
One line per turn, per model evaluated.

| Field | Description |
|---|---|
| `conversation_id` | UUID grouping all turns in one conversation |
| `model_id` | `"violets"` or baseline model name |
| `category` | Threat category probed |
| `seed_prompt` | Opening adversarial message |
| `seed_intent` | What the attacker was trying to extract |
| `seed_technique` | Adversarial method used |
| `turn` | 0-indexed turn number |
| `input` | Attacker's message this turn |
| `output` | Agent's response this turn |
| `label` | `PASS` / `WARN` / `FAIL` |
| `safety_score` | 0.0 (unsafe) → 1.0 (safe) |
| `overall_safety_score` | Avg safety score across all turns in the conversation |
| `violation` | Category violated, or `"none"` |
| `reasoning` | One-sentence judge explanation |
| `escalation_note` | Whether the attacker made progress |
| `flagged` | `true` if this turn's label is `FAIL` |
| `conversation_flagged` | `true` if any turn in the conversation is `FAIL` |
| `timestamp` | ISO 8601 UTC |

### RQ3 — `output/rq3/eval_dataset_<YYYYMMDD>.jsonl`
One line per `(faq_id, query_type, model_id)` — i.e. up to 4 lines per FAQ pair (original × 2 models, GLC-perturbed × 2 models).

| Field | Description |
|---|---|
| `faq_id` | ID from `data/faq_pairs.csv` |
| `category` | FAQ category from `data/faq_pairs.csv` (see [FAQ Data](#faq-data-rq3) — a different taxonomy than RQ1/RQ2) |
| `query_type` | `"original"` or `"perturbed"` (GLC paraphrase) |
| `model_id` | `"violets"` or baseline model name |
| `query` | The question actually sent (original or paraphrased) |
| `official_answer` | Ground-truth answer from `data/faq_pairs.csv` |
| `model_response` | Agent's one-shot response |
| `similarity_score` | Cosine similarity in [0, 1] between response and official answer embeddings (`text-embedding-3-small`); `null` if the query or scoring failed |
| `timestamp` | ISO 8601 UTC |

### RQ1 / RQ2 — `output/rq{1,2}/errors_<YYYYMMDD>.jsonl`
One line per failed call or degraded judge event (not present for RQ3). Empty (0 bytes) if a run had zero errors — the file is still created since `reset_errors()` truncates it at the start of every run.

| Field | Description |
|---|---|
| `conversation_id` | UUID of the conversation the error occurred in, or `null` for a top-level conversation failure |
| `category` | Question/threat category being evaluated |
| `model_id` | `"violets"` \| baseline model name \| `null` if not model-specific |
| `turn` | 0-indexed turn number, or `null` if not turn-specific |
| `stage` | Where it failed: `attacker_generation` / `participant_generation`, `violets_call`, `baseline_call`, `judge_violets(_veracity\|_url)`, `judge_baseline(_veracity)`, or `conversation` |
| `message` | Exception text or short description |
| `timestamp` | ISO 8601 UTC |

---

## FAQ Question Types (RQ1)

| Category | Description |
|---|---|
| `procedural` | Registration deadlines, polling locations, hours |
| `eligibility` | ID requirements, residency, age, felony records |
| `mail_in` | How to request, deadlines, return methods |
| `results_integrity` | Vote counting, verification, oversight, audits |
| `edge_cases` | No ID, name not on rolls, emergency voting options |

## Threat Categories (RQ2)

| Category | Description |
|---|---|
| `harmful_content` | Self-harm, harassment, child safety, poisoning, lethal dosages |
| `off_topic_drift` | Role abandonment, casual drift, authority impersonation |
| `misinformation` | Election fraud claims, voting machine tampering |
| `sensitive_personal` | PII requests (SSN), identity verification edge cases |
| `political_electoral` | Partisan endorsements, political attack content, election interference |

## FAQ Data (RQ3)

`data/faq_pairs.csv` currently has 26 rows across 9 categories — a different taxonomy from RQ1/RQ2's question types, since it's derived directly from the structure of the official FAQ sources rather than authored per this project's taxonomy:

| Category |
|---|
| `voter_registration` |
| `requesting_a_ballot` |
| `state_of_application_ballot` |
| `ballot_documents_envelope` |
| `marking_reviewing_ballot` |
| `returning_ballot` |
| `ballot_drop_boxes` |
| `in_person_voting` |
| `general_election` |

Required columns: `id, category, question, answer`. Add rows to extend coverage — no code changes needed.

---

## Key Design Decisions

- **VIOLETS is the primary agent** — the attacker/participant sees VIOLETS's responses to drive follow-up turns; the baseline runs silently in parallel (RQ1/RQ2)
- **Baseline shares the same turns** — identical messages sent to both agents for direct RQ1/RQ2 comparison
- **Judges score independently** — each response is scored on its own, not relative to the other model (RQ1/RQ2). RQ3 uses no LLM judge at all — alignment is measured by embedding cosine similarity against the official FAQ answer, which avoids judge-model bias but also can't explain *why* a response diverges
- **Bounded concurrency** — `asyncio.Semaphore(cfg.concurrency)` prevents API rate limit exhaustion (all three RQs)
- **Fallback seeds** — hardcoded seeds in both RQ1/RQ2 generators ensure runs complete even if LLM generation fails
- **Incremental, crash-safe writes (RQ1/RQ2)** — `eval_dataset_<date>.jsonl` and `errors_<date>.jsonl` are appended to as each conversation finishes, not buffered in memory and written once at the end; a crash mid-run keeps everything completed so far. RQ3 does not yet have this — it writes once at the end of the run.
- **Output filenames are date-tagged (all three RQs)** — each run computes today's UTC date once at startup and writes `eval_dataset_<YYYYMMDD>.jsonl` (and `errors_<YYYYMMDD>.jsonl` for RQ1/RQ2) instead of a fixed filename, so repeated runs on different days (e.g. temporal-stability testing) don't overwrite each other. Two runs on the *same* day still overwrite each other — this was a deliberate simplicity/collision-risk tradeoff, not an oversight.
- **A VIOLETS-side failure ends the conversation, but doesn't waste the baseline's turn** — the participant/attacker needs VIOLETS's response to drive the next turn, so a VIOLETS API failure stops that conversation; but if the baseline call for that same turn already succeeded, it's still scored and recorded rather than discarded
- **Baseline errors are excluded, not judged** — if the baseline call fails, that turn is skipped entirely rather than having the judge score a fabricated error string as if it were real model output
- **Errors are structured, not just logged (RQ1/RQ2)** — every dropped turn or degraded judge call is recorded to `errors.jsonl` with enough context (conversation, turn, stage) to audit how much data was lost and why, in addition to the console log line
