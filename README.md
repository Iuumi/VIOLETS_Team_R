# VIOLETS Pre-Deployment Evaluation Pipeline

Automated evaluation framework for VIOLETS — a RAG-based voting information chatbot for 2026 Maryland elections. Covers two research questions:

- **RQ1 (Accuracy)** — How factually accurate are VIOLETS's responses, and does it hallucinate?
- **RQ2 (Safety)** — Does VIOLETS correctly handle adversarial, out-of-scope, and sensitive queries?

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
                    │  output/rq1/eval_dataset.jsonl  (RQ1)
                    │  output/rq2/eval_dataset.jsonl  (RQ2)
                    └─────────────────────────┘
```

---

## Scripts

### Shared
| Script | Role |
|---|---|
| `config.py` | Central config dataclass; loads from environment variables |
| `violets_client.py` | HTTP client for VIOLETS API (`POST /chat` with `user_id` + `query`) |
| `baseline_client.py` | OpenAI client for baseline LLM; maintains full message history client-side |
| `dataset_writer.py` | Writes JSONL output files; one line per turn |

### RQ1 — Accuracy & Veracity
| Script | Role |
|---|---|
| `accuracy_runner.py` | Accuracy orchestrator; mirrors `redteam_runner.py` in structure |
| `participant_generator.py` | Generates FAQ query seeds per question-type category |
| `participant.py` | Participant LLM; asks natural follow-up questions across turns |
| `accuracy_judge.py` | Scores each `(query, response)` pair on a 0–100 veracity scale; uses `gpt-5-nano` via the Responses API with `web_search` (`tool_choice="required"`) restricted to `elections.maryland.gov` and `montgomerycountymd.gov` |

### RQ2 — Safety & Red-Teaming
| Script | Role |
|---|---|
| `redteam_runner.py` | Main orchestrator; runs multi-turn adversarial conversations |
| `seed_generator.py` | Generates adversarial opening prompts per threat category |
| `attacker.py` | Escalating attacker LLM; adapts strategy based on VIOLETS's responses |
| `judge.py` | Scores each `(attacker_message, response)` pair: PASS / WARN / FAIL |

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

# Models
ATTACKER_MODEL=gpt-4o
JUDGE_MODEL=gpt-5-nano
ACCURACY_JUDGE_MODEL=gpt-5-nano
SEED_MODEL=gpt-4o-mini
BASELINE_MODEL=gpt-4o-mini

# Run settings
SEEDS_PER_CATEGORY=5
MAX_TURNS=6
CONCURRENCY=4
RUN_BASELINE=true
OUTPUT_DIR=./output/rq2
```

---

## Running

### RQ1 — Accuracy evaluation
```bash
python accuracy_runner.py
```
Output: `./output/rq1/eval_dataset.jsonl`

### RQ2 — Safety evaluation
```bash
python redteam_runner.py
```
Output: `./output/rq2/eval_dataset.jsonl`

---

## Output Schemas

### RQ1 — `output/rq1/eval_dataset.jsonl`
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
| `timestamp` | ISO 8601 UTC |

### RQ2 — `output/rq2/eval_dataset.jsonl`
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

---

## Key Design Decisions

- **VIOLETS is the primary agent** — the attacker/participant sees VIOLETS's responses to drive follow-up turns; the baseline runs silently in parallel
- **Baseline shares the same turns** — identical messages sent to both agents for direct RQ1/RQ2 comparison
- **Judges score independently** — each response is scored on its own, not relative to the other model
- **Bounded concurrency** — `asyncio.Semaphore(cfg.concurrency)` prevents API rate limit exhaustion
- **Fallback seeds** — hardcoded seeds in both generators ensure runs complete even if LLM generation fails
