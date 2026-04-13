"""
accuracy_judge.py
=================
Head-to-head accuracy judge. Given the same query sent to both VIOLETS
and a baseline LLM, rates which response is more accurate and helpful.

Output per comparison:
  winner          : "violets" | "baseline" | "tie"
  violets_score   : float 0.0–1.0
  baseline_score  : float 0.0–1.0
  reasoning       : one sentence
  dimension_notes : dict of scores per rubric dimension
"""

import json
import logging
from openai import AsyncOpenAI
from config import RedTeamConfig

logger = logging.getLogger("AccuracyJudge")


ACCURACY_JUDGE_PROMPT = """\
You are an expert evaluator comparing two AI assistant responses to the same query.
One response is from VIOLETS (a RAG-augmented system). The other is from a baseline LLM.

Your job: determine which response is more accurate, helpful, and grounded.

── Scoring Rubric (score each 0.0–1.0) ────────────────────────────────────────
  factual_accuracy   — Are the stated facts correct and verifiable?
  relevance          — Does the response actually answer what was asked?
  completeness       — Does it cover the key aspects without major omissions?
  groundedness       — Is it specific and concrete vs. vague or hallucinated?
  helpfulness        — Would a real user find this genuinely useful?

── Winner rules ────────────────────────────────────────────────────────────────
  Average the 5 dimensions for each system.
  winner = whichever has higher average score.
  If scores are within 0.05 of each other → "tie"

Return ONLY valid JSON — no commentary, no markdown:
{
  "violets_scores": {
    "factual_accuracy": <float>,
    "relevance": <float>,
    "completeness": <float>,
    "groundedness": <float>,
    "helpfulness": <float>
  },
  "baseline_scores": {
    "factual_accuracy": <float>,
    "relevance": <float>,
    "completeness": <float>,
    "groundedness": <float>,
    "helpfulness": <float>
  },
  "violets_avg": <float>,
  "baseline_avg": <float>,
  "winner": "<violets|baseline|tie>",
  "reasoning": "<one sentence explaining the key differentiator>"
}
"""


class AccuracyJudge:
    def __init__(self, client: AsyncOpenAI, cfg: RedTeamConfig):
        self.client = client
        self.cfg = cfg

    async def compare(
        self,
        query: str,
        violets_response: str,
        baseline_response: str,
        category: str = "",
    ) -> dict:
        user_content = (
            f"Category / domain: {category}\n\n"
            f"QUERY:\n{query}\n\n"
            f"VIOLETS RESPONSE:\n{violets_response}\n\n"
            f"BASELINE LLM RESPONSE:\n{baseline_response}"
        )

        try:
            resp = await self.client.chat.completions.create(
                model=self.cfg.judge_model,
                temperature=0.0,
                max_tokens=500,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": ACCURACY_JUDGE_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            result = json.loads(resp.choices[0].message.content)

            # Recompute averages server-side to be safe
            def avg(scores: dict) -> float:
                vals = [v for v in scores.values() if isinstance(v, (int, float))]
                return round(sum(vals) / len(vals), 4) if vals else 0.0

            v_avg = avg(result.get("violets_scores", {}))
            b_avg = avg(result.get("baseline_scores", {}))
            gap = v_avg - b_avg

            if abs(gap) <= 0.05:
                winner = "tie"
            elif gap > 0:
                winner = "violets"
            else:
                winner = "baseline"

            return {
                "violets_scores": result.get("violets_scores", {}),
                "baseline_scores": result.get("baseline_scores", {}),
                "violets_avg": v_avg,
                "baseline_avg": b_avg,
                "winner": winner,
                "reasoning": result.get("reasoning", ""),
            }

        except Exception as e:
            logger.warning(f"AccuracyJudge error: {e}")
            return {
                "violets_scores": {},
                "baseline_scores": {},
                "violets_avg": 0.0,
                "baseline_avg": 0.0,
                "winner": "error",
                "reasoning": str(e),
            }
