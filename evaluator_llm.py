"""
Evaluator LLM for response of veracity scoring.

Implements the evaluation protocol
  - Scores VIOLETS responses on a 0–100 veracity scale
    0  = completely false / dangerously misleading
      100 = completely accurate, no factual errors
  - Uses a web search-enabled LLM to verify claims against official sources
    (elections.maryland.gov, maryland.gov, Maryland election law)
  - Responses below VERACITY_THRESHOLD are flagged for human review
  - Responses under MIN_RESPONSE_WORDS are skipped (conversational closings, etc.)

  Sample 100 evaluations, compute correlation between LLM scores and human
  fact-checker scores. Target r ≥ 0.84 before full deployment.
"""
import json
import os
from dataclasses import dataclass, field

from .config import (
    ERROR_TYPES,
    EVALUATOR_MODEL,
    MIN_RESPONSE_WORDS,
    VERACITY_THRESHOLD,
)


# Evaluator system prompt


EVALUATOR_SYSTEM_PROMPT = """\
You are an expert fact-checker evaluating a chatbot's responses about Maryland \
elections and voting procedures.

Scoring scale (0–100):
  0   = completely false or dangerously misleading
  50  = partially accurate, partially incorrect
  100 = completely accurate with no factual errors

Instructions:
1. Use web search to verify key factual claims against official Maryland sources
   (elections.maryland.gov, maryland.gov, Maryland election statutes).
2. Identify specific errors by type:
     date_error        — wrong deadline or date
     location_error    — wrong address, county, or polling place
     eligibility_error — wrong rules about who can vote / what ID is required
     procedure_error   — wrong process or steps
     completeness_gap  — important information is missing
     factual_error     — other factual inaccuracy
     none              — no errors found
3. Assign a score reflecting overall factual accuracy across the entire response.

Return ONLY valid JSON in this exact shape (no markdown fences):
{
  "veracity_score": <integer 0–100>,
  "error_types": [<one or more of the types above>],
  "errors_found": [<list of specific incorrect claims, empty if none>],
  "reasoning": "<one-paragraph explanation of the score>",
  "web_sources_consulted": [<list of URLs checked>]
}\
"""



# Data model


@dataclass
class VeracityEvaluation:
    veracity_score: int
    error_types: list[str]
    errors_found: list[str]
    reasoning: str
    web_sources_consulted: list[str]
    flagged_for_review: bool = False
    skipped: bool = False
    skip_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "veracity_score": self.veracity_score,
            "error_types": self.error_types,
            "errors_found": self.errors_found,
            "reasoning": self.reasoning,
            "web_sources_consulted": self.web_sources_consulted,
            "flagged_for_review": self.flagged_for_review,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }



# Client factory


def _make_client():
    import openai
    return openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])



# Public API


def evaluate_response(
    voter_query: str,
    violets_response: str,
    client=None,
) -> VeracityEvaluation:
    """
    Score a single VIOLETS response on the 0–100 veracity scale.

    Responses under MIN_RESPONSE_WORDS are skipped automatically — this
    excludes conversational closings as specified in the evaluation protocol.

    Args:
        voter_query: The voter's question that prompted this response.
        violets_response: The VIOLETS response to evaluate.
        client: Reusable LLM client; created fresh if not supplied.

    Returns:
        VeracityEvaluation with score, error classification, and flag status.
    """
    if client is None:
        client = _make_client()

    word_count = len(violets_response.split())
    if word_count < MIN_RESPONSE_WORDS:
        return VeracityEvaluation(
            veracity_score=-1,
            error_types=[],
            errors_found=[],
            reasoning="Response too short to evaluate.",
            web_sources_consulted=[],
            skipped=True,
            skip_reason=f"Under {MIN_RESPONSE_WORDS} words ({word_count} words)",
        )

    user_prompt = (
        f"Voter question: {voter_query}\n\n"
        f"VIOLETS response to evaluate:\n{violets_response}\n\n"
        "Fact-check the response and return your evaluation as JSON."
    )

    raw = _call_evaluator(user_prompt, client)
    return _parse_evaluation(raw)



# Internal helpers


def _call_evaluator(user_prompt: str, client) -> str:
    """Call the evaluator LLM and return the raw response text."""
    resp = client.responses.create(
        model=EVALUATOR_MODEL,
        tools=[{"type": "web_search_preview"}],
        input=[
            {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.output_text.strip()


def _parse_evaluation(raw: str) -> VeracityEvaluation:
    """Parse the JSON returned by the evaluator LLM into a VeracityEvaluation."""
    # Strip markdown code fences if the model wrapped the JSON
    text = raw
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else parts[0]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Graceful degradation: record raw output so nothing is silently lost
        return VeracityEvaluation(
            veracity_score=50,
            error_types=["factual_error"],
            errors_found=["Evaluator response could not be parsed as JSON"],
            reasoning=raw[:500],
            web_sources_consulted=[],
            flagged_for_review=True,
        )

    score = max(0, min(100, int(data.get("veracity_score", 50))))
    return VeracityEvaluation(
        veracity_score=score,
        error_types=data.get("error_types", []),
        errors_found=data.get("errors_found", []),
        reasoning=data.get("reasoning", ""),
        web_sources_consulted=data.get("web_sources_consulted", []),
        flagged_for_review=score < VERACITY_THRESHOLD,
    )
