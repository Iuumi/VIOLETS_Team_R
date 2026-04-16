"""
Participant LLM for synthetic voter query generation.

Queries are drawn from the five-category question-type categorization in config.py:
  procedural | eligibility | mail_in_voting | results_integrity | edge_case

The same client instance should be reused across calls to avoid re-initialising
the SDK on every dialogue.
"""
import os
from typing import Optional

from .config import (
    PARTICIPANT_MODEL,
    QUESTION_CATEGORIZATION,
)


# System prompt — finalising the exact wording is a RQ1 To-Do item


PARTICIPANT_SYSTEM_PROMPT = """\
You are roleplaying as a Maryland voter who needs help understanding elections
and voting procedures for the upcoming 2026 elections.

Creating a persona guidelines:
- You are a real person with a genuine question — not an AI assistant
- Speak naturally and conversationally, as if texting a help-line
- You may have partial knowledge or a common misconception
- Ask ONE focused question at a time (1–2 sentences)
- Do not volunteer answers or pretend to already know the information
- Your follow-up questions should flow naturally from what you just learned

Output only the voter's question — no preamble, no meta-commentary.\
"""


def _make_client():
    """Initialise and return the OpenAI client."""
    import openai
    return openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _call(prompt: str, client, max_tokens: int = 200) -> str:
    """Send a prompt to the Participant LLM and return the response text."""
    resp = client.chat.completions.create(
        model=PARTICIPANT_MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": PARTICIPANT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()



# Public API


def generate_initial_query(question_type: str, client=None) -> str:
    """
    Generate an opening voter question for the given question type.

    Args:
        question_type: One of the keys in QUESTION_CATEGORIZATION.
        client: Reusable LLM client; created fresh if not supplied.

    Returns:
        A natural-language voter question string.
    """
    if client is None:
        client = _make_client()

    info = QUESTION_CATEGORIZATION.get(question_type, QUESTION_CATEGORIZATION["procedural"])
    prompt = (
        f"Generate a realistic question a Maryland voter might ask about: "
        f"{info['description']}.\n"
        f"Reference examples (do NOT copy these exactly — write something different):\n"
        + "\n".join(f"  - {ex}" for ex in info["examples"][:2])
    )
    return _call(prompt, client)


def generate_followup_query(
    conversation_history: list[dict],
    violets_response: str,
    question_type: str,
    client=None,
) -> Optional[str]:
    """
    Generate a follow-up question based on VIOLETS's last response.

    Returns None if the conversation should naturally end (answer was complete).

    Args:
        conversation_history: Prior turns as OpenAI-style dicts.
        violets_response: The VIOLETS response the voter just received.
        question_type: Question category for context.
        client: Reusable LLM client; created fresh if not supplied.
    """
    if client is None:
        client = _make_client()

    history_text = "\n".join(
        f"{'Voter' if t['role'] == 'user' else 'VIOLETS'}: {t['content']}"
        for t in conversation_history
    )
    prompt = (
        f"The conversation so far:\n{history_text}\n\n"
        f"VIOLETS just answered: {violets_response}\n\n"
        "As the voter, do you have a natural follow-up question?\n"
        "- If the answer was complete and you are satisfied, reply with exactly: NO_FOLLOWUP\n"
        "- Otherwise, write one short follow-up question (1–2 sentences)."
    )
    result = _call(prompt, client, max_tokens=150)

    if result.strip().upper() == "NO_FOLLOWUP" or "no_followup" in result.lower():
        return None
    return result
