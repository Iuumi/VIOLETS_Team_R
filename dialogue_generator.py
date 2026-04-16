"""
A dialogue generator to show a multi-turn based conversation

Generates a single dialogue between the Participant LLM (synthetic voter)
and VIOLETS, following the evaluation protocol:

  Turn 1: Participant LLM generates an initial voter query
  Turn 2: VIOLETS responds
  Turn 3: Participant LLM optionally generates a follow-up
  Turn 4: VIOLETS responds
  ...up to TURNS_PER_DIALOGUE VIOLETS turns

The resulting Dialogue object carries all turns with timestamps and word counts,
and can be serialised to JSON for the results archive.
"""
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

from .config import QUESTION_CATEGORIZATION, TURNS_PER_DIALOGUE
from .participant_llm import generate_followup_query, generate_initial_query
from .violets_interface import VioletsBackend, query_violets



# Data models

@dataclass
class Turn:
    role: str        # "voter" | "violets"
    content: str
    word_count: int = 0
    timestamp: str = ""

    def __post_init__(self):
        self.word_count = len(self.content.split())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class Dialogue:
    id: str
    question_type: str
    turns: list[Turn] = field(default_factory=list)
    created_at: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    
    # Convenience accessors


    @property
    def violets_turns(self) -> list[Turn]:
        """All VIOLETS response turns in order."""
        return [t for t in self.turns if t.role == "violets"]

    def to_openai_history(self) -> list[dict]:
        """
        Return the conversation as OpenAI-style message dicts.
        Used to pass prior context back to both LLMs.
        """
        mapping = {"voter": "user", "violets": "assistant"}
        return [{"role": mapping[t.role], "content": t.content} for t in self.turns]

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# Public api section 


def generate_dialogue(
    question_type: str,
    violets_backend: Optional[VioletsBackend] = None,
    participant_client=None,
    max_turns: int = TURNS_PER_DIALOGUE,
) -> Dialogue:
    """
    Generate a single multi-turn dialogue for the given question type.

    Args:
        question_type: One of the keys in QUESTION_CATEGORIZATION.
        violets_backend: Backend to query VIOLETS; uses the registered default if None.
        participant_client: Reusable Participant LLM client; created fresh if None.
        max_turns: Maximum number of VIOLETS response turns (default TURNS_PER_DIALOGUE).

    Returns:
        A completed Dialogue with all turns populated.
    """
    dialogue = Dialogue(
        id=str(uuid.uuid4()),
        question_type=question_type,
        metadata={
            "max_turns": max_turns,
            "question_type_desc": QUESTION_CATEGORIZATION.get(question_type, {}).get("description", ""),
        },
    )

    # --- Turn 1: initial voter query ---
    initial_query = generate_initial_query(question_type, client=participant_client)
    dialogue.turns.append(Turn(role="voter", content=initial_query))

    # --- Turn 2: VIOLETS response ---
    violets_response = query_violets(
        message=initial_query,
        conversation_history=[],  # no prior history on first turn
        backend=violets_backend,
    )
    dialogue.turns.append(Turn(role="violets", content=violets_response))

    # --- Follow-up turns ---
    for _ in range(max_turns - 1):
        history = dialogue.to_openai_history()

        followup = generate_followup_query(
            conversation_history=history,
            violets_response=violets_response,
            question_type=question_type,
            client=participant_client,
        )
        if followup is None:
            break  # voter is satisfied — end dialogue naturally

        dialogue.turns.append(Turn(role="voter", content=followup))

        # Pass full history *before* the new voter message to VIOLETS
        history_before = dialogue.to_openai_history()[:-1]
        violets_response = query_violets(
            message=followup,
            conversation_history=history_before,
            backend=violets_backend,
        )
        dialogue.turns.append(Turn(role="violets", content=violets_response))

    return dialogue
