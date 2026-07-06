"""
baseline_client.py
==================
Client for the baseline model (e.g. GPT-4.1-nano / GPT-4o-mini / GPT-5-nano).

The client must:
  1. Maintain the full conversation history locally.
  2. Send the complete messages array every turn.
  3. Append each new (user, assistant) pair to its local history.

This produces outputs in the same JSONL schema as VIOLETS so both agents
can be compared directly for RQ1/RQ2.

Reasoning models (o1/o3/o4/gpt-5 family) take different call parameters than
regular chat models via the chat.completions endpoint:
  - max_completion_tokens instead of max_tokens (max_tokens is rejected outright)
  - no custom temperature — only the default (1) is supported
  - a much larger completion-token budget: reasoning models spend an
    invisible "reasoning" token budget before emitting any visible content,
    so a budget sized for a normal chat model (e.g. 512) is silently
    consumed entirely by reasoning and returns empty content with no error.
    Measured empirically against gpt-5-nano: realistic multi-sentence
    election-FAQ answers needed 4,400-5,800 reasoning tokens alone.
"""

import logging
from typing import Optional
from openai import AsyncOpenAI
from config import RedTeamConfig

logger = logging.getLogger("BaselineClient")

# Model name prefixes that identify an OpenAI reasoning model.
_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")

# Cap, not a fixed spend — the call stops as soon as the model is done.
# Sized with headroom above the ~5-6k reasoning tokens observed for
# realistic prompts so answers aren't truncated into empty content.
_REASONING_MODEL_MAX_COMPLETION_TOKENS = 8000

# Regular (non-reasoning) chat models use this fixed budget + temperature.
_CHAT_MODEL_MAX_TOKENS = 512
_CHAT_MODEL_TEMPERATURE = 0.7


def _is_reasoning_model(model: str) -> bool:
    return model.lower().startswith(_REASONING_MODEL_PREFIXES)


class BaselineClient:
    def __init__(self, cfg: RedTeamConfig, client: AsyncOpenAI):
        self.cfg = cfg
        self.client = client

    def new_session(self) -> "BaselineSession":
        """
        Create a new conversation session with a fresh message history.
        Call once per conversation; reuse the session for all turns.
        """
        return BaselineSession(
            client=self.client,
            model=self.cfg.baseline_model,
            system_prompt=self.cfg.baseline_system_prompt,
        )


class BaselineSession:
    """
    Represents one conversation with the baseline model.
    Maintains client-side message history across turns — the baseline has
    no server-side memory, so we must send the full history every call.
    """

    def __init__(self, client: AsyncOpenAI, model: str, system_prompt: str):
        self.client = client
        self.model = model
        self.is_reasoning_model = _is_reasoning_model(model)
        # Build the initial messages array with the system prompt
        self.history: list[dict] = [
            {"role": "system", "content": system_prompt}
        ]
        logger.debug(
            f"New baseline session | model={model} | reasoning_model={self.is_reasoning_model}"
        )

    async def chat(self, latest_user_message: str) -> Optional[str]:
        """
        Append the latest user message to history, call the baseline model
        with the full history, append the assistant reply, and return it.

        Returns None on any API or parsing failure — the caller treats a
        None response as "no baseline turn this round" rather than judging
        a fabricated error string as if it were real model output. An
        empty (but technically successful) response counts as a failure
        too, since that's exactly how an under-provisioned reasoning model
        call manifests.
        """
        self.history.append({"role": "user", "content": latest_user_message})

        if self.is_reasoning_model:
            call_kwargs = {"max_completion_tokens": _REASONING_MODEL_MAX_COMPLETION_TOKENS}
        else:
            call_kwargs = {"max_tokens": _CHAT_MODEL_MAX_TOKENS, "temperature": _CHAT_MODEL_TEMPERATURE}

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                **call_kwargs,
            )
            choice = resp.choices[0]
            if choice.finish_reason == "length":
                logger.warning(
                    f"Baseline response hit the token cap before finishing "
                    f"(model={self.model}, reasoning_model={self.is_reasoning_model})"
                )
            reply = choice.message.content.strip()
            if not reply:
                logger.error(
                    f"Baseline returned empty content "
                    f"(model={self.model}, reasoning_model={self.is_reasoning_model})"
                )
                reply = None
        except (IndexError, AttributeError) as parse_err:
            logger.error(f"Baseline response parsing error: {parse_err}")
            reply = None
        except Exception as e:
            logger.error(f"Baseline model error: {e}")
            reply = None

        # Persist the assistant's reply so it's included in the next turn
        if reply is not None:
            self.history.append({"role": "assistant", "content": reply})
        else:
            self.history.pop()  # remove the unpaired user message
        return reply
