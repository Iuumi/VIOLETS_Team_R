"""
VIOLETS Query Interface.

Defines the plug-in point for sending queries to VIOLETS and receiving responses.
The evaluation pipeline calls query_violets() — register your VIOLETS backend here
before running the pipeline.

Usage:
    from evaluation.violets_interface import set_backend
    from evaluation.rq1_pipeline import run_rq1_evaluation

    set_backend(MyVioletsBackend())
    results = run_rq1_evaluation(n_per_type=10)
"""
from typing import Protocol, Optional


class VioletsBackend(Protocol):
    """
    Protocol any VIOLETS backend must satisfy.

    Implement this in your own module and register it with set_backend().
    `conversation_history` follows OpenAI message format:
        [{"role": "user"|"assistant", "content": "..."}]
    """
    def query(self, message: str, conversation_history: list[dict]) -> str:
        """Send a voter message and return VIOLETS's response string."""
        ...


class StubVioletsBackend:
    """
    Stub backend for testing the evaluation pipeline without a live VIOLETS instance.
    Returns a plausible but static Maryland elections answer.
    """
    def query(self, message: str, conversation_history: list[dict]) -> str:
        return (
            "Thank you for your question about Maryland elections. "
            "To register to vote in Maryland you must be a U.S. citizen, "
            "a Maryland resident, and at least 18 years old by Election Day. "
            "The registration deadline is 21 days before the election. "
            "You can register online at elections.maryland.gov, by mail, "
            "or in person at your local Board of Elections office."
        )


_default_backend: Optional[VioletsBackend] = None


def set_backend(backend: VioletsBackend) -> None:
    """Register the VIOLETS backend to use for all evaluations."""
    global _default_backend
    _default_backend = backend


def query_violets(
    message: str,
    conversation_history: list[dict],
    backend: Optional[VioletsBackend] = None,
) -> str:
    """
    Send a message to VIOLETS and return its response.

    Args:
        message: The voter's current query.
        conversation_history: Prior turns as OpenAI-style dicts
            (does NOT include the current message).
        backend: Override the registered default for this call.

    Raises:
        RuntimeError: If no backend has been registered and none is passed.
    """
    b = backend or _default_backend
    if b is None:
        raise RuntimeError(
            "No VIOLETS backend registered. "
            "Call set_backend(your_backend) before running the evaluation, "
            "or pass a backend explicitly to query_violets(). "
            "For testing without a live instance, use StubVioletsBackend()."
        )
    return b.query(message, conversation_history)
