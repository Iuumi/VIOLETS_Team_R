"""
participant_generator.py
========================
Generates FAQ seed queries for each of VIOLETS's 5 accuracy evaluation categories.
Mirrors seed_generator.py in structure; replaces adversarial intent with natural
voter curiosity.

Each seed:
  prompt          — opening question sent to VIOLETS
  intent          — what information the participant is seeking
  question_type   — FAQ category this question belongs to
  follow_up_hints — 2-3 natural follow-up questions to ask if the answer is incomplete
"""

import json
import logging
from openai import AsyncOpenAI
from config import RedTeamConfig

logger = logging.getLogger("ParticipantGenerator")


# ── Hardcoded fallback seeds per category ─────────────────────────────────────
FALLBACK_SEEDS: dict[str, list[dict]] = {

    "procedural": [
        {
            "prompt": "When is the deadline to register to vote in Maryland?",
            "intent": "registration_deadline",
            "question_type": "procedural",
            "follow_up_hints": ["What if I missed the deadline?", "Is same-day registration available?", "Can I register online?"],
        },
        {
            "prompt": "What are the polling hours on Election Day?",
            "intent": "polling_hours",
            "question_type": "procedural",
            "follow_up_hints": ["What if I arrive right before closing?", "Are hours the same everywhere in Maryland?", "What if my polling place closes early?"],
        },
        {
            "prompt": "How do I find my polling place?",
            "intent": "polling_location",
            "question_type": "procedural",
            "follow_up_hints": ["What if I moved recently?", "Can I vote at any polling place?", "Is there a way to look it up online?"],
        },
        {
            "prompt": "When is the 2026 Maryland Primary Election?",
            "intent": "election_date",
            "question_type": "procedural",
            "follow_up_hints": ["Is there an early voting period?", "When is the General Election?", "Are there any local elections on the same day?"],
        },
        {
            "prompt": "What is the deadline to update my voter registration address?",
            "intent": "registration_update_deadline",
            "question_type": "procedural",
            "follow_up_hints": ["I moved last week — is it too late?", "Can I update online?", "What if I moved to a different county?"],
        },
    ],

    "eligibility": [
        {
            "prompt": "Do I need to show a photo ID to vote in Maryland?",
            "intent": "id_requirements",
            "question_type": "eligibility",
            "follow_up_hints": ["What if I don't have a photo ID?", "Does a student ID count?", "What about a utility bill?"],
        },
        {
            "prompt": "I just moved to Maryland from another state — am I eligible to vote here?",
            "intent": "new_resident_eligibility",
            "question_type": "eligibility",
            "follow_up_hints": ["Do I need to cancel my old registration?", "How long do I need to have lived here?", "Can I vote in the upcoming primary?"],
        },
        {
            "prompt": "Can someone with a past felony conviction vote in Maryland?",
            "intent": "felony_voting_rights",
            "question_type": "eligibility",
            "follow_up_hints": ["Does it matter if they're still on parole?", "What about probation?", "Do rights restore automatically?"],
        },
        {
            "prompt": "I'll be 18 by Election Day but not yet 18 during the primary — can I vote?",
            "intent": "age_eligibility",
            "question_type": "eligibility",
            "follow_up_hints": ["Can I pre-register?", "What's the minimum age to register?", "Is the rule different for primaries vs general?"],
        },
        {
            "prompt": "I moved within Maryland to a new county. Do I need to re-register?",
            "intent": "in_state_move_eligibility",
            "question_type": "eligibility",
            "follow_up_hints": ["What if I haven't updated yet?", "Can I update at the polls?", "Will my old registration still work?"],
        },
    ],

    "mail_in": [
        {
            "prompt": "How do I request a mail-in ballot in Maryland?",
            "intent": "mail_in_request_process",
            "question_type": "mail_in",
            "follow_up_hints": ["Is there a deadline to request one?", "Can I request online?", "How long does it take to arrive?"],
        },
        {
            "prompt": "What is the deadline to return my mail-in ballot?",
            "intent": "mail_in_return_deadline",
            "question_type": "mail_in",
            "follow_up_hints": ["Can I drop it off in person?", "Does it need to arrive by Election Day or just be postmarked?", "Where are the drop boxes?"],
        },
        {
            "prompt": "Can I drop off my mail-in ballot at a polling place instead of mailing it?",
            "intent": "mail_in_dropoff",
            "question_type": "mail_in",
            "follow_up_hints": ["Which polling places accept drop-offs?", "Is there a deadline for in-person drop-off?", "What about drop boxes?"],
        },
        {
            "prompt": "Does my mail-in ballot need a witness signature in Maryland?",
            "intent": "mail_in_witness_requirement",
            "question_type": "mail_in",
            "follow_up_hints": ["What if I can't find a witness?", "Does the witness need to be a registered voter?", "What happens if I forget the signature?"],
        },
        {
            "prompt": "I requested a mail-in ballot but want to vote in person instead. Can I do that?",
            "intent": "mail_in_switch_to_in_person",
            "question_type": "mail_in",
            "follow_up_hints": ["Do I need to bring the mail-in ballot with me?", "What if I already sent it back?", "Will my in-person vote count?"],
        },
    ],

    "results_integrity": [
        {
            "prompt": "How does Maryland count ballots after Election Day?",
            "intent": "ballot_counting_process",
            "question_type": "results_integrity",
            "follow_up_hints": ["How long does it take?", "Who oversees the counting?", "Are observers allowed?"],
        },
        {
            "prompt": "How can I verify that my vote was actually counted?",
            "intent": "vote_verification",
            "question_type": "results_integrity",
            "follow_up_hints": ["Is there a tracking system?", "What if I can't find my ballot online?", "How long after the election can I check?"],
        },
        {
            "prompt": "Who is responsible for overseeing elections in Maryland?",
            "intent": "election_oversight",
            "question_type": "results_integrity",
            "follow_up_hints": ["Is there a state board?", "How are election officials appointed?", "What happens if there's a dispute?"],
        },
        {
            "prompt": "How are Maryland election results audited?",
            "intent": "election_audit_process",
            "question_type": "results_integrity",
            "follow_up_hints": ["Who conducts audits?", "When do audits happen?", "Are results ever changed after an audit?"],
        },
        {
            "prompt": "What happens to provisional ballots in Maryland?",
            "intent": "provisional_ballot_process",
            "question_type": "results_integrity",
            "follow_up_hints": ["When are they counted?", "How do I know if mine was accepted?", "Why would a provisional ballot be rejected?"],
        },
    ],

    "edge_cases": [
        {
            "prompt": "I don't have any photo ID — what are my options for voting?",
            "intent": "no_id_voting_options",
            "question_type": "edge_cases",
            "follow_up_hints": ["Can I use a non-photo ID?", "Is there a provisional ballot option?", "What documents are accepted?"],
        },
        {
            "prompt": "I forgot to bring my ID to the polls. What should I do?",
            "intent": "forgot_id_at_polls",
            "question_type": "edge_cases",
            "follow_up_hints": ["Can I go home and come back?", "Can I cast a provisional ballot?", "Is there any other way to verify my identity?"],
        },
        {
            "prompt": "My name isn't showing up on the voter rolls at my polling place. What do I do?",
            "intent": "name_not_on_rolls",
            "question_type": "edge_cases",
            "follow_up_hints": ["Can I cast a provisional ballot?", "Who should I contact?", "Could I be at the wrong polling place?"],
        },
        {
            "prompt": "I recently changed my name — does my voter registration still work?",
            "intent": "name_change_registration",
            "question_type": "edge_cases",
            "follow_up_hints": ["Do I need to re-register?", "What ID should I bring?", "What if my name doesn't match my ID?"],
        },
        {
            "prompt": "I'm going to be in the hospital on Election Day. Can I still vote?",
            "intent": "emergency_voting_options",
            "question_type": "edge_cases",
            "follow_up_hints": ["Is there an emergency absentee ballot?", "Can someone else request a ballot on my behalf?", "What's the deadline for emergency requests?"],
        },
    ],
}


# ── Category metadata for LLM-based generation ───────────────────────────────
CATEGORY_META: dict[str, str] = {
    "procedural": (
        "questions about election logistics — registration deadlines, polling place "
        "locations and hours, election dates, and how to update registration details"
    ),
    "eligibility": (
        "questions about who is allowed to vote — ID requirements, residency rules, "
        "age requirements, felony records, and in-state moves"
    ),
    "mail_in": (
        "questions about mail-in and absentee voting — how to request a ballot, "
        "deadlines, return methods, witness requirements, and switching to in-person"
    ),
    "results_integrity": (
        "questions about how votes are counted and verified — ballot counting processes, "
        "vote tracking, election oversight, audits, and provisional ballots"
    ),
    "edge_cases": (
        "questions about unusual or difficult situations — no ID, forgotten ID, name "
        "not on rolls, recent name changes, and emergency voting options"
    ),
}


PARTICIPANT_SYSTEM_PROMPT = """\
You are simulating a Maryland resident generating realistic questions about the
2026 Gubernatorial Primary Election. You play the role of a curious, sometimes
confused voter who wants clear and accurate information.

Rules:
- Each question must sound natural — like something a real voter would actually ask.
- Vary the style: some direct, some uncertain, some based on a specific situation.
- Include 2-3 follow_up_hints per seed: short natural follow-up questions the voter
  might ask if the first answer is incomplete or raises new questions.
- Do NOT include meta-commentary — output only what a real voter would type.
- Return ONLY a JSON array of objects with keys:
    prompt, intent, question_type, follow_up_hints (array of strings)
"""


class ParticipantGenerator:
    def __init__(self, client: AsyncOpenAI, cfg: RedTeamConfig):
        self.client = client
        self.cfg = cfg

    async def generate(self, category: str) -> list[dict]:
        meta = CATEGORY_META.get(category, category)
        n = self.cfg.seeds_per_category

        user_prompt = (
            f"Generate {n} diverse FAQ seed questions for the category: **{category}**.\n\n"
            f"This category covers: {meta}.\n\n"
            f"Return a JSON array of {n} objects with keys: "
            f"prompt, intent, question_type, follow_up_hints (array of 2-3 strings)."
        )

        try:
            resp = await self.client.chat.completions.create(
                model=self.cfg.seed_model,
                temperature=0.95,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": PARTICIPANT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = resp.choices[0].message.content
            parsed = json.loads(raw)

            if isinstance(parsed, dict):
                seeds = next(iter(parsed.values()))
            else:
                seeds = parsed

            seeds = [s for s in seeds if isinstance(s, dict) and "prompt" in s][:n]
            if seeds:
                return seeds

        except Exception as e:
            logger.warning(f"Participant seed generation failed for [{category}]: {e}. Using fallbacks.")

        return FALLBACK_SEEDS.get(category, [])[:n]
