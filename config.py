"""
RQ1 Evaluation Configuration 

All tuneable parameters live here so the pipeline can be adjusted
without touching the evaluation logic.
"""
import os


# The model configuration

# Participant LLM: generates synthetic voter queries (simulates study participants
PARTICIPANT_PROVIDER = "openai"
PARTICIPANT_MODEL = os.getenv("PARTICIPANT_MODEL", "gpt-4o")

# Evaluator LLM: scores response veracity using web search
# GPT-4o Search Preview recommended (Hackenburg et al., 2025)
EVALUATOR_PROVIDER = os.getenv("EVALUATOR_PROVIDER", "openai")
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "gpt-4o-search-preview")


# Evaluation Parameters

# Responses scoring below this threshold are flagged for human review.
# TODO (RQ1 To-Do): Finalize exact threshold (e.g., < 70?)
VERACITY_THRESHOLD = int(os.getenv("VERACITY_THRESHOLD", "70"))

# Minimum word count for a VIOLETS response to be evaluated.
# Short responses (greetings, conversational closings) are excluded per protocol.
MIN_RESPONSE_WORDS = int(os.getenv("MIN_RESPONSE_WORDS", "30"))


# Sampling section


# Dialogues generated per question type
DIALOGUES_PER_TYPE = int(os.getenv("DIALOGUES_PER_TYPE", "10"))

# Max turns per dialogue (1 initial + N-1 follow-ups)
TURNS_PER_DIALOGUE = int(os.getenv("TURNS_PER_DIALOGUE", "3"))

# Sample size for human-LLM correlation validation
# (Hackenburg et al. sampled 100 evaluations to check r = 0.84)
VALIDATOR_SAMPLE_SIZE = int(os.getenv("VALIDATOR_SAMPLE_SIZE", "100"))


# Putting out the output


OUTPUT_DIR = os.getenv("OUTPUT_DIR", "evaluation/results")


# Question type categorization 


QUESTION_CATEGORIZATION: dict[str, dict] = {
    "procedural": {
        "description": "Registration deadlines, polling locations, hours",
        "examples": [
            "What is the deadline to register to vote in Maryland?",
            "Where is my polling place for the 2026 primary?",
            "What are the polling hours on Election Day?",
            "Can I register to vote on Election Day in Maryland?",
        ],
    },
    "eligibility": {
        "description": "ID requirements, eligibility after moving",
        "examples": [
            "What ID do I need to bring to vote in Maryland?",
            "I recently moved to Maryland — am I eligible to vote?",
            "Do I need to re-register if I moved within my county?",
            "Are 17-year-olds allowed to vote in Maryland primaries?",
        ],
    },
    "mail_in_voting": {
        "description": "How to apply, deadlines, drop-off locations",
        "examples": [
            "How do I request a mail-in ballot in Maryland?",
            "What is the deadline to return my mail-in ballot?",
            "Where can I drop off my mail-in ballot?",
            "Can I still vote in person if I already got a mail-in ballot?",
        ],
    },
    "results_integrity": {
        "description": "How counting works, how to verify results",
        "examples": [
            "How are votes counted in Maryland elections?",
            "How can I verify that my mail-in ballot was received and counted?",
            "What happens if there is a recount in Maryland?",
            "How does Maryland ensure the security of its elections?",
        ],
    },
    "edge_case": {
        "description": "No ID, same-day address changes, provisional ballots",
        "examples": [
            "What happens if I forget my ID on Election Day?",
            "I moved this morning — can I still vote at my old polling place?",
            "What is a provisional ballot and when would I need one?",
            "I got a mail-in ballot but I want to vote in person — what do I do?",
        ],
    },
}


# Error-Type Classification Rubric

# TODO (RQ1 To-Do): Finalize this rubric with the full research team

ERROR_TYPES = [
    "date_error",         # Wrong deadline or date
    "location_error",     # Wrong address, county, or polling place
    "eligibility_error",  # Wrong rules about who can vote or what ID is required
    "procedure_error",    # Wrong process or steps described
    "completeness_gap",   # Important information missing from response
    "factual_error",      # General factual inaccuracy not covered above
    "none",               # No errors detected
]
