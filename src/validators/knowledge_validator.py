"""
Knowledge Grounding Validator  v_kb(c_k, q_k, a_k, r_k) → (pass: bool, reason: str)

Checks whether the answerer's rationale is consistent with the retrieved
medical knowledge passages (NLI-style, via LLM judge on TEXT_MODEL).

Response format:
  Large models (GPT, Qwen ≥ 9B): JSON  {"consistent": bool, "verdict": "...", "reason": "...", ...}
  Small models (medgemma, ≤ 4B):  free-text  first line = CONSISTENT / INCONSISTENT / UNVERIFIABLE

Unverifiable verdict → soft-pass: the KB is silent on this finding, which does not mean
the rationale is wrong. Only INCONSISTENT triggers a retry.
"""
import json
import logging
import re

from src.models.openai_client import text_completion
from src.config import VALIDATOR_TEXT_MODEL

logger = logging.getLogger(__name__)

# Models known to produce unreliable JSON; use plain free-text format instead.
# Matched as substrings of the lowercased model name.
_FREE_TEXT_MODELS = {"medgemma", "gemma-4b", "gemma-2b", "gemma-1b", "llava"}


def _prefers_json(model: str) -> bool:
    m = (model or "").lower()
    return not any(s in m for s in _FREE_TEXT_MODELS)


_SYSTEM = (
    "You are a medical knowledge consistency checker. "
    "Your job is to verify that a radiologist's rationale is factually accurate and consistent with the provided references. "
    "You are NOT assessing whether a finding is specific to one diagnosis — cumulative evidence handles discrimination."
)

_PROMPT_BODY = """A radiologist made the following claim about a chest X-ray finding:

FINDING: {finding}
ASSESSMENT: {answer_str}
RATIONALE: "{rationale}"

MEDICAL REFERENCES:
{knowledge_context}

Evaluate CONSISTENCY: Is the rationale factually consistent with how the references describe this finding?
- PRESENT claims: does the rationale describe features that match the reference description?
- ABSENT claims: does the rationale correctly describe absence per the references?
- If the references do not address this finding at all: verdict = UNVERIFIABLE

Note: You are checking factual accuracy only. Do NOT penalize findings for being non-specific — it is
normal and expected that individual radiology findings are consistent with multiple diagnoses."""

_PROMPT_JSON = _PROMPT_BODY + """

Respond in JSON:
{{
  "consistent": true or false,
  "verdict": "Consistent" or "Inconsistent" or "Unverifiable",
  "reason": "<specific reason — cite reference text if inconsistent>",
  "correction": "<what the rationale should say instead, if inconsistent>"
}}"""

_PROMPT_TEXT = _PROMPT_BODY + """

Respond with exactly one word on the first line: CONSISTENT, INCONSISTENT, or UNVERIFIABLE
Second line: brief reason (one sentence)."""


def _parse_json(raw: str) -> tuple[str | None, str, str]:
    """Returns (verdict, reason, correction) or (None, "", "") on failure.
    verdict is one of: "consistent", "inconsistent", "unverifiable", None.
    """
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            verdict    = str(data.get("verdict", "")).strip().lower()
            reason     = str(data.get("reason", ""))
            correction = str(data.get("correction", ""))
            # normalise
            if "inconsistent" in verdict:
                return "inconsistent", reason, correction
            if "unverifiable" in verdict:
                return "unverifiable", reason, ""
            return "consistent", reason, ""
    except (json.JSONDecodeError, ValueError):
        pass
    return None, "", ""


def _parse_text(raw: str) -> tuple[str | None, str, str]:
    """Returns (verdict, reason, correction) or (None, "", "") if no signal."""
    lines = raw.strip().split("\n", 1)
    first = lines[0].strip().upper()
    reason = lines[1].strip() if len(lines) > 1 else ""
    if "INCONSISTENT" in first:
        return "inconsistent", reason, ""
    if "UNVERIFIABLE" in first:
        return "unverifiable", reason, ""
    if "CONSISTENT" in first:
        return "consistent", reason, ""
    return None, reason, ""


def validate(
    finding: str,
    answer: bool,
    rationale: str,
    knowledge_context: str,
    model: str | None = None,
) -> tuple[bool, str, dict]:
    """
    Check knowledge grounding via LLM NLI-style check.

    Returns:
        (passed: bool, feedback: str, details: dict)
        details keys: verdict, reason, correction, trigger
    """
    if not knowledge_context or knowledge_context.startswith("(no relevant"):
        logger.debug("No knowledge context available; soft-passing v_kb.")
        return True, "", {"trigger": "no_kb"}

    _model = model or VALIDATOR_TEXT_MODEL
    use_json = _prefers_json(_model)

    answer_str = "PRESENT" if answer else "ABSENT"
    prompt = (_PROMPT_JSON if use_json else _PROMPT_TEXT).format(
        finding=finding,
        answer_str=answer_str,
        rationale=rationale,
        knowledge_context=knowledge_context,
    )

    raw = text_completion(prompt, system=_SYSTEM, model=_model, max_tokens=350)

    verdict, reason, correction = _parse_json(raw) if use_json else _parse_text(raw)

    if verdict is None:
        # No parseable verdict — soft-pass (model gave no inconsistency signal)
        logger.debug(f"v_kb: no verdict signal in response; soft-passing. raw={raw[:80]!r}")
        return True, "", {"trigger": "no_signal", "reason": reason,
                          "response_format": "json" if use_json else "text"}

    details: dict = {
        "verdict": verdict,
        "reason": reason,
        "correction": correction,
        "response_format": "json" if use_json else "text",
    }

    if verdict == "unverifiable":
        # KB is silent on this finding — not evidence of error; soft-pass.
        logger.debug(f"v_kb: Unverifiable for '{finding}'; soft-passing.")
        details["trigger"] = "unverifiable"
        return True, "", details

    if verdict == "consistent":
        details["trigger"] = "pass"
        return True, "", details

    # inconsistent
    details["trigger"] = "inconsistent"
    return False, (
        f"Knowledge grounding failed: {reason} "
        f"Suggested correction: {correction}"
    ), details
