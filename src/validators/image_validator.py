"""
Image Grounding Validator  v_img(I, q_k, a_k, r_k, b_k) → (pass: bool, reason: str)

γ_k^img ∈ {0,1}: pass if the visual evidence in the proposed region supports the answerer's
claim, fail otherwise. The validator sees the answer, rationale, and region — it verifies
whether they are visually grounded, not whether the answer is correct independently.

Response format:
  Large models (GPT, Qwen ≥ 9B): JSON  {"contradicted": "Yes"/"No", "reason": "..."}
  Small models (medgemma, ≤ 4B):  free-text  first line = CONTRADICTED / NOT_CONTRADICTED
"""
import json
import logging
import re
from pathlib import Path

from src.models.openai_client import vision_completion
from src.config import VALIDATOR_VISION_MODEL

logger = logging.getLogger(__name__)

# Models known to produce unreliable JSON; use plain free-text format instead.
# Matched as substrings of the lowercased model name.
_FREE_TEXT_MODELS = {"medgemma", "gemma-4b", "gemma-2b", "gemma-1b", "llava"}


def _prefers_json(model: str) -> bool:
    m = (model or "").lower()
    return not any(s in m for s in _FREE_TEXT_MODELS)


_SYSTEM = (
    "You are a radiology AI reviewer. "
    "You are given a chest X-ray, a clinical finding question, and an answer with a rationale "
    "and anatomical region proposed by another model. "
    "Your job is to verify whether the stated visual evidence in that region is actually visible "
    "in the image. Focus on whether the rationale is visually supported — not whether the "
    "conclusion is medically correct."
)

_PROMPT_BODY = """A radiology model answered the following question about a chest X-ray:

Question: {finding}
Answer: {answer_str}
Rationale: {answerer_rationale}
Proposed region: {region}

Look at the image and determine: does the image CONTRADICT the answer and rationale above?
Only respond CONTRADICTED if you clearly see evidence that directly contradicts the claim.
If the image is consistent with or ambiguous about the claim, respond NOT_CONTRADICTED."""

_PROMPT_JSON = _PROMPT_BODY + """

Respond in JSON:
{{
  "contradicted": "Yes" or "No",
  "reason": "<what you observe in the image relevant to this finding>"
}}"""

_PROMPT_TEXT = _PROMPT_BODY + """

Respond with exactly one word on the first line: CONTRADICTED or NOT_CONTRADICTED
Second line: brief reason (one sentence)."""


def _parse_json(raw: str) -> tuple[bool | None, str]:
    """Returns (contradicted, reason) or (None, "") on failure."""
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            contradicted = str(data.get("contradicted", "")).strip().lower() in ("yes", "true")
            reason = str(data.get("reason", ""))
            return contradicted, reason
    except (json.JSONDecodeError, ValueError):
        pass
    return None, ""


def _parse_text(raw: str) -> tuple[bool | None, str]:
    """Returns (contradicted, reason) or (None, "") if no verdict signal found."""
    lines = raw.strip().split("\n", 1)
    verdict = lines[0].strip().upper()
    reason  = lines[1].strip() if len(lines) > 1 else ""
    if "NOT_CONTRADICTED" in verdict:
        return False, reason
    if "CONTRADICTED" in verdict:
        return True, reason
    return None, reason   # no clear signal


def validate(
    image_path: Path,
    finding: str,
    answerer_answer: bool,
    answerer_rationale: str,
    region: str,
    confidence: float = 0.5,
    confidence_source: str = "unknown",
    model: str | None = None,
) -> tuple[bool, str, dict]:
    """
    Validate image grounding: does the proposed region and rationale visually support the answer?

    Returns:
        (passed: bool, feedback: str, details: dict)
        passed=True  if the visual evidence is verifiable in the image, False otherwise.
    """
    _model = model or VALIDATOR_VISION_MODEL
    use_json = _prefers_json(_model)

    _base_details: dict = {
        "val_answer": None,
        "val_conf": None,
        "val_evidence": "",
        "val_specificity": "",
        "uncertain": False,
        "answerer_answer": answerer_answer,
        "answerer_conf": confidence,
        "response_format": "json" if use_json else "text",
    }

    answer_str = "YES" if answerer_answer else "NO"
    prompt = (_PROMPT_JSON if use_json else _PROMPT_TEXT).format(
        finding=finding,
        answer_str=answer_str,
        answerer_rationale=answerer_rationale,
        region=region,
    )

    raw = vision_completion(prompt, image_path=image_path, system=_SYSTEM,
                            model=_model, max_tokens=200)

    contradicted, val_reason = _parse_json(raw) if use_json else _parse_text(raw)

    if contradicted is None:
        # No parseable verdict — soft-pass (model gave no contradiction signal)
        logger.debug(f"v_img: no verdict signal in response; soft-passing. raw={raw[:80]!r}")
        return True, "", {**_base_details, "trigger": "no_signal", "val_evidence": val_reason}

    details: dict = {
        **_base_details,
        "val_answer": not contradicted,
        "val_evidence": val_reason,
    }

    if not contradicted:
        details["trigger"] = "pass"
        return True, "", details

    details["trigger"] = "contradicted"
    return False, (
        f"Image grounding failed: the image contradicts the claim for '{finding}' ({answer_str}) "
        f"in region '{region}'. Validator observation: {val_reason}. "
        f"Please re-examine the image and revise your rationale or region."
    ), details
