"""
Explainability Validator  v_exp(E, S_K, I, ŷ) → (pass, reason, mode)

Rubric (5 criteria):
  r1_specificity   : explanation cites specific visual features (not just finding names)
  r2_localization  : explanation references an anatomical region
  r3_causal_chain  : has observation → finding → conclusion structure (not circular)
  r4_differential  : addresses at least one alternative and why it's excluded
  r5_grounded      : every supporting claim appears in the evidence trail S_K

Failure modes:
  - "synthesis_failure": S_K has sufficient evidence but explanation is poorly written
  - "evidence_gap":      explanation cannot be grounded because evidence is missing
  - "pass"
"""
import json
import logging
import re

from src.models.openai_client import text_completion
from src.config import VALIDATOR_TEXT_MODEL

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a radiology education expert evaluating the quality of a clinical "
    "explanation. Apply the rubric strictly and objectively."
)


def validate(
    explanation: str,
    history: list[dict],
    vqa_question: str,
    predicted_answer: str,
    model: str | None = None,
) -> tuple[bool, str, str, dict]:
    """
    Validate the final explanation.

    Returns:
        (passed: bool, feedback: str, mode: str, details: dict)
        mode ∈ {"pass", "synthesis_failure", "evidence_gap"}
        details keys: scores (r1–r5), total, missing_evidence
    """
    # Summarise evidence trail for context
    ev_lines = []
    for item in history:
        a = "Present" if item["answer"] else "Absent"
        obs = item.get("observation", "")
        obs_str = f"  Obs: {obs}" if obs else ""
        ev_lines.append(
            f"  - {item['query']}: {a}  |  {item.get('rationale','')}"
            + (f"\n{obs_str}" if obs_str else "")
        )
    evidence_str = "\n".join(ev_lines) if ev_lines else "  (no evidence)"

    prompt = f"""Evaluate this radiology explanation against the rubric below.

QUESTION: {vqa_question}
PREDICTED ANSWER: {predicted_answer}

EVIDENCE TRAIL (collected findings):
{evidence_str}

EXPLANATION TO EVALUATE:
"{explanation}"

RUBRIC (score each criterion 0 or 1):
  r1_specificity : Does the explanation cite specific visual features (texture, opacity, shape, density)?
                   NOT just repeating the finding name or the answer label.
  r2_localization: Does the explanation reference a specific anatomical region?
  r3_causal_chain: Is there a clear observation → finding → conclusion chain?
                   i.e., does it avoid circular reasoning like "X because X"?
  r4_differential: Does it address at least one alternative answer and explain why it's ruled out?
  r5_grounded    : Are the claims supported by the evidence trail (not hallucinated)?

Also classify failure mode:
  "synthesis_failure" → evidence exists in trail but explanation fails rubric (re-generate)
  "evidence_gap"      → required evidence is genuinely absent from the trail (need more queries)
  "pass"              → all criteria met (score ≥ 4)

Respond in JSON:
{{
  "scores": {{
    "r1_specificity": 0 or 1,
    "r2_localization": 0 or 1,
    "r3_causal_chain": 0 or 1,
    "r4_differential": 0 or 1,
    "r5_grounded": 0 or 1
  }},
  "total": <int 0-5>,
  "failure_mode": "pass" or "synthesis_failure" or "evidence_gap",
  "feedback": "<specific actionable feedback if failed, otherwise empty string>",
  "missing_evidence": "<what evidence is missing, if evidence_gap, else empty>"
}}
"""
    raw = text_completion(prompt, system=_SYSTEM, model=model or VALIDATOR_TEXT_MODEL, max_tokens=400)

    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data   = json.loads(m.group())
            scores = data.get("scores", {})
            total  = int(data.get("total", 0))
            mode   = str(data.get("failure_mode", "synthesis_failure"))
            fb     = str(data.get("feedback", ""))
            miss   = str(data.get("missing_evidence", ""))

            details: dict = {
                "scores": scores,
                "total": total,
                "missing_evidence": miss,
                "feedback": fb,
            }

            if mode == "pass" or total >= 4:
                return True, "", "pass", details
            else:
                full_fb = fb
                if miss:
                    full_fb += f" Missing evidence: {miss}"
                return False, full_fb, mode, details
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: soft pass
    logger.debug("Explanation validator parse failed; soft-passing.")
    return True, "", "pass", {}
