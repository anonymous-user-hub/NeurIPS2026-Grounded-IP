"""
Predictor  f(S_K) → ŷ

Given the evidence history S_K, the original VQA question, options, and
optional metadata / retrieved knowledge, produce a prediction over the answer
options (A/B/C/D). Named "predictor" rather than "classifier" because it also
returns calibrated confidence and free-text reasoning, going beyond simple classification.

Design decision: the predictor does NOT see the image.
  • The image is the source of raw visual signal; the answerer (φ) already
    summarised it into verified binary findings in S_K.
  • Giving the predictor direct image access lets it bypass the evidence chain
    and re-derive an answer on its own — which defeats the purpose of the IP
    framework and empirically hurts accuracy (the predictor ignores carefully
    collected evidence in favour of its own visual assessment).
  • Prediction must be driven entirely by the structured, validated evidence
    trail, making the decision process transparent and auditable.
"""
import json
import logging
import math
import re
from pathlib import Path

from src.models.openai_client import text_completion, text_completion_logprobs

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a radiologist synthesising structured evidence from a systematic "
    "chest X-ray evaluation. Select the best answer based solely on the "
    "verified findings provided — do not speculate beyond the evidence."
)


def predict(
    vqa_question: str,
    vqa_options: list[str],
    history: list[dict],
    metadata: dict | None = None,
    knowledge_context: str = "",
    model: str | None = None,
) -> dict:
    """
    Map evidence history S_K to an answer choice using text reasoning only.

    The predictor sees:
      - the original clinical question and answer options
      - the full evidence trail S_K (query, answer, observation, rationale,
        region, grounding flags for each step)
      - optional patient metadata (e.g. clinical indication)
      - optional retrieved knowledge context

    The predictor does NOT see the raw image — prediction must be grounded
    in the collected evidence, not a fresh visual assessment.

    Args:
        vqa_question:      the original clinical question
        vqa_options:       answer options list
        history:           evidence trail (list of EvidenceItem dicts)
        metadata:          patient metadata (Indication, etc.)
        knowledge_context: optional KB text relevant to the final decision

    Returns:
      { predicted_answer: str (e.g. "A"),
        confidence: float,
        reasoning: str }
    """
    # Format evidence
    evidence_lines = []
    for item in history:
        a = "Yes" if item["answer"] else "No"
        conf = item.get("confidence", "?")
        obs  = item.get("observation", "")
        rat  = item.get("rationale", "")
        reg  = item.get("region", "")
        img_ok = "✓" if item.get("img_grounded") else "✗(unverified)"
        kb_ok  = "✓" if item.get("kb_grounded") else "✗(unverified)"
        obs_line = f"\n    Observation: {obs}" if obs else ""
        evidence_lines.append(
            f"  Finding: {item['query']}\n"
            f"    Answer: {a}  (confidence: {conf}){obs_line}\n"
            f"    Rationale: {rat}\n"
            f"    Region: {reg}\n"
            f"    Image grounded: {img_ok}  |  Knowledge grounded: {kb_ok}"
        )
    evidence_str = "\n".join(evidence_lines) if evidence_lines else "  (no evidence collected)"

    opts_str = "\n".join(vqa_options)
    ind = (metadata or {}).get("Indication", "")
    meta_str = f"Clinical indication: {ind}\n" if ind else ""

    kb_block = ""
    if knowledge_context and not knowledge_context.startswith("(no relevant"):
        kb_block = f"\nRelevant medical knowledge:\n{knowledge_context}\n"

    option_letters = [opt[0].upper() for opt in vqa_options if opt]
    assessment_schema = "\n".join(
        f'    "{L}": "<supports | argues_against | neutral> — <1 sentence why>"'
        for L in option_letters
    )

    prompt = f"""{meta_str}
QUESTION: {vqa_question}

OPTIONS:
{opts_str}

COLLECTED EVIDENCE (from structured query loop — grounded findings marked ✓):
{evidence_str}
{kb_block}
=== STEP 1 — OPTION-BY-OPTION ASSESSMENT ===
For each option, explicitly state whether the collected evidence SUPPORTS it, ARGUES AGAINST it,
or is NEUTRAL (insufficient evidence). Weight grounded findings (✓) more heavily than
unverified ones (✗). Consider:
  • Which findings are characteristic of this option? Are they present (Yes) or absent (No)?
  • Which findings argue against this option because they would be expected but are absent?
  • Do any findings rule this option in or out decisively?

=== STEP 2 — SELECT THE BEST ANSWER ===
Choose the option best supported (or least ruled out) by the evidence.
If evidence is ambiguous, prefer the option consistent with the most grounded positive findings.

Reminder — the question you are answering:
QUESTION: {vqa_question}
OPTIONS:
{opts_str}

Respond in JSON:
{{
  "option_assessment": {{
{assessment_schema}
  }},
  "predicted_answer": "<letter: {'/'.join(option_letters)}>",
  "confidence": <float 0.0–1.0>,
  "reasoning": "<1–2 sentences citing the specific findings that most strongly determined the answer>"
}}
"""
    raw = text_completion(prompt, system=_SYSTEM, max_tokens=512, model=model)
    return _parse_classification(raw, vqa_options)


_PROBA_SYSTEM = (
    "You are a radiologist answering a multiple-choice question from structured evidence. "
    "Output only the single letter of the best answer."
)


def predict_proba(
    vqa_question: str,
    vqa_options: list[str],
    history: list[dict],
    metadata: dict | None = None,
    model: str | None = None,
    predictor_T: float | None = None,
) -> dict[str, float]:
    """
    Return a probability distribution over answer options derived from token log-probs.

    Used by the querier for MI-based candidate scoring. Called twice per candidate
    with a hypothetical Yes or No answer appended to the history, to estimate
    H(Y | A=Yes) and H(Y | A=No).

    Method: present the original clinical question and options unchanged, ask the
    model to output exactly one letter, then read the API's token log-probabilities
    for each answer letter. This gives the model's actual belief distribution, not
    a self-reported number (which LLMs cannot reliably generate).

    Args:
        vqa_question:  the original clinical question (used verbatim — do NOT rephrase)
        vqa_options:   answer options list (e.g. ["A. Pneumonia", "B. CHF", ...])
        history:       evidence trail — may include a hypothetical finding appended by querier
        metadata:      patient metadata
        predictor_T:   calibrated temperature scalar; post-hoc scaling applied to log-probs

    Returns:
        dict mapping option letter → probability (normalised, sums to 1.0).
        Example: {"A": 0.55, "B": 0.25, "C": 0.12, "D": 0.08}
    """
    evidence_lines = []
    for item in history:
        a = "Yes" if item["answer"] else "No"
        rat = item.get("rationale", "")
        evidence_lines.append(
            f"  • {item['query']}  →  {a}"
            + (f"  ({rat[:80]})" if rat else "")
        )
    evidence_str = "\n".join(evidence_lines) if evidence_lines else "  (no evidence yet)"

    valid_letters = [opt[0].upper() for opt in vqa_options if opt]
    opts_str = "\n".join(vqa_options)
    ind = (metadata or {}).get("Indication", "")
    meta_str = f"Clinical indication: {ind}\n" if ind else ""
    letters_str = "/".join(valid_letters)

    # Keep the original question and options verbatim — only add the evidence and
    # instruction to output a single letter.
    prompt = (
        f"{meta_str}"
        f"QUESTION: {vqa_question}\n\n"
        f"OPTIONS:\n{opts_str}\n\n"
        f"EVIDENCE COLLECTED SO FAR:\n{evidence_str}\n\n"
        f"Reminder — QUESTION: {vqa_question}\n"
        f"OPTIONS:\n{opts_str}\n\n"
        f"Based on the evidence above, answer the question.\n"
        f"Reply with ONLY the single letter ({letters_str})."
    )

    if predictor_T is not None:
        # v3: calibrated temperature scaling
        log_probs = text_completion_logprobs(
            prompt=prompt,
            answer_letters=valid_letters,
            system=_PROBA_SYSTEM,
            temperature=1.0,
            model=model,
        )
        return _logprobs_to_proba(log_probs, valid_letters, T=predictor_T)
    else:
        # v1: API at T=1.0; no post-hoc scaling (identity)
        log_probs = text_completion_logprobs(
            prompt=prompt,
            answer_letters=valid_letters,
            system=_PROBA_SYSTEM,
            temperature=1.0,
            model=model,
        )
        return _logprobs_to_proba(log_probs, valid_letters)


def _logprobs_to_proba(log_probs: dict[str, float], valid_letters: list[str], T: float = 1.0) -> dict[str, float]:
    """Convert log-probabilities to a normalised probability distribution via softmax."""
    scaled = {L: log_probs[L] / T for L in valid_letters}
    max_lp = max(scaled.values())
    probs = {L: math.exp(scaled[L] - max_lp) for L in valid_letters}
    total = sum(probs.values())
    return {L: probs[L] / total for L in valid_letters}


def _dist_entropy(log_probs: dict[str, float], valid_letters: list[str], T: float) -> float:
    """Shannon entropy (bits) of softmax(log_probs / T)."""
    probs = _logprobs_to_proba(log_probs, valid_letters, T)
    return -sum(p * math.log2(p) for p in probs.values() if p > 1e-12)


def _entropy_target_T(
    log_probs: dict[str, float],
    valid_letters: list[str],
    target_bits: float = 1.0,
) -> float:
    """Binary-search for T such that H(softmax(log_probs / T)) ≈ target_bits.

    High T flattens the distribution (→ max entropy = log2(N)).
    Low T sharpens it (→ 0 bits).
    If the distribution is already at or above target_bits even at T=1, return 1.0.
    Search range: [1.0, 1000.0]; 30 iterations gives precision < 0.001.
    """
    if _dist_entropy(log_probs, valid_letters, T=1.0) >= target_bits:
        return 1.0

    lo, hi = 1.0, 1000.0
    for _ in range(30):
        mid = (lo + hi) / 2.0
        if _dist_entropy(log_probs, valid_letters, mid) < target_bits:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _parse_classification(raw: str, options: list[str]) -> dict:
    valid_letters = [opt[0].upper() for opt in options if opt]
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            ans = str(data.get("predicted_answer", "A")).strip().upper()
            if ans not in valid_letters:
                ans = valid_letters[0] if valid_letters else "A"
            return {
                "predicted_answer": ans,
                "confidence": float(data.get("confidence", 0.5)),
                "reasoning": str(data.get("reasoning", "")),
                "option_assessment": data.get("option_assessment", {}),
            }
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: find first letter A/B/C/D
    m = re.search(r"\b([A-D])\b", raw)
    ans = m.group(1) if m else (valid_letters[0] if valid_letters else "A")
    return {"predicted_answer": ans, "confidence": 0.5, "reasoning": raw[:150], "option_assessment": {}}
