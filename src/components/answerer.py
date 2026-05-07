"""
Answerer  φ(I, q_k, S_{k-1}, c_k, m) → (a_k, r_k, b_k)

Given the image and a binary finding question:
  - a_k ∈ {True, False}   (Yes / No)
  - r_k  : one-sentence rationale citing visual features
  - b_k  : anatomical region description (e.g. "right lower lobe")
             – we use free-text region rather than pixel bbox
               since the VLM cannot produce pixel coordinates reliably
  - confidence: float in [0, 1] — derived from token logprobs, NOT self-reported.
                P(chosen) / (P(Yes) + P(No)) at the exact answer token position.
                Falls back to self-reported value only if logprob extraction fails.

Supports optional retry with validator feedback.
"""
import json
import logging
import math
import re
from pathlib import Path

from src.models.openai_client import vision_completion_with_logprobs

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are an expert radiologist interpreting chest X-rays. "
    "You give precise, calibrated answers grounded in specific visual evidence. "
    "You never over-claim certainty — subtle or equivocal findings are reported as such."
)

_PROMPT_TEMPLATE = """Image: chest X-ray (see above)
{meta_line}
{kb_block}
FINDING QUESTION: {question}

Previous Q&A context (for reference only — do NOT let prior answers bias your visual assessment):
{history_str}

{feedback_block}
--- INSTRUCTIONS ---
Step 1 — OBSERVE: In 1–2 sentences, describe exactly what you see in the image region relevant \
to this question. Mention density, borders, symmetry, and any reference structures (heart border, \
hemidiaphragm, costophrenic angle). Do this BEFORE deciding Yes or No.

Step 2 — DECIDE: Based solely on your Step 1 description, answer Yes or No.

Step 3 — RATIONALE: Write 2–3 sentences that will help a downstream reader (who cannot see the image) \
reconstruct your assessment. Include:
  (a) the visual appearance of the finding in this image — texture, density, borders, distribution, size;
  (b) the specific features that drove your Yes/No decision and what they argue for or against;
  (c) whether this is a typical or atypical/subtle presentation, and what alternative interpretation \
      you considered and rejected.

Step 4 — CALIBRATE your confidence using this scale:
  0.90–1.00 : Finding is unambiguous — clearly present or clearly absent with no alternative explanation.
  0.70–0.89 : Finding is likely present/absent — characteristic features visible but not textbook-perfect.
  0.50–0.69 : Equivocal — some features suggest it, others argue against; genuinely uncertain.
  0.30–0.49 : Finding probably absent/present but the image is low-quality or the feature is very subtle.
  < 0.30    : Use only if the image is uninterpretable for this specific finding.

Important calibration notes:
- In a typical diagnostic CXR cohort, ~30–40% of targeted binary questions have a positive answer.
  If you find yourself answering No more than 70% of the time overall, you are being systematically
  conservative — check whether a subtle version of the finding could be present.
- Partial or atypical presentations still count as Yes (e.g., mild cardiomegaly is still cardiomegaly).
- Absence of the finding should be supported by a positive description of what IS there instead.
- Never default to 0.95 confidence unless the finding is textbook-obvious or textbook-absent.

Reminder — the question you are answering:
FINDING QUESTION: {question}

Respond in valid JSON with exactly these keys:
{{
  "observation": "<Step 1: 1–2 sentences on what you see in the relevant region>",
  "answer": "Yes" or "No",
  "rationale": "<Step 3: 2–3 sentences — visual appearance, decision-driving features, alternative considered>",
  "region": "<anatomical location, or 'N/A' if not applicable>",
  "confidence": <float 0.0–1.0 per the calibration scale above>
}}
"""


# Instruction-tuned local models (e.g. Qwen3.5) produce extremely peaked token
# distributions: the chosen token's logprob is near 0 while the alternative is
# absent from top_logprobs (defaulting to e^{-100} ≈ 0), saturating conf to 1.000.
# Cap at 0.95 so the value remains discriminative for MI scoring and stopping.
_MAX_LOGPROB_CONF = 0.95


def _logprob_confidence(
    raw_text: str,
    token_logprob_list: list,
    answer_str: str,
    answerer_T: float = 1.0,
) -> tuple[float, float, float] | None:
    """
    Extract calibrated confidence from token logprobs at the Yes/No answer position.

    Finds the token spanning the "Yes"/"No" value in the JSON `"answer"` field,
    reads P(Yes) and P(No) from top_logprobs, and returns:

        (confidence, p_yes, p_no)

    where confidence = P(chosen) / (P(Yes) + P(No)), capped at _MAX_LOGPROB_CONF.

    Returns None if the answer token cannot be located in the logprob list.
    """
    if not token_logprob_list:
        return None

    m = re.search(r'"answer"\s*:\s*"(Yes|No)"', raw_text, re.IGNORECASE)
    if not m:
        return None
    char_pos = m.start(1)

    cumulative = ""
    for tok_obj in token_logprob_list:
        tok_start = len(cumulative)
        cumulative += tok_obj.token
        if tok_start <= char_pos < len(cumulative):
            lp_chosen = tok_obj.logprob

            target = "No" if answer_str.lower() == "yes" else "Yes"
            lp_other = -100.0
            try:
                for entry in tok_obj.top_logprobs:
                    if entry.token.strip().lower() == target.lower():
                        lp_other = entry.logprob
                        break
            except (AttributeError, TypeError):
                pass

            p_chosen = math.exp(lp_chosen / answerer_T)
            p_other  = math.exp(lp_other  / answerer_T)
            total = p_chosen + p_other
            if total > 1e-9:
                conf  = min(p_chosen / total, _MAX_LOGPROB_CONF)
                p_yes = conf        if answer_str.lower() == "yes" else 1.0 - conf
                p_no  = 1.0 - p_yes
                logger.debug(
                    f"Logprob confidence: P({answer_str})={p_chosen:.3f} "
                    f"P({target})={p_other:.3f} → conf={conf:.3f}"
                )
                return conf, p_yes, p_no
            # complement not found in top_logprobs — cap and infer complement
            conf  = min(p_chosen, _MAX_LOGPROB_CONF)
            p_yes = conf        if answer_str.lower() == "yes" else 1.0 - conf
            p_no  = 1.0 - p_yes
            return conf, p_yes, p_no

    return None


def answer_query(
    image_path: Path,
    finding: str,
    history: list[dict],
    knowledge_context: str = "",
    metadata: dict | None = None,
    feedback: str | None = None,
    model: str | None = None,
    answerer_T: float | None = None,
) -> dict:
    """
    Answer a binary finding question about the image.

    Returns dict:
      { answer: bool, observation: str, rationale: str, region: str,
        confidence: float, confidence_source: str, raw: str }

    confidence_source is "logprob" when extracted from token logprobs (preferred),
    or "self_reported" when falling back to the model's own float.
    """
    meta_line = ""
    if metadata:
        ind = metadata.get("Indication", "")
        if ind:
            meta_line = f"Clinical indication: {ind}"

    kb_block = ""
    if knowledge_context:
        kb_block = f"Relevant radiology references:\n{knowledge_context}\n"

    hist_lines = []
    for item in history[-5:]:   # show last 5 steps (important context with k_max=15)
        a = "Yes" if item["answer"] else "No"
        hist_lines.append(f"  - {item['query']}: {a}")
    history_str = "\n".join(hist_lines) if hist_lines else "  (none)"

    feedback_block = ""
    if feedback:
        feedback_block = (
            f"VALIDATOR FEEDBACK (your previous answer was rejected):\n{feedback}\n"
            "Please correct your answer based on this feedback."
        )

    # In open_set mode the query is already a full question; avoid double-wrapping
    if finding.strip().endswith("?"):
        question = finding.strip()
    else:
        question = f"Is there {finding} visible in this chest X-ray?"

    prompt = _PROMPT_TEMPLATE.format(
        meta_line=meta_line,
        kb_block=kb_block,
        question=question,
        history_str=history_str,
        feedback_block=feedback_block,
    )

    raw, token_logprob_list, _ = vision_completion_with_logprobs(
        prompt=prompt,
        image_path=image_path,
        system=_SYSTEM,
        max_tokens=768,
        model=model,
        capture_thinking=False,
    )
    return _parse_answer(raw, token_logprob_list, finding, answerer_T=answerer_T if answerer_T is not None else 1.0)


def _parse_answer(raw: str, token_logprob_list: list, finding: str, answerer_T: float = 1.0) -> dict:
    """Parse the JSON response from the answerer, replacing self-reported confidence
    with logprob-derived confidence wherever possible."""
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            answer_str    = str(data.get("answer", "No")).strip()
            answer_bool   = answer_str.lower() in ("yes", "true", "1")
            self_reported = float(data.get("confidence", 0.5))

            lp_result = _logprob_confidence(raw, token_logprob_list, answer_str, answerer_T=answerer_T)
            if lp_result is not None:
                confidence, lp_p_yes, lp_p_no = lp_result
                confidence_source = "logprob_v3" if answerer_T != 1.0 else "logprob"
            else:
                confidence        = self_reported
                confidence_source = "self_reported"
                lp_p_yes          = None
                lp_p_no           = None
                logger.debug(f"Logprob confidence unavailable for '{finding}'; using self-reported.")

            return {
                "answer":                 answer_bool,
                "observation":            str(data.get("observation", "")),
                "rationale":              str(data.get("rationale", "")),
                "region":                 str(data.get("region", "N/A")),
                "confidence":             confidence,
                "confidence_source":      confidence_source,
                "self_reported_confidence": self_reported,
                "lp_p_yes":               lp_p_yes,
                "lp_p_no":                lp_p_no,
                "raw":                    raw,
            }
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: step-by-step format (e.g. medgemma-1.5-4b outputs "**Step 2:** No" or
    # "**Step 2: Answer**\nYes" instead of JSON).  Find "Step 2" and take the first
    # Yes/No within 150 chars; only then fall back to a wider raw scan.
    step2_m = re.search(r"step\s*2\b(.{0,150})", raw, re.IGNORECASE | re.DOTALL)
    if step2_m:
        yn = re.search(r"\b(yes|no)\b", step2_m.group(1), re.IGNORECASE)
        answer = yn.group(1).lower() == "yes" if yn else ("yes" in raw.lower()[:200])
    else:
        answer = "yes" in raw.lower()[:200]
    logger.debug(f"JSON parse failed for finding '{finding}'; falling back to keyword.")
    return {
        "answer":                 answer,
        "observation":            "",
        "rationale":              raw[:200],
        "region":                 "N/A",
        "confidence":             0.5,
        "confidence_source":      "fallback",
        "self_reported_confidence": None,
        "lp_p_yes":               None,
        "lp_p_no":                None,
        "raw":                    raw,
    }
