"""
Explanation generator  ε(S_K, ŷ) → E

Produces a structured clinical explanation that:
  1. Cites specific findings from the verified evidence trail S_K
  2. References anatomical regions reported in the trail
  3. Builds a finding → reasoning → diagnosis chain
  4. Addresses at least one differential

Design decision: ε does NOT receive the raw image.
  • All visual information is already captured in the evidence trail (query, answer,
    rationale, region) produced by the verified answerer calls.
  • Giving ε direct image access would allow it to make fresh visual claims that
    bypass the grounded evidence chain, undermining the interpretability guarantee.
  • If the evidence trail is insufficient for a good explanation, v_exp will detect
    this as an evidence_gap and trigger extra query steps — the correct remedy.

Supports retry with v_exp feedback.
"""
import json
import logging
import re

from src.models.openai_client import text_completion

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a senior radiologist writing a concise, evidence-based explanation "
    "for a clinical decision. "
    "You are operating in EVIDENCE-BOUND mode: every visual or anatomical claim "
    "in your explanation must trace verbatim to a PRESENT finding in the verified "
    "evidence trail. Do NOT introduce any visual observation, anatomical feature, "
    "severity qualifier, or radiological descriptor that is not explicitly stated "
    "in the rationale of a PRESENT finding. Violations constitute hallucination."
)

_PROMPT_TEMPLATE = """You have answered a chest X-ray multiple-choice question using a systematic \
evidence-gathering process. Write a clinical explanation based solely on the verified findings below.

QUESTION: {question}
OPTIONS:
{opts_str}
SELECTED ANSWER: {answer_letter}. {answer_text}

{evidence_str}

{feedback_block}
⛔ STRICT RULES — violations constitute hallucination:
- You may ONLY describe visual findings that appear in the PRESENT FINDINGS list above.
- Do NOT paraphrase, generalise, or infer any visual feature beyond what is explicitly
  written in a PRESENT finding's Rationale. Do not add severity, extent, or location
  details that are not stated there.
- ABSENT findings may only be used to rule out alternative diagnoses — never as positive
  evidence for the selected answer.
- Do NOT refer to the image directly. Do NOT add any observation not in the list above.
- Every sentence that states a finding must end with [N] citing the entry number.

Structure your explanation as follows:
1. Supporting findings: cite each PRESENT finding [N] that supports the selected answer and
   explain the clinical link. Use only the text from that entry's Rationale.
2. Ruling out alternatives: cite ABSENT findings [N] that exclude the most plausible
   alternative option.
3. Conclusion: one sentence naming the answer and the strength of the evidence.

Keep the total explanation under 130 words.

Respond in JSON:
{{
  "explanation": "<structured explanation with [N] citations>",
  "supporting_findings": ["<entry N text>", ...],
  "differential_ruled_out": "<alternative option and which ABSENT finding [N] rules it out>"
}}
"""


def generate_explanation(
    vqa_question: str,
    vqa_options: list[str],
    predicted_answer: str,
    history: list[dict],
    feedback: str | None = None,
    model: str | None = None,
) -> dict:
    """
    Generate a text-only explanation from the evidence trail.

    ε does NOT receive the image — it operates on text evidence only.

    Returns:
      { explanation: str, supporting_findings: list[str],
        differential_ruled_out: str, raw: str }
    """
    # Resolve answer text from predicted letter
    answer_text = ""
    for opt in vqa_options:
        m = re.match(r"^" + re.escape(predicted_answer) + r"[.):\s]\s*(.*)", opt, re.IGNORECASE)
        if m:
            answer_text = m.group(1).strip()
            break

    # Only expose fully grounded steps to the LLM.
    # Steps that failed validation (img_grounded=False or kb_grounded=False) must
    # not appear in the explanation prompt — they were never accepted as evidence.
    grounded = [
        item for item in history
        if item.get("img_grounded") and item.get("kb_grounded")
    ]

    # Separate into PRESENT and ABSENT for unambiguous framing.
    present = [(i + 1, item) for i, item in enumerate(grounded) if item["answer"]]
    absent  = [(i + 1, item) for i, item in enumerate(grounded) if not item["answer"]]

    def _fmt_present(idx: int, item: dict) -> str:
        obs = item.get("observation", "").strip()
        obs_line = f"\n       Observation: {obs}" if obs else ""
        return (
            f"  [{idx}] PRESENT — {item['query']}\n"
            f"       Region: {item.get('region', 'N/A')}{obs_line}\n"
            f"       Rationale: {item.get('rationale', '')}"
        )

    def _fmt_absent(idx: int, item: dict) -> str:
        return (
            f"  [{idx}] ABSENT  — {item['query']}\n"
            f"       Region: {item.get('region', 'N/A')}"
        )

    present_str = "\n".join(_fmt_present(i, it) for i, it in present) or "  (none)"
    absent_str  = "\n".join(_fmt_absent(i, it)  for i, it in absent)  or "  (none)"
    evidence_str = (
        f"PRESENT FINDINGS (image-validated + KB-validated — cite these to support diagnosis):\n"
        f"{present_str}\n\n"
        f"ABSENT FINDINGS (confirmed NOT present — use only to rule out alternatives):\n"
        f"{absent_str}"
    )
    if not grounded:
        evidence_str = "  (no fully grounded evidence collected)"

    opts_str = "\n".join(vqa_options)

    feedback_block = ""
    if feedback:
        feedback_block = (
            f"VALIDATOR FEEDBACK (previous explanation rejected):\n{feedback}\n"
            "Please revise your explanation to address the above issues."
        )

    prompt = _PROMPT_TEMPLATE.format(
        question=vqa_question,
        opts_str=opts_str,
        answer_letter=predicted_answer,
        answer_text=answer_text,
        evidence_str=evidence_str,
        feedback_block=feedback_block,
    )

    raw = text_completion(prompt, system=_SYSTEM, max_tokens=400, model=model)

    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            return {
                "explanation": str(data.get("explanation", "")),
                "supporting_findings": data.get("supporting_findings", []),
                "differential_ruled_out": str(data.get("differential_ruled_out", "")),
                "raw": raw,
            }
    except (json.JSONDecodeError, ValueError):
        pass

    return {
        "explanation": raw[:400],
        "supporting_findings": [],
        "differential_ruled_out": "",
        "raw": raw,
    }
