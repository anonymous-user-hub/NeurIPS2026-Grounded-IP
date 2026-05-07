"""
LLM-judge prompt templates for VHR, FHR, and EXP evaluation.

VHR (Visual Hallucination Rate, lower is better):
    Score 1–5 where 1 = fully visually grounded, 5 = contradicted by image.
    VHR = fraction scoring >= 4.

FHR (Factual Hallucination Rate, lower is better):
    Score 1–5 where 1 = fully factually supported, 5 = severely incorrect.
    FHR = fraction scoring >= 4.

EXP (Explainability, higher is better):
    Score 1–5 where 1 = excellent clinical reasoning, 5 = uninformative.
    EXP = fraction scoring <= 2.
"""

# ── VHR ───────────────────────────────────────────────────────────────────────

VHR_SYSTEM = "You are an expert medical vision-language evaluator."

VHR_PROMPT = """\
Given a medical image, a medical question, the ground-truth answer, and a model-generated diagnosis, assess whether the generated diagnosis is adequately grounded in the visual evidence from the image.

Use the following scale:
1 = Fully visually grounded: all visual claims are clearly supported by the image and consistent with the ground-truth answer.
2 = Mostly grounded: the main diagnosis is visually supported, with only minor imprecision or missing details.
3 = Partially grounded: the diagnosis is partially supported, but some visual claims are unsupported, insufficiently justified, or inconsistent with the image.
4 = Visually ungrounded: at least one important visual claim is not supported by the image or contradicts the ground-truth answer.
5 = Contradicted by the image: the diagnosis contains major visual claims that directly contradict the image and could mislead clinical interpretation.

Medical question:
{question}

Ground-truth answer:
{ground_truth}

Generated diagnosis:
{diagnosis}

Evaluate visual grounding by comparing the generated diagnosis with both the medical image and the ground-truth answer. Focus only on whether the visual claims in the diagnosis are supported by the image. Do not evaluate general medical factuality unless it is directly tied to visual evidence in the image.

Return your judgment in the following format:

Score: 1/2/3/4/5
Explanation: Briefly justify the score by identifying which visual claims are supported, unsupported, or contradicted by the image."""


# ── FHR ───────────────────────────────────────────────────────────────────────

FHR_SYSTEM = "You are an expert medical factuality evaluator."

FHR_PROMPT = """\
Given a medical question, the ground-truth answer, and a model-generated diagnosis, assess whether the generated diagnosis contains unsupported or factually incorrect medical claims.

Use the following scale:
1 = Fully factually supported: all claims are accurate and consistent with the ground-truth answer.
2 = Mostly supported: the main diagnosis is correct, with only minor imprecision or missing details.
3 = Partially unsupported: the diagnosis is partially correct, but some claims are unsupported, insufficiently justified, or inconsistent with the ground-truth answer.
4 = Factually incorrect: at least one important claim contradicts the ground-truth answer.
5 = Severely incorrect: the diagnosis contains major factual errors that could mislead clinical interpretation.

Question:
{question}

Ground-truth answer:
{ground_truth}

Generated diagnosis:
{diagnosis}

Evaluate factual correctness by comparing the generated diagnosis with the ground-truth answer. Do not evaluate visual grounding; focus only on whether the textual claims in the diagnosis are factually correct and supported by the ground truth.

Return your judgment in the following format:

Score: 1/2/3/4/5
Explanation: Briefly justify the score by identifying which claims are correct, unsupported, or inconsistent with the ground-truth answer."""


# ── EXP ───────────────────────────────────────────────────────────────────────

EXP_SYSTEM = "You are an expert clinical reasoning evaluator."

EXP_PROMPT = """\
Given a medical question, the answer options, the correct answer, and a model-generated explanation, assess whether the explanation provides clear and clinically coherent reasoning that adequately supports the stated diagnosis.

Use the following scale:
1 = Excellent: the reasoning is clear, clinically specific, and explicitly connects observable findings to the diagnosis; it explains why the correct answer is preferred over the alternatives.
2 = Good: the reasoning is mostly valid and clinically relevant, with only minor gaps in depth or completeness.
3 = Adequate: the reasoning is partially valid but vague, generic, or missing key diagnostic steps; it weakly supports the conclusion.
4 = Poor: the reasoning is superficial, tangential to the correct diagnosis, or contains notable logical errors.
5 = Uninformative: the explanation contains no valid clinical reasoning, is self-contradictory, or does not explain the stated answer.

Question:
{question}

Answer options:
{options}

Correct answer:
{ground_truth}

Model-generated explanation:
{explanation}

Evaluate the clinical reasoning quality of the explanation. Focus on whether the reasoning adequately supports the correct answer with sound clinical logic. Do not evaluate factual correctness or visual grounding separately — assess only reasoning quality and coherence.

Return your judgment in the following format:

Score: 1/2/3/4/5
Explanation: Briefly justify the score by identifying the strengths and weaknesses of the reasoning."""
