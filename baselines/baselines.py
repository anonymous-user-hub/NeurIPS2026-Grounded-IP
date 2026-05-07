"""
Baseline methods for Ground-IP comparison.

All baselines share the same interface:
    run(item, image_path) → BaselineResult

Implemented baselines:
  1. DirectPredict   — single-turn direct answer
  2. ChainOfThought  — single-turn CoT
  3. FixedChecklist  — multi-turn, fixed query order (systematic radiology checklist)
  4. RandomChecklist — multi-turn, randomized query order (control)
  5. GroundIPNoVal   — Ground-IP without validators (ablation)
"""
import json
import logging
import re
import random
import time
from dataclasses import dataclass
from pathlib import Path

from src.models.openai_client import vision_completion, text_completion
from src.query_set.refine_queries import load_queries
from src.knowledge_base.retriever import retrieve_as_text
from src.components.answerer import answer_query
from src.components.predictor import predict
from src.components.explanation_generator import generate_explanation

logger = logging.getLogger(__name__)

# ── Fixed radiology checklist (5 key CXR questions) ──────────────────────────
FIXED_CHECKLIST = [
    "cardiomegaly",
    "pleural effusion",
    "focal consolidation or airspace opacity",
    "pneumothorax",
    "pulmonary edema or interstitial markings",
]


@dataclass
class BaselineResult:
    item_id:          str
    baseline_name:    str
    question:         str
    options:          list[str]
    correct_answer:   str
    predicted_answer: str
    confidence:       float
    correct:          bool
    reasoning:        str
    num_queries:      int = 0
    api_calls:        int = 0
    elapsed_sec:      float = 0.0
    error:            str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ── Shared option parsing ─────────────────────────────────────────────────────

def _parse_answer(raw: str, options: list[str]) -> tuple[str, float]:
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            ans  = str(data.get("answer", data.get("predicted_answer", "A"))).strip().upper()
            conf = float(data.get("confidence", 0.5))
            valid = [o[0].upper() for o in options if o]
            if ans not in valid:
                ans = valid[0] if valid else "A"
            return ans, conf
    except Exception:
        pass
    m = re.search(r"\b([A-D])\b", raw)
    ans = m.group(1).upper() if m else "A"
    return ans, 0.5


# ── Baseline 1: Direct Predict ────────────────────────────────────────────────

class DirectPredict:
    """Single-turn: directly ask the VLM to answer the multiple-choice question."""
    name = "direct_predict"

    def run(self, item: dict, image_path: Path) -> BaselineResult:
        t0 = time.time()
        question = item["question"]
        options  = item["options"]
        opts_str = "\n".join(options)
        ind = (item.get("metadata") or {}).get("Indication", "")
        meta = f"Clinical indication: {ind}\n\n" if ind else ""

        prompt = f"""{meta}Look at this chest X-ray and answer the following multiple-choice question.

QUESTION: {question}
OPTIONS:
{opts_str}

Respond in JSON:
{{"answer": "<letter A/B/C/D>", "confidence": <float 0-1>, "reasoning": "<brief>"}}
"""
        try:
            raw = vision_completion(prompt, image_path=image_path, max_tokens=200)
            ans, conf = _parse_answer(raw, options)
            return BaselineResult(
                item_id=item["id"], baseline_name=self.name,
                question=question, options=options,
                correct_answer=item["correct_answer"],
                predicted_answer=ans, confidence=conf,
                correct=(ans == item["correct_answer"].upper()),
                reasoning=raw[:200], api_calls=1,
                elapsed_sec=time.time() - t0,
            )
        except Exception as e:
            return BaselineResult(
                item_id=item["id"], baseline_name=self.name,
                question=question, options=options,
                correct_answer=item["correct_answer"],
                predicted_answer="A", confidence=0.0, correct=False,
                reasoning="", error=str(e), elapsed_sec=time.time() - t0,
            )


# ── Baseline 2: Chain of Thought ──────────────────────────────────────────────

class ChainOfThought:
    """Single-turn: CoT prompt asking to reason step by step before answering."""
    name = "chain_of_thought"

    def run(self, item: dict, image_path: Path) -> BaselineResult:
        t0 = time.time()
        question = item["question"]
        options  = item["options"]
        opts_str = "\n".join(options)
        ind = (item.get("metadata") or {}).get("Indication", "")
        meta = f"Clinical indication: {ind}\n\n" if ind else ""

        prompt = f"""{meta}Examine this chest X-ray carefully and answer the following multiple-choice question.

QUESTION: {question}
OPTIONS:
{opts_str}

Think step by step:
1. Systematically describe what you see: cardiac size and borders, lung fields (upper/middle/lower zones bilaterally), pleural spaces, mediastinum, and any other notable features.
2. For each answer option, identify which observed findings support or contradict it.
3. Select the option best supported by the radiological evidence.

Respond in JSON:
{{
  "step_by_step": "<your systematic reasoning through findings and options>",
  "answer": "<letter A/B/C/D>",
  "confidence": <float 0-1>
}}
"""
        try:
            raw = vision_completion(prompt, image_path=image_path, max_tokens=600)
            ans, conf = _parse_answer(raw, options)
            return BaselineResult(
                item_id=item["id"], baseline_name=self.name,
                question=question, options=options,
                correct_answer=item["correct_answer"],
                predicted_answer=ans, confidence=conf,
                correct=(ans == item["correct_answer"].upper()),
                reasoning=raw[:400], api_calls=1,
                elapsed_sec=time.time() - t0,
            )
        except Exception as e:
            return BaselineResult(
                item_id=item["id"], baseline_name=self.name,
                question=question, options=options,
                correct_answer=item["correct_answer"],
                predicted_answer="A", confidence=0.0, correct=False,
                reasoning="", error=str(e), elapsed_sec=time.time() - t0,
            )


# ── Baseline 3: Fixed Checklist ───────────────────────────────────────────────

class FixedChecklist:
    """
    Multi-turn: iterate through a fixed 5-item radiology checklist,
    collect answers, then classify.
    """
    name = "fixed_checklist"

    def __init__(self, checklist: list[str] = FIXED_CHECKLIST, use_kb: bool = True):
        self.checklist = checklist
        self.use_kb = use_kb

    def run(self, item: dict, image_path: Path) -> BaselineResult:
        t0 = time.time()
        question = item["question"]
        options  = item["options"]
        api_calls = 0
        history = []

        for finding in self.checklist:
            kb_ctx = retrieve_as_text(finding) if self.use_kb else ""
            try:
                ans_dict = answer_query(
                    image_path=image_path, finding=finding,
                    history=history, knowledge_context=kb_ctx,
                    metadata=item.get("metadata"),
                )
                api_calls += 1
                history.append({
                    "query": finding,
                    "answer": ans_dict["answer"],
                    "rationale": ans_dict["rationale"],
                    "region": ans_dict["region"],
                    "confidence": ans_dict["confidence"],
                    "img_grounded": True,
                    "kb_grounded": True,
                })
            except Exception as e:
                logger.warning(f"Fixed checklist answerer failed for '{finding}': {e}")

        clf = {}
        try:
            clf = predict(question, options, history, item.get("metadata"))
            api_calls += 1
            ans, conf = clf["predicted_answer"], clf["confidence"]
        except Exception as e:
            ans, conf = "A", 0.0

        return BaselineResult(
            item_id=item["id"], baseline_name=self.name,
            question=question, options=options,
            correct_answer=item["correct_answer"],
            predicted_answer=ans, confidence=conf,
            correct=(ans == item["correct_answer"].upper()),
            reasoning=clf.get("reasoning", ""),
            num_queries=len(self.checklist), api_calls=api_calls,
            elapsed_sec=time.time() - t0,
        )


# ── Baseline 4: Random Checklist ──────────────────────────────────────────────

class RandomChecklist:
    """
    Multi-turn: same as FixedChecklist but with randomized query order.
    Used as a control to test whether query ordering matters.
    """
    name = "random_checklist"

    def __init__(self, n_queries: int = 5, seed: int | None = None, use_kb: bool = True):
        self.n_queries = n_queries
        self.seed = seed
        self.use_kb = use_kb
        self._all_queries = load_queries(refined=True)

    def run(self, item: dict, image_path: Path) -> BaselineResult:
        t0 = time.time()
        rng = random.Random(self.seed)
        checklist = rng.sample(self._all_queries, min(self.n_queries, len(self._all_queries)))

        # Reuse FixedChecklist logic
        fixed = FixedChecklist(checklist=checklist, use_kb=self.use_kb)
        result = fixed.run(item, image_path)
        result.baseline_name = self.name
        result.elapsed_sec = time.time() - t0
        return result


# ── Baseline 5: Ground-IP without validators ──────────────────────────────────

def run_no_validators(item: dict, image_path: Path, k_max: int = 15, k_min: int = 10,
                      querier_mode: str = "open_set") -> BaselineResult:
    """Ablation: Grounded-IP without v_img or v_kb."""
    from src.pipeline import run as grounded_ip_run
    t0 = time.time()
    result = grounded_ip_run(
        item=item, image_path=image_path,
        k_max=k_max, k_min=k_min, use_validators=False,
        querier_mode=querier_mode,
    )
    return BaselineResult(
        item_id=result.item_id,
        baseline_name="grounded_ip_no_val",
        question=result.question,
        options=result.options,
        correct_answer=result.correct_answer,
        predicted_answer=result.predicted_answer,
        confidence=result.confidence,
        correct=result.correct,
        reasoning=result.explanation,
        num_queries=result.num_queries,
        api_calls=result.api_calls,
        elapsed_sec=time.time() - t0,
    )
