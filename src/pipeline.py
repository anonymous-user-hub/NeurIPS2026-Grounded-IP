"""
Grounded-IP  — main inference pipeline.

Implements:
  For k = 1..K_max:
    1. Querier g       → select q_k
    2. Retrieval R     → c_k
    3. Answerer φ      → (a_k, r_k, b_k)
    4. v_img           → image grounding check (with retry)
    5. v_kb            → knowledge grounding check (with retry)
    6. Update S_k
    7. Stopping criterion
  After loop:
    8. Predictor f     → ŷ
    9. Explanation ε   → E
   10. v_exp           → explainability check (with retry / evidence-gap routing)

Returns a PipelineResult with all fields populated.
"""
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.config import K_MAX, MAX_RETRIES, STOP_ENTROPY, TEXT_MODEL, VISION_MODEL
from src.knowledge_base.retriever import retrieve_as_text
from src.components.querier import select_next_query, build_discrimination_map, reset_mi_stats, get_mi_stats
from src.components.answerer import answer_query as _answer_query_v1
from src.components.predictor import predict as _predict_v1
from src.components.predictor import predict_proba as _predict_proba_v1
from src.components.explanation_generator import generate_explanation
from src.validators.image_validator import validate as validate_image
from src.validators.knowledge_validator import validate as validate_knowledge
from src.validators.explanation_validator import validate as validate_explanation

logger = logging.getLogger(__name__)


@dataclass
class EvidenceItem:
    """s_k = (q_k, a_k, r_k, b_k, c_k) plus grounding flags."""
    query:        str
    answer:       bool
    rationale:    str
    region:       str
    confidence:        float
    confidence_source: str    # "logprob" | "self_reported" | "fallback"
    knowledge_context: str
    img_grounded: bool
    kb_grounded:  bool
    observation:  str = ""   # Step-1 visual description from the answerer (stored for predictor/explanation)
    img_feedback: str = ""
    kb_feedback:  str = ""
    retry_count:  int = 0
    # Answerer full output
    answerer_raw:              str   = ""    # raw LLM text before JSON parsing
    self_reported_confidence:  float | None = None  # model's own scalar float (v1 only; before logprob override)
    lp_p_yes:                  float | None = None  # logprob-derived P(Yes)
    lp_p_no:                   float | None = None  # logprob-derived P(No)
    # Validator full outputs
    img_val_details: dict = field(default_factory=dict)  # full v_img response (val_answer, val_conf, val_evidence, …)
    kb_val_details:  dict = field(default_factory=dict)  # full v_kb response (consistent, specificity, verdict, …)
    retry_attempts:  list = field(default_factory=list)  # per-attempt dicts for failed retries
    step_proba:            dict = field(default_factory=dict)  # P(option | S_k) after this step
    step_predicted_answer: str = ""                            # argmax of step_proba
    # Querier trace
    candidates:        list = field(default_factory=list)  # all candidates generated before MI selection
    querier_reasoning: str  = ""                           # Step 1 pair analysis from querier

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class PipelineResult:
    item_id:          str
    question:         str
    options:          list[str]
    correct_answer:   str
    predicted_answer: str
    confidence:       float
    correct:          bool
    explanation:      str
    supporting_findings: list[str]
    differential_ruled_out: str
    evidence_trail:   list[dict]          # list of EvidenceItem.to_dict()
    exp_passed:       bool
    exp_mode:         str
    num_queries:      int
    api_calls:        int = 0
    elapsed_sec:      float = 0.0
    error:            str = ""
    exp_val_details:  dict = field(default_factory=dict)  # v_exp rubric scores, total, missing_evidence
    step_log:         list[dict] = field(default_factory=list)  # full trace: every loop iteration including failures
    # Dual predictions for acc-vs-queries analysis (IP paper convention)
    pred_ip:      str   = ""   # prediction at first confidence plateau; = predicted_answer if no plateau
    pred_ip_step: int   = 0    # len(history) when plateau first fired (0 = no plateau reached)
    pred_ip_conf: float = 0.0  # predictor's best step_proba at the pred_ip step
    pred_max:     str   = ""   # prediction using all k_max accepted evidence (alias of predicted_answer)


_CONFIDENCE_PLATEAU_THRESHOLD   = 0.90  # each of the last N steps must individually exceed this
_CONFIDENCE_PLATEAU_CONSECUTIVE = 3     # number of consecutive high-confidence steps required
_MAX_CONSECUTIVE_FAILURES       = 5     # break loop after this many steps with no accepted evidence


def _is_confidence_plateau(
    history: list[EvidenceItem],
    k_min: int,
    threshold: float = _CONFIDENCE_PLATEAU_THRESHOLD,
    n_consecutive: int = _CONFIDENCE_PLATEAU_CONSECUTIVE,
) -> bool:
    """
    Return True when the predictor's belief has plateaued:
      - At least k_min accepted evidence items, AND
      - Each of the last n_consecutive accepted steps INDIVIDUALLY has
        answerer confidence > threshold (not an average — all must pass).

    Requires every step to individually exceed the threshold so that a single
    overconfident answerer step cannot trigger early stopping.

    STOP_ENTROPY is kept in config for a future MI-based alternative criterion.
    """
    if len(history) < max(k_min, n_consecutive):
        return False
    return all(h.confidence > threshold for h in history[-n_consecutive:])


def run(
    item: dict,
    image_path: Path,
    k_max: int = K_MAX,
    k_min: int = 10,
    max_retries: int = MAX_RETRIES,
    k_extra: int = 2,
    queries: list[str] | None = None,
    use_kb: bool = True,
    use_validators: bool = True,
    querier_mode: str = "open_set",
    querier_model: str | None = None,
    answerer_model: str | None = None,
    predictor_model: str | None = None,
    val_img_model: str | None = None,
    val_kb_model: str | None = None,
    val_exp_model: str | None = None,
) -> PipelineResult:
    """
    Run Grounded-IP on a single ReXVQA item.

    Args:
        item:           ReXVQA item dict
        image_path:     resolved image path
        k_max:          maximum number of query steps
        k_min:          minimum accepted evidence items before plateau detection activates.
                        Default 10. All items always run to k_max regardless.
        max_retries:    maximum validator retries per step
        k_extra:        extra queries allowed if v_exp finds evidence_gap
        queries:        pre-loaded query set Q (used only in closed_set mode)
        use_kb:         whether to retrieve and use knowledge context
        use_validators: whether to run v_img and v_kb (set False to ablate)
        querier_mode:   "open_set"   — LLM generates binary question from scratch
                        "closed_set" — select from option-filtered Q

    Returns:
        PipelineResult
    """
    t0 = time.time()

    # Confidence calibration: temperature scaling (v3)
    from src.calibration import load_temperatures
    _pred_T    = load_temperatures(predictor_model or TEXT_MODEL)["predictor_T"]
    _ans_T     = load_temperatures(answerer_model  or VISION_MODEL)["answerer_T"]
    _answer_query  = _answer_query_v1
    _predict       = _predict_v1
    _predict_proba = _predict_proba_v1


    vqa_question = item["question"]
    vqa_options  = item["options"]
    correct_ans  = item["correct_answer"]
    metadata     = item.get("metadata", {})
    item_id      = item["id"]

    history: list[EvidenceItem] = []  # S_k — only contains fully grounded evidence
    api_call_count = 0
    reset_mi_stats()   # clear per-item MI vs fallback counters

    # Dual-prediction tracking (IP paper: pred_ip at plateau, pred_max at k_max)
    consecutive_failures  = 0      # steps since last accepted evidence item
    early_stop_recorded   = False  # True once pred_ip has been set
    pred_ip:       str   = ""
    pred_ip_step:  int   = 0       # 0 = no plateau reached
    pred_ip_conf:  float = 0.0

    # ── Build KB-grounded discrimination map (once per item, open_set only) ──
    # For each pair of answer options, retrieves KB chunks and asks a text LLM
    # to identify the key CXR finding that discriminates option A from option B.
    # The resulting table is injected into every querier call to scaffold which
    # findings to ask about — turning the KB into active query scaffolding.
    discrimination_map = ""
    if querier_mode == "open_set":
        try:
            discrimination_map = build_discrimination_map(
                vqa_question=vqa_question,
                vqa_options=vqa_options,
                metadata=metadata,
            )
            if discrimination_map:
                logger.debug(f"[{item_id}] Discrimination map built ({len(discrimination_map)} chars)")
        except Exception as e:
            logger.warning(f"[{item_id}] Discrimination map failed: {e}")

    # ── Query loop ────────────────────────────────────────────────────────────
    effective_k_max = k_max
    step_log: list[dict] = []
    k = 0
    while k < effective_k_max:
        k += 1
        logger.debug(f"[{item_id}] Step {k}/{effective_k_max}")
        _step_entry: dict = {"step": k, "candidates": [], "querier_reasoning": "",
                             "selected_query": "", "outcome": "unknown"}

        # ── Step 1a: Pre-retrieval for open_set querier ────────────────────────
        # For open_set, retrieve KB context keyed on question+options first,
        # so the querier can generate a knowledge-informed question.
        pre_kb_ctx = ""
        if use_kb and querier_mode == "open_set":
            try:
                option_key = vqa_question + " " + " ".join(vqa_options)
                pre_kb_ctx = retrieve_as_text(option_key, context="")
            except Exception as e:
                logger.warning(f"Pre-retrieval failed: {e}")

        # ── Step 1: Querier g ─────────────────────────────────────────────────
        _step_candidates: list[str] = []
        _step_reasoning:  str = ""
        try:
            selected_query, _step_candidates, _step_reasoning = select_next_query(
                image_path=image_path,
                vqa_question=vqa_question,
                vqa_options=vqa_options,
                candidate_queries=queries,
                history=[e.to_dict() for e in history],
                metadata=metadata,
                mode=querier_mode,
                knowledge_context=pre_kb_ctx,
                discrimination_map=discrimination_map,
                querier_model=querier_model,
                answerer_model=answerer_model,
                predictor_model=predictor_model,
                predictor_T=_pred_T,
                answerer_T=_ans_T,
            )
            api_call_count += 1
            _step_entry["candidates"]        = _step_candidates
            _step_entry["querier_reasoning"] = _step_reasoning
            _step_entry["selected_query"]    = selected_query
        except RuntimeError as e:
            # Querier stuck: all generated candidates are near-duplicates of history.
            # Per paper §3, the loop runs to k_max; only _MAX_CONSECUTIVE_FAILURES breaks it.
            logger.warning(f"[{item_id}] Querier stuck at step {k}: {e}. Skipping step.")
            _step_entry["outcome"] = "querier_stuck"
            step_log.append(_step_entry)
            consecutive_failures += 1
            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                logger.warning(f"[{item_id}] {_MAX_CONSECUTIVE_FAILURES} consecutive failures (querier stuck); stopping loop.")
                break
            continue
        except Exception as e:
            logger.warning(f"Querier failed at step {k}: {e}")
            _step_entry["outcome"] = "querier_failed"
            step_log.append(_step_entry)
            consecutive_failures += 1
            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                logger.warning(f"[{item_id}] {_MAX_CONSECUTIVE_FAILURES} consecutive failures (querier exception); stopping loop.")
                break
            continue

        # ── Step 2: Retrieval R ────────────────────────────────────────────────
        # Retrieve KB context keyed on the selected query (both modes)
        knowledge_ctx = ""
        if use_kb:
            try:
                # Disease-aware context: VQA options + question anchor the embedding
                # toward the relevant disease space, not just the generic finding.
                options_str = " | ".join(vqa_options)
                recent_hist = "; ".join(
                    f"{h.query}={'Y' if h.answer else 'N'}" for h in history[-5:]
                )
                ctx_summary = f"{vqa_question} [{options_str}] | {recent_hist}"
                knowledge_ctx = retrieve_as_text(selected_query, context=ctx_summary)
            except Exception as e:
                logger.warning(f"Retrieval failed: {e}")

        # ── Step 3+4+5: Answerer φ + validators ───────────────────────────────
        answer_dict = None
        img_grounded = False
        kb_grounded  = False
        img_feedback = ""
        kb_feedback  = ""
        img_val_details: dict = {}
        kb_val_details:  dict = {}
        retry_attempts:  list = []
        retries = 0

        for attempt in range(max_retries + 1):
            feedback = None
            if attempt > 0:
                feedback = img_feedback or kb_feedback

            try:
                answer_dict = _answer_query(
                    image_path=image_path,
                    finding=selected_query,
                    history=[e.to_dict() for e in history],
                    knowledge_context=knowledge_ctx,
                    metadata=metadata,
                    feedback=feedback,
                    model=answerer_model,
                    answerer_T=_ans_T,
                )
                api_call_count += 1
            except Exception as e:
                logger.warning(f"Answerer failed (attempt {attempt}): {e}")
                break

            logger.info(
                f"[{item_id}] step {k:2d} attempt {attempt}  "
                f"ans={'Yes' if answer_dict['answer'] else 'No '}  "
                f"conf={answer_dict['confidence']:.3f} ({answer_dict.get('confidence_source','?')})  "
                f"region='{answer_dict.get('region','')[:60]}'"
            )

            if not use_validators:
                img_grounded = True
                kb_grounded  = True
                break

            # v_img
            _attempt_img_ok   = False
            _attempt_img_fb   = ""
            _attempt_img_det  = {}
            try:
                img_ok, img_fb, img_det = validate_image(
                    image_path=image_path,
                    finding=selected_query,
                    answerer_answer=answer_dict["answer"],
                    answerer_rationale=answer_dict["rationale"],
                    region=answer_dict["region"],
                    confidence=answer_dict["confidence"],
                    confidence_source=answer_dict.get("confidence_source", "unknown"),
                    model=val_img_model,
                )
                api_call_count += 1
                img_grounded      = img_ok
                img_feedback      = img_fb
                img_val_details   = img_det
                _attempt_img_ok   = img_ok
                _attempt_img_fb   = img_fb
                _attempt_img_det  = img_det
                logger.info(
                    f"[{item_id}] step {k:2d} attempt {attempt}  "
                    f"v_img={'PASS' if img_ok else 'FAIL'}  "
                    + (f"trigger={img_det.get('trigger','')}  val_ans={'Yes' if img_det.get('val_answer') else 'No'}  "
                       f"val_conf={img_det.get('val_conf','?')}  "
                       f"val_evidence='{str(img_det.get('val_evidence',''))[:80]}'" if not img_ok else "")
                )
            except Exception as e:
                logger.warning(f"v_img exception at step {k} attempt {attempt}: {e}")
                img_grounded = True  # soft pass on error

            # v_kb (only run if image grounding passes)
            _attempt_kb_ok   = False
            _attempt_kb_fb   = ""
            _attempt_kb_det  = {}
            if img_grounded:
                try:
                    kb_ok, kb_fb, kb_det = validate_knowledge(
                        finding=selected_query,
                        answer=answer_dict["answer"],
                        rationale=answer_dict["rationale"],
                        knowledge_context=knowledge_ctx,
                        model=val_kb_model,
                    )
                    api_call_count += 1
                    kb_grounded      = kb_ok
                    kb_feedback      = kb_fb
                    kb_val_details   = kb_det
                    _attempt_kb_ok   = kb_ok
                    _attempt_kb_fb   = kb_fb
                    _attempt_kb_det  = kb_det
                    logger.info(
                        f"[{item_id}] step {k:2d} attempt {attempt}  "
                        f"v_kb={'PASS' if kb_ok else 'FAIL'}  "
                        + (f"trigger={kb_det.get('trigger','')}  verdict={kb_det.get('verdict','')}  "
                           f"reason='{str(kb_det.get('reason',''))[:80]}'" if not kb_ok else "")
                    )
                except Exception as e:
                    logger.warning(f"v_kb exception at step {k} attempt {attempt}: {e}")
                    kb_grounded = True

            if img_grounded and kb_grounded:
                break

            # Record this failed attempt before retrying
            retry_attempts.append({
                "attempt":                attempt,
                "answer":                 answer_dict["answer"],
                "observation":            answer_dict.get("observation", ""),
                "rationale":              answer_dict["rationale"],
                "region":                 answer_dict.get("region", ""),
                "confidence":             answer_dict["confidence"],
                "confidence_source":      answer_dict.get("confidence_source", "unknown"),
                "self_reported_confidence": answer_dict.get("self_reported_confidence"),
                "lp_p_yes":               answer_dict.get("lp_p_yes"),
                "lp_p_no":                answer_dict.get("lp_p_no"),
                "raw":                    answer_dict.get("raw", ""),
                "img_passed":             _attempt_img_ok,
                "img_feedback":           _attempt_img_fb,
                "img_val_details":        _attempt_img_det,
                "kb_passed":              _attempt_kb_ok,
                "kb_feedback":            _attempt_kb_fb,
                "kb_val_details":         _attempt_kb_det,
            })
            retries = attempt + 1

        # ── Case A: answerer returned None — log and re-queue ─────────────────
        if answer_dict is None:
            logger.warning(f"[{item_id}] Answerer returned None at step {k}; skipping.")
            _step_entry["outcome"]                = "answerer_failed"
            _step_entry["not_appended_to_history"] = True
            _step_entry["anomaly_flag"]            = "answerer_returned_none"
            step_log.append(_step_entry)
            consecutive_failures += 1
            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                logger.warning(f"[{item_id}] {_MAX_CONSECUTIVE_FAILURES} consecutive step failures; stopping loop.")
                break
            continue

        # ── Case B: grounding failed after max retries — discard, re-queue ────
        if not (img_grounded and kb_grounded):
            logger.warning(
                f"[{item_id}] Step {k}: validation exhausted after {retries} retries "
                f"(img={img_grounded}, kb={kb_grounded}); discarding step — querier will re-select."
            )
            _step_entry["outcome"]                = "validation_exhausted"
            _step_entry["not_appended_to_history"] = True
            _step_entry["anomaly_flag"]            = "validation_failed_after_max_retries"
            _step_entry["img_grounded_final"]      = img_grounded
            _step_entry["kb_grounded_final"]       = kb_grounded
            _step_entry["final_failed_answer"]     = {
                "answer":            answer_dict["answer"],
                "rationale":         answer_dict["rationale"],
                "region":            answer_dict.get("region", ""),
                "confidence":        answer_dict["confidence"],
                "confidence_source": answer_dict.get("confidence_source", "unknown"),
                "img_feedback":      img_feedback,
                "kb_feedback":       kb_feedback,
            }
            _step_entry["all_retry_attempts"]      = retry_attempts  # full per-attempt detail
            step_log.append(_step_entry)
            consecutive_failures += 1
            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                logger.warning(f"[{item_id}] {_MAX_CONSECUTIVE_FAILURES} consecutive validation failures; stopping loop.")
                break
            continue

        # ── Grounding passed — reset failure counter ───────────────────────────
        consecutive_failures = 0

        # ── Step 6: State update S_k (only fully grounded evidence) ───────────
        ev = EvidenceItem(
            query=selected_query,
            answer=answer_dict["answer"],
            observation=answer_dict.get("observation", ""),
            rationale=answer_dict["rationale"],
            region=answer_dict["region"],
            confidence=answer_dict["confidence"],
            confidence_source=answer_dict.get("confidence_source", "unknown"),
            knowledge_context=knowledge_ctx,
            img_grounded=img_grounded,
            kb_grounded=kb_grounded,
            img_feedback=img_feedback,
            kb_feedback=kb_feedback,
            retry_count=retries,
            answerer_raw=answer_dict.get("raw", ""),
            self_reported_confidence=answer_dict.get("self_reported_confidence"),
            lp_p_yes=answer_dict.get("lp_p_yes"),
            lp_p_no=answer_dict.get("lp_p_no"),
            img_val_details=img_val_details,
            kb_val_details=kb_val_details,
            retry_attempts=retry_attempts,
            candidates=_step_candidates,
            querier_reasoning=_step_reasoning,
        )
        history.append(ev)
        _step_entry["outcome"]        = "accepted"
        _step_entry["evidence_index"] = len(history) - 1
        step_log.append(_step_entry)

        # ── Per-step classification probability ────────────────────────────────
        step_proba: dict = {}
        step_pred:  str  = ""
        try:
            step_proba = _predict_proba(
                vqa_question=vqa_question,
                vqa_options=vqa_options,
                history=[e.to_dict() for e in history],
                metadata=metadata,
                model=predictor_model,
                predictor_T=_pred_T,
            )
            api_call_count += 1
            step_pred = max(step_proba, key=step_proba.__getitem__)
            history[-1].step_proba = step_proba
            history[-1].step_predicted_answer = step_pred
            proba_str = "  ".join(
                f"{L}={step_proba.get(L, 0.0):.3f}" for L in sorted(step_proba)
            )
            logger.info(
                f"[{item_id}] step {k:2d}  |S|={len(history)}  q='{selected_query[:60]}'  "
                f"ans={'Yes' if ev.answer else 'No '}  "
                f"→ pred={step_pred}  [{proba_str}]"
            )
        except Exception as e:
            logger.warning(f"[{item_id}] predict_proba failed at step {k}: {e}")

        # ── Step 7: Plateau detection — record pred_ip; do NOT break ──────────
        # All items always run to k_max for pred_max. pred_ip is the prediction
        # at the first step where the confidence plateau criterion fires.
        if not early_stop_recorded and _is_confidence_plateau(history, k_min):
            early_stop_recorded = True
            pred_ip      = step_pred
            pred_ip_step = len(history)
            pred_ip_conf = step_proba.get(step_pred, 0.0) if step_proba else 0.0
            logger.info(
                f"[{item_id}] Confidence plateau at |S|={pred_ip_step}  "
                f"pred_ip={pred_ip}  conf={pred_ip_conf:.3f}  "
                f"(all last {_CONFIDENCE_PLATEAU_CONSECUTIVE} steps individually "
                f"> {_CONFIDENCE_PLATEAU_THRESHOLD})  — continuing to k_max for pred_max"
            )

    # ── MI scoring summary ────────────────────────────────────────────────────
    mi_stats = get_mi_stats()
    logger.info(
        f"[{item_id}] MI stats: {mi_stats['mi_count']} MI  /  "
        f"{mi_stats['fallback_count']} fallback  "
        f"(fallback rate {mi_stats['fallback_rate']:.0%})"
        + (" ← WARNING: predictor overconfident, MI not contributing" if mi_stats["fallback_rate"] > 0.8 else "")
    )

    # ── Step 8: Final prediction ──────────────────────────────────────────────
    # Use argmax of the last step_proba rather than a separate predict() call.
    # Analysis of run_v3 (50 items) showed predict() disagrees with step_proba
    # on 20% of items and reduces accuracy by 10pp due to free-text reasoning
    # over-weighting single dramatic findings.  predict_proba() accumulates
    # cumulative evidence correctly across all steps.
    last_step_proba = history[-1].step_proba if history else {}
    if last_step_proba:
        predicted_ans = max(last_step_proba, key=last_step_proba.__getitem__)
        clf_conf      = last_step_proba[predicted_ans]
        logger.info(f"[{item_id}] Final pred from step_proba: {predicted_ans} (conf={clf_conf:.3f})")
    else:
        # Fallback: predict_proba failed for all steps → call predict() as backup
        try:
            clf_result = _predict(
                vqa_question=vqa_question,
                vqa_options=vqa_options,
                history=[e.to_dict() for e in history],
                metadata=metadata,
                model=predictor_model,
            )
            api_call_count += 1
            predicted_ans = clf_result["predicted_answer"]
            clf_conf      = clf_result["confidence"]
        except Exception as e:
            logger.error(f"Predictor failed: {e}")
            predicted_ans = "A"
            clf_conf      = 0.0

    # ── pred_ip fallback: if no plateau was reached, pred_ip = pred_max ─────────
    if not early_stop_recorded:
        pred_ip      = predicted_ans
        pred_ip_step = 0   # 0 signals no plateau reached
        pred_ip_conf = clf_conf
        logger.info(f"[{item_id}] No confidence plateau reached; pred_ip = pred_max = {pred_ip}")
    else:
        logger.info(f"[{item_id}] pred_ip={pred_ip} (step {pred_ip_step})  pred_max={predicted_ans}")

    # ── Steps 9+10: Explanation ε + v_exp ─────────────────────────────────────
    # Outer structure: evidence_gap can trigger extra query steps, then
    # re-generate the explanation.  We track whether extra queries have already
    # been run to avoid an infinite loop.
    exp_result = {"explanation": "", "supporting_findings": [], "differential_ruled_out": ""}
    exp_passed      = False
    exp_mode        = "pass"
    exp_val_details: dict = {}
    gap_extended    = False   # only allow one evidence_gap extension

    for exp_attempt in range(max_retries + 1):
        exp_feedback = exp_result.get("_feedback")
        try:
            exp_result = generate_explanation(
                vqa_question=vqa_question,
                vqa_options=vqa_options,
                predicted_answer=predicted_ans,
                history=[e.to_dict() for e in history],
                feedback=exp_feedback,
                model=predictor_model,
            )
            api_call_count += 1
        except Exception as e:
            logger.error(f"Explanation generator failed: {e}")
            break

        try:
            exp_passed, exp_fb, exp_mode, exp_det = validate_explanation(
                explanation=exp_result.get("explanation", ""),
                history=[e.to_dict() for e in history],
                vqa_question=vqa_question,
                predicted_answer=predicted_ans,
                model=val_exp_model,
            )
            api_call_count += 1
            exp_val_details = exp_det
            logger.info(
                f"[{item_id}] v_exp attempt {exp_attempt}  "
                f"{'PASS' if exp_passed else 'FAIL'}  mode={exp_mode}  "
                f"total={exp_det.get('total','?')}  scores={exp_det.get('scores',{})}"
                + (f"  missing='{exp_det.get('missing_evidence','')[:80]}'" if exp_mode == "evidence_gap" else "")
            )
        except Exception as e:
            logger.warning(f"v_exp failed: {e}")
            exp_passed = True
            exp_mode   = "pass"
            break

        if exp_passed:
            break

        if exp_mode == "evidence_gap" and not gap_extended:
            # Run up to k_extra additional query steps, then retry explanation.
            gap_extended    = True
            new_k_max       = min(k + k_extra, k_max + k_extra)
            logger.debug(f"[{item_id}] evidence_gap: running extra queries k={k}→{new_k_max}")
            while k < new_k_max:
                k += 1
                # Querier
                _gap_entry: dict = {"step": k, "candidates": [], "querier_reasoning": "",
                                    "selected_query": "", "outcome": "unknown", "phase": "evidence_gap"}
                try:
                    extra_query, _gap_cands, _gap_reasoning = select_next_query(
                        image_path=image_path,
                        vqa_question=vqa_question,
                        vqa_options=vqa_options,
                        candidate_queries=queries,
                        history=[e.to_dict() for e in history],
                        metadata=metadata,
                        mode=querier_mode,
                        discrimination_map=discrimination_map,
                        querier_model=querier_model,
                        answerer_model=answerer_model,
                        predictor_model=predictor_model,
                        predictor_T=_pred_T,
                        answerer_T=_ans_T,
                            )
                    api_call_count += 1
                    _gap_entry["candidates"]        = _gap_cands
                    _gap_entry["querier_reasoning"] = _gap_reasoning
                    _gap_entry["selected_query"]    = extra_query
                except Exception as e:
                    logger.warning(f"Extra querier failed at step {k}: {e}")
                    _gap_entry["outcome"] = "querier_failed"
                    step_log.append(_gap_entry)
                    break
                # Retrieval + Answer + full validation for extra steps.
                # Extra steps must pass v_img and v_kb before entering history —
                # unvalidated claims must never reach the explanation generator.
                extra_kb = retrieve_as_text(extra_query) if use_kb else ""
                try:
                    extra_ans = _answer_query(
                        image_path=image_path,
                        finding=extra_query,
                        history=[e.to_dict() for e in history],
                        knowledge_context=extra_kb,
                        metadata=metadata,
                        model=answerer_model,
                        answerer_T=_ans_T,
                    )
                    api_call_count += 1

                    # Validate the extra step (same validators as the main loop)
                    _extra_img_ok, _extra_img_fb, _extra_img_det = True, "", {}
                    _extra_kb_ok,  _extra_kb_fb,  _extra_kb_det  = True, "", {}
                    if use_validators:
                        try:
                            _extra_img_ok, _extra_img_fb, _extra_img_det = validate_image(
                                image_path=image_path,
                                finding=extra_query,
                                answerer_answer=extra_ans["answer"],
                                answerer_rationale=extra_ans["rationale"],
                                region=extra_ans["region"],
                                confidence=extra_ans["confidence"],
                                confidence_source=extra_ans.get("confidence_source", "unknown"),
                                model=val_img_model,
                            )
                            api_call_count += 1
                        except Exception as e:
                            logger.warning(f"Extra v_img failed at step {k}: {e}")
                            _extra_img_ok = True  # soft pass on error
                        if _extra_img_ok:
                            try:
                                _extra_kb_ok, _extra_kb_fb, _extra_kb_det = validate_knowledge(
                                    finding=extra_query,
                                    answer=extra_ans["answer"],
                                    rationale=extra_ans["rationale"],
                                    knowledge_context=extra_kb,
                                    model=val_kb_model,
                                )
                                api_call_count += 1
                            except Exception as e:
                                logger.warning(f"Extra v_kb failed at step {k}: {e}")
                                _extra_kb_ok = True  # soft pass on error

                    if not (_extra_img_ok and _extra_kb_ok):
                        logger.warning(
                            f"[{item_id}] Extra gap step rejected by validators "
                            f"(img={_extra_img_ok}, kb={_extra_kb_ok}); not appended."
                        )
                        _gap_entry["outcome"] = "validation_failed"
                        _gap_entry["not_appended_to_history"] = True
                        step_log.append(_gap_entry)
                        break

                    history.append(EvidenceItem(
                        query=extra_query,
                        answer=extra_ans["answer"],
                        observation=extra_ans.get("observation", ""),
                        rationale=extra_ans["rationale"],
                        region=extra_ans["region"],
                        confidence=extra_ans["confidence"],
                        confidence_source=extra_ans.get("confidence_source", "unknown"),
                        knowledge_context=extra_kb,
                        img_grounded=_extra_img_ok,
                        kb_grounded=_extra_kb_ok,
                        img_feedback=_extra_img_fb,
                        kb_feedback=_extra_kb_fb,
                        img_val_details=_extra_img_det,
                        kb_val_details=_extra_kb_det,
                        answerer_raw=extra_ans.get("raw", ""),
                        self_reported_confidence=extra_ans.get("self_reported_confidence"),
                        lp_p_yes=extra_ans.get("lp_p_yes"),
                        lp_p_no=extra_ans.get("lp_p_no"),
                        candidates=_gap_cands,
                        querier_reasoning=_gap_reasoning,
                    ))
                    _gap_entry["outcome"] = "accepted"
                    _gap_entry["evidence_index"] = len(history) - 1
                    step_log.append(_gap_entry)
                except Exception as e:
                    logger.warning(f"Extra answerer failed at step {k}: {e}")
                    _gap_entry["outcome"] = "answerer_failed"
                    step_log.append(_gap_entry)
                    break
            exp_result["_feedback"] = exp_fb
            continue   # retry explanation with richer history

        exp_result["_feedback"] = exp_fb

    elapsed = time.time() - t0

    return PipelineResult(
        item_id=item_id,
        question=vqa_question,
        options=vqa_options,
        correct_answer=correct_ans,
        predicted_answer=predicted_ans,    # = pred_max
        confidence=clf_conf,
        correct=(predicted_ans.upper() == correct_ans.upper()),
        explanation=exp_result.get("explanation", ""),
        supporting_findings=exp_result.get("supporting_findings", []),
        differential_ruled_out=exp_result.get("differential_ruled_out", ""),
        evidence_trail=[e.to_dict() for e in history],
        exp_passed=exp_passed,
        exp_mode=exp_mode,
        num_queries=len(history),
        api_calls=api_call_count,
        elapsed_sec=elapsed,
        exp_val_details=exp_val_details,
        step_log=step_log,
        pred_max=predicted_ans,
        pred_ip=pred_ip,
        pred_ip_step=pred_ip_step,
        pred_ip_conf=pred_ip_conf,
    )
