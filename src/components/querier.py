"""
Querier  g(I, S_{k-1}, m) -> q_k

Two modes:

  open_set  (default): The VLM generates N candidate binary questions from
            scratch (in a single call), each conditioned on the image,
            evidence history, VQA question/options, and retrieved KB context.
            All N candidates are scored by mutual information I(A_q ; Y | S);
            the candidate maximising expected information gain is selected:

              q* = argmax_q  I(A_q ; Y | S_{k-1})
                 = argmax_q  [ H(Y|S) - P(Yes)·H(Y|Yes,S) - P(No)·H(Y|No,S) ]

            Per candidate: 1 vision probe (answerer) + 2 text calls (predictor
            with hypothetical Yes/No answers) to estimate H(Y|Yes) and H(Y|No).

  closed_set: Selects from the pre-built query set Q (407 image-only queries).
            Queries are filtered to those semantically related to the diseases
            and findings mentioned in the answer options, then scored by MI
            (same criterion as open_set).

In both modes, already-asked queries are excluded before scoring.
"""
import json
import logging
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.models.openai_client import vision_completion, text_completion

logger = logging.getLogger(__name__)

_N_CANDIDATES = 5        # number of candidates to generate / score per step
_MAX_PROBE_POOL = 15     # max candidates to score in closed_set mode
_KB_TOP_R_PER_OPTION = 6 # KB chunks retrieved per answer option for the discrimination map (was 3)

_SYSTEM = (
    "You are a senior radiologist conducting a systematic diagnostic evaluation "
    "of a chest X-ray.  Your goal is to gather the most informative evidence "
    "to answer the clinical question."
)


# ── KB-grounded discrimination map ───────────────────────────────────────────

_DISCRIMINATION_MAP_PROMPT = """You are building a differential diagnosis framework for a chest X-ray question.

QUESTION: {vqa_question}
OPTIONS:
{opts_str}

{meta_str}
KNOWLEDGE BASE FACTS PER OPTION:
{option_facts}

Task: For each pair of options, identify the 1-2 CXR findings that BEST discriminate between them.
A good discriminating finding is:
  - Pathophysiologically specific: PRESENT in one condition, ABSENT (or clearly different) in the other
  - Directly visible on chest X-ray
  - Likely to appear POSITIVE in a real case (not an extreme or rare manifestation)

Output format — one line per pair, no other text:
[X] vs [Y]: <finding present in X, absent in Y> (X→YES) | <finding present in Y, absent in X> (Y→YES)

Only include pairs where you can identify a clear, specific discriminating finding.
"""

_DISCRIMINATION_SYSTEM = (
    "You are a radiologist expert in differential diagnosis. "
    "You reason from pathophysiology to specific CXR findings."
)


def build_discrimination_map(
    vqa_question: str,
    vqa_options: list[str],
    metadata: dict | None = None,
) -> str:
    """
    Build a KB-grounded per-pair discrimination map for the given options.

    For each answer option, retrieves KB chunks describing its CXR findings.
    Then uses a text LLM to reason: "what single finding best separates option A
    from option B?" for each pair.

    Returns a formatted string ready to be injected into the querier prompt.
    Called once per pipeline item (before the query loop) and reused across all k steps.
    """
    from src.knowledge_base.retriever import retrieve_as_text

    def _strip_prefix(opt: str) -> str:
        return re.sub(r"^[A-Da-d][.)]\s*", "", opt).strip()

    # Per-option KB retrieval
    option_facts_parts = []
    for opt in vqa_options:
        label = opt[0].upper()
        condition = _strip_prefix(opt)
        kb_text = retrieve_as_text(condition, top_r=_KB_TOP_R_PER_OPTION)
        if kb_text.startswith("(no relevant"):
            kb_text = "(no specific KB facts found)"
        option_facts_parts.append(f"[{label}] {condition}:\n{kb_text}")

    opts_str = "\n".join(vqa_options)
    ind = (metadata or {}).get("Indication", "")
    meta_str = f"Clinical indication: {ind}\n" if ind else ""

    prompt = _DISCRIMINATION_MAP_PROMPT.format(
        vqa_question=vqa_question,
        opts_str=opts_str,
        meta_str=meta_str,
        option_facts="\n\n".join(option_facts_parts),
    )

    try:
        raw = text_completion(prompt, system=_DISCRIMINATION_SYSTEM, max_tokens=700)
        return raw.strip()
    except Exception as e:
        logger.warning(f"Discrimination map generation failed: {e}")
        return ""


# ── Closed-set helpers ────────────────────────────────────────────────────────

def _filter_by_options(queries: list[str], vqa_options: list[str]) -> list[str]:
    """
    Filter the full query set to those related to diseases/findings mentioned
    in the answer options.  Returns ALL matching queries (no arbitrary cap).
    Falls back to all queries if no keyword matches are found.

    Keywords: every word ≥4 chars in the option texts.
    """
    option_words: set[str] = set()
    for opt in vqa_options:
        text = re.sub(r"^[A-Da-d][.)]\s*", "", opt)   # strip "A. " prefix
        option_words.update(w for w in re.findall(r"[a-z]{4,}", text.lower()))

    matched = [q for q in queries if any(w in q.lower() for w in option_words)]
    if not matched:
        logger.debug("No keyword matches found; returning full query set.")
        return queries
    logger.debug(f"Filtered to {len(matched)}/{len(queries)} queries matching options.")
    return matched


# ── Candidate generation (open_set) ──────────────────────────────────────────

_OPEN_SET_CANDIDATES_PROMPT = """You are a radiologist answering this chest X-ray question by gathering targeted evidence.

CLINICAL QUESTION: {vqa_question}
OPTIONS:
{opts_str}

{meta_str}

Evidence gathered so far (do NOT repeat or rephrase these):
{hist_str}

{discrimination_block}{kb_block}{stuck_block}
=== STEP 1 — PAIR ANALYSIS (write this out) ===
Reason through the following and OUTPUT your analysis explicitly:

  a) QUICK GLANCE: Which 1-2 options look most plausible given the image and why?
  b) HARDEST PAIR: Which pair of options is most difficult to separate — what finding do they share?
  c) DISCRIMINATING FINDINGS: For each confusable pair, identify the single CXR finding that is
     PRESENT in one option's pathophysiology and ABSENT in the other's.
     This finding must be something you have reason to believe IS OR COULD BE PRESENT in this image —
     not a finding you already know is absent.

Format your Step 1 output exactly as:
PAIR ANALYSIS:
a) <your reasoning>
b) <your reasoning>
c) <PairX vs PairY>: <discriminating finding> (<PairX>→YES, <PairY>→NO)
   <PairA vs PairB>: <discriminating finding> (<PairA>→YES, <PairB>→NO)
   (one line per confusable pair)

=== STEP 2 — GENERATE {n} BINARY QUESTIONS ===
Using your pair analysis above, generate exactly {n} binary (yes/no) questions.

Each question MUST:
1. Target a DIFFERENT confusable pair — do not ask two questions about the same pair.
2. Ask about a finding EXPECTED TO BE PRESENT in at least one plausible diagnosis.
3. Have asymmetric discrimination: YES strongly favours one specific option, NO favours another.
4. Be specific: name the anatomical structure AND the visual feature (density, border, distribution).
5. Not duplicate or rephrase anything already in "Evidence gathered so far".

Do NOT ask about:
  - Findings shared by all options (e.g., "Is there an opacity?" when all options can cause opacity)
  - Findings you can already see are absent from a quick scan of the image
  - Generic checklist items not tied to discriminating the specific options given

Correct examples (options: Pulmonary edema vs Chronic ILD):
  ✓ "Are there Kerley B lines at the lateral costophrenic angles?"           YES→edema, NO→ILD
  ✓ "Is the interstitial pattern basilar-predominant with apical sparing?"   YES→edema, NO→ILD
  ✗ "Is there an interstitial pattern?"  — shared by both, not discriminating

Correct examples (options: Pneumonia vs Atelectasis):
  ✓ "Are there air bronchograms within the consolidation?"           YES→pneumonia, NO→atelectasis
  ✓ "Is there ipsilateral mediastinal shift toward the opacity?"     YES→atelectasis, NO→pneumonia
  ✗ "Is there an opacity?"  — shared, not discriminating

Correct examples (options: Pneumothorax vs Bullous emphysema):
  ✓ "Is there a sharp visceral pleural line with absent lung markings beyond it?"  YES→pneumothorax
  ✓ "Are the lucent areas bilateral and distributed throughout both lung fields?"   YES→emphysema
  ✗ "Is there increased lucency?"  — shared, not discriminating

CRITICAL — re-read before generating your final list:
The clinical question you must answer is: {vqa_question}
Your {n} questions MUST DISTINGUISH between these specific options: {opts_str}
Each question must target a finding PRESENT in one option and ABSENT in another — not shared by all.
If most evidence so far points toward one option, include at least one question asking about a finding
whose ABSENCE would argue against that leading option.

Reply with your Step 1 analysis followed by a numbered list of {n} questions:
PAIR ANALYSIS:
a) ...
b) ...
c) ...

1. <question>?
2. <question>?
3. <question>?
4. <question>?
5. <question>?
"""


def _exhausted_categories(history: list[dict]) -> str:
    """
    Derive a short plain-English summary of the diagnostic topics already covered,
    to inject into the stuck-warning prompt block.  Uses the key content words from
    each asked query (4+ chars, non-stopword).
    """
    _stop = {"there", "with", "that", "this", "from", "into", "than", "have",
              "does", "area", "both", "each", "such", "over", "also", "been",
              "which", "their", "upper", "lower", "right", "left", "zone",
              "field", "lung", "lobe", "region", "aspect", "visible", "show",
              "sign", "seen", "markings", "either", "within", "absence"}
    topics: list[str] = []
    for item in history:
        words = [w for w in item["query"].lower().split()
                 if len(w) >= 4 and w.rstrip("?.,") not in _stop]
        if words:
            topics.append(", ".join(dict.fromkeys(words[:4])))  # dedup, keep order
    return "; ".join(topics) if topics else "none"


def _extract_reasoning(raw: str) -> str:
    """Extract the PAIR ANALYSIS block from the querier's raw response."""
    m = re.search(r"PAIR ANALYSIS:(.*?)(?=^\s*\d+[\.\):])", raw, re.DOTALL | re.MULTILINE)
    if m:
        return m.group(1).strip()
    # Fallback: everything before the first numbered question
    m = re.search(r"^(.+?)(?=^\s*1[\.\):])", raw, re.DOTALL | re.MULTILINE)
    if m:
        return m.group(1).strip()
    return ""


def _generate_open_set_candidates(
    image_path: Path,
    vqa_question: str,
    vqa_options: list[str],
    history: list[dict],
    knowledge_context: str = "",
    metadata: dict | None = None,
    n: int = _N_CANDIDATES,
    stuck_hint: str = "",
    discrimination_map: str = "",
    model: str | None = None,
) -> tuple[list[str], str]:
    """Generate n diverse candidate binary questions in a single VLM call.

    Returns:
        (candidates, reasoning) where reasoning is the querier's Step 1 pair analysis.
    """
    hist_lines = []
    for item in history:
        a = "Yes" if item["answer"] else "No"
        hist_lines.append(f"  - {item['query']}  ->  {a}")
    hist_str = "\n".join(hist_lines) if hist_lines else "  (none yet)"

    opts_str = "\n".join(vqa_options)
    ind = (metadata or {}).get("Indication", "")
    meta_str = f"Clinical indication: {ind}" if ind else ""

    discrimination_block = ""
    if discrimination_map:
        discrimination_block = (
            f"KB-GROUNDED DISCRIMINATION MAP (pathophysiology -> key CXR findings per option pair):\n"
            f"{discrimination_map}\n\n"
            f"Use the discrimination map above to choose WHICH pairs to target and WHICH findings "
            f"to ask about. Prefer findings listed as discriminating over generic checklist items.\n\n"
        )

    kb_block = ""
    if knowledge_context and not knowledge_context.startswith("(no relevant"):
        kb_block = f"Relevant medical knowledge:\n{knowledge_context}\n"

    stuck_block = ""
    if stuck_hint:
        stuck_block = (
            f"STUCK WARNING: Your previous attempt generated questions too similar to "
            f"already-asked ones and all were rejected.\n"
            f"Exhausted topics so far: {stuck_hint}\n"
            f"You MUST ask about a completely DIFFERENT anatomical region or diagnostic "
            f"category — do not rephrase, split into left/right, or otherwise vary any "
            f"topic already listed above.\n\n"
        )

    prompt = _OPEN_SET_CANDIDATES_PROMPT.format(
        vqa_question=vqa_question,
        opts_str=opts_str,
        meta_str=meta_str,
        hist_str=hist_str,
        discrimination_block=discrimination_block,
        kb_block=kb_block,
        stuck_block=stuck_block,
        n=n,
    )

    raw = vision_completion(prompt, image_path=image_path, system=_SYSTEM, max_tokens=1024, model=model)
    reasoning = _extract_reasoning(raw)
    if reasoning:
        logger.info(f"[QUERIER] Pair analysis:\n{reasoning}")
    candidates = _parse_candidate_list(raw, n)
    return candidates, reasoning


def _clean_question(q: str) -> str:
    q = q.strip().strip('"').strip("'").rstrip(",").strip()
    if not q.endswith("?"):
        q = q.rstrip(".") + "?"
    return q


def _parse_candidate_list(raw: str, n: int) -> list[str]:
    """Parse binary questions from a VLM response in any common format.

    Tries in order:
      1. Numbered list  (1. / 1) / 1: / 1-)
      2. JSON array     (["q1", "q2", ...])
      3. JSON object    ({"questions": [...]})
      4. Bullet points  (- / * / •)
      5. Any line > 10 chars ending with "?"
    """
    candidates: list[str] = []

    def _add(q: str) -> None:
        q = _clean_question(q)
        if len(q) > 10:
            candidates.append(q)

    # 1. Numbered list
    for line in raw.strip().split("\n"):
        m = re.match(r"^\d+[.):\-]\s*(.+)", line.strip())
        if m:
            _add(m.group(1))
    if candidates:
        return candidates[:n]

    # 2 & 3. JSON array or object
    try:
        parsed = json.loads(raw.strip())
        items: list = []
        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    items = v
                    break
        for item in items:
            if isinstance(item, str):
                _add(item)
    except (json.JSONDecodeError, ValueError):
        pass
    if candidates:
        return candidates[:n]

    # 4. Bullet points
    for line in raw.strip().split("\n"):
        m = re.match(r"^[\*\-•]\s*(.+)", line.strip())
        if m:
            _add(m.group(1))
    if candidates:
        return candidates[:n]

    # 5. Any line > 10 chars ending with "?"
    for line in raw.strip().split("\n"):
        line = line.strip().strip('"').strip("'").rstrip(",")
        if len(line) > 10 and line.endswith("?"):
            candidates.append(line)

    return candidates[:n]


# ── Informativeness scoring via mutual information (both modes) ───────────────

def _compute_mi_score(
    image_path: Path,
    query: str,
    vqa_question: str,
    vqa_options: list[str],
    history: list[dict],
    metadata: dict | None,
    answerer_model: str | None = None,
    predictor_model: str | None = None,
    predictor_T: float | None = None,
    answerer_T: float | None = None,
) -> float:
    """
    Estimate I(A_q ; Y | S_{k-1}) for a candidate binary query.

    MI(A_q ; Y | S) = H(Y | S) - [ P(Yes)·H(Y | Yes, S) + P(No)·H(Y | No, S) ]

    Since H(Y | S) is constant across all candidates, we return the negative
    conditional entropy:

        score = -[ P_yes · H(Y|Yes,S) + P_no · H(Y|No,S) ]

    Higher score = more mutual information = better query to ask next.

    Steps:
      1. Answerer probe  → P(Yes | I, q, S)
      2. Classifier call with hypothetical evidence S + {q: Yes} → P(Y | Yes, S)
      3. Classifier call with hypothetical evidence S + {q: No}  → P(Y | No, S)
      4. Compute H(Y | Yes), H(Y | No) and return weighted average (negated).

    The two predictor calls are text-only (no image), so they are cheap.
    The answerer probe requires one vision call.
    """
    # Import here to avoid circular import at module load time
    from src.components.answerer import answer_query
    from src.components.predictor import predict_proba

    # Step 1: probe answerer → P(Yes | I, q, S)
    probe = answer_query(
        image_path=image_path,
        finding=query,
        history=history,
        knowledge_context="",   # no KB for probe
        metadata=metadata,
        model=answerer_model,
        answerer_T=answerer_T,
    )
    # probe["confidence"] is the model's confidence in its OWN answer (Yes or No).
    # Convert to P(Yes): if answer=Yes, P(Yes)=conf; if answer=No, P(Yes)=1-conf.
    probe_conf = probe["confidence"]
    p_yes = probe_conf if probe["answer"] else (1.0 - probe_conf)
    # Clamp to avoid log(0)
    p_yes = max(min(p_yes, 1.0 - 1e-9), 1e-9)
    p_no  = 1.0 - p_yes

    # Hypothetical evidence items for Yes and No.
    # Use the probe's actual rationale for the direction the probe answered;
    # use a generic placeholder for the opposite direction.
    # This avoids the contradiction of using an absence-rationale for a Yes hypothetical
    # while still giving the predictor real, query-specific signal for at least one direction —
    # which is what differentiates MI scores across candidates.
    actual_rationale = probe.get("rationale", "")
    if probe["answer"]:   # probe says Yes
        yes_rationale = actual_rationale or f"Finding confirmed present: {query.rstrip('?')}."
        no_rationale  = f"Finding absent: {query.rstrip('?')}."
    else:                 # probe says No (the common case)
        yes_rationale = f"Finding present: {query.rstrip('?')}."
        no_rationale  = actual_rationale or f"Finding confirmed absent: {query.rstrip('?')}."

    hyp_yes = dict(
        query=query, answer=True,
        rationale=yes_rationale,
        region=probe.get("region", "N/A"),
        confidence=p_yes,
        img_grounded=False, kb_grounded=False,
    )
    hyp_no = dict(
        query=query, answer=False,
        rationale=no_rationale,
        region=probe.get("region", "N/A"),
        confidence=p_no,
        img_grounded=False, kb_grounded=False,
    )

    # Step 2–3: predictor with hypothetical Yes / No — run both in parallel
    with ThreadPoolExecutor(max_workers=2) as _ex:
        _f_yes = _ex.submit(predict_proba, vqa_question, vqa_options, history + [hyp_yes], metadata, predictor_model, predictor_T)
        _f_no  = _ex.submit(predict_proba, vqa_question, vqa_options, history + [hyp_no],  metadata, predictor_model, predictor_T)
        proba_yes = _f_yes.result()
        proba_no  = _f_no.result()

    # Step 4: entropy
    h_yes = -sum(p * math.log2(p) for p in proba_yes.values() if p > 0)
    h_no  = -sum(p * math.log2(p) for p in proba_no.values()  if p > 0)

    conditional_entropy = p_yes * h_yes + p_no * h_no
    mi_score = -conditional_entropy   # higher = more informative

    q_short = repr(query[:60])
    logger.info(
        f"[MI]  q={q_short}  "
        f"P_yes={p_yes:.2f}  H_yes={h_yes:.3f}  H_no={h_no:.3f}  "
        f"H(Y|A)={conditional_entropy:.4f}  MI={-conditional_entropy:.4f}"
    )
    return mi_score, p_yes   # also return p_yes for overconfidence fallback


_MI_DEGENERATE_THRESHOLD = 0.01   # bits; if all MI scores within this range, fall back

# Per-thread counters — each worker thread gets its own mi_count / fallback_count.
# reset_mi_stats() initialises them on the calling thread; get_mi_stats() reads them.
# getattr(..., 0) handles threads that never called reset_mi_stats().
_mi_local = threading.local()


def reset_mi_stats() -> None:
    _mi_local.mi_count       = 0
    _mi_local.fallback_count = 0


def get_mi_stats() -> dict:
    mi  = getattr(_mi_local, "mi_count",       0)
    fb  = getattr(_mi_local, "fallback_count", 0)
    tot = mi + fb
    return {
        "mi_count":       mi,
        "fallback_count": fb,
        "fallback_rate":  fb / tot if tot > 0 else 0.0,
    }


def _select_most_informative(
    candidates: list[str],
    image_path: Path,
    vqa_question: str,
    vqa_options: list[str],
    history: list[dict],
    metadata: dict | None,
    answerer_model: str | None = None,
    predictor_model: str | None = None,
    predictor_T: float | None = None,
    answerer_T: float | None = None,
) -> str:
    """
    Score candidates by mutual information I(A_q ; Y | S_{k-1}).
    Select the query that maximises expected information gain.

    Overconfidence fallback: if the model is so confident on every hypothetical
    that all MI scores are within _MI_DEGENERATE_THRESHOLD bits of each other
    (near-zero conditional entropy regardless of query), the MI criterion is
    uninformative.  In that case, fall back to selecting by |P_yes − 0.5|
    (maximum answerer uncertainty), which is cheaper and still better than random.

    Falls back to the first candidate if all MI computations fail.
    """
    mi_scores: dict[str, float] = {}
    p_yes_scores: dict[str, float] = {}

    # Score all candidates in parallel — each _compute_mi_score is independent.
    # Inner parallelism (Yes/No predict_proba) already handled inside _compute_mi_score.
    # Max workers = N candidates (typically 5); I/O-bound so threads are appropriate.
    with ThreadPoolExecutor(max_workers=len(candidates)) as ex:
        future_to_q = {
            ex.submit(
                _compute_mi_score,
                image_path, q, vqa_question, vqa_options, history, metadata,
                answerer_model, predictor_model,
                predictor_T, answerer_T,
            ): q
            for q in candidates
        }
        for future in as_completed(future_to_q):
            q = future_to_q[future]
            try:
                mi, p_yes = future.result()
                mi_scores[q]    = mi
                p_yes_scores[q] = p_yes
            except Exception as e:
                logger.warning(f"MI scoring failed for candidate {q!r}: {e}")

    if not mi_scores:
        logger.warning("All MI scores failed; returning first candidate.")
        return candidates[0]

    mi_vals = list(mi_scores.values())
    mi_range = max(mi_vals) - min(mi_vals)

    if mi_range < _MI_DEGENERATE_THRESHOLD:
        # Degenerate: predictor is overconfident on every hypothetical.
        # Fall back to answerer uncertainty: select candidate with P_yes closest to 0.5.
        _mi_local.fallback_count = getattr(_mi_local, "fallback_count", 0) + 1
        fb = _mi_local.fallback_count
        mi = getattr(_mi_local, "mi_count", 0)
        logger.info(
            f"[MI-FALLBACK] MI scores degenerate (range={mi_range:.4f}); "
            f"using |P_yes-0.5| fallback  (fallback_total={fb}, "
            f"mi_total={mi}, rate={fb/(mi+fb):.0%})"
        )
        best_query = min(p_yes_scores, key=lambda q: abs(p_yes_scores[q] - 0.5))
        logger.debug(f"Fallback selected: {best_query!r}  (|P_yes-0.5|={abs(p_yes_scores[best_query]-0.5):.3f})")
    else:
        _mi_local.mi_count = getattr(_mi_local, "mi_count", 0) + 1
        best_query = max(mi_scores, key=mi_scores.__getitem__)
        logger.debug(f"[MI-SELECT] q={best_query!r}  MI≈{mi_scores[best_query]:.3f}  range={mi_range:.3f}")

    return best_query


# ── Public interface ──────────────────────────────────────────────────────────

def select_next_query(
    image_path: Path,
    vqa_question: str,
    vqa_options: list[str],
    candidate_queries: list[str],
    history: list[dict],
    metadata: dict | None = None,
    mode: str = "open_set",
    knowledge_context: str = "",
    discrimination_map: str = "",
    querier_model: str | None = None,
    answerer_model: str | None = None,
    predictor_model: str | None = None,
    predictor_T: float | None = None,
    answerer_T: float | None = None,
) -> str:
    """
    Select or generate the next binary query via test-time informativeness scoring.

    Both modes generate / retrieve a pool of candidates, then score each by
    mutual information I(A_q ; Y | S_{k-1}).  The candidate maximising expected
    information gain is selected.

    Args:
        image_path:         chest X-ray
        vqa_question:       the clinical question to answer
        vqa_options:        answer options (A, B, C, D)
        candidate_queries:  full query set Q (used in closed_set mode only)
        history:            evidence items collected so far
        metadata:           patient metadata
        mode:               "open_set"   — VLM generates N candidates, scores all
                            "closed_set" — filters Q by option keywords, scores top N
        knowledge_context:  pre-retrieved KB text (used in open_set mode)
        discrimination_map: KB-grounded pair-wise finding table from build_discrimination_map();
                            injected into the querier prompt to scaffold query generation.

    Returns:
        (selected_query, candidates, querier_reasoning)
        - selected_query:    the chosen binary question
        - candidates:        all generated/filtered candidates before MI scoring
        - querier_reasoning: Step 1 pair analysis text (open_set only; "" for closed_set)
    """
    asked = {item["query"] for item in history}

    def _is_duplicate(q: str) -> bool:
        """True if q is an exact match or semantically near-duplicate of any asked query."""
        if q in asked:
            return True
        q_words = {w for w in q.lower().split() if len(w) >= 4}
        for h in asked:
            h_words = {w for w in h.lower().split() if len(w) >= 4}
            union = q_words | h_words
            if union and len(q_words & h_words) / len(union) >= 0.5:
                return True
        return False

    querier_reasoning = ""
    all_candidates: list[str] = []

    if mode == "open_set":
        # Generate N candidates in one call, deduplicate, then score
        candidates, querier_reasoning = _generate_open_set_candidates(
            image_path=image_path,
            vqa_question=vqa_question,
            vqa_options=vqa_options,
            history=history,
            knowledge_context=knowledge_context,
            metadata=metadata,
            n=_N_CANDIDATES,
            discrimination_map=discrimination_map,
            model=querier_model,
        )
        all_candidates = list(candidates)
        # Remove exact-match and semantic near-duplicates (Jaccard ≥ 0.5 on 4-char words)
        candidates = [q for q in candidates if not _is_duplicate(q)]
        if not candidates:
            # Retry with an explicit stuck warning so the model knows to change topic
            logger.warning("All generated candidates duplicate history; regenerating.")
            stuck_hint = _exhausted_categories(history)
            candidates, querier_reasoning = _generate_open_set_candidates(
                image_path=image_path,
                vqa_question=vqa_question,
                vqa_options=vqa_options,
                history=history,
                knowledge_context=knowledge_context,
                metadata=metadata,
                n=_N_CANDIDATES,
                stuck_hint=stuck_hint,
                discrimination_map=discrimination_map,
                model=querier_model,
            )
            all_candidates = list(candidates)
            candidates = [q for q in candidates if not _is_duplicate(q)]
        if not candidates:
            # Hard fallback: single-query generation
            logger.warning("Candidate generation failed; falling back to single-query mode.")
            fallback_q = _generate_single_query(
                image_path=image_path,
                vqa_question=vqa_question,
                vqa_options=vqa_options,
                history=history,
                knowledge_context=knowledge_context,
                metadata=metadata,
                model=querier_model,
            )
            if not _is_duplicate(fallback_q):
                return fallback_q, [fallback_q], querier_reasoning
            raise RuntimeError(
                "Querier stuck: all generated queries are near-duplicates of history. "
                "Stopping query loop early."
            )

    else:  # closed_set
        # Filter query pool by option keywords, remove asked, score a capped pool
        filtered = _filter_by_options(candidate_queries, vqa_options)
        remaining = [q for q in filtered if not _is_duplicate(q)]
        if not remaining:
            logger.warning("All candidate queries exhausted.")
            remaining = filtered[:1] if filtered else candidate_queries[:1]
        candidates = remaining[:_MAX_PROBE_POOL]
        all_candidates = list(candidates)

    selected = _select_most_informative(
        candidates=candidates,
        image_path=image_path,
        vqa_question=vqa_question,
        vqa_options=vqa_options,
        history=history,
        metadata=metadata,
        answerer_model=answerer_model,
        predictor_model=predictor_model,
        predictor_T=predictor_T,
        answerer_T=answerer_T,
    )
    return selected, all_candidates, querier_reasoning


# ── Legacy fallback (single-query generation) ─────────────────────────────────

_SINGLE_QUERY_PROMPT = """You are a radiologist answering this chest X-ray question through targeted evidence gathering.

CLINICAL QUESTION: {vqa_question}
OPTIONS:
{opts_str}

{meta_str}

Evidence gathered so far:
{hist_str}

{kb_block}

Identify the pair of options that is HARDEST to separate given the current evidence.
Generate ONE binary (yes/no) question that:
1. Targets a finding EXPECTED TO BE PRESENT in one of those two options but not the other.
2. Has NOT been asked yet (check evidence gathered so far).
3. Is answerable by direct visual inspection of the X-ray.
4. Is specific — name the anatomical structure and the visual feature.

Reply with ONLY the binary question itself (a single sentence ending with "?").
"""


def _generate_single_query(
    image_path: Path,
    vqa_question: str,
    vqa_options: list[str],
    history: list[dict],
    knowledge_context: str = "",
    metadata: dict | None = None,
    model: str | None = None,
) -> str:
    """Fallback: generate a single binary question (used when candidate generation fails)."""
    hist_lines = []
    for item in history:
        a = "Yes" if item["answer"] else "No"
        hist_lines.append(f"  • {item['query']}  →  {a}")
    hist_str = "\n".join(hist_lines) if hist_lines else "  (none yet)"

    opts_str = "\n".join(vqa_options)
    ind = (metadata or {}).get("Indication", "")
    meta_str = f"Clinical indication: {ind}" if ind else ""

    kb_block = ""
    if knowledge_context and not knowledge_context.startswith("(no relevant"):
        kb_block = f"Relevant medical knowledge:\n{knowledge_context}\n"

    prompt = _SINGLE_QUERY_PROMPT.format(
        vqa_question=vqa_question,
        opts_str=opts_str,
        meta_str=meta_str,
        hist_str=hist_str,
        kb_block=kb_block,
    )

    raw = vision_completion(prompt, image_path=image_path, system=_SYSTEM, max_tokens=256, model=model)

    # Try JSON extraction first (model may output {"question": "..."} or ["..."])
    q = raw.strip()
    try:
        parsed = json.loads(q)
        if isinstance(parsed, dict):
            q = str(parsed.get("question", parsed.get("query", ""))).strip()
        elif isinstance(parsed, list) and parsed:
            q = str(parsed[0]).strip()
    except (json.JSONDecodeError, ValueError):
        # Not JSON — try to extract the first line ending with "?"
        for line in q.split("\n"):
            line = line.strip().strip('"').strip("'")
            if len(line) > 10 and line.endswith("?"):
                q = line
                break

    q = _clean_question(q)
    if len(q) < 15:
        raise RuntimeError(f"Single-query fallback returned unusable output: {q!r}")
    return q
