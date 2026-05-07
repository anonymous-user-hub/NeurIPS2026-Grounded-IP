"""
Chain-of-Thought prompting baseline — Differential Diagnosis on ReXVQA.

The model reasons step-by-step through radiological findings, interprets each
finding, evaluates all answer choices, then commits to a final answer.
Response is JSON {"explanation": "<4-step reasoning>", "answer": "X"}.
Logprobs are extracted from the exact character position of the answer letter.

Concurrent requests via ThreadPoolExecutor so vLLM can batch them on GPU.

Usage:
    # Full test-split run:
    python scripts/run_cot_prompt.py \
        --model Qwen/Qwen3.5-9B \
        --base_url http://localhost:8000/v1

    # Quick smoke-test (10 items, OpenAI):
    python scripts/run_cot_prompt.py \
        --model gpt-4o \
        --max_items 10 --batch_size 10
"""
import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cot_prompt")

SAVE_CODE_LIST = [
    "scripts/run_cot_prompt.py",
    "job/run_cot_prompt.job",
    "src/models/openai_client.py",
    "src/data/rexvqa_dataset.py",
    "src/config.py",
]

SYSTEM = "You are a board-certified radiologist with expertise in chest X-ray interpretation."

# ── Prompt template ───────────────────────────────────────────────────────────
#
# Design rationale:
#   - Step 1 grounds the model in concrete visual findings before any diagnosis.
#     Listing findings first prevents anchoring on the answer choices.
#   - Step 2 bridges radiology → pathophysiology. Asking for an interpretation of
#     each finding forces explicit evidence linking rather than pattern matching.
#   - Step 3 tests all options rather than committing early. This is the key CoT
#     advantage for multi-class DD: the model must rule out plausible distractors.
#   - Step 4 integrates steps 1–3 into a single best-fit conclusion.
#   - "Answer: <letter>" on the very last line enables exact logprob extraction.

PROMPT_COT = """\
You are interpreting a chest X-ray to answer the differential diagnosis question below.

QUESTION: {question}

OPTIONS:
{options}

Reason through the following four steps, then respond in JSON.

Step 1 — Radiological findings:
Systematically describe the key findings on this chest X-ray. Cover each of the following as relevant: lung parenchyma (opacities, consolidation, interstitial markings, hyperinflation), pleural spaces (effusion, pneumothorax), cardiac silhouette (size, contour), mediastinum and hila (width, lymphadenopathy), vascularity (pulmonary vascular prominence), and bones/soft tissues (if pertinent).

Step 2 — Pathophysiological interpretation:
For each significant finding identified above, state the underlying pathological process it most likely represents and why.

Step 3 — Evaluation of each answer choice:
Go through options A, B, C, and D one by one. For each, state whether the radiological findings support or argue against it, and why.

Step 4 — Conclusion:
Integrate your findings and reasoning. Identify which diagnosis best explains the overall pattern on this chest X-ray, accounting for what is present and what is absent.

Respond in valid JSON with exactly these two keys — all four steps in the explanation, answer last:
{{
  "explanation": "<Step 1 findings. Step 2 interpretation. Step 3 option analysis. Step 4 conclusion.>",
  "answer": "<A, B, C, or D>"
}}"""


def get_prompt(item: dict) -> str:
    opts_str = "\n".join(item["options"])
    return PROMPT_COT.format(question=item["question"], options=opts_str)


# ── Response parsing ──────────────────────────────────────────────────────────

def _letter_fallback(raw_text: str, options: list[str] | None = None) -> tuple[str, int]:
    """
    Multi-strategy fallback for models that don't output 'Answer: X'.
    Returns (pred_letter, char_pos), or ('?', -1) if nothing found.
    Strategies in order:
      1. Natural-language conclusion indicators scanned over full text.
      2. Step 4 / Conclusion section targeted scan.
      3. Last standalone letter in the final 2000 chars.
      4. Option-text matching against the tail of the response.
    """
    # Strategy 1: natural-language conclusion patterns, last match wins
    for pat in [
        r"(?i)(?:the\s+)?(?:correct\s+)?(?:answer|diagnosis|option|choice)\s+is\s*[:\-]?\s*\**([A-D])\b",
        r"(?i)(?:therefore|thus|hence|so)[^A-D]{0,30}([A-D])\b",
        r"(?i)most\s+likely[^A-D]{0,30}([A-D])\b",
        r"(?i)best\s+(?:fits?|explains?|matches?|supported)[^A-D]{0,30}([A-D])\b",
        r"(?i)([A-D])\s+is\s+(?:the\s+)?(?:most\s+likely|correct|best)\b",
        r"(?i)(?:I\s+would\s+)?(?:choose|select|go\s+with)\s+(?:option\s+)?([A-D])\b",
        r"(?i)(?:my\s+)?(?:final\s+)?(?:answer|conclusion)\s*[:\-]\s*\**([A-D])\b",
        r"\*\*\s*([A-D])\s*[\.\*\)]",
        r"\(([A-D])\)",
    ]:
        matches = list(re.finditer(pat, raw_text))
        if matches:
            m = matches[-1]
            return m.group(1).upper(), m.start(1)

    # Strategy 2: scan the Step 4 / Conclusion section specifically
    step4_m = re.search(r"[Ss]tep\s*4[^\n]*\n(.*?)$", raw_text, re.DOTALL)
    if step4_m:
        seg = step4_m.group(1)
        m = re.search(r"\b([A-D])\b", seg)
        if m:
            return m.group(1).upper(), step4_m.start(1) + m.start(1)

    # Strategy 3: last standalone letter in the final 2000 chars (extended from 500)
    tail = raw_text[-2000:]
    matches = list(re.finditer(r"\b([A-D])\b", tail))
    if matches:
        m = matches[-1]
        return m.group(1).upper(), len(raw_text) - len(tail) + m.start(1)

    # Strategy 4: option-text overlap in the tail
    if options:
        tail_lower = raw_text[-800:].lower()
        best_letter, best_score = "?", 0.0
        for opt in options:
            letter = opt[0].upper()
            words = [w for w in re.split(r"\W+", opt[3:].lower()) if len(w) > 3]
            if words:
                score = sum(1 for w in words if w in tail_lower) / len(words)
                if score > best_score:
                    best_score, best_letter = score, letter
        if best_score > 0:
            return best_letter, -1
    return "?", -1


def _forced_extraction(raw_text: str, item: dict) -> str:
    """Last-resort: ask the text model to extract the answer letter from raw_text."""
    from src.models.openai_client import text_completion
    opts_str = "\n".join(item.get("options", []))
    prompt = (
        f"A model was asked this multiple-choice question:\n\n"
        f"QUESTION: {item.get('question', '')}\n\nOPTIONS:\n{opts_str}\n\n"
        f"The model responded:\n\n{raw_text}\n\n"
        "What is the model's final answer? Reply with ONLY a single uppercase letter: A, B, C, or D."
    )
    try:
        result = text_completion(prompt, max_tokens=10)
        m = re.search(r"\b([A-D])\b", result)
        if m:
            return m.group(1).upper()
    except Exception:
        pass
    return "?"


def parse_json_response(raw_text: str, options: list[str] | None = None) -> tuple[str, int, str]:
    """
    Parse a JSON response with 'answer' and 'explanation' fields.
    Returns (pred_letter, char_pos, explanation).
    char_pos is the index of the answer letter in raw_text for logprob alignment.
    Falls back to _letter_fallback if JSON is malformed or missing.
    """
    try:
        m = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if m:
            data = json.loads(m.group())
            answer_val = str(data.get("answer", "")).strip()
            letter_m = re.search(r"[A-D]", answer_val, re.IGNORECASE)
            if letter_m:
                pred = letter_m.group(0).upper()
                explanation = str(data.get("explanation", "")).strip()
                pos_m = re.search(r'"answer"\s*:\s*"([A-D])', raw_text, re.IGNORECASE)
                char_pos = pos_m.start(1) if pos_m else -1
                return pred, char_pos, explanation
    except (json.JSONDecodeError, ValueError, AttributeError):
        pass
    pred, char_pos = _letter_fallback(raw_text, options)
    return pred, char_pos, raw_text.strip()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_items(split: str, task: str, max_items: int | None) -> list[dict]:
    from src.data.rexvqa_dataset import _resolve_image_path
    path = ROOT / f"data/Radiology/ReXVQA/metadata/{split}_vqa_data.json"
    raw = json.loads(path.read_text())
    items = []
    for item_id, it in raw.items():
        if task.lower() not in it.get("task_name", "").lower():
            continue
        paths = [_resolve_image_path(p) for p in it.get("ImagePath", [])]
        paths = [p for p in paths if p.exists()]
        if not paths:
            continue
        it = dict(it)
        it["id"] = item_id
        it["image_paths"] = paths
        it["metadata"] = {k: it.get(k) for k in [
            "Indication", "Comparison", "Findings", "Impression",
            "PatientSex", "PatientAge", "ImageViewPosition",
        ]}
        items.append(it)
        if max_items and len(items) >= max_items:
            break
    return items


# ── Code backup ───────────────────────────────────────────────────────────────

def save_code(out_dir: Path) -> None:
    backup_dir = out_dir / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for rel in SAVE_CODE_LIST:
        src = ROOT / rel
        if src.exists():
            shutil.copy2(src, backup_dir / src.name)
        else:
            logger.warning(f"save_code: {rel} not found, skipping.")


# ── Per-item inference ────────────────────────────────────────────────────────

_MAX_TOKENS: int | None = None
_CAPTURE_THINKING: bool = False


def process_item(item: dict) -> dict:
    from src.models.openai_client import vision_completion_with_logprobs, char_to_token_logprobs
    from src.data.rexvqa_dataset import get_primary_image

    img = get_primary_image(item) or item["image_paths"][0]
    prompt = get_prompt(item)
    t0 = time.time()

    try:
        raw_text, token_logprob_list, thinking = vision_completion_with_logprobs(
            prompt=prompt,
            system=SYSTEM,
            image_path=img,
            max_tokens=_MAX_TOKENS,
            capture_thinking=_CAPTURE_THINKING,
        )

        pred, char_pos, explanation = parse_json_response(raw_text, item.get("options"))

        logprobs = char_to_token_logprobs(
            raw_text=raw_text,
            token_logprob_list=token_logprob_list,
            char_pos=char_pos,
            answer_letters=["A", "B", "C", "D"],
        )

        if pred == "?":
            pred = _forced_extraction(raw_text, item)
            char_pos = -1
            logger.warning(
                f"Regex parsing failed for {item['id']}; forced extraction → {pred!r}"
            )
        elif char_pos < 0:
            logger.warning(f"No char_pos for {item['id']} (pred={pred!r})")

        gt = item.get("correct_answer", "").upper()
        return {
            "id": item["id"],
            "question": item["question"],
            "options": item["options"],
            "gt_answer": gt,
            "pred_answer": pred,
            "explanation": explanation,
            "thinking": thinking,
            "logprobs": logprobs,
            "correct": pred == gt,
            "elapsed_sec": round(time.time() - t0, 2),
            "error": None,
        }

    except Exception as e:
        logger.error(f"Item {item['id']} failed: {e}")
        uniform_lp = -1.386  # log(0.25)
        return {
            "id": item["id"],
            "question": item["question"],
            "options": item["options"],
            "gt_answer": item.get("correct_answer", "").upper(),
            "pred_answer": "?",
            "explanation": "",
            "thinking": "",
            "logprobs": {"A": uniform_lp, "B": uniform_lp,
                         "C": uniform_lp, "D": uniform_lp},
            "correct": False,
            "elapsed_sec": round(time.time() - t0, 2),
            "error": str(e),
        }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-4o",
                   help="HF model ID or OpenAI model name")
    p.add_argument("--base_url", default="http://localhost:8000/v1",
                   help="vLLM server URL (ignored for OpenAI models)")
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p.add_argument("--task", default="Differential Diagnosis")
    p.add_argument("--batch_size", type=int, default=128,
                   help="Concurrent requests to vLLM.")
    p.add_argument("--max_tokens", type=int, default=None,
                   help="Max output tokens. Default None = let vLLM use (model_len - input_tokens).")
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--run_name", default=None,
                   help="Suffix for output folder (default: model_slug_cot)")
    p.add_argument("--resume_dir", default=None,
                   help="Resume into an existing output directory (absolute or relative to saved/results/cot_prompt/).")
    p.add_argument("--with_thinking", action="store_true",
                   help="Enable thinking for Qwen3/3.5 models (default: off for fair comparison with Grounded-IP)")
    return p.parse_args()


def run(args):
    global _MAX_TOKENS, _CAPTURE_THINKING
    _MAX_TOKENS = args.max_tokens
    _CAPTURE_THINKING = args.with_thinking

    os.environ["VISION_MODEL"] = args.model
    os.environ["TEXT_MODEL"] = args.model
    from src.models.openai_client import configure_local
    configure_local(base_url=args.base_url, api_key=os.getenv("LOCAL_API_KEY", "EMPTY"))

    model_slug = args.model.replace("/", "_").replace(".", "-")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    if args.resume_dir:
        p = Path(args.resume_dir)
        out_dir = p if p.is_absolute() else ROOT / "saved" / "results" / "cot_prompt" / p
        run_name = out_dir.name
    else:
        run_label = args.run_name or f"{model_slug}_cot"
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
        job_prefix = f"job{slurm_job_id}_" if slurm_job_id else ""
        run_name = f"{timestamp}_{job_prefix}{run_label}"
        out_dir = ROOT / "saved" / "results" / "cot_prompt" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"
    meta_path = out_dir / "meta.json"

    save_code(out_dir)

    meta = vars(args) | {"timestamp": timestamp, "run_name": run_name,
                         "prompt_cot": PROMPT_COT}
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info(f"Model:   {args.model}  @  {args.base_url}")
    logger.info(f"Split:   {args.split}  |  task: {args.task}  |  mode: cot")
    logger.info(f"Output:  {out_dir}")

    items = load_items(args.split, args.task, args.max_items)
    logger.info(f"Loaded {len(items)} {args.task} items from {args.split} split")
    if not items:
        logger.error("No items found — check data path and task filter.")
        return

    # Resume: skip item IDs already written to results.jsonl
    done_ids: set[str] = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            try:
                done_ids.add(json.loads(line)["id"])
            except Exception:
                pass
        if done_ids:
            logger.info(f"Resuming: {len(done_ids)} items already done, skipping.")
    pending = [it for it in items if it["id"] not in done_ids]
    logger.info(f"Pending: {len(pending)} items  |  batch_size={args.batch_size}")

    n_correct = 0
    n_total = 0
    t_start = time.time()

    # Submit in chunks to bound peak memory: each completed future holds the full
    # response (CoT text + thinking + logprobs) until consumed by as_completed.
    # Submitting all 4025 at once lets GBs accumulate before the first is consumed.
    chunk_size = args.batch_size * 4
    i = 0
    with open(results_path, "a") as fout:
        with ThreadPoolExecutor(max_workers=args.batch_size) as pool:
            for chunk_start in range(0, len(pending), chunk_size):
                chunk = pending[chunk_start:chunk_start + chunk_size]
                futures = {pool.submit(process_item, it): it for it in chunk}
                for fut in as_completed(futures):
                    i += 1
                    r = fut.result()
                    fout.write(json.dumps(r) + "\n")
                    fout.flush()
                    n_total += 1
                    if r["correct"]:
                        n_correct += 1
                    elapsed = time.time() - t_start
                    logger.info(
                        f"[{i:>5}/{len(pending)}]  "
                        f"pred={r['pred_answer']} gt={r['gt_answer']}  "
                        f"{'✓' if r['correct'] else '✗'}  "
                        f"acc={n_correct/n_total:.3f}  "
                        f"({elapsed/n_total:.2f}s/item avg)"
                    )

    # Final summary over all results (includes previously-resumed items)
    all_results = []
    for line in results_path.read_text().splitlines():
        try:
            all_results.append(json.loads(line))
        except Exception:
            pass

    n_ok = sum(r["correct"] for r in all_results)
    n_all = len(all_results)
    summary = {
        "model": args.model,
        "split": args.split,
        "task": args.task,
        "mode": "cot",
        "n_total": n_all,
        "n_correct": n_ok,
        "accuracy": round(n_ok / n_all, 4) if n_all else 0.0,
        "n_errors": sum(1 for r in all_results if r.get("error")),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"\nAccuracy: {n_ok}/{n_all} = {summary['accuracy']:.4f}")
    logger.info(f"Results  → {results_path}")
    logger.info(f"Summary  → {summary_path}")


if __name__ == "__main__":
    run(parse_args())
