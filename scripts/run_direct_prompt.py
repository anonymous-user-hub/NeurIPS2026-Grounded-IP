"""
Direct prompting baseline — Differential Diagnosis on ReXVQA.

One vision call per item. Response is always JSON {"answer": "X", "explanation": "..."}.
Logprobs are extracted via exact char-to-token alignment at the answer letter position.

Concurrent requests via ThreadPoolExecutor so vLLM can batch them on GPU.

Usage:
    # Full test-split run (4025 DD items):
    python scripts/run_direct_prompt.py \
        --model Qwen/Qwen3.5-9B \
        --base_url http://localhost:8000/v1

    # Quick smoke-test (10 items, using OpenAI directly — no vLLM needed):
    python scripts/run_direct_prompt.py \
        --model gpt-4o \
        --max_items 10 --batch_size 10

Prompt modes (--mode):
    answer_first  [default]  JSON with "answer" field first, then "explanation".
                             Model commits to the answer before writing the rationale.
    explain_first            JSON with "explanation" field first, then "answer".
                             Model writes full reasoning before committing to the answer.
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
logger = logging.getLogger("direct_prompt")

SAVE_CODE_LIST = [
    "scripts/run_direct_prompt.py",
    "job/run_direct_prompt.job",
    "src/models/openai_client.py",
    "src/data/rexvqa_dataset.py",
    "src/config.py",
]

SYSTEM = "You are a radiology AI assistant specializing in chest X-ray interpretation."

# ── Prompt templates ──────────────────────────────────────────────────────────

PROMPT_ANSWER_FIRST = """\
Look at this chest X-ray and answer the multiple-choice question below.

QUESTION: {question}

OPTIONS:
{options}

Respond in valid JSON with exactly these two keys — answer first, then explanation:
{{
  "answer": "<A, B, C, or D>",
  "explanation": "<your reasoning for this choice>"
}}"""

PROMPT_EXPLAIN_FIRST = """\
Look at this chest X-ray and answer the multiple-choice question below.

QUESTION: {question}

OPTIONS:
{options}

Respond in valid JSON with exactly these two keys — explanation first, then answer:
{{
  "explanation": "<your full reasoning>",
  "answer": "<A, B, C, or D>"
}}"""


def get_prompt(item: dict, mode: str) -> str:
    opts_str = "\n".join(item["options"])
    template = PROMPT_ANSWER_FIRST if mode == "answer_first" else PROMPT_EXPLAIN_FIRST
    return template.format(question=item["question"], options=opts_str)


# ── Response parsing ──────────────────────────────────────────────────────────

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


def _letter_fallback(raw_text: str, options: list[str] | None = None) -> tuple[str, int]:
    """
    Multi-strategy fallback for models that don't output a clear answer letter.
    Returns (pred_letter, char_pos), or ('?', -1) if nothing found.
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
    # Strategy 2: last standalone uppercase letter in the final 2000 chars
    tail = raw_text[-2000:]
    matches = list(re.finditer(r"\b([A-D])\b", tail))
    if matches:
        m = matches[-1]
        return m.group(1).upper(), len(raw_text) - len(tail) + m.start(1)
    # Strategy 3: option-text overlap against response tail
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

_MODE: str = "answer_first"
_MAX_TOKENS: int | None = None
_CAPTURE_THINKING: bool = False


def process_item(item: dict) -> dict:
    from src.models.openai_client import vision_completion_with_logprobs, char_to_token_logprobs
    from src.data.rexvqa_dataset import get_primary_image

    img = get_primary_image(item) or item["image_paths"][0]
    prompt = get_prompt(item, _MODE)
    t0 = time.time()

    try:
        raw_text, token_logprob_list, reasoning = vision_completion_with_logprobs(
            prompt=prompt,
            system=SYSTEM,
            image_path=img,
            max_tokens=_MAX_TOKENS,
            capture_thinking=_CAPTURE_THINKING,
        )

        pred, char_pos, explanation = parse_json_response(raw_text, item.get("options"))

        # Extract logprobs from the exact token at char_pos in raw_text
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
            "thinking": reasoning,
            "logprobs": logprobs,
            "correct": pred == gt,
            "elapsed_sec": round(time.time() - t0, 2),
            "error": None,
        }

    except Exception as e:
        logger.error(f"Item {item['id']} failed: {e}")
        uniform_lp = -1.386   # log(0.25)
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
                   help="HF model ID or OpenAI model name "
                        "(e.g. Qwen/Qwen3.5-9B for vLLM, gpt-4o for OpenAI)")
    p.add_argument("--base_url", default="http://localhost:8000/v1",
                   help="vLLM server URL (ignored for OpenAI models)")
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p.add_argument("--task", default="Differential Diagnosis")
    p.add_argument("--mode", default="answer_first",
                   choices=["answer_first", "explain_first"])
    p.add_argument("--batch_size", type=int, default=128,
                   help="Concurrent requests to vLLM. Match to --max-num-seqs on server. "
                        "B200: 256 for 4B/9B, 128 for 30B.")
    p.add_argument("--max_tokens", type=int, default=None,
                   help="Max output tokens. Default None = let vLLM use (model_len - input_tokens).")
    p.add_argument("--max_items", type=int, default=None,
                   help="Cap on items (for quick tests; omit for full split)")
    p.add_argument("--run_name", default=None,
                   help="Suffix for output folder (default: model_slug + mode)")
    p.add_argument("--with_thinking", action="store_true",
                   help="Enable thinking for Qwen3/3.5 models (default: off for fair comparison with Grounded-IP)")
    return p.parse_args()


def run(args):
    global _MODE, _MAX_TOKENS, _CAPTURE_THINKING
    _MODE = args.mode
    _MAX_TOKENS = args.max_tokens
    _CAPTURE_THINKING = args.with_thinking

    os.environ["VISION_MODEL"] = args.model
    os.environ["TEXT_MODEL"] = args.model
    from src.models.openai_client import configure_local
    configure_local(base_url=args.base_url, api_key=os.getenv("LOCAL_API_KEY", "EMPTY"))

    model_slug = args.model.replace("/", "_").replace(".", "-")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_label = args.run_name or f"{model_slug}_{args.mode}"
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    job_prefix = f"job{slurm_job_id}_" if slurm_job_id else ""
    run_name = f"{timestamp}_{job_prefix}{run_label}"

    out_dir = ROOT / "saved" / "results" / "direct_prompt" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"
    meta_path = out_dir / "meta.json"

    save_code(out_dir)

    meta = vars(args) | {"timestamp": timestamp, "run_name": run_name,
                         "prompt_answer_first": PROMPT_ANSWER_FIRST,
                         "prompt_explain_first": PROMPT_EXPLAIN_FIRST}
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info(f"Model:   {args.model}  @  {args.base_url}")
    logger.info(f"Split:   {args.split}  |  task: {args.task}  |  mode: {args.mode}")
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
        "mode": args.mode,
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
