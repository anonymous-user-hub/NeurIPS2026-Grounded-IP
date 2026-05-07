"""
eval_exp.py — Explainability (EXP) evaluator.

For each result item with a non-empty explanation, a text LLM judge scores (1–5)
the clinical reasoning quality of the explanation:
  1 = Excellent reasoning  ...  5 = Uninformative / no valid reasoning.

EXP = fraction of scored items with score <= 2 (good or excellent reasoning).
Higher is better.

Usage:
    python evals/eval_exp.py \\
        --results saved/results/direct_prompt/<run>/results.jsonl \\
        [--model gpt-4o-mini] [--batch_size 32]

Output (same directory as --results):
    exp_scores.jsonl    — per-item scores
    exp_summary.json    — aggregate EXP
"""
import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("eval_exp")


def parse_score(text: str) -> int | None:
    m = re.search(r"Score:\s*([1-5])", text)
    return int(m.group(1)) if m else None


def gt_full_text(options: list[str], gt_letter: str) -> str:
    return next((o for o in options if o.upper().startswith(gt_letter.upper() + ".")), gt_letter)


def evaluate_one(item: dict, model: str) -> dict:
    from src.models.openai_client import text_completion
    from evals.prompts import EXP_SYSTEM, EXP_PROMPT

    item_id = item["id"]
    explanation = (item.get("explanation") or "").strip()

    if not explanation:
        return {"id": item_id, "score": None, "judge_text": "",
                "skipped": True, "skip_reason": "empty_explanation"}

    options = item.get("options", [])
    options_str = "\n".join(options)

    prompt = EXP_PROMPT.format(
        question=item["question"],
        options=options_str,
        ground_truth=gt_full_text(options, item.get("gt_answer", "")),
        explanation=explanation,
    )
    try:
        judge_text = text_completion(
            prompt=prompt,
            system=EXP_SYSTEM,
            model=model,
            max_tokens=300,
        )
        score = parse_score(judge_text)
        if score is None:
            logger.warning(f"Unparseable score for {item_id}: {judge_text[:80]!r}")
        return {"id": item_id, "score": score, "judge_text": judge_text, "skipped": False}
    except Exception as exc:
        logger.error(f"Item {item_id} failed: {exc}")
        return {"id": item_id, "score": None, "judge_text": "",
                "skipped": True, "skip_reason": str(exc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Path to results.jsonl")
    ap.add_argument("--model", default="gpt-4o-mini", help="Text judge model")
    ap.add_argument("--batch_size", type=int, default=32, help="Concurrent API requests")
    ap.add_argument("--output", default=None,
                    help="Output JSONL path (default: <results_dir>/exp_scores.jsonl)")
    args = ap.parse_args()

    results_path = Path(args.results)
    out_dir = results_path.parent
    out_path = Path(args.output) if args.output else out_dir / "exp_scores.jsonl"
    summary_path = out_dir / "exp_summary.json"

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    os.environ.setdefault("TEXT_MODEL", args.model)

    # Load results
    items = []
    for line in results_path.read_text().splitlines():
        try:
            items.append(json.loads(line))
        except Exception:
            pass
    logger.info(f"Loaded {len(items)} items from {results_path}")

    # Resume
    done_ids: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done_ids.add(json.loads(line)["id"])
            except Exception:
                pass
        if done_ids:
            logger.info(f"Resuming: {len(done_ids)} already scored")
    pending = [it for it in items if it["id"] not in done_ids]
    logger.info(f"Pending: {len(pending)}  |  batch_size={args.batch_size}  |  model={args.model}")

    t0 = time.time()
    with open(out_path, "a") as fout:
        with ThreadPoolExecutor(max_workers=args.batch_size) as pool:
            futures = {pool.submit(evaluate_one, it, args.model): it for it in pending}
            for i, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                fout.write(json.dumps(r) + "\n")
                fout.flush()
                if i % 100 == 0 or i == len(pending):
                    logger.info(f"[{i:>5}/{len(pending)}]  elapsed={time.time()-t0:.0f}s")

    # Aggregate
    all_rows = []
    for line in out_path.read_text().splitlines():
        try:
            all_rows.append(json.loads(line))
        except Exception:
            pass

    scored = [r for r in all_rows if r.get("score") is not None]
    skipped = [r for r in all_rows if r.get("skipped")]
    good = [r for r in scored if r["score"] <= 2]

    exp = len(good) / len(scored) if scored else float("nan")
    summary = {
        "metric": "EXP",
        "description": "fraction of scored items with reasoning score <= 2 (higher is better)",
        "model": args.model,
        "n_total": len(all_rows),
        "n_scored": len(scored),
        "n_skipped": len(skipped),
        "exp": round(exp, 4),
        "exp_pct": round(exp * 100, 2),
        "score_distribution": {str(s): sum(1 for r in scored if r["score"] == s)
                                for s in range(1, 6)},
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    logger.info(f"\nEXP = {exp * 100:.2f}%  ({len(good)}/{len(scored)} scored, "
                f"{len(skipped)} skipped)")
    logger.info(f"Scores  → {out_path}")
    logger.info(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
