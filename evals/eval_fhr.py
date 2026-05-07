"""
eval_fhr.py — Factual Hallucination Rate (FHR) evaluator.

For each result item with a non-empty explanation, a text LLM judge scores (1–5)
whether the explanation contains factually incorrect or unsupported medical claims:
  1 = Fully factually supported  ...  5 = Severely incorrect.

FHR = fraction of scored items with score >= 4.  Lower is better.

Usage:
    python evals/eval_fhr.py \\
        --results saved/results/direct_prompt/<run>/results.jsonl \\
        [--model gpt-4o-mini] [--batch_size 32]

Output (same directory as --results):
    fhr_scores.jsonl    — per-item scores
    fhr_summary.json    — aggregate FHR
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
logger = logging.getLogger("eval_fhr")


def parse_score(text: str) -> int | None:
    m = re.search(r"Score:\s*([1-5])", text)
    return int(m.group(1)) if m else None


def gt_full_text(options: list[str], gt_letter: str) -> str:
    return next((o for o in options if o.upper().startswith(gt_letter.upper() + ".")), gt_letter)


def evaluate_one(item: dict, model: str) -> dict:
    from src.models.openai_client import text_completion
    from evals.prompts import FHR_SYSTEM, FHR_PROMPT

    item_id = item["id"]
    explanation = (item.get("explanation") or "").strip()

    if not explanation:
        return {"id": item_id, "score": None, "judge_text": "",
                "skipped": True, "skip_reason": "empty_explanation"}

    prompt = FHR_PROMPT.format(
        question=item["question"],
        ground_truth=gt_full_text(item.get("options", []), item.get("gt_answer", "")),
        diagnosis=explanation,
    )
    try:
        judge_text = text_completion(
            prompt=prompt,
            system=FHR_SYSTEM,
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
                    help="Output JSONL path (default: <results_dir>/fhr_scores.jsonl)")
    args = ap.parse_args()

    results_path = Path(args.results)
    out_dir = results_path.parent
    out_path = Path(args.output) if args.output else out_dir / "fhr_scores.jsonl"
    summary_path = out_dir / "fhr_summary.json"

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
    hallucinated = [r for r in scored if r["score"] >= 4]

    fhr = len(hallucinated) / len(scored) if scored else float("nan")
    summary = {
        "metric": "FHR",
        "description": "fraction of scored items with factuality score >= 4 (lower is better)",
        "model": args.model,
        "n_total": len(all_rows),
        "n_scored": len(scored),
        "n_skipped": len(skipped),
        "fhr": round(fhr, 4),
        "fhr_pct": round(fhr * 100, 2),
        "score_distribution": {str(s): sum(1 for r in scored if r["score"] == s)
                                for s in range(1, 6)},
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    logger.info(f"\nFHR = {fhr * 100:.2f}%  ({len(hallucinated)}/{len(scored)} scored, "
                f"{len(skipped)} skipped)")
    logger.info(f"Scores  → {out_path}")
    logger.info(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
