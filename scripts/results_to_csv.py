"""
Convert a Grounded-IP results.json to a clean per-sample CSV.

Columns:
  item_id, question, options, gt,
  pred_max, pred_ip, pred_ip_step, pred_ip_conf,
  correct (== correct_max), correct_max, correct_ip,
  num_queries, exp_passed, exp_mode, elapsed_sec,
  pred_step_1 … pred_step_N   (step_predicted_answer at each evidence step)

Usage:
  python scripts/results_to_csv.py <results.json>
  python scripts/results_to_csv.py <results_dir/>          # auto-finds results.json
  python scripts/results_to_csv.py --all                   # convert all results.json under saved/results/
  python scripts/results_to_csv.py <results.json> --out my_output.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd


GROUNDED_IP_KEYS = ("grounded_ip", "ground_ip")  # new name first, then legacy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", help="Path to results.json or its parent directory.")
    p.add_argument("--all", dest="all_runs", action="store_true",
                   help="Convert every results.json under saved/results/.")
    p.add_argument("--out", default=None,
                   help="Output CSV path. Defaults to <results_dir>/grounded_ip_clean.csv.")
    return p.parse_args()


def find_key(results: dict) -> str | None:
    for key in GROUNDED_IP_KEYS:
        if key in results:
            return key
    return None


def convert(results_path: Path, out_path: Path) -> int:
    with open(results_path) as f:
        results = json.load(f)

    key = find_key(results)
    if key is None:
        print(f"  skip (no grounded_ip/ground_ip key): {results_path}")
        return 0

    items = results[key]
    if not items:
        print(f"  skip (empty list): {results_path}")
        return 0

    max_steps = max((len(r.get("evidence_trail", [])) for r in items), default=0)

    rows = []
    for r in items:
        trail = r.get("evidence_trail", [])
        options = r.get("options", [])

        pred_max = r.get("pred_max") or r.get("predicted_answer")
        pred_ip  = r.get("pred_ip")  or pred_max
        gt       = r.get("correct_answer")
        # correct == correct_max (kept for backward compat; same value).
        correct_max = r.get("correct_max") if r.get("correct_max") is not None \
                      else r.get("correct")
        correct_ip  = r.get("correct_ip")
        if correct_max is None and pred_max and gt:
            correct_max = int(pred_max == gt)
        if correct_ip is None and pred_ip and gt:
            correct_ip = int(pred_ip == gt)
        row = {
            "item_id":      r.get("item_id"),
            "question":     r.get("question"),
            "options":      " | ".join(options) if isinstance(options, list) else str(options),
            "gt":           gt,
            "pred_max":     pred_max,
            "pred_ip":      pred_ip,
            "pred_ip_step": r.get("pred_ip_step", 0),
            "pred_ip_conf": round(r.get("pred_ip_conf", 0.0), 3),
            "correct":      int(bool(correct_max)) if correct_max is not None else None,  # == correct_max
            "correct_max":  int(bool(correct_max)) if correct_max is not None else None,
            "correct_ip":   int(bool(correct_ip))  if correct_ip  is not None else None,
            "num_queries":  r.get("num_queries"),
            "exp_passed":   r.get("exp_passed"),
            "exp_mode":     r.get("exp_mode"),
            "elapsed_sec":  round(r.get("elapsed_sec", 0), 1),
        }

        for k in range(max_steps):
            step_pred = trail[k].get("step_predicted_answer") if k < len(trail) else None
            row[f"pred_step_{k+1}"] = step_pred

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  {len(df)} rows -> {out_path}")
    return len(df)


def main():
    args = parse_args()

    if args.all_runs:
        script_dir = Path(__file__).parent
        results_root = script_dir.parent / "saved" / "results"
        paths = sorted(results_root.rglob("results.json"))
        if not paths:
            print("No results.json found under", results_root)
            return
        total = 0
        for p in paths:
            out = p.parent / "grounded_ip_clean.csv"
            total += convert(p, out)
        print(f"\nDone. {total} total rows across {len(paths)} files.")
        return

    if not args.path:
        print("Provide a path or use --all. Run with -h for help.")
        return

    path = Path(args.path)
    if path.is_dir():
        path = path / "results.json"
    if not path.exists():
        raise FileNotFoundError(path)

    out = Path(args.out) if args.out else path.parent / "grounded_ip_clean.csv"
    convert(path, out)


if __name__ == "__main__":
    main()
