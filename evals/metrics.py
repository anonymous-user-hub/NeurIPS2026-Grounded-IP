"""
Evaluation metrics for Grounded-IP and baselines on ReXVQA.

Bootstrap utilities
-------------------
bootstrap_scores(values)     — shared core for VHR, FHR, EXP: takes pre-computed per-item
                               binary flags and returns (mean%, std%).
bootstrap_accuracy(gt, pred) — accuracy wrapper: computes (gt==pred), then calls bootstrap_scores.

All four paper metrics (Accuracy, VHR, FHR, EXP) use the same bootstrap_scores core.
"""
import numpy as np
from collections import defaultdict


def bootstrap_scores(
    values,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Shared bootstrap for any per-item binary metric (VHR, FHR, EXP).

    Args:
        values:      array-like of per-item binary scores (0.0 or 1.0)
        n_bootstrap: number of bootstrap resamples
        seed:        random seed

    Returns:
        (mean_pct, std_pct)  — both in % (multiplied by 100)
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = arr[rng.integers(0, n, size=(n_bootstrap, n))].mean(axis=1)
    return float(arr.mean() * 100), float(boot.std(ddof=1) * 100)


def bootstrap_accuracy(
    gt,
    pred,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Bootstrap mean accuracy and std.

    Args:
        gt:   array-like of ground-truth labels
        pred: array-like of predicted labels (same type as gt)

    Returns:
        (mean_acc%, std_acc%)
    """
    gt   = np.asarray(gt)
    pred = np.asarray(pred)
    correct = (gt == pred).astype(float)
    return bootstrap_scores(correct, n_bootstrap=n_bootstrap, seed=seed)


def compute_accuracy(results: list[dict]) -> dict:
    """Accuracy for pred_max (predicted_answer) with bootstrap mean±std."""
    gt   = [r.get("correct_answer", "").upper() for r in results]
    pred = [r.get("predicted_answer", "").upper() for r in results]
    mean_pct, std_pct = bootstrap_accuracy(gt, pred)
    n_correct = sum(g == p for g, p in zip(gt, pred))
    return {
        "accuracy":     mean_pct / 100,
        "accuracy_pct": mean_pct,
        "accuracy_std": std_pct,
        "n":            len(results),
        "n_correct":    n_correct,
    }


def compute_pred_ip_accuracy(results: list[dict]) -> dict:
    """Accuracy for pred_ip (prediction at confidence plateau) with bootstrap mean±std."""
    rows = [r for r in results if r.get("pred_ip")]
    if not rows:
        return {}
    gt   = [r.get("correct_answer", "").upper() for r in rows]
    pred = [r.get("pred_ip", "").upper() for r in rows]
    mean_pct, std_pct = bootstrap_accuracy(gt, pred)
    n_correct = sum(g == p for g, p in zip(gt, pred))
    n_plateau = sum(1 for r in rows if r.get("pred_ip_step", 0) > 0)
    return {
        "pred_ip_accuracy":     mean_pct / 100,
        "pred_ip_accuracy_pct": mean_pct,
        "pred_ip_accuracy_std": std_pct,
        "n":                    len(rows),
        "n_correct":            n_correct,
        "n_plateau_reached":    n_plateau,
        "plateau_rate":         n_plateau / len(rows) if rows else 0.0,
        "avg_plateau_step":     (
            sum(r.get("pred_ip_step", 0) for r in rows if r.get("pred_ip_step", 0) > 0) /
            max(n_plateau, 1)
        ),
    }


def compute_per_task_accuracy(results: list[dict]) -> dict[str, dict]:
    by_task = defaultdict(list)
    for r in results:
        task = r.get("task_name", "Unknown")
        by_task[task].append(r)

    out = {}
    for task, items in by_task.items():
        out[task] = compute_accuracy(items)
    return out


def compute_query_efficiency(results: list[dict]) -> dict:
    """Average number of queries used per prediction."""
    query_counts = [r.get("num_queries", 0) for r in results]
    if not query_counts:
        return {}
    return {
        "avg_queries": sum(query_counts) / len(query_counts),
        "min_queries": min(query_counts),
        "max_queries": max(query_counts),
    }


def compute_grounding_stats(results: list[dict]) -> dict:
    """For Ground-IP results: compute grounding rates + validator fire rates."""
    img_pass, kb_pass, total_steps = 0, 0, 0
    img_fired, kb_fired = 0, 0   # steps where validator rejected at least once
    for r in results:
        trail = r.get("evidence_trail", [])
        for step in trail:
            total_steps += 1
            if step.get("img_grounded"):
                img_pass += 1
            if step.get("kb_grounded"):
                kb_pass += 1
            # validator "fired" = rejected first attempt (retry_count > 0)
            if step.get("retry_count", 0) > 0:
                if not step.get("img_grounded") or step.get("img_feedback"):
                    img_fired += 1
                if not step.get("kb_grounded") or step.get("kb_feedback"):
                    kb_fired += 1
    if total_steps == 0:
        return {}
    return {
        "img_grounding_rate":  img_pass / total_steps,
        "kb_grounding_rate":   kb_pass  / total_steps,
        "img_validator_fire_rate": img_fired / total_steps,
        "kb_validator_fire_rate":  kb_fired  / total_steps,
        "total_steps": total_steps,
    }


def compute_explanation_stats(results: list[dict]) -> dict:
    """Explanation validator pass rate (only for Ground-IP)."""
    has_exp = [r for r in results if "exp_passed" in r]
    if not has_exp:
        return {}
    passed = sum(1 for r in has_exp if r.get("exp_passed"))
    modes  = defaultdict(int)
    for r in has_exp:
        modes[r.get("exp_mode", "unknown")] += 1
    return {
        "exp_pass_rate": passed / len(has_exp),
        "exp_modes": dict(modes),
        "n": len(has_exp),
    }


def compute_per_step_accuracy(results: list[dict]) -> dict:
    """
    For each query step k, compute classifier accuracy using step_proba stored
    in the evidence trail.  Only items that reached step k are included.

    Returns a dict like:
      {"k1": {"accuracy": 0.6, "n": 20, "n_correct": 12},
       "k2": {"accuracy": 0.65, "n": 20, "n_correct": 13}, ...}
    """
    max_k = max((len(r.get("evidence_trail", [])) for r in results), default=0)
    if max_k == 0:
        return {}

    step_accs = {}
    for k in range(1, max_k + 1):
        correct = total = 0
        for r in results:
            trail = r.get("evidence_trail", [])
            if len(trail) < k:
                continue
            step_proba = trail[k - 1].get("step_proba", {})
            if not step_proba:
                continue
            pred = max(step_proba, key=step_proba.get).upper()
            gt   = r.get("correct_answer", "").upper()
            total += 1
            if pred == gt:
                correct += 1
        if total > 0:
            step_accs[f"k{k}"] = {"accuracy": correct / total, "n": total, "n_correct": correct}

    return step_accs


def compute_all(results: list[dict], method_name: str = "") -> dict:
    """
    Compute all metrics for a set of results.

    VHR, FHR, EXP are not computed here — they require a separate judge model call
    (see evals/eval_vhr.py, eval_fhr.py, eval_exp.py). Run those scripts and merge
    the per-item scores back using merge_hallucination_scores().
    """
    return {
        "method": method_name,
        **compute_accuracy(results),
        "pred_ip": compute_pred_ip_accuracy(results),
        "per_task": compute_per_task_accuracy(results),
        **compute_query_efficiency(results),
        "grounding": compute_grounding_stats(results),
        "explanation": compute_explanation_stats(results),
        "per_step_accuracy": compute_per_step_accuracy(results),
        "avg_confidence": sum(r.get("confidence", 0) for r in results) / len(results) if results else 0,
        "avg_api_calls": sum(r.get("api_calls", 0) for r in results) / len(results) if results else 0,
        "avg_elapsed_sec": sum(r.get("elapsed_sec", 0) for r in results) / len(results) if results else 0,
        # VHR / FHR / EXP: populated by merge_hallucination_scores() after running eval scripts
        "vhr": None,
        "fhr": None,
        "exp": None,
    }


def merge_hallucination_scores(
    summary: dict,
    vhr_scores: list[dict] | None = None,
    fhr_scores: list[dict] | None = None,
    exp_scores: list[dict] | None = None,
) -> dict:
    """
    Merge per-item VHR/FHR/EXP scores into a compute_all() summary dict.

    Each *_scores list is a list of {"id": ..., "score": int|None} dicts
    from evals/eval_vhr.py, eval_fhr.py, eval_exp.py.

    Updates summary["vhr"], summary["fhr"], summary["exp"] in place and returns it.
    """
    def _compute(scores: list[dict], bad_fn) -> dict:
        scored = [r for r in scores if r.get("score") is not None]
        if not scored:
            return {"n_scored": 0, "mean_pct": float("nan"), "std_pct": float("nan")}
        values = [1.0 if bad_fn(r["score"]) else 0.0 for r in scored]
        mean_pct, std_pct = bootstrap_scores(values)
        return {
            "n_scored":      len(scored),
            "n_skipped":     len(scores) - len(scored),
            "mean_pct":      round(mean_pct, 2),
            "std_pct":       round(std_pct, 2),
            "score_distribution": {
                str(s): sum(1 for r in scored if r["score"] == s)
                for s in range(1, 6)
            },
        }

    if vhr_scores is not None:
        summary["vhr"] = _compute(vhr_scores, lambda s: s >= 4)
    if fhr_scores is not None:
        summary["fhr"] = _compute(fhr_scores, lambda s: s >= 4)
    if exp_scores is not None:
        # EXP: fraction with score <= 2 (good reasoning; higher is better)
        summary["exp"] = _compute(exp_scores, lambda s: s <= 2)
    return summary


def print_summary(metrics: dict):
    print(f"\n{'='*60}")
    print(f"Method: {metrics.get('method', 'unknown')}")
    print(f"{'='*60}")
    print(f"Accuracy        : {metrics['accuracy']*100:.1f}%  ({metrics['n_correct']}/{metrics['n']})")
    print(f"Avg queries     : {metrics.get('avg_queries', 'N/A'):.1f}" if 'avg_queries' in metrics else "Avg queries     : N/A")
    print(f"Avg API calls   : {metrics.get('avg_api_calls', 0):.1f}")
    print(f"Avg elapsed (s) : {metrics.get('avg_elapsed_sec', 0):.1f}")

    if metrics.get("grounding"):
        g = metrics["grounding"]
        print(f"Img grounding   : {g.get('img_grounding_rate',0)*100:.1f}%  "
              f"(validator fired: {g.get('img_validator_fire_rate',0)*100:.1f}%)")
        print(f"KB grounding    : {g.get('kb_grounding_rate',0)*100:.1f}%  "
              f"(validator fired: {g.get('kb_validator_fire_rate',0)*100:.1f}%)")

    if metrics.get("explanation"):
        e = metrics["explanation"]
        print(f"Exp pass rate   : {e.get('exp_pass_rate',0)*100:.1f}%")

    per_step = metrics.get("per_step_accuracy", {})
    if per_step:
        print("\nPer-step accuracy (classify_proba @ step k):")
        for k, stats in sorted(per_step.items()):
            bar = "█" * int(stats["accuracy"] * 20)
            print(f"  {k}: {stats['accuracy']*100:5.1f}%  {bar:<20}  (n={stats['n']})")

    print("\nPer-task accuracy:")
    for task, stats in sorted(metrics.get("per_task", {}).items()):
        print(f"  {task:<35} {stats['accuracy']*100:.1f}%  (n={stats['n']})")
