"""
Unified Ground-IP experiment runner.

Usage examples:

  # Run baselines + Grounded-IP, Differential Diagnosis, gpt-4o, 20 items
  python scripts/run_rexvqa.py \
      --run_name run_diffdiag_gpt4o \
      --vision_model gpt-4o --text_model gpt-4o \
      --task "Differential Diagnosis" \
      --positive_only \
      --n 20 --seed 42 --k_max 15 --k_min 10

  # Run Grounded-IP only (skip baselines when iterating on the pipeline)
  python scripts/run_rexvqa.py \
      --run_name run_grounded_ip_only \
      --vision_model gpt-4o --text_model gpt-4o \
      --task "Differential Diagnosis" --positive_only \
      --methods grounded_ip \
      --n 20 --seed 42 --k_max 15 --k_min 10

  # Ablation: no validators
  python scripts/run_rexvqa.py \
      --run_name run_ablation_novalidators \
      --vision_model gpt-4o --text_model gpt-4o \
      --task "Differential Diagnosis" --positive_only \
      --methods direct_predict cot grounded_ip_no_val grounded_ip \
      --n 20 --seed 42 --k_max 15 --k_min 10
"""
import argparse, json, logging, os, random, shutil, sys
import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Key source files copied into <run_dir>/backup/ at the start of each run.
# Mirrors the save_code pattern used in echo_foundation_model.
SAVE_CODE_LIST = [
    "src/pipeline.py",
    "src/config.py",
    "src/models/openai_client.py",
    "src/components/querier.py",
    "src/components/answerer.py",
    "src/components/predictor.py",
    "src/components/explanation_generator.py",
    "src/validators/image_validator.py",
    "src/validators/knowledge_validator.py",
    "src/validators/explanation_validator.py",
    "baselines/baselines.py",
    "scripts/run_rexvqa.py",
    "job/run_qwen.job",
]


def save_code(run_dir: Path) -> None:
    backup_dir = run_dir / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for rel_path in SAVE_CODE_LIST:
        src = ROOT / rel_path
        if src.exists():
            shutil.copy2(src, backup_dir / src.name)
        else:
            logger.warning(f"save_code: {rel_path} not found, skipping.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("exp")


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name",      required=True)
    p.add_argument("--vision_model",  default="gpt-4o")
    p.add_argument("--text_model",    default="gpt-4o-mini")
    p.add_argument("--local_model",   default=None,
                   help="Model name served by local vLLM (e.g. 'Qwen/Qwen3.5-4B'). "
                        "Overrides --vision_model and --text_model and routes all "
                        "API calls to --local_base_url.")
    p.add_argument("--local_base_url", default=None,
                   help="Base URL of local vLLM server (default: http://localhost:8000/v1)")
    p.add_argument("--validator_base_url", default=None,
                   help="Base URL of a second vLLM server for validator models "
                        "(e.g. http://localhost:8001/v1 for MedGemma). "
                        "If omitted, validators use VALIDATOR_BASE_URL env var or default routing.")
    p.add_argument("--task",          default="Differential Diagnosis",
                   help="Filter by task_name substring (e.g. 'Differential Diagnosis')")
    p.add_argument("--question_prefix", default=None,
                   help="Comma-separated list of question prefixes to keep "
                        "(e.g. 'Is there,Are there')")
    p.add_argument("--positive_only", action="store_true",
                   help="Exclude items where correct answer contains "
                        "normal/no/absent/intact/unremarkable/none")
    p.add_argument("--n",             type=int, default=50)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--difficulty_csv", default=None,
                   help="Path to a difficulty-sorted CSV (e.g. "
                        "saved/results/2026-04-29-DifficultSet/difficult_set_sorted.csv). "
                        "When provided, items are selected in difficulty order (hardest first, "
                        "row 0 = rank-0 hardest) instead of random sampling. --n controls "
                        "how many top-k hardest items to run. --seed is ignored.")
    p.add_argument("--k_max",         type=int, default=25)
    p.add_argument("--querier_mode",  default="open_set",
                   choices=["open_set", "closed_set"],
                   help="open_set: LLM generates binary question from scratch (default). "
                        "closed_set: select from option-filtered query set Q.")
    p.add_argument("--k_min",         type=int, default=15,
                   help="Min queries before early stopping. Set equal to "
                        "k_max to disable early stopping entirely.")
    p.add_argument("--split",         default="test")
    p.add_argument("--methods",       nargs="+",
                   default=["direct_predict", "cot", "fixed_checklist", "grounded_ip"])
    p.add_argument("--verbose",       action="store_true")
    # Per-component model overrides (all default to global VISION_MODEL / TEXT_MODEL)
    p.add_argument("--querier_model",   default=None,
                   help="Model for querier g (candidate generation). Default: VISION_MODEL.")
    p.add_argument("--answerer_model",  default=None,
                   help="Model for answerer φ (binary visual Q&A). Default: VISION_MODEL.")
    p.add_argument("--predictor_model", default=None,
                   help="Model for predictor f + explanation ε (text-only). Default: TEXT_MODEL.")
    p.add_argument("--val_img_model",   default=None,
                   help="Model for image validator v_img. Default: VALIDATOR_VISION_MODEL.")
    p.add_argument("--val_kb_model",    default=None,
                   help="Model for KB validator v_kb. Default: VALIDATOR_TEXT_MODEL.")
    p.add_argument("--val_exp_model",   default=None,
                   help="Model for explanation validator v_exp. Default: VALIDATOR_TEXT_MODEL.")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of items to process in parallel (default: 1 = sequential). "
                        "Safe range for one B200 with two vLLM servers: 4–8.")
    return p.parse_args()


# ── item selection ────────────────────────────────────────────────────────────

NEGATIVE_TOKENS = {
    "no ", "none", "absent", "normal", "intact",
    "unremarkable", "without", "not seen", "not present",
    "no evidence", "no acute",
}


def item_is_positive(item: dict) -> bool:
    """Return True if the correct answer option describes a real positive finding."""
    letter = item.get("correct_answer", "A")
    opts   = item.get("options", [])
    idx    = ord(letter.upper()) - ord("A")
    if idx < 0 or idx >= len(opts):
        return True
    text = opts[idx].lower()
    return not any(tok in text for tok in NEGATIVE_TOKENS)


def select_items(args) -> list[dict]:
    from src.data.rexvqa_dataset import _resolve_image_path

    data = json.load(open(f"data/Radiology/ReXVQA/metadata/{args.split}_vqa_data.json"))
    prefixes = [p.strip().lower() for p in args.question_prefix.split(",")
                ] if args.question_prefix else []

    pool = []
    for item_id, it in data.items():
        q = it.get("question", "").lower()

        # task filter
        if args.task and args.task.lower() not in it.get("task_name", "").lower():
            continue

        # question prefix filter
        if prefixes and not any(q.startswith(p) for p in prefixes):
            continue

        # positive finding filter
        if args.positive_only and not item_is_positive(it):
            continue

        # image existence check
        paths = [_resolve_image_path(p) for p in it.get("ImagePath", [])]
        paths = [p for p in paths if p.exists()]
        if not paths:
            continue

        it = dict(it)
        it["id"]          = item_id
        it["image_paths"] = paths
        it["metadata"]    = {k: it.get(k) for k in
            ["Indication", "Comparison", "Findings", "Impression",
             "PatientSex", "PatientAge", "ImageViewPosition"]}
        pool.append(it)

    logger.info(f"Pool size after filters: {len(pool)}")

    if args.difficulty_csv:
        # Select items in difficulty order (hardest first) from the sorted CSV.
        import pandas as pd
        rank_df = pd.read_csv(args.difficulty_csv)
        ordered_ids = list(rank_df["id"])          # already sorted hardest → easiest
        pool_by_id  = {it["id"]: it for it in pool}
        selected = [pool_by_id[iid] for iid in ordered_ids
                    if iid in pool_by_id][: args.n]
        logger.info(f"Difficulty-ordered selection: top-{args.n} hardest "
                    f"(n_correct range {rank_df['n_correct'].iloc[0]}–"
                    f"{rank_df['n_correct'].iloc[len(selected)-1] if selected else '?'})")
    else:
        random.seed(args.seed)
        selected = random.sample(pool, min(args.n, len(pool)))

    return selected


# ── run ───────────────────────────────────────────────────────────────────────

def run_all(args):
    # Set model env vars BEFORE importing anything that reads them
    os.environ["VISION_MODEL"] = args.vision_model
    os.environ["TEXT_MODEL"]   = args.text_model

    # Local vLLM override: redirect all API calls to the local endpoint
    if args.local_model:
        os.environ["VISION_MODEL"] = args.local_model
        os.environ["TEXT_MODEL"]   = args.local_model
        from src.models.openai_client import configure_local, configure_model_endpoint
        from src.config import LOCAL_BASE_URL, VALIDATOR_BASE_URL, VALIDATOR_VISION_MODEL, VALIDATOR_TEXT_MODEL
        configure_local(
            base_url=args.local_base_url or LOCAL_BASE_URL,
            api_key=os.getenv("LOCAL_API_KEY", "EMPTY"),
        )
        # Route validator models to a separate endpoint if one is specified
        val_url = args.validator_base_url or VALIDATOR_BASE_URL
        if val_url:
            for val_model in set([VALIDATOR_VISION_MODEL, VALIDATOR_TEXT_MODEL]):
                if val_model and not val_model.startswith("gpt-"):
                    configure_model_endpoint(val_model, val_url, api_key="EMPTY")

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from src.config import (
        RESULTS_DIR, VISION_MODEL, TEXT_MODEL,
        VALIDATOR_VISION_MODEL, VALIDATOR_TEXT_MODEL,
        LOCAL_BASE_URL,
    )
    from src.data.rexvqa_dataset import get_primary_image
    from src.pipeline import run as grounded_ip_run
    from baselines.baselines import (
        DirectPredict, ChainOfThought, FixedChecklist,
        RandomChecklist, run_no_validators,
    )
    from evals.metrics import compute_all, print_summary

    # output directory: YYYY-MM-DD_HH-MM_[jobNNNNNN_]<run_name>
    timestamp_prefix = datetime.now().strftime("%Y-%m-%d_%H-%M")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    job_prefix = f"job{slurm_job_id}_" if slurm_job_id else ""
    out_dir = RESULTS_DIR / f"{timestamp_prefix}_{job_prefix}{args.run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path     = out_dir / "results.json"
    summary_path = out_dir / "summary.json"
    meta_path    = out_dir / "meta.json"

    # snapshot all key source files into backup/ for reproducibility
    save_code(out_dir)

    # save run metadata
    meta = vars(args)
    meta["timestamp"] = datetime.now().isoformat()
    from src.calibration import load_temperatures
    main_model = args.local_model or args.vision_model
    meta["calibration_temperatures"] = load_temperatures(main_model)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    _endpoint = f" @ {args.local_base_url or LOCAL_BASE_URL}" if args.local_model else " (OpenAI)"
    logger.info(f"Run: {args.run_name}")
    logger.info("Model assignments:")
    logger.info(f"  querier (g)    : {args.querier_model   or VISION_MODEL}{_endpoint}")
    logger.info(f"  answerer (φ)   : {args.answerer_model  or VISION_MODEL}{_endpoint}")
    logger.info(f"  predictor (f)  : {args.predictor_model or TEXT_MODEL}{_endpoint}")
    logger.info(f"  explanation (ε): {args.predictor_model or TEXT_MODEL}{_endpoint}")
    _val_url = args.validator_base_url or (VALIDATOR_BASE_URL if args.local_model else "")
    _val_endpoint = f" @ {_val_url}" if _val_url else " (OpenAI)"
    logger.info(f"  v_img          : {args.val_img_model or VALIDATOR_VISION_MODEL}{_val_endpoint}")
    logger.info(f"  v_kb           : {args.val_kb_model  or VALIDATOR_TEXT_MODEL}{_val_endpoint}")
    logger.info(f"  v_exp          : {args.val_exp_model or VALIDATOR_TEXT_MODEL}{_val_endpoint}")
    logger.info(f"Output: {out_dir}")

    selected = select_items(args)
    logger.info(f"Selected {len(selected)} items (seed={args.seed})")
    for i, it in enumerate(selected):
        logger.info(f"  [{i+1:2d}] [{it['correct_answer']}]  "
                    f"{it.get('task_name',''):<30}  {it['question'][:55]}")

    # method map
    method_map = {}
    for m in args.methods:
        if m == "direct_predict":
            obj = DirectPredict()
            method_map[m] = lambda it, img, _o=obj: _o.run(it, img).to_dict()
        elif m in ("cot", "chain_of_thought"):
            obj = ChainOfThought()
            method_map["chain_of_thought"] = lambda it, img, _o=obj: _o.run(it, img).to_dict()
        elif m == "fixed_checklist":
            obj = FixedChecklist()
            method_map[m] = lambda it, img, _o=obj: _o.run(it, img).to_dict()
        elif m == "random_checklist":
            obj = RandomChecklist(n_queries=args.k_max, seed=args.seed)
            method_map[m] = lambda it, img, _o=obj: _o.run(it, img).to_dict()
        elif m == "grounded_ip_no_val":
            method_map[m] = lambda it, img: run_no_validators(
                it, img, k_max=args.k_max, k_min=args.k_min,
                querier_mode=args.querier_mode).to_dict()
        elif m == "grounded_ip":
            def _gip(it, img, _q=None, _k=args.k_max, _km=args.k_min,
                     _qmode=args.querier_mode,
                     _qrm=args.querier_model, _arm=args.answerer_model,
                     _prm=args.predictor_model, _vim=args.val_img_model,
                     _vkm=args.val_kb_model, _vem=args.val_exp_model):
                r = grounded_ip_run(it, img, k_max=_k, k_min=_km, queries=_q,
                                    querier_mode=_qmode,
                                    querier_model=_qrm, answerer_model=_arm,
                                    predictor_model=_prm, val_img_model=_vim,
                                    val_kb_model=_vkm, val_exp_model=_vem)
                return {
                    "item_id": r.item_id, "baseline_name": "grounded_ip",
                    "question": r.question, "options": r.options,
                    "correct_answer": r.correct_answer,
                    "predicted_answer": r.predicted_answer, "confidence": r.confidence,
                    "pred_max": r.pred_max,
                    "pred_ip": r.pred_ip,
                    "pred_ip_step": r.pred_ip_step,
                    "pred_ip_conf": r.pred_ip_conf,
                    "correct": r.pred_max == r.correct_answer,      # == correct_max
                    "correct_max": r.pred_max == r.correct_answer,
                    "correct_ip": r.pred_ip == r.correct_answer,
                    "explanation": r.explanation,
                    "supporting_findings": r.supporting_findings,
                    "differential_ruled_out": r.differential_ruled_out,
                    "evidence_trail": r.evidence_trail,
                    "exp_passed": r.exp_passed, "exp_mode": r.exp_mode,
                    "num_queries": r.num_queries, "api_calls": r.api_calls,
                    "elapsed_sec": r.elapsed_sec,
                    "task_name": it.get("task_name", ""),
                }
            method_map["grounded_ip"] = _gip

    all_results = {m: [] for m in method_map}
    results_lock = threading.Lock()

    def process_item(i: int, item: dict) -> dict:
        """Run all methods on one item. Returns {method_name: result_dict}."""
        img = get_primary_image(item) or item["image_paths"][0]
        logger.info(f"\n{'='*65}")
        logger.info(f"Item {i+1}/{len(selected)}: [{item['correct_answer']}] "
                    f"{item['question'][:60]}")
        logger.info(f"Task: {item.get('task_name','')}  |  "
                    f"Options: {[o[:30] for o in item['options']]}")
        item_results = {}
        for method_name, fn in method_map.items():
            try:
                r = fn(item, img)
                r.setdefault("task_name", item.get("task_name", ""))
                r["image_paths"] = [str(p) for p in item["image_paths"]]
                item_results[method_name] = r
                status = "✓" if r.get("correct") else "✗"
                nq = r.get("num_queries", "-")
                logger.info(
                    f"  [{method_name:<25}] → {r.get('predicted_answer')} "
                    f"(gt={item['correct_answer']}) {status}  "
                    f"[{r.get('api_calls',0)} calls"
                    + (f", {nq} queries" if nq != "-" else "")
                    + f", {r.get('elapsed_sec',0):.1f}s]"
                )
            except Exception as e:
                logger.error(f"  [{method_name}] FAILED: {e}", exc_info=True)
        return item_results

    def save_incremental():
        """Write results.json and per-method CSVs. Must be called under results_lock."""
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        for method_name, method_results in all_results.items():
            if not method_results:
                continue
            max_steps = max(len(r.get("evidence_trail", [])) for r in method_results)
            csv_path = out_dir / f"{method_name}_clean.csv"
            with open(csv_path, "w", newline="") as cf:
                step_cols = [f"pred_step_{k+1}" for k in range(max_steps)]
                fieldnames = ["item_id", "question", "options", "gt",
                              "pred_max", "pred_ip", "pred_ip_step", "pred_ip_conf",
                              "correct_max", "correct_ip",
                              "num_queries", "exp_passed", "exp_mode",
                              "elapsed_sec"] + step_cols
                writer = csv.DictWriter(cf, fieldnames=fieldnames)
                writer.writeheader()
                for r in method_results:
                    trail = r.get("evidence_trail", [])
                    row = {
                        "item_id":      r.get("item_id"),
                        "question":     r.get("question"),
                        "options":      " | ".join(r.get("options", [])),
                        "gt":           r.get("correct_answer"),
                        "pred_max":     r.get("pred_max") or r.get("predicted_answer"),
                        "pred_ip":      r.get("pred_ip"),
                        "pred_ip_step": r.get("pred_ip_step"),
                        "pred_ip_conf": r.get("pred_ip_conf"),
                        "correct_max":  int(bool(r.get("correct_max") if "correct_max" in r else r.get("correct"))),
                        "correct_ip":   int(bool(r.get("correct_ip"))),
                        "num_queries":  r.get("num_queries"),
                        "exp_passed":   r.get("exp_passed"),
                        "exp_mode":     r.get("exp_mode"),
                        "elapsed_sec":  round(r.get("elapsed_sec", 0), 1),
                    }
                    for k in range(max_steps):
                        ev = trail[k] if k < len(trail) else {}
                        row[f"pred_step_{k+1}"] = ev.get("step_predicted_answer")
                    writer.writerow(row)

    workers = args.workers
    logger.info(f"Processing {len(selected)} items with workers={workers}")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_item, i, item): i
                   for i, item in enumerate(selected)}
        for future in as_completed(futures):
            item_results = future.result()
            with results_lock:
                for method_name, r in item_results.items():
                    all_results[method_name].append(r)
                save_incremental()

    # summary
    print(f"\n{'#'*65}")
    print(f"  RESULTS — {args.run_name}")
    print(f"  vision={args.vision_model}  text={args.text_model}  n={len(selected)}")
    print(f"{'#'*65}")
    summary = {}
    for method_name, results in all_results.items():
        if not results:
            continue
        m = compute_all(results, method_name=method_name)
        summary[method_name] = m
        print_summary(m)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Results → {out_path}")
    logger.info(f"Summary → {summary_path}")


if __name__ == "__main__":
    run_all(parse_args())
