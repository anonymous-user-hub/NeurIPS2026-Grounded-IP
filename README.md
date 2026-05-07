# Grounded-IP: Grounded Information Pursuit for Interpretable Medical VQA

## Abstract

Current medical vision-language models (VLMs) have shown increasing promise for Medical Visual Question Answering (Med-VQA), but they typically offer limited insight into how an answer is derived: their explanations tend to be either post-hoc or non-sequential, failing to reflect the underlying decision-making process. In high-stakes medical settings, accurate predictions alone are insufficient: clinicians must be able to understand and verify the decision-making process. In this paper, we propose **Grounded-IP**, an *interpretable-by-design* Med-VQA framework based on Information Pursuit (IP) with *verified grounding*. Rather than producing a one-shot answer, Grounded-IP treats medical image understanding as an iterative evidence-seeking process. Given a clinical question, it adaptively selects a compact sequence of clinically meaningful queries based on the query–answer history, making the decision-making process transparent. To prevent unsupported intermediate claims from propagating, each intermediate query–answer pair is verified through both (1) *image grounding* to validate support from the medical image and (2) *knowledge grounding* to validate consistency with medical knowledge. The final output includes both a prediction and a traceable evidence trail of verified query–answer pairs, allowing users to inspect how the answer was reached and whether each step is grounded. Experiments on the chest X-ray benchmark ReXVQA show that Grounded-IP is comparable with state-of-the-art Med-VQA models while providing interpretable and grounded decision-making.

---

## Setup

```bash
pip install -r requirements.txt
```

Add your API key to a `.env` file at the project root:

```
OPENAI_API_KEY=sk-...
```

Build the knowledge base (run once):

```bash
python scripts/build_kb.py
```

---

## Usage

**Grounded-IP (proposed method):**

```bash
python scripts/run_rexvqa.py --run_name my_run
```

Defaults: `--vision_model gpt-4o`, `--task "Differential Diagnosis"`, `--split test`, `--n 50`, `--seed 42`.

**Baselines:**

```bash
# Direct prompting
python scripts/run_direct_prompt.py

# Chain-of-thought
python scripts/run_cot_prompt.py
```

Defaults: `--model gpt-4o`, `--task "Differential Diagnosis"`, `--split test`.

**Evaluation (compute accuracy and grounding metrics on saved results):**

```bash
python scripts/results_to_csv.py saved/results/<run_name>/results.json
```

Results are saved to `saved/results/<run_name>/`.

---

## Repository Structure

```
src/
  pipeline.py              # main pipeline (10-step evidence loop)
  config.py                # hyperparameters and paths
  components/              # querier, answerer, predictor, explanation generator
  validators/              # image validator (v_img), knowledge validator (v_kb), explanation validator (v_exp)
  knowledge_base/          # FAISS index construction and retrieval
  models/                  # API client (OpenAI / local vLLM)
  data/                    # dataset loader (ReXVQA)
  eval/                    # VHR / FHR / EXP evaluation judges
evals/
  metrics.py               # accuracy, grounding rate, explanation metrics
scripts/
  build_kb.py              # one-time knowledge base setup
  run_rexvqa.py            # Grounded-IP runner
  run_direct_prompt.py     # direct prompting baseline
  run_cot_prompt.py        # chain-of-thought baseline
  results_to_csv.py        # convert results to CSV
```
