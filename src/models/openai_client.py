"""
Thin wrapper around the OpenAI API.
Handles:
  - text completions  (TEXT_MODEL)
  - vision completions (VISION_MODEL, with base64 image)
  - embeddings        (EMBED_MODEL)
  - retry on transient errors
  - 16-bit PNG → 8-bit JPEG conversion before encoding
"""
import base64
import io
import math
import time
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from openai import OpenAI, RateLimitError, APIError

from src.config import OPENAI_API_KEY, VISION_MODEL, TEXT_MODEL, EMBED_MODEL, LOCAL_BASE_URL, LOCAL_API_KEY

logger = logging.getLogger(__name__)

# Process-level overrides set by configure_local() for local vLLM endpoints
_local_base_url: str | None = None
_local_api_key:  str | None = None
_clients: dict[tuple, OpenAI] = {}

# Per-model URL overrides: model_name → (base_url, api_key)
# Populated by configure_model_endpoint(); takes priority over _local_base_url.
_model_url_map: dict[str, tuple[str, str]] = {}

# Models for which we've already emitted a one-time warning (e.g. no logprobs support)
_warned_models: set[str] = set()

# OpenAI model name prefixes — always routed to real OpenAI, never to local endpoint
_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4", "gpt-5", "text-embedding-", "davinci", "babbage")


def configure_local(base_url: str, api_key: str = "EMPTY") -> None:
    """
    Configure the default local vLLM endpoint.  Only calls whose model name does NOT
    match an OpenAI prefix are redirected to this endpoint; OpenAI-native models
    (gpt-*, text-embedding-*, etc.) always use the real OpenAI API.
    """
    global _local_base_url, _local_api_key, _clients
    _local_base_url = base_url
    _local_api_key  = api_key
    _clients.clear()
    logger.info(f"Local model endpoint configured: {base_url}")


def configure_model_endpoint(model: str, base_url: str, api_key: str = "EMPTY") -> None:
    """
    Register a per-model URL override.  Calls using exactly this model name will
    be routed to base_url regardless of the global _local_base_url.  Use this to
    point validator models at a separate vLLM server (e.g. MedGemma on port 8001).
    """
    global _model_url_map, _clients
    _model_url_map[model] = (base_url, api_key)
    _clients.clear()
    logger.info(f"Per-model endpoint registered: {model} → {base_url}")


def _is_openai_model(model: str) -> bool:
    return any(model.startswith(p) for p in _OPENAI_PREFIXES)


def _is_thinking_model(model: str) -> bool:
    """Qwen3 / Qwen3.5 run in thinking mode by default via vLLM."""
    return "Qwen3" in model or "qwen3" in model.lower()


def get_client(model: str | None = None) -> OpenAI:
    """
    Return the appropriate OpenAI client for the given model name.
    Priority order:
      1. Per-model override (_model_url_map) — e.g. validator on a separate server
      2. Global local endpoint (_local_base_url) — for non-OpenAI models
      3. Real OpenAI API — for gpt-*, text-embedding-*, etc.
    """
    if model and model in _model_url_map:
        b, k = _model_url_map[model]
    elif (_local_base_url is not None) and (model is not None) and not _is_openai_model(model):
        b, k = _local_base_url, _local_api_key or OPENAI_API_KEY
    else:
        b, k = None, OPENAI_API_KEY
    cache_key = (b, k)
    if cache_key not in _clients:
        kwargs: dict = {"api_key": k}
        if b:
            kwargs["base_url"] = b
        _clients[cache_key] = OpenAI(**kwargs)
    return _clients[cache_key]


# ── Image utilities ────────────────────────────────────────────────────────────

def load_image_as_base64(image_path: str | Path, max_side: int = 1024) -> str:
    """
    Load a chest X-ray PNG (possibly 16-bit) and return a base64-encoded JPEG
    suitable for the OpenAI vision API.
    """
    img = Image.open(str(image_path))

    # 16-bit grayscale → 8-bit
    if img.mode in ("I;16", "I;16B", "I"):
        arr = np.array(img, dtype=np.uint16)
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = ((arr.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")

    # Grayscale → RGB (OpenAI expects RGB)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize: keep aspect ratio, longest side ≤ max_side
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── API calls ──────────────────────────────────────────────────────────────────

def _retry(fn, retries: int = 3, wait: float = 2.0):
    for attempt in range(retries):
        try:
            return fn()
        except RateLimitError:
            if attempt < retries - 1:
                time.sleep(wait * (attempt + 1))
            else:
                raise
        except APIError as e:
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                raise


def _uses_max_completion_tokens(model: str) -> bool:
    """gpt-5.x and o-series models require max_completion_tokens instead of max_tokens."""
    return model.startswith(("o1", "o3", "o4", "gpt-5"))


def _scale_max_tokens(max_tokens: int | None, model: str) -> int | None:
    """Scale up max_completion_tokens for reasoning models.

    gpt-5 and o-series models spend a shared token budget on internal thinking
    before producing visible content.  A tight limit (e.g. 768) is entirely
    consumed by thinking, leaving the content field empty.  We multiply by 8
    (floor 8192) so the model can think AND produce a full JSON response.
    Only applied when max_tokens is explicitly set — None means "no limit".
    """
    if max_tokens is None or not _uses_max_completion_tokens(model):
        return max_tokens
    return max(max_tokens * 8, 8192)


def text_completion(
    prompt: str,
    system: str = "You are a helpful radiology AI assistant.",
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> str:
    model = model or TEXT_MODEL
    eff = _scale_max_tokens(max_tokens, model)
    if eff is None:
        extra = {} if _uses_max_completion_tokens(model) else {"temperature": temperature}
    else:
        extra = {"max_completion_tokens": eff} if _uses_max_completion_tokens(model) \
                else {"max_tokens": eff, "temperature": temperature}
    # Thinking models: disable thinking so the token budget is not consumed before
    # the visible response, and so structured outputs (JSON, single letters) are
    # produced directly without a reasoning preamble that breaks parsers.
    if not _is_openai_model(model) and _is_thinking_model(model):
        extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    def _call():
        resp = get_client(model).chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            **extra,
        )
        return resp.choices[0].message.content.strip()
    return _retry(_call)


def text_completion_logprobs(
    prompt: str,
    answer_letters: list[str],
    system: str = "You are a helpful radiology AI assistant.",
    model: str | None = None,
    temperature: float = 1.0,
) -> dict[str, float]:
    """
    Generate a single-token response and return log-probabilities for each
    answer letter (e.g. ["A","B","C","D"]).

    The model is asked to output exactly one letter.  We read the raw token
    log-probabilities from the API response — these are the model's actual
    belief scores, not self-reported values.

    Args:
        prompt:         the full prompt (should instruct model to output one letter)
        answer_letters: list of valid single-character answer keys, e.g. ["A","B","C","D"]
        system:         system message
        model:          override model (default TEXT_MODEL)
        temperature:    use > 0 to soften the distribution (default 1.0); temperature=0
                        can collapse all probability mass onto one token.

    Returns:
        dict[letter → log_prob].  Letters not found in top_logprobs are assigned
        log-prob = -100 (effectively zero probability after exp).
    """
    model = model or TEXT_MODEL
    # o-series and gpt-5 do not support logprobs; fall back to uniform for those
    if _uses_max_completion_tokens(model):
        logger.warning(f"Model {model} does not support logprobs; returning uniform distribution.")
        lp = math.log(1.0 / len(answer_letters))
        return {L: lp for L in answer_letters}

    # OpenAI API caps temperature at 2.0, but at T>1 distributions become too noisy
    # for MI scoring. Local vLLM models need T=7.0 to overcome extreme overconfidence
    # post-RLHF; OpenAI models are already calibrated at T=1.0.
    if _is_openai_model(model):
        temperature = min(temperature, 1.0)

    extra: dict = {"max_tokens": 1, "temperature": temperature, "logprobs": True, "top_logprobs": 20}
    # Thinking models (Qwen3/3.5): thinking tokens share the max_tokens budget.
    # With max_tokens=1, thinking would consume the entire budget leaving no visible
    # token → logprobs.content is empty → IndexError → uniform fallback → MI=-2.
    # Disable thinking so the model outputs the answer letter directly.
    if not _is_openai_model(model) and _is_thinking_model(model):
        extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    def _call():
        resp = get_client(model).chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            **extra,
        )
        return resp.choices[0]

    choice = _retry(_call)

    # Extract log-probs for the first output token.
    # top_logprobs is sorted highest → lowest: use first-match-wins per letter
    # so that a later low-prob token (e.g. ' a') doesn't overwrite the correct entry.
    result: dict[str, float] = {L: -100.0 for L in answer_letters}
    try:
        top = choice.logprobs.content[0].top_logprobs   # list of TopLogprob objects
        for entry in top:
            tok = entry.token.strip().upper()
            if tok in result and result[tok] == -100.0:
                result[tok] = entry.logprob
    except (AttributeError, IndexError, TypeError) as e:
        logger.warning(f"Logprob extraction failed: {e}; using uniform fallback.")
        lp = math.log(1.0 / len(answer_letters))
        return {L: lp for L in answer_letters}

    return result


def vision_completion(
    prompt: str,
    image_path: str | Path | None = None,
    image_b64: str | None = None,
    system: str = "You are a radiology AI assistant specialized in chest X-ray interpretation.",
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> str:
    """
    Send a vision-capable completion. Provide either image_path or image_b64.
    """
    model = model or VISION_MODEL
    if image_path is not None:
        image_b64 = load_image_as_base64(image_path)

    content: list = [{"type": "text", "text": prompt}]
    if image_b64:
        content.insert(0, {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"},
        })

    eff = _scale_max_tokens(max_tokens, model)
    if eff is None:
        extra = {} if _uses_max_completion_tokens(model) else {"temperature": temperature}
    else:
        extra = {"max_completion_tokens": eff} if _uses_max_completion_tokens(model) \
                else {"max_tokens": eff, "temperature": temperature}
    # Thinking models: disable thinking to prevent the reasoning chain from consuming
    # the entire token budget before any visible output is produced, and to ensure
    # structured (JSON / numbered-list) formats are followed directly.
    if not _is_openai_model(model) and _is_thinking_model(model):
        extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    def _call():
        resp = get_client(model).chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": content},
            ],
            **extra,
        )
        return resp.choices[0].message.content.strip()

    return _retry(_call)


def vision_completion_with_logprobs(
    prompt: str,
    image_path: str | Path | None = None,
    image_b64: str | None = None,
    system: str = "You are a radiology AI assistant specialized in chest X-ray interpretation.",
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    capture_thinking: bool = True,
) -> tuple[str, list, str]:
    """
    Vision completion with token-level log-probabilities.

    Returns:
        (raw_text, token_logprob_list, reasoning_content)
        raw_text            — model's visible output (message.content); for thinking models
                              this is the clean post-thinking text, so char_to_token_logprobs
                              alignment is always correct.
        token_logprob_list  — logprobs for visible tokens (choice.logprobs.content).
        reasoning_content   — thinking chain from vLLM's reasoning_content field (Qwen3 etc.);
                              empty string for non-thinking models.

    capture_thinking=True (default): thinking runs normally for Qwen3/3.5 models; the
        internal reasoning chain is returned as reasoning_content.  Logprob extraction is
        unaffected because vLLM separates thinking tokens from logprobs.content.
    capture_thinking=False: thinking is suppressed (enable_thinking=False) for terse output.
        Only use this if you specifically need to prevent thinking (e.g. token-budget tests).
    """
    model = model or VISION_MODEL
    if image_path is not None:
        image_b64 = load_image_as_base64(image_path)

    # OpenAI API caps temperature at 2.0, but T>1 is too noisy for MI scoring.
    # Local vLLM models need T=7.0 to overcome post-RLHF overconfidence; OpenAI
    # models are calibrated well enough at T=1.0.
    if _is_openai_model(model):
        temperature = min(temperature, 1.0)

    content: list = [{"type": "text", "text": prompt}]
    if image_b64:
        content.insert(0, {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"},
        })

    supports_logprobs = not _uses_max_completion_tokens(model)
    eff = _scale_max_tokens(max_tokens, model)
    if eff is None:
        extra: dict = {} if _uses_max_completion_tokens(model) \
                      else {"temperature": temperature, "logprobs": True, "top_logprobs": 20}
    else:
        extra: dict = {"max_completion_tokens": eff} if _uses_max_completion_tokens(model) \
                      else {"max_tokens": eff, "temperature": temperature,
                            "logprobs": True, "top_logprobs": 20}
    # Suppress thinking for Qwen3/3.5 unless the caller wants the reasoning chain.
    # capture_thinking=True: thinking runs normally; reasoning_content returned as 3rd value.
    # capture_thinking=False: thinking disabled so model outputs terse clean text (needed for
    #   JSON responses in the answerer).
    if not _is_openai_model(model) and _is_thinking_model(model) and not capture_thinking:
        extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    if not supports_logprobs and model not in _warned_models:
        logger.warning(f"Model {model} does not support logprobs; logprobs will be uniform fallback.")
        _warned_models.add(model)

    def _call():
        resp = get_client(model).chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": content},
            ],
            **extra,
        )
        return resp.choices[0]

    choice = _retry(_call)
    raw_text = choice.message.content or ""
    token_logprob_list: list = []
    if supports_logprobs:
        try:
            token_logprob_list = list(choice.logprobs.content or [])
        except (AttributeError, TypeError):
            logger.warning("Logprobs not available in response.")

    # Extract thinking chain returned by vLLM in the non-standard reasoning_content field.
    reasoning_content = ""
    if capture_thinking:
        try:
            reasoning_content = (choice.message.model_extra or {}).get("reasoning_content", "") or ""
        except AttributeError:
            pass

    return raw_text, token_logprob_list, reasoning_content


def char_to_token_logprobs(
    raw_text: str,
    token_logprob_list: list,
    char_pos: int,
    answer_letters: list[str],
) -> dict[str, float]:
    """
    Extract per-letter log-probabilities from the token at a specific character position.

    Walk token_logprob_list accumulating text until the token spanning char_pos
    is found, then read its top_logprobs.  This is exact: no guessing or scanning.

    Args:
        raw_text:           unstripped model output (same string the tokens reconstruct)
        token_logprob_list: choice.logprobs.content
        char_pos:           character index in raw_text of the answer letter
        answer_letters:     e.g. ["A", "B", "C", "D"]

    Returns:
        dict[letter → log_prob]; absent letters get -100 (≈ zero probability).
    """
    logprobs: dict[str, float] = {L: -100.0 for L in answer_letters}
    if char_pos < 0 or not token_logprob_list:
        return logprobs
    cumulative = ""
    for tok_obj in token_logprob_list:
        tok_start = len(cumulative)
        cumulative += tok_obj.token
        if tok_start <= char_pos < len(cumulative):
            # top_logprobs is sorted highest → lowest probability.
            # Use first-match-wins so a later low-probability token (e.g. ' a')
            # that also strips to 'A' does not overwrite the correct high-prob entry.
            for entry in tok_obj.top_logprobs:
                t = entry.token.strip().upper()
                if t in logprobs and logprobs[t] == -100.0:
                    logprobs[t] = entry.logprob
            break
    return logprobs


def get_embedding(text: str, model: str | None = None) -> np.ndarray:
    model = model or EMBED_MODEL
    text = text.replace("\n", " ")
    def _call():
        resp = get_client(model).embeddings.create(input=[text], model=model)
        return np.array(resp.data[0].embedding, dtype=np.float32)
    return _retry(_call)


def get_embeddings_batch(texts: list[str], model: str | None = None) -> np.ndarray:
    model = model or EMBED_MODEL
    texts = [t.replace("\n", " ") for t in texts]
    def _call():
        resp = get_client(model).embeddings.create(input=texts, model=model)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)
    return _retry(_call)
