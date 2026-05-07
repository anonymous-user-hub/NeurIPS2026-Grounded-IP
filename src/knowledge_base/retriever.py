"""
Knowledge retrieval function R(B, q_k, S_{k-1}).

Loads the FAISS index from disk (once, cached) and retrieves
the top-r most relevant knowledge chunks for a given query + context.
"""
import json
import logging
from pathlib import Path

import faiss
import numpy as np

from src.config import KB_INDEX_PATH, KB_CHUNKS_PATH, TOP_R
from src.models.openai_client import get_embedding

logger = logging.getLogger(__name__)

_index:  faiss.IndexFlatIP | None = None
_chunks: list[dict] | None = None


def _load_db():
    global _index, _chunks
    if _index is not None:
        return
    if not KB_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base index not found at {KB_INDEX_PATH}. "
            "Run scripts/build_kb.py first."
        )
    _index = faiss.read_index(str(KB_INDEX_PATH))
    with open(KB_CHUNKS_PATH) as f:
        _chunks = json.load(f)
    logger.info(f"Loaded knowledge base: {_index.ntotal} chunks")


def retrieve(query: str, context: str = "", top_r: int = TOP_R) -> list[dict]:
    """
    Retrieve top-r knowledge chunks relevant to (query, context).

    Args:
        query:   the current binary question q_k
        context: optional summary of current history S_{k-1}
        top_r:   number of chunks to retrieve

    Returns:
        list of chunk dicts with keys: text, source, chunk_id, score
    """
    _load_db()
    # Combine query and context into a single retrieval string
    search_text = query
    if context:
        search_text = f"{query}\n{context}"
    emb = get_embedding(search_text)
    emb = emb.reshape(1, -1)
    faiss.normalize_L2(emb)

    scores, indices = _index.search(emb, top_r)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = dict(_chunks[idx])
        chunk["score"] = float(score)
        results.append(chunk)
    return results


def retrieve_as_text(query: str, context: str = "", top_r: int = TOP_R) -> str:
    """
    Retrieve and format chunks as a single string for prompt injection.
    """
    chunks = retrieve(query, context, top_r)
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[Reference {i} – {c['source']}]\n{c['text']}")
    return "\n\n".join(parts) if parts else "(no relevant knowledge found)"
