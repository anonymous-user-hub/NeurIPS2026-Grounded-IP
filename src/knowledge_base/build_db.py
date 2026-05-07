"""
Build a FAISS vector database from the radiology knowledge text files.

Steps:
  1. Load text files → clean text
  2. Chunk into overlapping passages
  3. Embed with OpenAI text-embedding-3-small (batched)
  4. Build FAISS IndexFlatIP (inner product = cosine on L2-normalized vectors)
  5. Save index + chunk metadata to disk

Usage:
    python scripts/build_kb.py
    # or directly:
    from src.knowledge_base.build_db import build_and_save
    build_and_save()
"""
import json
import logging
import math
from pathlib import Path

import faiss
import numpy as np

from src.config import (
    KNOWLEDGE_FILES, KB_DIR, KB_INDEX_PATH, KB_CHUNKS_PATH,
    KB_EMBEDS_PATH, CHUNK_SIZE, CHUNK_OVERLAP,
)
from src.models.openai_client import get_embeddings_batch

logger = logging.getLogger(__name__)

EMBED_BATCH = 64   # chunks per OpenAI embed API call


# ── Text chunking ──────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 30]


def _read_file(fpath: Path) -> str:
    """Read text from a .txt or .pdf file."""
    if fpath.suffix.lower() == ".pdf":
        import pypdf
        reader = pypdf.PdfReader(str(fpath))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    return fpath.read_text(encoding="utf-8", errors="ignore")


def load_and_chunk(files: list[Path], chunk_size: int = CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Returns list of chunk dicts:
      { "text": str, "source": str, "chunk_id": int }
    """
    import re
    all_chunks = []
    for fpath in files:
        if not fpath.exists():
            logger.warning(f"Knowledge file not found: {fpath}")
            continue
        text = _read_file(fpath)
        # basic cleaning: collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        chunks = _chunk_text(text, chunk_size, overlap)
        for i, c in enumerate(chunks):
            all_chunks.append({
                "text": c,
                "source": fpath.name,
                "chunk_id": len(all_chunks),
            })
        logger.info(f"  {fpath.name}: {len(chunks)} chunks")
    return all_chunks


# ── Embedding & FAISS ──────────────────────────────────────────────────────────

def embed_chunks(chunks: list[dict], batch_size: int = EMBED_BATCH) -> np.ndarray:
    """Embed all chunks; returns float32 array of shape (N, D)."""
    texts = [c["text"] for c in chunks]
    n_batches = math.ceil(len(texts) / batch_size)
    all_embs = []
    for i in range(n_batches):
        batch = texts[i * batch_size: (i + 1) * batch_size]
        logger.info(f"  Embedding batch {i+1}/{n_batches} ({len(batch)} chunks)")
        embs = get_embeddings_batch(batch)
        all_embs.append(embs)
    return np.vstack(all_embs)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build inner-product index (cosine sim on L2-normalized vectors)."""
    embs = embeddings.copy()
    faiss.normalize_L2(embs)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index


def build_and_save(
    files: list[Path] | None = None,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> tuple[faiss.IndexFlatIP, list[dict]]:
    files = files or KNOWLEDGE_FILES
    KB_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading and chunking knowledge files...")
    chunks = load_and_chunk(files, chunk_size, overlap)
    logger.info(f"Total chunks: {len(chunks)}")

    logger.info("Embedding chunks...")
    embeddings = embed_chunks(chunks)

    logger.info("Building FAISS index...")
    index = build_faiss_index(embeddings)

    # Save
    faiss.write_index(index, str(KB_INDEX_PATH))
    with open(KB_CHUNKS_PATH, "w") as f:
        json.dump(chunks, f, indent=2)
    np.save(str(KB_EMBEDS_PATH), embeddings)

    logger.info(f"Saved index  → {KB_INDEX_PATH}")
    logger.info(f"Saved chunks → {KB_CHUNKS_PATH}")
    return index, chunks
