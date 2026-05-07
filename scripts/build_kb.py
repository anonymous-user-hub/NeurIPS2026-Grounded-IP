"""
Build the knowledge vector database from radiology tutorial text files.

This must be run ONCE before running any evaluation.

Usage:
    python scripts/build_kb.py

What it does:
  1. Loads the 3 radiology tutorial TXT files from RadKnowledge/
  2. Chunks each into overlapping ~400-char passages
  3. Embeds all passages using OpenAI text-embedding-3-small
  4. Builds a FAISS inner-product index (cosine similarity on normalized vectors)
  5. Saves to saved/knowledge_base/

Cost estimate: ~300-500 chunks × ~400 chars ≈ very low cost (<$0.01)
"""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)

from src.knowledge_base.build_db import build_and_save

if __name__ == "__main__":
    print("Building knowledge base from radiology tutorials...")
    index, chunks = build_and_save()
    print(f"\nDone! Index contains {index.ntotal} passages.")
    print("You can now run scripts/run_rexvqa.py")
