"""
Query set loader: reads the refined image-only query list.
Falls back to the full 520 list if the refined file doesn't exist.
"""
from pathlib import Path
from src.config import QUERY_FILE, REFINED_QUERY_FILE


def load_queries(refined: bool = True) -> list[str]:
    path = REFINED_QUERY_FILE if (refined and REFINED_QUERY_FILE.exists()) else QUERY_FILE
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                queries.append(line)
    return queries


def format_query_as_question(finding: str) -> str:
    """
    Convert a short finding phrase into a binary yes/no question.
    e.g. "focal consolidation" → "Is there focal consolidation visible in the chest X-ray?"
    """
    return f"Is there {finding} visible in this chest X-ray?"
