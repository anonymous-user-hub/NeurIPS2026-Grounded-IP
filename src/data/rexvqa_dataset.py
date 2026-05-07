"""
ReXVQA dataset loader.

Each item is a dict with keys:
  id, study_id, question, options, correct_answer, correct_answer_explanation,
  task_name, category, image_paths (list[Path]), metadata (dict)

ImagePath in JSON: "../deid_png/PATIENT/ACCESSION/studies/.../instances/XXX.png"
Resolved against: IMAGE_DIR (ReXGradient-160K/deid_png/)
"""
import json
import logging
from pathlib import Path
from typing import Iterator

from src.config import REXVQA_DIR, IMAGE_DIR

logger = logging.getLogger(__name__)

METADATA_KEYS = [
    "PatientID", "AccessionNumber", "PatientSex", "PatientAge",
    "Indication", "Comparison", "Findings", "Impression",
    "ImageViewPosition", "Manufacturer", "StudyDate",
]


def _resolve_image_path(rel_path: str) -> Path:
    """
    Convert relative path like "../deid_png/P/A/studies/.../X.png"
    to an absolute path under IMAGE_DIR.
    """
    # strip leading "../deid_png/"
    parts = Path(rel_path).parts
    # find "deid_png" index and take everything after it
    try:
        idx = parts.index("deid_png")
        rel = Path(*parts[idx + 1:])
    except ValueError:
        rel = Path(rel_path)
    return IMAGE_DIR / rel


def load_split(split: str = "valid") -> dict:
    """
    Load a ReXVQA split.

    Args:
        split: "train" | "valid" | "test"

    Returns:
        dict mapping item_id → item_dict
    """
    fname_map = {"train": "train_vqa_data.json", "valid": "valid_vqa_data.json",
                 "test": "test_vqa_data.json"}
    path = REXVQA_DIR / "metadata" / fname_map[split]
    with open(path) as f:
        raw = json.load(f)

    items = {}
    for item_id, item in raw.items():
        image_paths = []
        for rel in item.get("ImagePath", []):
            abs_p = _resolve_image_path(rel)
            if abs_p.exists():
                image_paths.append(abs_p)
            else:
                logger.debug(f"Image not found: {abs_p}")

        metadata = {k: item.get(k) for k in METADATA_KEYS}

        items[item_id] = {
            "id": item_id,
            "study_id": item.get("study_id", ""),
            "question": item.get("question", ""),
            "options": item.get("options", []),
            "correct_answer": item.get("correct_answer", ""),
            "correct_answer_explanation": item.get("correct_answer_explanation", ""),
            "task_name": item.get("task_name", ""),
            "category": item.get("category", ""),
            "subcategory": item.get("subcategory", ""),
            "image_paths": image_paths,
            "metadata": metadata,
        }
    return items


def iter_split(
    split: str = "valid",
    task_filter: str | None = None,
    max_items: int | None = None,
    require_image: bool = True,
) -> Iterator[dict]:
    """
    Iterate over ReXVQA items.

    Args:
        split: dataset split
        task_filter: if set, only yield items where task_name matches this string
        max_items: maximum number of items to yield
        require_image: skip items with no resolved image paths
    """
    items = load_split(split)
    count = 0
    for item in items.values():
        if require_image and not item["image_paths"]:
            continue
        if task_filter and task_filter.lower() not in item["task_name"].lower():
            continue
        yield item
        count += 1
        if max_items and count >= max_items:
            break


def get_primary_image(item: dict) -> Path | None:
    """
    Return the PA-view image if available, else the first image.
    """
    views = item["metadata"].get("ImageViewPosition") or []
    if isinstance(views, str):
        views = [views]
    paths = item["image_paths"]
    if not paths:
        return None
    for path, view in zip(paths, views):
        if isinstance(view, str) and "PA" in view.upper():
            return path
    return paths[0]
