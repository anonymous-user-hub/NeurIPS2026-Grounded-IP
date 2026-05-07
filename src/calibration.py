"""Load per-model calibrated temperatures from saved/calibration/."""
import json, re
from pathlib import Path

ROOT = Path(__file__).parent.parent
_CALIBRATION_DIR = ROOT / "saved" / "calibration"
_CONSOLIDATED = _CALIBRATION_DIR / "temperatures_consolidated.json"
_CACHE: dict[str, dict[str, float]] = {}

def load_temperatures(model: str) -> dict[str, float]:
    """Return {predictor_T, answerer_T} for model; falls back to 1.0 if not found."""
    if model in _CACHE:
        return _CACHE[model]
    result = {"predictor_T": 1.0, "answerer_T": 1.0}
    if _CONSOLIDATED.exists():
        data = json.loads(_CONSOLIDATED.read_text())
        if model in data:
            result = {
                "predictor_T": float(data[model].get("predictor_T", 1.0)),
                "answerer_T":  float(data[model].get("answerer_T",  1.0)),
            }
            _CACHE[model] = result
            return result
    # fall back to per-run folders
    safe = re.sub(r"[^A-Za-z0-9._-]", "--", model)
    matches = sorted(_CALIBRATION_DIR.glob(f"*calibrate-{safe}"))
    if matches:
        f = matches[-1] / "temperatures.json"
        if f.exists():
            data = json.loads(f.read_text())
            result = {
                "predictor_T": float(data.get("predictor_T", 1.0)),
                "answerer_T":  float(data.get("answerer_T",  1.0)),
            }
    _CACHE[model] = result
    return result
