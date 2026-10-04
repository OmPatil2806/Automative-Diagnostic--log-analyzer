"""Load the trained anomaly detector once per server process."""

from datetime import datetime
from pathlib import Path

import streamlit as st

from analyzer.detection.anomaly import AnomalyDetector
from analyzer.pipeline import DEFAULT_MODEL_PATH


class ModelNotFoundError(FileNotFoundError):
    pass


@st.cache_resource(show_spinner="Loading the anomaly detection model...")
def load_detector(path: str = str(DEFAULT_MODEL_PATH)) -> AnomalyDetector:
    """The trained detector, loaded once and shared by all pages and users."""
    if not Path(path).exists():
        raise ModelNotFoundError(f"No trained model at {path}. Run `python scripts/train_model.py`.")
    return AnomalyDetector.load(Path(path))


def model_info(path: str = str(DEFAULT_MODEL_PATH)) -> dict:
    """Facts about the loaded model, for display."""
    detector = load_detector(path)
    file = Path(path)
    return {
        "file": file.name,
        "size_mb": round(file.stat().st_size / 1_000_000, 1),
        "modified": datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d"),
        "detectors": list(detector.models),
        "vehicles_with_baseline": len(set(detector.baseline.maf_k) | set(detector.baseline.stft_offset)
                                      | set(detector.baseline.battery_coef)),
        "alarm_rule": f"{detector.persistence_ratio:.0%} of the last {detector.persistence_window} readings",
    }
