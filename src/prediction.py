"""
Prediction utilities — load the trained model & preprocessing pipeline,
transform raw input, and produce a prediction.

This module is intentionally UI-agnostic so that it can be reused in
batch scoring scripts, REST APIs, or unit tests.
"""

from pathlib import Path
from typing import Dict, Optional, Any

import joblib
import pandas as pd

from .config import MODEL_PATH, PIPELINE_PATH, SEVERITY_CLASSES


# -----------------------------------------------------------------------------
# Lazy-loading caches (Streamlit safe)
# -----------------------------------------------------------------------------
def load_model():
    """
    Load the trained classification model from disk.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}\n"
            "Please place your trained model as 'accident_severity_model.pkl' "
            "inside the 'models/' folder."
        )
    return joblib.load(MODEL_PATH)


def load_preprocessing_pipeline():
    """
    Load the fitted preprocessing pipeline or label encoder from disk.
    """
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessing pipeline file not found at: {PIPELINE_PATH}\n"
            "Please place your fitted pipeline as 'preprocessing_pipeline.pkl' "
            "inside the 'models/' folder."
        )
    return joblib.load(PIPELINE_PATH)


# -----------------------------------------------------------------------------
# Prediction entry-point
# -----------------------------------------------------------------------------
def run_prediction(model: Any,
                   pipeline: Any,
                   input_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Apply the preprocessing pipeline to `input_df` and produce a prediction.
    """
    if input_df is None or input_df.shape[0] == 0:
        raise ValueError("Input DataFrame is empty — cannot run prediction.")

    # ------------------------------------------------------------------
    # Apply preprocessing if a separate pipeline was provided.
    # ------------------------------------------------------------------
    try:
        if pipeline is not None and hasattr(pipeline, "transform"):
            # Try to transform the dataframe
            processed = pipeline.transform(input_df)
        else:
            processed = input_df
    except Exception:
        # If it fails (e.g. pipeline is a LabelEncoder that expects 1D array),
        # just use the raw input_df directly for the model.
        processed = input_df

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    try:
        predicted_label = model.predict(processed)[0]
    except Exception as e:
        raise ValueError(f"Model prediction failed: {e}") from e

    # ------------------------------------------------------------------
    # Map raw label → human-readable class name
    # ------------------------------------------------------------------
    if isinstance(predicted_label, str):
        predicted_class = predicted_label
    else:
        # Check if pipeline is actually a label encoder
        if pipeline is not None and hasattr(pipeline, "inverse_transform"):
            try:
                # inverse_transform expects a 2D array or list of labels
                predicted_class = pipeline.inverse_transform([predicted_label])[0]
            except Exception:
                # Fallback to index mapping if inverse_transform fails
                idx = int(predicted_label)
                predicted_class = SEVERITY_CLASSES[idx] if 0 <= idx < len(SEVERITY_CLASSES) else str(predicted_label)
        else:
            # Fallback to index mapping
            idx = int(predicted_label)
            predicted_class = SEVERITY_CLASSES[idx] if 0 <= idx < len(SEVERITY_CLASSES) else str(predicted_label)

    # ------------------------------------------------------------------
    # Probabilities (if supported)
    # ------------------------------------------------------------------
    probabilities: Optional[Dict[str, float]] = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(processed)[0]
            classes = getattr(model, "classes_", None)
            if classes is not None and len(classes) == len(proba):
                probabilities = {
                    (SEVERITY_CLASSES[c] if isinstance(c, int)
                     and 0 <= c < len(SEVERITY_CLASSES) else str(c)): float(p)
                    for c, p in zip(classes, proba)
                }
            else:
                probabilities = {
                    SEVERITY_CLASSES[i] if i < len(SEVERITY_CLASSES)
                    else f"Class_{i}": float(p)
                    for i, p in enumerate(proba)
                }
        except Exception:
            probabilities = None

    return {
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "probabilities": probabilities,
    }
