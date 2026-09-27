"""
Feature engineering utilities.

These functions replicate the time-derived features that were
created during model training. They are kept separate from the UI
so that the same logic can be reused in training notebooks and
in production inference.
"""

from datetime import datetime
from typing import Dict

import pandas as pd

from .config import EXPECTED_FEATURES, time_period_from_hour


def derive_time_features(time_str: str) -> Dict[str, object]:
    """
    Given a 'HH:MM' string, return the four engineered time features.

    Parameters
    ----------
    time_str : str
        Time in 'HH:MM' format (e.g. '14:30').

    Returns
    -------
    dict
        {
            "Hour": int,
            "Minute": int,
            "Minutes_since_midnight": int,
            "Time_period": str,
        }

    Raises
    ------
    ValueError
        If `time_str` is not in a valid 'HH:MM' format.
    """
    if not time_str or not isinstance(time_str, str):
        raise ValueError("Time input must be a non-empty 'HH:MM' string.")

    try:
        t = datetime.strptime(time_str, "%H:%M")
    except ValueError:
        # Try the 'HH:MM:SS' variant as a fallback
        try:
            t = datetime.strptime(time_str, "%H:%M:%S")
        except ValueError as e:
            raise ValueError(
                f"Invalid time format: '{time_str}'. Expected 'HH:MM'."
            ) from e

    hour = t.hour
    minute = t.minute
    minutes_since_midnight = hour * 60 + minute
    time_period = time_period_from_hour(hour)

    return {
        "Hour": hour,
        "Minute": minute,
        "Minutes_since_midnight": minutes_since_midnight,
        "Time_period": time_period,
    }


def build_input_dataframe(raw_input: Dict[str, object],
                          time_features: Dict[str, object]) -> pd.DataFrame:
    """
    Combine raw user inputs with derived time features and return a
    single-row DataFrame whose columns match `EXPECTED_FEATURES`.

    Parameters
    ----------
    raw_input : dict
        All raw categorical/numeric values collected from the UI.
    time_features : dict
        The four engineered time features returned by `derive_time_features`.

    Returns
    -------
    pd.DataFrame
        One-row DataFrame whose columns exactly match `EXPECTED_FEATURES`.

    Raises
    ------
    ValueError
        If any expected feature is missing from the combined dict.
    """
    combined = {**raw_input, **time_features}

    # Remove the raw 'Time' string column — the model expects engineered features only
    combined.pop("Time", None)

    missing = [f for f in EXPECTED_FEATURES if f not in combined]
    if missing:
        raise ValueError(
            "Missing required features for model input: " + ", ".join(missing)
        )

    # Build the row in the exact order expected by the model
    row = {feature: combined[feature] for feature in EXPECTED_FEATURES}
    return pd.DataFrame([row])