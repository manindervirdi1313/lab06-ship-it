"""Prediction engine for late food deliveries."""

import json
from pathlib import Path

import joblib
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "late_orders.joblib"
INFO_PATH = BASE_DIR / "models" / "model_info.json"


# Load model and metadata once
model = joblib.load(MODEL_PATH)

with INFO_PATH.open("r", encoding="utf-8") as file:
    model_info = json.load(file)


# Model configuration
THRESHOLD = float(model_info["threshold"])
FEATURES = (
    model_info["numeric_features"]
    + model_info["categorical_features"]
)

LATE_CLASS = 1

if list(model.feature_names_in_) != [
    "distance_km",
    "items",
    "hour",
    "prep_minutes",
    "courier_months",
    "weather",
    "day_type",
    "vehicle",
]:
    raise ValueError("Model feature configuration does not match.")

if LATE_CLASS not in model.classes_:
    raise ValueError("Late-delivery class 1 is missing from the model.")


def predict_order(
    distance_km,
    items,
    hour,
    prep_minutes,
    courier_months,
    weather,
    day_type,
    vehicle,
):
    """Predict late-delivery probability and voucher eligibility."""

    # Validate numeric inputs
    numeric_values = {
        "distance_km": distance_km,
        "items": items,
        "hour": hour,
        "courier_months": courier_months,
    }

    if prep_minutes is not None:
        numeric_values["prep_minutes"] = prep_minutes

    for name, value in numeric_values.items():
        if isinstance(value, bool):
            raise ValueError(f"{name} must be numeric.")

        try:
            numeric_values[name] = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be numeric.") from None

        if not pd.notna(numeric_values[name]):
            raise ValueError(f"{name} must be a valid number.")

    if numeric_values["distance_km"] <= 0:
        raise ValueError("Distance must be greater than zero.")

    if numeric_values["items"] < 1:
        raise ValueError("Items must be at least 1.")

    if not numeric_values["items"].is_integer():
        raise ValueError("Items must be a whole number.")

    if not numeric_values["courier_months"] >= 0:
        raise ValueError("Courier months cannot be negative.")

    if not 0 <= numeric_values["hour"] <= 23:
        raise ValueError("Hour must be between 0 and 23.")

    if prep_minutes is not None:
        if numeric_values["prep_minutes"] < 0:
            raise ValueError("Preparation time cannot be negative.")

    # Validate categorical inputs
    categorical_values = {
        "weather": weather,
        "day_type": day_type,
        "vehicle": vehicle,
    }

    for name, value in categorical_values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string.")

    # Build input DataFrame in exact model feature order
    input_data = {
        "distance_km": numeric_values["distance_km"],
        "items": int(numeric_values["items"]),
        "hour": numeric_values["hour"],
        "prep_minutes": (
            numeric_values.get("prep_minutes")
        ),
        "courier_months": numeric_values["courier_months"],
        "weather": weather.strip(),
        "day_type": day_type.strip(),
        "vehicle": vehicle.strip(),
    }

    input_df = pd.DataFrame([input_data], columns=[
        "distance_km",
        "items",
        "hour",
        "prep_minutes",
        "courier_months",
        "weather",
        "day_type",
        "vehicle",
    ])

    # Calculate late-delivery probability
    probabilities = model.predict_proba(input_df)[0]

    late_index = list(model.classes_).index(LATE_CLASS)
    late_probability = float(probabilities[late_index])

    # Apply business decision threshold
    is_late = late_probability >= THRESHOLD

    voucher_decision = (
        "Issue voucher" if is_late else "No voucher"
    )

    return {
        "late_probability": round(late_probability, 4),
        "late_percentage": round(late_probability * 100, 2),
        "prediction": "Late delivery" if is_late else "On-time delivery",
        "voucher_decision": voucher_decision,
        "threshold": THRESHOLD,
    }