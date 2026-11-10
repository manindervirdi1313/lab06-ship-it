---
title: QuickBite Late Order Check
emoji: 🚚
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Predict late food deliveries and recommend vouchers
---

# QuickBite Late Order Check

A machine learning application that predicts the probability of late food deliveries and recommends whether a customer voucher should be issued.

## What it does

The application uses a trained Logistic Regression model to estimate late-delivery probability from eight order and courier features. It recommends a voucher when the predicted probability reaches the 30% decision threshold.

Hugging Face Space: https://huggingface.co/spaces/Maninder3513/late-delivery-predictor

## Features

- Predicts late-delivery probability.
- Accepts eight delivery-related features.
- Validates numeric and categorical inputs.
- Supports missing preparation-time values.
- Provides a voucher recommendation.
- Interactive Gradio interface.

## Model information

- Algorithm: Logistic Regression
- Scikit-learn version: 1.9.1
- Random state: 42
- Decision threshold: 0.30

## Input features

| Feature | Accepted values |
|---|---|
| distance_km | 0.1–30 |
| items | Integer, 1–20 |
| hour | Integer, 10–22 |
| prep_minutes | 1–60 or empty |
| courier_months | 0–120 |
| weather | clear, rain, snow |
| day_type | weekday, weekend |
| vehicle | bike, scooter, car |

## Model evaluation

| Metric | Result |
|---|---:|
| Accuracy | 84.13% |
| Precision | 49.07% |
| Recall | 45.30% |
| F1 Score | 47.11% |
| ROC-AUC | 79.62% |

The model's estimated policy cost was $1,392, compared with $1,755 when flagging nobody and $3,000 when flagging everybody.

## How to run locally

Install dependencies:

```bash
pip install -r requirements-dev.txt
```

Launch:

```bash
gradio app.py
```

Run tests:

```bash
python -m pytest -v
```

Check code quality:

```bash
python -m ruff format --check .
python -m ruff check .
```

## Project structure

- `app.py` – Gradio interface
- `predict.py` – Prediction and validation logic
- `models/` – Trained model and metadata
- `tests/` – Automated tests
- `requirements.txt` – Runtime dependencies
- `requirements-dev.txt` – Development dependencies

## Group members and contributions

Document each member's actual contribution before submission:

- Eric Hammond – [Add verified contribution]
- Maninder Singh Virdi – [Add verified contribution]
- Tochukwu Adaobi Okafor – [Add verified contribution]
- Ugochi Igwegbe – [Add verified contribution]

## Limitations

The model was trained on a specific delivery dataset and may not generalize to all restaurants, locations, couriers, or customer populations. Its predictions are estimates, not guarantees. Voucher recommendations should be reviewed alongside business policies and operational context.

## Tests

Automated tests cover prediction outputs, decision thresholds, input validation, and deployment configuration.
