---
title: Late Delivery Prediction System
sdk: gradio
sdk_version: 6.31.0
python_version: "3.11"
app_file: app.py
pinned: false
---



\# Late Delivery Prediction System



A machine learning application that predicts the probability of late food deliveries and recommends whether a customer voucher should be issued.



\## Features



\- Predicts late-delivery probability.

\- Uses a trained Logistic Regression model.

\- Accepts eight delivery-related features.

\- Handles missing preparation-time values through model preprocessing.

\- Provides an automated voucher recommendation.

\- Interactive Gradio web interface.



\## Model Information



\- Algorithm: Logistic Regression

\- Scikit-learn version: 1.9.1

\- Random state: 42

\- Decision threshold: 0.30



\## Input Features



1\. Distance (km)

2\. Number of items

3\. Order hour

4\. Preparation time (minutes)

5\. Courier experience (months)

6\. Weather

7\. Day type

8\. Vehicle



\## Business Decision Rule



A voucher is recommended when the predicted late-delivery probability is greater than or equal to 30%.



\## Model Evaluation



| Metric | Result |

|---|---:|

| Accuracy | 84.13% |

| Precision | 49.07% |

| Recall | 45.30% |

| F1 Score | 47.11% |

| ROC-AUC | 79.62% |



\## Project Structure



\- `app.py` – Gradio application

\- `predict.py` – Prediction engine

\- `models/` – Trained model and metadata

\- `tests/` – Automated tests

\- `requirements.txt` – Runtime dependencies

\- `requirements-dev.txt` – Development dependencies

\- `environment.yml` – Conda environment

\- `pyproject.toml` – Testing and linting configuration



\## Run Locally



Install dependencies:



```bash

pip install -r requirements.txt

```



Launch:



```bash

python app.py

```



Run tests:



```bash

pytest -v

```

