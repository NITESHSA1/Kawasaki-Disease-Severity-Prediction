# Kawasaki Disease Severity Prediction

End-to-end ML pipeline to predict Kawasaki disease severity (Mild / Severe / Unknown) using MGWO feature selection, XGBoost and AdaBoost, with a Flask web UI and API.

## Overview

This repository contains an end-to-end machine learning pipeline and Flask web application for predicting Kawasaki disease severity (Mild, Severe, Unknown) using MGWO-based feature selection with XGBoost and AdaBoost classifiers on a clinical dataset. The workflow is organized into Python "sessions":

- **SESSION_2**: Data cleaning & preprocessing (`SESSION_2.py`)
- **SESSION_3**: Feature matrix preparation (`SESSION_3.py`)
- **SESSION_4**: MGWO feature selection (`SESSION_4.py`)
- **SESSION_5**: Train XGBoost & AdaBoost models (`SESSION_5.py`)
- **SESSION_6**: Compute ensemble weights (`SESSION_6.py`)
- **SESSION_7**: Evaluation (confusion matrices, ROC) (`SESSION_7.py`)
- **SESSION_8**: Final models for deployment (`SESSION_8.py`)
- **app.py**: Flask API + web UI

## Workflow Description

`SESSION_2.py` performs data cleaning and preprocessing of `KAWASAKI_ORIGINAL.csv` (handling missing values, encoding categorical variables, scaling, and outlier detection) and saves cleaned outputs for later use.

`SESSION_3.py` builds the main feature matrix and train/test splits, while `SESSION_4.py` applies Modified Grey Wolf Optimizer (MGWO) to select an optimal subset of features and generates a reduced feature matrix.

`SESSION_5.py` trains individual XGBoost and AdaBoost models on the reduced features, computes classification metrics, and stores a comparison table; `SESSION_6.py` uses validation F1 scores to compute ensemble weights for XGBoost and AdaBoost and saves them to a CSV file.

`SESSION_7.py` evaluates the individual models and the weighted ensemble on a held-out test set, exporting confusion matrix images and multiclass ROC curves for each model to dedicated folders in the project directory.

`SESSION_8.py` retrains final deployment models using the same data splits, saves the trained estimators (`KD_final_xgb_model.joblib`, `KD_final_ada_model.joblib`), the label encoder, and normalized ensemble weights (`KD_final_ensemble_weights_XGB_ADA.npz`), and prints the feature order required for inference.

## Installation

All scripts use project-relative paths via `BASE_DIR = os.path.dirname(os.path.abspath(__file__))`, so you only need to place `KAWASAKI_ORIGINAL.csv` in the repository root.

```bash
git clone https://github.com/NITESHSA1/Kawasaki-Disease-Severity-Prediction.git
cd Kawasaki-Disease-Severity-Prediction
python -m venv venv
venv\\Scripts\\activate   # on Windows (use source venv/bin/activate on Linux/Mac)
pip install -r requirements.txt
```

## Running the Pipeline

1. Place `KAWASAKI_ORIGINAL.csv` in the root folder.
2. Run the sessions in order:

```bash
python SESSION_2.py  # cleaning & preprocessing
python SESSION_3.py  # feature matrix
python SESSION_4.py  # MGWO feature selection
python SESSION_5.py  # base models
python SESSION_6.py  # ensemble weights
python SESSION_7.py  # evaluation & plots
python SESSION_8.py  # final models for deployment
```

## Running the Flask Web App

After `SESSION_8.py` has created the final models:

```bash
python app.py
```

- **Web UI**: http://127.0.0.1:5000/ui
- **API docs/health**: http://127.0.0.1:5000/

### API Example

Send a POST request to `http://127.0.0.1:5000/predict` with a JSON body:

```json
{
  "features": {
    "feature1": 0.5,
    "feature2": 0.3,
    ...
  }
}
```

Response:

```json
{
  "severity": "Mild",
  "confidence": 0.85,
  "probabilities": {
    "Mild": 0.85,
    "Severe": 0.10,
    "Unknown": 0.05
  },
  "models": {
    "xgb": "Mild",
    "ada": "Mild"
  }
}
```

## Repository Structure

- `SESSION_2.py` – data cleaning & preprocessing
- `SESSION_3.py` – feature matrix creation
- `SESSION_4.py` – MGWO feature selection
- `SESSION_5.py` – XGBoost and AdaBoost training
- `SESSION_6.py` – ensemble weights computation
- `SESSION_7.py` – evaluation, confusion matrices, ROC curves
- `SESSION_8.py` – final training & saving artifacts
- `app.py` – Flask API and web UI
- `requirements.txt` – Python dependencies
- `LICENSE` – MIT License

## Project Results

After running all sessions, the pipeline generates:
- Cleaned and processed datasets
- MGWO-reduced feature matrix
- Trained XGBoost and AdaBoost models
- Ensemble weights based on validation F1 scores
- Confusion matrices and ROC curves
- Final deployment-ready models

## NOTICE

Copyright © 2026 NITESHSA1. All rights reserved.

This repository is provided for personal learning and academic purposes only.
You may read and study the code, but you may **not** copy, modify, reuse, or distribute any part of this code in your own projects, publications, or repositories without prior written permission from the author.

Any unauthorized use, reproduction, or distribution of this code is strictly prohibited and may violate applicable laws.

## License

MIT License - See LICENSE file for details. The MIT License grants permission for use and modification with attribution, but the NOTICE above adds restrictions for academic integrity.

---

**Anurag Institutions ML Project** | Powered by Flask & scikit-learn
