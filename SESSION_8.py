# ==========================================
# SESSION 8: Deployment & Interpretation Prep
# (Save models + simple prediction function)
# ==========================================

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# ------------------ 1. Configuration ------------------
# Use project folder (script location) instead of hardcoded path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "KD_MGWO_REDUCED_MATRIX.csv")
WEIGHTS_SESSION6_PATH = os.path.join(BASE_DIR, "KD_Ensemble_Weights_Session6.csv")

# Output model paths
XGB_MODEL_PATH = os.path.join(BASE_DIR, "KD_final_xgb_model.joblib")
ADA_MODEL_PATH = os.path.join(BASE_DIR, "KD_final_ada_model.joblib")
ENCODER_PATH = os.path.join(BASE_DIR, "KD_final_label_encoder.joblib")
WEIGHTS_PATH = os.path.join(BASE_DIR, "KD_final_ensemble_weights_XGB_ADA.npz")

def train_final_models():
    print("SESSION 8 STARTED")
    print("=" * 60)
    print("Deployment: Training final models and saving artifacts")
    print("=" * 60)

    print(f"[INFO] Loading data from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Data file not found: {DATA_PATH}")
        return []

    df = pd.read_csv(DATA_PATH)
    target_col = "Long-Term Effects"

    if target_col not in df.columns:
        print(f"[ERROR] Target column '{target_col}' not found in data.")
        return []

    X = df.drop(columns=[target_col])
    y_text = df[target_col].values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_text)
    classes = le.classes_
    print(f"[INFO] Encoded classes: {dict(zip(range(len(classes)), classes))}")

    # Save label encoder
    joblib.dump(le, ENCODER_PATH)
    print(f"[INFO] Saved label encoder to: {ENCODER_PATH}")

    # Train/val/test split (same as previous sessions)
    print("\n[INFO] Splitting into Train/Val/Test (60/20/20)...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X.values, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, stratify=y_train_full, random_state=42
    )
    print(f"[INFO] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ------------- XGBoost -------------
    print("\n[STEP] Training final XGBoost model...")
    sample_weights_xgb = compute_sample_weight(class_weight="balanced", y=y_train)
    xgb_clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.5,
        objective="multi:softprob",
        num_class=len(classes),
        eval_metric=["mlogloss", "merror"],
        random_state=42,
        tree_method="hist"
    )
    xgb_clf.fit(
        X_train,
        y_train,
        sample_weight=sample_weights_xgb,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    joblib.dump(xgb_clf, XGB_MODEL_PATH)
    print(f"[INFO] Saved XGBoost model to: {XGB_MODEL_PATH}")

    # ------------- AdaBoost -------------
    print("\n[STEP] Training final AdaBoost model...")
    base_tree = DecisionTreeClassifier(
        max_depth=2,
        min_samples_leaf=10,
        random_state=42
    )
    ada_clf = AdaBoostClassifier(
        estimator=base_tree,
        n_estimators=200,
        learning_rate=0.8,
        random_state=42
    )
    sample_weights_ada = compute_sample_weight(class_weight="balanced", y=y_train)
    ada_clf.fit(X_train, y_train, sample_weight=sample_weights_ada)
    joblib.dump(ada_clf, ADA_MODEL_PATH)
    print(f"[INFO] Saved AdaBoost model to: {ADA_MODEL_PATH}")

    # ------------- Ensemble weights (XGB + ADA) -------------
    print("\n[INFO] Loading Session 6 ensemble weights...")
    if not os.path.exists(WEIGHTS_SESSION6_PATH):
        print(f"[ERROR] Could not find: {WEIGHTS_SESSION6_PATH}")
        print("[HINT] Make sure KD_Ensemble_Weights_Session6.csv is in the same folder.")
        print("SESSION 8 STOPPED BEFORE SAVING WEIGHTS.")
        print("=" * 60)
        return list(X.columns)

    w_df = pd.read_csv(WEIGHTS_SESSION6_PATH)
    print("[INFO] Read Session 6 weights:")
    print(w_df)

    w_xgb_raw = float(w_df.loc[w_df["Model"] == "XGBoost", "Weight"].values[0])
    w_ada_raw = float(w_df.loc[w_df["Model"] == "AdaBoost", "Weight"].values[0])
    denom = w_xgb_raw + w_ada_raw
    w_xgb = w_xgb_raw / denom
    w_ada = w_ada_raw / denom

    print(f"[DEBUG] Raw weights: XGB={w_xgb_raw}, ADA={w_ada_raw}, denom={denom}")
    np.savez(WEIGHTS_PATH, w_xgb=w_xgb, w_ada=w_ada)
    print(f"[INFO] Saved normalized ensemble weights (XGB+ADA) to: {WEIGHTS_PATH}")

    print("\nSESSION 8 TRAINING & SAVING COMPLETED ✅")
    print("=" * 60)

    # Return column order for prediction use
    feature_names = list(X.columns)
    print(f"[INFO] Feature order for deployment: {feature_names}")
    return feature_names

def load_artifacts():
    """Load models, encoder, and weights for inference."""
    if not os.path.exists(XGB_MODEL_PATH):
        raise FileNotFoundError(f"XGBoost model not found at {XGB_MODEL_PATH}")
    if not os.path.exists(ADA_MODEL_PATH):
        raise FileNotFoundError(f"AdaBoost model not found at {ADA_MODEL_PATH}")
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Label encoder not found at {ENCODER_PATH}")
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Ensemble weights file not found at {WEIGHTS_PATH}")

    xgb_clf = joblib.load(XGB_MODEL_PATH)
    ada_clf = joblib.load(ADA_MODEL_PATH)
    le = joblib.load(ENCODER_PATH)

    data = np.load(WEIGHTS_PATH)
    w_xgb = float(data["w_xgb"])
    w_ada = float(data["w_ada"])

    return xgb_clf, ada_clf, le, w_xgb, w_ada

def predict_single_patient(feature_values, feature_names):
    """
    feature_values: list or 1D array of length = n_features
    feature_names: list of column names in the same order as KD_MGWO_REDUCED_MATRIX (without target).

    Returns: predicted_class_name, proba_dict
    """
    xgb_clf, ada_clf, le, w_xgb, w_ada = load_artifacts()

    # Ensure numpy array shape (1, n_features)
    x_array = np.array(feature_values, dtype=float).reshape(1, -1)

    # Get probabilities from each model
    p_xgb = xgb_clf.predict_proba(x_array)
    p_ada = ada_clf.predict_proba(x_array)

    # Weighted ensemble
    p_ens = w_xgb * p_xgb + w_ada * p_ada
    pred_idx = int(np.argmax(p_ens, axis=1)[0])
    pred_class = le.inverse_transform([pred_idx])[0]

    # Build probability dict: {class_name: prob}
    class_probs = {}
    for idx, class_name in enumerate(le.classes_):
        class_probs[class_name] = float(p_ens[0, idx])

    print("\n[INFERENCE] Single-patient prediction")
    print("-------------------------------------")
    print("Input features (ordered):")
    for name, val in zip(feature_names, feature_values):
        print(f"  {name}: {val}")
    print(f"\nPredicted class: {pred_class}")
    print("Class probabilities:")
    for cls, prob in class_probs.items():
        print(f"  {cls}: {prob:.4f}")

    return pred_class, class_probs

if __name__ == "__main__":
    # 1) Train and save models + encoder + weights
    feature_names = train_final_models()

    # If weights saved correctly, WEIGHTS_PATH will exist
    if os.path.exists(WEIGHTS_PATH):
        # 2) Example: dummy single-patient prediction (replace with real values)
        # Make sure the length matches len(feature_names)
        example_values = [0] * len(feature_names)
        predict_single_patient(example_values, feature_names)
    else:
        print("\n[WARN] Ensemble weights file not found; skipping demo prediction.")
        print(f"[CHECK] Expected weights at: {WEIGHTS_PATH}")
