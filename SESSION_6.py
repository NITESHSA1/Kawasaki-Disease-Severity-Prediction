# ==========================================
# SESSION 6: Ensemble Weights (XGBoost + AdaBoost only)
# ==========================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# ------------------ 1. Configuration ------------------
# Project root = folder where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "KD_MGWO_REDUCED_MATRIX.csv")
WEIGHTS_OUT_PATH = os.path.join(BASE_DIR, "KD_Ensemble_Weights_Session6.csv")
METRICS_OUT_PATH = os.path.join(BASE_DIR, "KD_Session6_metrics_XGB_ADA.csv")

def load_data():
    print("SESSION 6 STARTED")
    print("=" * 60)
    print(f"[INFO] Loading data from: {DATA_PATH}")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    target_col = "Long-Term Effects"
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in data.")

    X = df.drop(columns=[target_col])
    y_text = df[target_col].values

    # Label encode targets
    le = LabelEncoder()
    y = le.fit_transform(y_text)
    classes = le.classes_
    print(f"[INFO] Encoded classes: {dict(zip(range(len(classes)), classes))}")

    return X, y, le, classes

def split_data(X, y):
    print("\n[INFO] Splitting into Train/Val/Test (60/20/20)...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X.values, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, stratify=y_train_full, random_state=42
    )
    print(f"[INFO] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_xgboost(X_train, y_train, X_val, y_val, num_classes):
    print("\n[STEP] Training XGBoost...")
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    xgb_clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.5,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric=["mlogloss", "merror"],
        random_state=42,
        tree_method="hist"
    )

    xgb_clf.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return xgb_clf

def train_adaboost(X_train, y_train):
    print("\n[STEP] Training AdaBoost...")
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

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    ada_clf.fit(X_train, y_train, sample_weight=sample_weights)

    return ada_clf

def evaluate_model(name, model, X_val, y_val):
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro")
    prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_val, y_pred, average="macro", zero_division=0)

    print(f"\n[RESULTS] {name} on Validation Set")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")

    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    }

def compute_and_save_weights(metrics_xgb, metrics_ada):
    print("\n[STEP] Computing ensemble weights from validation F1-scores...")

    f1_xgb = metrics_xgb["F1"]
    f1_ada = metrics_ada["F1"]

    raw_xgb = f1_xgb
    raw_ada = f1_ada
    denom = raw_xgb + raw_ada

    if denom == 0:
        # Fallback: equal weights if something went wrong
        w_xgb = 0.5
        w_ada = 0.5
    else:
        w_xgb = raw_xgb / denom
        w_ada = raw_ada / denom

    print(f"[INFO] Raw F1 scores: XGB={raw_xgb:.4f}, ADA={raw_ada:.4f}")
    print(f"[INFO] Normalized weights: w_XGB={w_xgb:.4f}, w_ADA={w_ada:.4f}")

    w_df = pd.DataFrame({
        "Model": ["XGBoost", "AdaBoost"],
        "Weight": [w_xgb, w_ada]
    })
    w_df.to_csv(WEIGHTS_OUT_PATH, index=False)
    print(f"[INFO] Saved ensemble weights to: {WEIGHTS_OUT_PATH}")

    return w_df

def save_metrics_table(metrics_xgb, metrics_ada):
    print("\n[STEP] Saving Session 6 metrics table...")
    metrics_df = pd.DataFrame([metrics_xgb, metrics_ada])
    metrics_df.to_csv(METRICS_OUT_PATH, index=False)
    print(f"[INFO] Saved metrics to: {METRICS_OUT_PATH}")

def main():
    # 1) Load data and split
    X, y, le, classes = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 2) Train models
    xgb_clf = train_xgboost(X_train, y_train, X_val, y_val, num_classes=len(classes))
    ada_clf = train_adaboost(X_train, y_train)

    # 3) Evaluate on validation set
    metrics_xgb = evaluate_model("XGBoost", xgb_clf, X_val, y_val)
    metrics_ada = evaluate_model("AdaBoost", ada_clf, X_val, y_val)

    # 4) Compute and save ensemble weights (XGB + ADA only)
    compute_and_save_weights(metrics_xgb, metrics_ada)

    # 5) Save metrics CSV
    save_metrics_table(metrics_xgb, metrics_ada)

    print("\nSESSION 6 COMPLETED ✅")
    print("=" * 60)

if __name__ == "__main__":
    main()
