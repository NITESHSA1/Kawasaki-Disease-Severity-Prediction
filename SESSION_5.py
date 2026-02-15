# ==========================================
# SESSION 5: Individual Classifier Models
# XGBoost + AdaBoost on MGWO-Reduced Features
# ==========================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier
import joblib

print("=" * 60)
print("SESSION_5 STARTED ✅")
print("=" * 60)

# ---------- 1. Paths & data loading ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

reduced_path = os.path.join(BASE_DIR, "KD_MGWO_REDUCED_MATRIX.csv")

print(f"\n[INFO] Loading reduced matrix from:\n{reduced_path}")
df = pd.read_csv(reduced_path)
print(f"[INFO] Reduced matrix shape: {df.shape}")
print(f"[INFO] Columns: {df.columns.tolist()}")

target_col = "Long-Term Effects"
if target_col not in df.columns:
    raise ValueError(f"Column '{target_col}' not found in KD_MGWO_REDUCED_MATRIX.csv")

# ---------- 2. Split X, y ----------
X = df.drop(columns=[target_col])
y_text = df[target_col].values

print("\n[INFO] Encoding target labels...")
le = LabelEncoder()
y = le.fit_transform(y_text)
classes = le.classes_
print(f"[INFO] Classes encoded as: {dict(zip(classes, range(len(classes))))}")

print("\n[INFO] Performing train/test split (80/20, stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

average_for_auc = "macro"

def evaluate_model(name, y_true, y_pred, y_proba):
    print(f"\n[INFO] Evaluating {name}...")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(
            y_true,
            y_proba,
            multi_class="ovr",
            average=average_for_auc
        )
    except ValueError:
        auc = np.nan

    print(f"\n{name} - Classification Report:")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

    print(f"[RESULT] {name} -> "
          f"Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, "
          f"F1: {f1:.4f}, AUC: {auc:.4f}")

    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "AUC": auc
    }

results = []

# =========================================================
# 3. XGBoost Classifier with Regularization
# =========================================================
print("\n" + "=" * 60)
print("[STEP] Training XGBoost (this may take 1–3 minutes)...")
print("=" * 60)

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
    eval_set=[(X_test, y_test)],
    verbose=True
)

print("[INFO] XGBoost training finished. Predicting...")
y_pred_xgb = xgb_clf.predict(X_test)
y_proba_xgb = xgb_clf.predict_proba(X_test)

res_xgb = evaluate_model("XGBoost", y_test, y_pred_xgb, y_proba_xgb)
results.append(res_xgb)

xgb_model_path = os.path.join(BASE_DIR, "KD_XGBoost_model.json")
xgb_clf.save_model(xgb_model_path)
print(f"[INFO] XGBoost model saved to: {xgb_model_path}")

# =========================================================
# 4. AdaBoost Classifier (no 'algorithm' argument)
# =========================================================
print("\n" + "=" * 60)
print("[STEP] Training AdaBoost...")
print("=" * 60)

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

print("[INFO] AdaBoost training finished. Predicting...")
y_pred_ada = ada_clf.predict(X_test)
y_proba_ada = ada_clf.predict_proba(X_test)

res_ada = evaluate_model("AdaBoost", y_test, y_pred_ada, y_proba_ada)
results.append(res_ada)

ada_model_path = os.path.join(BASE_DIR, "KD_AdaBoost_model.pkl")
joblib.dump(ada_clf, ada_model_path)
print(f"[INFO] AdaBoost model saved to: {ada_model_path}")

# =========================================================
# 5. Performance comparison table
# =========================================================
results_df = pd.DataFrame(results)
comp_path = os.path.join(BASE_DIR, "KD_Week5_model_comparison.csv")
results_df.to_csv(comp_path, index=False)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 60)
print(results_df)

print(f"\n[INFO] Comparison table saved to: {comp_path}")
print("\nSESSION 5 COMPLETED ✅")
print("=" * 60)
