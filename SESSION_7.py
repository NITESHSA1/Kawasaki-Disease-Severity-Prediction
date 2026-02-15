# ==========================================
# SESSION 7: Model Evaluation & ROC Analysis
# (XGBoost + AdaBoost only)
# ==========================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier

# ---------- Helpers ----------

def compute_metrics(name, y_true, y_pred, y_proba, classes):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        auc_macro = roc_auc_score(
            y_true,
            y_proba,
            multi_class="ovr",
            average="macro"
        )
    except ValueError:
        auc_macro = np.nan

    print(f"\n{name} - Classification Report")
    print("--------------------------------")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    print(f"[RESULT] {name} -> Acc: {acc:.4f}, Prec: {prec:.4f}, "
          f"Rec: {rec:.4f}, F1: {f1:.4f}, AUC (macro): {auc_macro:.4f}")

    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "AUC": auc_macro
    }

def plot_and_save_cm(y_true, y_pred, classes, title, filename, out_dir):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved confusion matrix: {save_path}")

def plot_multiclass_roc(y_true_bin, y_proba, classes, title, filename, out_dir):
    fpr = {}
    tpr = {}
    roc_auc_vals = {}
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc_vals[i] = auc(fpr[i], tpr[i])

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, class_name in enumerate(classes):
        ax.plot(
            fpr[i],
            tpr[i],
            lw=2,
            label=f"Class {class_name} (AUC = {roc_auc_vals[i]:.2f})"
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved ROC curves: {save_path}")

def main():
    print("SESSION 7 STARTED")
    print("=" * 60)
    print("Model Evaluation, ROC & Confusion Matrix Analysis (XGB + ADA)")
    print("=" * 60)

    # ------------------ 1. Paths & data ------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(BASE_DIR, "KD_MGWO_REDUCED_MATRIX.csv")
    print(f"[INFO] Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    target_col = "Long-Term Effects"
    X = df.drop(columns=[target_col])
    y_text = df[target_col].values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_text)
    classes = le.classes_
    n_classes = len(classes)
    print(f"[INFO] Encoded classes: {dict(zip(range(len(classes)), classes))}")

    # ------------------ 2. Train / Val / Test split ------------------
    print("\n[INFO] Splitting into Train/Val/Test (60/20/20)...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X.values, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, stratify=y_train_full, random_state=42
    )
    print(f"[INFO] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ------------------ 3. Train XGBoost ------------------
    print("\n[STEP] Training XGBoost (Session 7)...")
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

    # ------------------ 4. Train AdaBoost (no 'algorithm') ------------------
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    print("\n[STEP] Training AdaBoost (Session 7)...")
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

    # ------------------ 5. Load ensemble weights (XGB + ADA) ------------------
    weights_path = os.path.join(BASE_DIR, "KD_Ensemble_Weights_Session6.csv")
    print(f"\n[INFO] Loading ensemble weights from: {weights_path}")
    w_df = pd.read_csv(weights_path)

    w_xgb_raw = float(w_df.loc[w_df["Model"] == "XGBoost", "Weight"].values[0])
    w_ada_raw = float(w_df.loc[w_df["Model"] == "AdaBoost", "Weight"].values[0])
    denom = w_xgb_raw + w_ada_raw
    w_xgb = w_xgb_raw / denom
    w_ada = w_ada_raw / denom

    print("[INFO] Ensemble Weights (Session 7, XGB+ADA normalized):")
    print(f"  XGBoost: {w_xgb:.3f}")
    print(f"  AdaBoost: {w_ada:.3f}")

    # ------------------ 6. Predict on TEST set ------------------
    print("\n[STEP] Predicting on TEST set...")

    p_xgb_test = xgb_clf.predict_proba(X_test)
    p_ada_test = ada_clf.predict_proba(X_test)

    # Weighted ensemble probabilities (2 models)
    p_ens_test = w_xgb * p_xgb_test + w_ada * p_ada_test

    y_pred_xgb = np.argmax(p_xgb_test, axis=1)
    y_pred_ada = np.argmax(p_ada_test, axis=1)
    y_pred_ens = np.argmax(p_ens_test, axis=1)

    # ------------------ 7. Metrics for all models ------------------
    results = []
    results.append(compute_metrics("XGBoost", y_test, y_pred_xgb, p_xgb_test, classes))
    results.append(compute_metrics("AdaBoost", y_test, y_pred_ada, p_ada_test, classes))
    results.append(compute_metrics("Weighted Ensemble (XGB+ADA)", y_test, y_pred_ens, p_ens_test, classes))

    metrics_df = pd.DataFrame(results)
    metrics_path = os.path.join(BASE_DIR, "KD_Session7_metrics_comparison.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n[INFO] Session 7 metrics comparison saved to: {metrics_path}")

    # ------------------ 8. Confusion matrices ------------------
    print("\n[STEP] Generating confusion matrices...")

    cm_dir = os.path.join(BASE_DIR, "Session7_confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    plot_and_save_cm(y_test, y_pred_xgb, classes, "XGBoost - Confusion Matrix", "cm_xgb.png", cm_dir)
    plot_and_save_cm(y_test, y_pred_ada, classes, "AdaBoost - Confusion Matrix", "cm_ada.png", cm_dir)
    plot_and_save_cm(y_test, y_pred_ens, classes, "Ensemble (XGB+ADA) - Confusion Matrix", "cm_ensemble.png", cm_dir)

    # ------------------ 9. ROC curves per class ------------------
    print("\n[STEP] Plotting ROC curves per class...")

    roc_dir = os.path.join(BASE_DIR, "Session7_ROC_curves")
    os.makedirs(roc_dir, exist_ok=True)

    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    plot_multiclass_roc(y_test_bin, p_xgb_test, classes, "XGBoost - ROC Curves", "roc_xgb.png", roc_dir)
    plot_multiclass_roc(y_test_bin, p_ada_test, classes, "AdaBoost - ROC Curves", "roc_ada.png", roc_dir)
    plot_multiclass_roc(y_test_bin, p_ens_test, classes, "Ensemble (XGB+ADA) - ROC Curves", "roc_ensemble.png", roc_dir)

    print("\nSESSION 7 COMPLETED ✅")
    print("=" * 60)

if __name__ == "__main__":
    main()
