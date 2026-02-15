# ==========================================
# SESSION 3: Feature Matrix Preparation
# From cleaned Kawasaki dataset
# ==========================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# -------------------------
# 0. Project root / paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLEAN_DATA_PATH = os.path.join(BASE_DIR, "KAWASAKI-CLEANED.csv")
MATRIX_OUT_PATH = os.path.join(BASE_DIR, "KD_ADASYN_feature_matrix.csv")   # or KD_preprocessed_feature_matrix.csv
TRAIN_OUT_PATH = os.path.join(BASE_DIR, "KD_train_matrix.csv")
TEST_OUT_PATH = os.path.join(BASE_DIR, "KD_test_matrix.csv")

TARGET_COL = "Long-Term Effects"   # make sure this matches your column name


def main():
    print("SESSION 3 STARTED")
    print("=" * 60)
    print(f"[INFO] Loading cleaned data from: {CLEAN_DATA_PATH}")

    if not os.path.exists(CLEAN_DATA_PATH):
        raise FileNotFoundError(f"Cleaned data file not found: {CLEAN_DATA_PATH}")

    # 1) Load cleaned data
    df = pd.read_csv(CLEAN_DATA_PATH)
    print("[INFO] Cleaned data shape:", df.shape)

    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in cleaned data.")

    # 2) Separate features and target
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    print("[INFO] Feature matrix shape (X):", X.shape)
    print("[INFO] Target vector shape (y):", y.shape)

    # (Optional) If you already encoded everything in Session 2, you might be done here.
    # We just save a combined matrix with target as last column.

    feature_matrix = pd.concat([X, y], axis=1)
    feature_matrix.to_csv(MATRIX_OUT_PATH, index=False)
    print(f"[INFO] Full feature matrix saved to: {MATRIX_OUT_PATH}")

    # 3) Train-test split for later sessions (60/40 or 80/20; here 80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(TRAIN_OUT_PATH, index=False)
    test_df.to_csv(TEST_OUT_PATH, index=False)

    print(f"[INFO] Train matrix saved to: {TRAIN_OUT_PATH}")
    print(f"[INFO] Test matrix saved to:  {TEST_OUT_PATH}")

    print("\nSESSION 3 COMPLETED ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
