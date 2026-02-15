# ==========================================
# Data Cleaning & Preprocessing Pipeline
# Kawasaki Disease Dataset
# ==========================================

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -------------------------
# 0. Set project / output directory
#    (folder where this script is located)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = BASE_DIR
os.makedirs(output_dir, exist_ok=True)
print("Output directory:", os.path.abspath(output_dir))

# -------------------------
# 1. Load original dataset
# -------------------------
file_path = os.path.join(BASE_DIR, "KAWASAKI_ORIGINAL.csv")
df = pd.read_csv(file_path)

print("Original shape:", df.shape)

# ----------------------------------------
# 2. Handle missing values
#    - mean for numeric
#    - "Unknown" for categorical
# ----------------------------------------
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols_all = df.select_dtypes(include=["object"]).columns

# Fill numeric NaNs with mean
for col in numeric_cols:
    mean_val = df[col].mean()
    df[col] = df[col].fillna(mean_val)

# Fill categorical NaNs with a label
for col in cat_cols_all:
    df[col] = df[col].fillna("Unknown")

print("\nMissing values after imputation:\n", df.isna().sum())

# ---------------------------------------------------
# 3. Remove duplicate entries and inconsistent records
# ---------------------------------------------------
before_dup = df.shape[0]
df = df.drop_duplicates()
after_dup = df.shape[0]
print(f"\nRemoved {before_dup - after_dup} duplicate rows.")

df = df[df["Age at Diagnosis"] >= 0]
df = df[df["Fever Duration"] >= 0]

print("Shape after removing inconsistent records:", df.shape)

# -------------------------------
# 4. Encode categorical variables
# -------------------------------
categorical_cols = [
    "Gender", "Ethnicity", "Location", "Symptoms",
    "Laboratory Tests", "Echocardiography",
    "Treatment Approach", "Clinical Outcomes",
    "Complications", "Follow-up Visits", "Long-Term Effects"
]

df_cat = df[categorical_cols].astype(str)
df_num = df.drop(columns=categorical_cols)

ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
cat_encoded = ohe.fit_transform(df_cat)

encoded_feature_names = ohe.get_feature_names_out(categorical_cols)
df_cat_encoded = pd.DataFrame(cat_encoded, columns=encoded_feature_names, index=df.index)

df_encoded = pd.concat(
    [df_num.reset_index(drop=True), df_cat_encoded.reset_index(drop=True)],
    axis=1
)

print("\nShape after encoding:", df_encoded.shape)

# ----------------------------------------------------
# 5. Z-score normalization on numerical attributes
# ----------------------------------------------------
numeric_cols_encoded = df_num.select_dtypes(include=["int64", "float64"]).columns

scaler = StandardScaler()
df_encoded[numeric_cols_encoded] = scaler.fit_transform(df_encoded[numeric_cols_encoded])

# ----------------------------------------------------
# 6. Outlier detection using IQR
# ----------------------------------------------------
def iqr_outlier_mask(series, k=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    return (series < lower) | (series > upper)

outlier_mask = np.zeros(len(df_encoded), dtype=bool)
for col in numeric_cols_encoded:
    col_mask = iqr_outlier_mask(df_encoded[col])
    outlier_mask = outlier_mask | col_mask

print(f"\nTotal outliers detected (any numeric col, IQR): {outlier_mask.sum()}")

df_encoded["is_outlier"] = outlier_mask.astype(int)

# --------------------------------------
# 7. Save cleaned & preprocessed outputs
# --------------------------------------
cleaned_path = os.path.join(output_dir, "KAWASAKI-CLEANED.csv")
matrix_path = os.path.join(output_dir, "KD_preprocessed_feature_matrix.csv")

df.to_csv(cleaned_path, index=False)
df_encoded.to_csv(matrix_path, index=False)

print("\nCleaned CSV saved to:", os.path.abspath(cleaned_path))
print("Feature matrix saved to:", os.path.abspath(matrix_path))
print("SCRIPT FINISHED SUCCESSFULLY")
