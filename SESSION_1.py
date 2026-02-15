# ================================
# Session 1: Dataset Acquisition & Understanding
# Kawasaki Disease (KD) Dataset
# ================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Recommended for nicer plots
sns.set(style="whitegrid")

# ------------------------
# 1. Load the KD dataset
# ------------------------
# Make sure KAWASAKI_ORIGINAL.csv is in the same folder as this notebook/script
file_path = "KAWASAKI_ORIGINAL.csv"
df = pd.read_csv(file_path)

print("=== BASIC DATA INFO ===")
print("Shape (rows, columns):", df.shape)       # Expected: (2752, 15)
print("\nColumn names:\n", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nMissing values per column:\n", df.isna().sum())

# -------------------------------
# 2. Feature-type mapping table
# -------------------------------
# Manually define semantic types for documentation/report
feature_types = {
    "Patient ID": ("numeric", "demographic id"),
    "Date of Diagnosis": ("categorical/date", "temporal"),
    "Age at Diagnosis": ("numeric", "demographic"),
    "Gender": ("categorical", "demographic"),
    "Ethnicity": ("categorical", "demographic"),
    "Location": ("categorical", "demographic"),
    "Fever Duration": ("numeric", "clinical"),
    "Symptoms": ("categorical", "clinical"),
    "Laboratory Tests": ("categorical", "lab"),
    "Echocardiography": ("categorical", "lab/imaging"),
    "Treatment Approach": ("categorical", "treatment"),
    "Clinical Outcomes": ("categorical", "outcome"),
    "Complications": ("categorical", "outcome"),
    "Follow-up Visits": ("categorical", "follow-up"),
    "Long-Term Effects": ("categorical", "outcome severity"),
}

feature_summary = pd.DataFrame(
    [{"Feature": k, "DataType": v[0], "Category": v[1]} for k, v in feature_types.items()]
)

print("\n=== FEATURE-TYPE MAPPING TABLE ===")
print(feature_summary)

# (Optional) Save this mapping as a CSV for documentation
# feature_summary.to_csv("feature_type_mapping.csv", index=False)

# --------------------------------------------
# 3. Exploratory Data Analysis (EDA) - Numeric
# --------------------------------------------
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

print("\n=== NUMERIC COLUMNS ===")
print(numeric_cols.tolist())

print("\n=== DESCRIPTIVE STATISTICS (NUMERIC) ===")
print(df[numeric_cols].describe())

# ---------------------------------------------
# 4. Exploratory Data Analysis (EDA) - Categorical
# ---------------------------------------------
cat_cols = [
    "Gender", "Ethnicity", "Symptoms", "Laboratory Tests", "Echocardiography",
    "Treatment Approach", "Clinical Outcomes", "Complications",
    "Follow-up Visits", "Long-Term Effects",
]

print("\n=== VALUE COUNTS FOR CATEGORICAL FEATURES ===")
for col in cat_cols:
    if col in df.columns:
        print(f"\n--- {col} ---")
        print(df[col].value_counts(dropna=False))

# ------------------------------------------------
# 5. Class balance: Clinical Outcomes distribution
# ------------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(
    data=df,
    x="Clinical Outcomes",
    order=df["Clinical Outcomes"].value_counts().index
)
plt.title("Clinical Outcomes Class Distribution")
plt.xlabel("Clinical Outcome")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ----------------------------------------
# 6. Histograms for numeric distributions
# ----------------------------------------
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# -----------------------------
# 7. Boxplots for numeric data
# -----------------------------
for col in numeric_cols:
    plt.figure(figsize=(4, 4))
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot of {col}")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------
# 8. Correlation heatmap (numeric features only)
# ------------------------------------------------------
if len(numeric_cols) > 1:
    plt.figure(figsize=(4, 3))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.show()




