# ==========================================
# Session 4: MGWO Feature Selection (KD)
# ==========================================
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ---------- 0. Project root & load data ----------
# Use project folder (script location) instead of hardcoded path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(
    BASE_DIR,
    "KD_preprocessed_feature_matrix.csv"  # or KD_ADASYN_balanced_feature_matrix.csv
)
df = pd.read_csv(data_path)

print("Loaded shape:", df.shape)

# drop non‑numeric identifier / date columns
drop_cols = ["Patient ID", "Date of Diagnosis"]
for c in drop_cols:
    if c in df.columns:
        df = df.drop(columns=[c])

# one‑hot long‑term effect columns
lt_cols = [
    "Long-Term Effects_Mild",
    "Long-Term Effects_Severe",
    "Long-Term Effects_Unknown"
]
for c in lt_cols:
    if c not in df.columns:
        raise ValueError(f"Missing column in matrix: {c}")

# ---------- rebuild y from one‑hots ----------
y_idx = df[lt_cols].values.argmax(axis=1)
idx_to_label = {0: "Mild", 1: "Severe", 2: "Unknown"}
y_labels = np.array([idx_to_label[i] for i in y_idx])
classes_, y = np.unique(y_labels, return_inverse=True)

# X = all remaining numeric features
X = df.drop(columns=lt_cols).values
feature_names = df.drop(columns=lt_cols).columns.tolist()
n_features = X.shape[1]

print("Number of features used:", n_features)
print("Classes:", classes_)

# ---------- 1. MGWO hyperparameters ----------
n_wolves = 20
max_iter = 40
min_features = 3

# fitness: 1 - accuracy + feature penalty
def fitness(mask):
    if mask.sum() < min_features:
        return 1e6
    X_sub = X[:, mask == 1]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    accs = []
    for tr, va in skf.split(X_sub, y):
        model = XGBClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            eval_metric="mlogloss",
        )
        model.fit(X_sub[tr], y[tr])
        preds = model.predict(X_sub[va])
        accs.append(accuracy_score(y[va], preds))
    acc = np.mean(accs)
    feat_ratio = mask.sum() / n_features
    return 1.0 - acc + 0.1 * feat_ratio  # lower is better

# ---------- 2. Helpers: nonlinear a(t) + OBL ----------
def nonlinear_a(t, T):
    # cosine-based nonlinear convergence factor (2 -> 0)
    return 2 * np.cos((t / T) * (np.pi / 2))

def opposition(position):
    # opposition in [0,1] search space
    return 1.0 - position

# ---------- 3. Initialization ----------
rng = np.random.default_rng(42)
positions = rng.random((n_wolves, n_features))          # continuous [0,1]
binaries = (positions > 0.5).astype(int)

# OBL on initial population
for i in range(n_wolves):
    opp = opposition(positions[i])
    opp_bin = (opp > 0.5).astype(int)
    f1 = fitness(binaries[i])
    f2 = fitness(opp_bin)
    if f2 < f1:
        positions[i] = opp
        binaries[i] = opp_bin

fitness_vals = np.array([fitness(b) for b in binaries])
idx_sorted = np.argsort(fitness_vals)
alpha_pos, beta_pos, delta_pos = positions[idx_sorted[:3]]
alpha_bin, beta_bin, delta_bin = binaries[idx_sorted[:3]]
alpha_fit, beta_fit, delta_fit = fitness_vals[idx_sorted[:3]]

print("Initial alpha fitness:", alpha_fit, "features:", alpha_bin.sum())

# ---------- 4. Main MGWO loop ----------
for t in range(max_iter):
    a = nonlinear_a(t, max_iter)

    for i in range(n_wolves):
        for j in range(n_features):
            r1, r2 = rng.random(), rng.random()
            A1 = 2 * a * r1 - a
            C1 = 2 * r2

            r1, r2 = rng.random(), rng.random()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2

            r1, r2 = rng.random(), rng.random()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2

            D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
            D_beta  = abs(C2 * beta_pos[j]  - positions[i, j])
            D_delta = abs(C3 * delta_pos[j] - positions[i, j])

            X1 = alpha_pos[j] - A1 * D_alpha
            X2 = beta_pos[j]  - A2 * D_beta
            X3 = delta_pos[j] - A3 * D_delta

            positions[i, j] = (X1 + X2 + X3) / 3.0

        # clamp
        positions[i] = np.clip(positions[i], 0, 1)

        # OBL step
        opp = opposition(positions[i])
        cur_bin = (positions[i] > 0.5).astype(int)
        opp_bin = (opp > 0.5).astype(int)
        f_cur = fitness(cur_bin)
        f_opp = fitness(opp_bin)
        if f_opp < f_cur:
            positions[i] = opp
            cur_bin = opp_bin
            f_cur = f_opp

        binaries[i] = cur_bin
        fitness_vals[i] = f_cur

    idx_sorted = np.argsort(fitness_vals)
    alpha_pos, beta_pos, delta_pos = positions[idx_sorted[:3]]
    alpha_bin, beta_bin, delta_bin = binaries[idx_sorted[:3]]
    alpha_fit, beta_fit, delta_fit = fitness_vals[idx_sorted[:3]]

    print(f"Iter {t+1}/{max_iter} | Alpha fitness: {alpha_fit:.4f}, "
          f"features: {alpha_bin.sum()}")

# ---------- 5. Output selected features ----------
best_mask = alpha_bin
selected_features = [f for f, m in zip(feature_names, best_mask) if m == 1]

print("\nSelected features by MGWO:")
for f in selected_features:
    print(" -", f)

# feature selection frequency
freq = binaries.sum(axis=0) / n_wolves
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Selection_Frequency": freq
}).sort_values("Selection_Frequency", ascending=False)

out_sel = os.path.join(BASE_DIR, "MGWO_selected_features.csv")
importance_df.to_csv(out_sel, index=False)
print("\nMGWO feature importance table saved to:", out_sel)

# reduced matrix for modeling
X_best = X[:, best_mask == 1]
reduced_df = pd.DataFrame(X_best, columns=selected_features)
reduced_df["Long-Term Effects"] = y_labels
out_mat = os.path.join(BASE_DIR, "KD_MGWO_reduced_matrix.csv")
reduced_df.to_csv(out_mat, index=False)
print("Reduced feature matrix saved to:", out_mat)
