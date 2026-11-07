"""
Cervical cancer prediction: XGBoost vs. a small neural network, with SHAP
values for interpretability.

Dataset: UCI "Cervical Cancer (Risk Factors)" - 858 patient records with
demographic, behavioural and medical-history risk factors, plus four
diagnostic results per patient (Hinselmann, Schiller, Citology, Biopsy).

Target: the biopsy result, the histological diagnostic standard.

Features: the 28 risk-factor columns only. The three screening tests are
other diagnostic results for the same patient, and the Dx:* flags record
diagnoses that may have been made at the same time, so all of them are
excluded: using them could leak the answer.

Evaluation: 10x repeated stratified 5-fold cross-validation (50 test folds).
Imputation, scaling and class weights are fitted on each training fold only.
The same pipeline trained on shuffled labels shows what chance looks like.

Usage:
    python train.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N_SPLITS, N_REPEATS = 5, 10

# ---------------------------------------------------------------------------
# 1. Load data: biopsy target, risk-factor features
# ---------------------------------------------------------------------------
df = pd.read_csv("data/risk_factors_cervical_cancer.csv")
df = df.replace("?", np.nan).apply(pd.to_numeric, errors="coerce")

diagnostic_cols = ["Hinselmann", "Schiller", "Citology", "Biopsy", "Dx:Cancer", "Dx:CIN", "Dx:HPV", "Dx"]
feature_cols = [c for c in df.columns if c not in diagnostic_cols]
X = df[feature_cols]
y = df["Biopsy"].astype(int)

print(f"Dataset: {len(df)} patients, {len(feature_cols)} risk-factor features")
print(f"Biopsy-positive: {y.sum()} ({y.mean():.1%})")


# ---------------------------------------------------------------------------
# 2. Models. Everything that learns from data is fitted on the training fold.
# ---------------------------------------------------------------------------
def fit_xgboost(X_train, y_train):
    imputer = SimpleImputer(strategy="median").fit(X_train)
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    model.fit(imputer.transform(X_train), y_train)
    return lambda X_new: model.predict_proba(imputer.transform(X_new))[:, 1], model, imputer


def fit_network(X_train, y_train):
    """A 32-16 multilayer perceptron. Positives are oversampled within the training fold to balance the classes."""
    imputer = SimpleImputer(strategy="median").fit(X_train)
    scaler = StandardScaler().fit(imputer.transform(X_train))
    Z, t = scaler.transform(imputer.transform(X_train)), y_train.values
    pos = np.flatnonzero(t == 1)
    extra = np.random.default_rng(RANDOM_STATE).choice(pos, size=(t == 0).sum() - len(pos))
    net = MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=300, random_state=RANDOM_STATE)
    net.fit(np.vstack([Z, Z[extra]]), np.concatenate([t, t[extra]]))
    return lambda X_new: net.predict_proba(scaler.transform(imputer.transform(X_new)))[:, 1]


def fold_scores(y_true, proba):
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "recall": recall_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
    }


# ---------------------------------------------------------------------------
# 3. Repeated stratified cross-validation, plus a shuffled-label control
# ---------------------------------------------------------------------------
folds = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE).split(X, y))
y_shuffled = pd.Series(np.random.default_rng(RANDOM_STATE).permutation(y.values), index=y.index)

runs = {
    "XGBoost": (fit_xgboost, y),
    "Neural network": (fit_network, y),
    "XGBoost, shuffled labels (chance)": (fit_xgboost, y_shuffled),
}
results = {}
for name, (fit, target) in runs.items():
    rows = []
    for k, (tr, te) in enumerate(folds):
        predict = fit(X.iloc[tr], target.iloc[tr])[0] if fit is fit_xgboost else fit(X.iloc[tr], target.iloc[tr])
        rows.append(fold_scores(target.iloc[te], predict(X.iloc[te])))
    res = pd.DataFrame(rows)
    results[name] = {m: {"mean": float(res[m].mean()), "sd": float(res[m].std())} for m in res.columns}
    print(f"\n=== {name} ({len(folds)} test folds) ===")
    for m in res.columns:
        print(f"  {m:9s} {res[m].mean():.3f} ± {res[m].std():.3f}")

print(f"\nPR AUC of a random ranking = positive rate = {y.mean():.3f}")
results["positive_rate"] = float(y.mean())
with open("results.json", "w") as f:
    json.dump(results, f, indent=1)

# ---------------------------------------------------------------------------
# 4. SHAP: which risk factors the XGBoost model relies on (refit on all data)
# ---------------------------------------------------------------------------
_, final_model, final_imputer = fit_xgboost(X, y)
X_all = pd.DataFrame(final_imputer.transform(X), columns=feature_cols)
# XGBoost computes exact TreeSHAP values itself; the last column is the bias term
shap_values = final_model.get_booster().predict(xgb.DMatrix(X_all), pred_contribs=True)[:, :-1]

mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols).sort_values(ascending=False)
print("\n=== Top 10 features by mean |SHAP value| (XGBoost) ===")
print(mean_abs_shap.head(10).round(4).to_string())

top = mean_abs_shap.index[:12][::-1]
fig, ax = plt.subplots(figsize=(8, 5.5))
for i, col in enumerate(top):
    vals = shap_values[:, feature_cols.index(col)]
    feat = X_all[col].values
    span = feat.max() - feat.min()
    colour = (feat - feat.min()) / span if span else np.zeros_like(feat)
    jitter = np.random.default_rng(i).uniform(-0.25, 0.25, len(vals))
    pts = ax.scatter(vals, i + jitter, c=colour, cmap="coolwarm", s=8, alpha=0.7, vmin=0, vmax=1)
ax.axvline(0, color="#999", lw=0.8)
ax.set_yticks(range(len(top)), top)
ax.set_xlabel("SHAP value (impact on predicted risk of a positive biopsy)")
cbar = fig.colorbar(pts, ax=ax, ticks=[0, 1])
cbar.ax.set_yticklabels(["low", "high"])
cbar.set_label("Feature value")
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150)
print("\nSaved results.json and shap_summary.png")
