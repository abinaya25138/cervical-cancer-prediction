"""
Cervical cancer prediction: XGBoost vs. a logistic-regression baseline and a
small neural network, with SHAP values for interpretability.

Dataset: UCI "Cervical Cancer (Risk Factors)" - 858 patient records with
demographic, behavioural and medical-history risk factors, plus four
diagnostic results per patient (Hinselmann, Schiller, Citology, Biopsy).

Target: the biopsy result, the histological diagnostic standard.

Features: the 28 risk-factor columns only. The three screening tests are
other diagnostic results for the same patient, and the Dx:* flags record
diagnoses that may have been made at the same time, so all of them are
excluded: using them could leak the answer.

Evaluation: 10x repeated stratified 5-fold cross-validation (50 test folds).
Everything that learns from data - imputation, scaling, class weights, the
decision threshold and SHAP values - is computed from each training fold only.

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N_SPLITS, N_REPEATS = 5, 10
TARGET_RECALL = 0.80  # screening goal: catch at least 80% of positive biopsies

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
# 2. Models. Each fit_* returns a function that scores new patients.
# ---------------------------------------------------------------------------
def fit_xgboost(X_train, y_train, missing_indicators=False):
    imputer = SimpleImputer(strategy="median", add_indicator=missing_indicators).fit(X_train)
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
    predict = lambda X_new: model.predict_proba(imputer.transform(X_new))[:, 1]
    predict.model, predict.imputer = model, imputer
    return predict


def fit_logistic(X_train, y_train):
    imputer = SimpleImputer(strategy="median").fit(X_train)
    scaler = StandardScaler().fit(imputer.transform(X_train))
    model = LogisticRegression(class_weight="balanced", C=0.1, max_iter=5000)
    model.fit(scaler.transform(imputer.transform(X_train)), y_train)
    return lambda X_new: model.predict_proba(scaler.transform(imputer.transform(X_new)))[:, 1]


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


def recall_threshold(fit, X_train, y_train):
    """Pick the score cutoff that reaches TARGET_RECALL, using out-of-fold scores inside the training fold."""
    oof = np.zeros(len(y_train))
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    for tr, va in inner.split(X_train, y_train):
        oof[va] = fit(X_train.iloc[tr], y_train.iloc[tr])(X_train.iloc[va])
    positive_scores = np.sort(oof[y_train.values == 1])
    return positive_scores[int(np.floor((1 - TARGET_RECALL) * len(positive_scores)))]


# ---------------------------------------------------------------------------
# 3. Repeated stratified cross-validation
# ---------------------------------------------------------------------------
folds = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE).split(X, y))
y_shuffled = pd.Series(np.random.default_rng(RANDOM_STATE).permutation(y.values), index=y.index)

runs = {
    "XGBoost": (fit_xgboost, y),
    "XGBoost + missingness indicators": (lambda Xt, yt: fit_xgboost(Xt, yt, missing_indicators=True), y),
    "Logistic regression": (fit_logistic, y),
    "Neural network": (fit_network, y),
    "XGBoost, shuffled labels (chance)": (fit_xgboost, y_shuffled),
}
fold_auc = {name: [] for name in runs}
summary = {}
for name, (fit, target) in runs.items():
    rows = []
    for tr, te in folds:
        scores = fit(X.iloc[tr], target.iloc[tr])(X.iloc[te])
        rows.append({"roc_auc": roc_auc_score(target.iloc[te], scores),
                     "pr_auc": average_precision_score(target.iloc[te], scores),
                     "accuracy": accuracy_score(target.iloc[te], scores >= 0.5)})
    res = pd.DataFrame(rows)
    fold_auc[name] = res["roc_auc"].values
    summary[name] = {
        m: {"mean": float(res[m].mean()), "sd": float(res[m].std()),
            "p2.5": float(res[m].quantile(0.025)), "p97.5": float(res[m].quantile(0.975))}
        for m in res.columns
    }
    a = summary[name]["roc_auc"]
    print(f"{name:36s} ROC AUC {a['mean']:.3f} ± {a['sd']:.3f}  (95% of folds: {a['p2.5']:.2f} to {a['p97.5']:.2f})"
          f"   PR AUC {summary[name]['pr_auc']['mean']:.3f}   accuracy at 0.5 {summary[name]['accuracy']['mean']:.1%}")
print(f"PR AUC of a random ranking = positive rate = {y.mean():.3f}")
print(f"Accuracy of predicting 'negative' for everyone = {1 - y.mean():.1%}")

# Why missingness indicators are not used: the gain comes from a data-collection artifact
hc_missing = X["Hormonal Contraceptives"].isna()
artifact = {"patients_missing_hc": int(hc_missing.sum()), "positive_rate_missing": float(y[hc_missing].mean()),
            "positive_rate_reported": float(y[~hc_missing].mean())}
print(f"Patients with no contraceptive history recorded: {hc_missing.sum()}, biopsy-positive {y[hc_missing].mean():.1%} "
      f"vs {y[~hc_missing].mean():.1%} for the rest")

# Paired comparison on identical folds
paired = {}
for other in ("XGBoost + missingness indicators", "Logistic regression", "Neural network", "XGBoost, shuffled labels (chance)"):
    diff = fold_auc["XGBoost"] - fold_auc[other]
    paired[other] = {"mean_diff": float(diff.mean()), "xgboost_wins": float((diff > 0).mean())}
    print(f"XGBoost vs {other:34s} mean AUC difference {diff.mean():+.3f}, XGBoost higher in {(diff > 0).mean():.0%} of folds")

# ---------------------------------------------------------------------------
# 4. Screening threshold: what it costs to catch 80% of positive biopsies
# ---------------------------------------------------------------------------
rows = []
for tr, te in folds:
    X_tr, y_tr = X.iloc[tr], y.iloc[tr]
    cutoff = recall_threshold(fit_xgboost, X_tr, y_tr)
    flagged = fit_xgboost(X_tr, y_tr)(X.iloc[te]) >= cutoff
    y_te = y.iloc[te].values
    rows.append({
        "recall": flagged[y_te == 1].mean(),
        "precision": y_te[flagged].mean() if flagged.any() else 0.0,
        "flagged_share": flagged.mean(),
        "accuracy": (flagged == y_te).mean(),
    })
thr = pd.DataFrame(rows)
threshold = {k: {"mean": float(thr[k].mean()), "sd": float(thr[k].std())} for k in thr.columns}
print(f"\nThreshold chosen inside each training fold for {TARGET_RECALL:.0%} recall, applied to the test fold:")
print(f"  recall {thr['recall'].mean():.2f} ± {thr['recall'].std():.2f}, precision {thr['precision'].mean():.3f} "
      f"(base rate {y.mean():.3f}), patients flagged {thr['flagged_share'].mean():.0%}, accuracy {thr['accuracy'].mean():.1%}")

# ---------------------------------------------------------------------------
# 5. SHAP values and how stable the top risk factors are across folds
# ---------------------------------------------------------------------------
def shap_values(predict, X_new):
    """Exact TreeSHAP values from XGBoost itself; the last column is the bias term."""
    Z = predict.imputer.transform(X_new)
    return predict.model.get_booster().predict(xgb.DMatrix(Z), pred_contribs=True)[:, :-1]


def readable(name):
    return name.replace("missingindicator_", "") + " (missing)" if name.startswith("missingindicator_") else name


top5_counts = pd.Series(dtype=float)
for tr, te in folds:
    predict = fit_xgboost(X.iloc[tr], y.iloc[tr])
    names = [readable(n) for n in predict.imputer.get_feature_names_out()]
    fold_importance = pd.Series(np.abs(shap_values(predict, X.iloc[te])).mean(axis=0), index=names)
    for name in fold_importance.nlargest(5).index:
        top5_counts[name] = top5_counts.get(name, 0) + 1
stability = (top5_counts / len(folds)).sort_values(ascending=False)
print("\nShare of the 50 folds in which each risk factor is in the SHAP top five:")
print(stability[stability > 0].round(2).to_string())

final = fit_xgboost(X, y)
final_names = [readable(n) for n in final.imputer.get_feature_names_out()]
X_all = pd.DataFrame(final.imputer.transform(X), columns=final_names)
sv = shap_values(final, X)
mean_abs_shap = pd.Series(np.abs(sv).mean(axis=0), index=final_names).sort_values(ascending=False)

top = mean_abs_shap.index[:12][::-1]
fig, ax = plt.subplots(figsize=(8, 5.5))
for i, col in enumerate(top):
    vals, feat = sv[:, final_names.index(col)], X_all[col].values
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

with open("results.json", "w") as f:
    json.dump({"cross_validation": summary, "missingness_artifact": artifact, "paired_vs_xgboost": paired, "threshold_80pct_recall": threshold,
               "shap_top5_share": stability[stability > 0].round(3).to_dict(), "positive_rate": float(y.mean())},
              f, indent=1)
print("\nSaved results.json and shap_summary.png")
