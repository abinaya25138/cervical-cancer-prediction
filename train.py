"""
Cervical cancer risk prediction: XGBoost vs. a small DNN, with SHAP for
interpretability.

Dataset: UCI "Cervical Cancer (Risk Factors)" - 858 patient records with
demographic, behavioral, and medical-history features, plus 4 independent
diagnostic screening results (Hinselmann, Schiller, Citology, Biopsy).

Ground truth: Hinselmann, Schiller, and Citology are each a clinician's
screening read of the same patient (colposcopy x2 + cytology) - majority
vote across those 3 gives a consensus label, which is what we predict.
Biopsy (the histological gold standard) and the Dx:* columns are prior
diagnosis flags, so they're excluded from the feature set as label leakage.

Usage:
    python train.py
"""
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load and build the consensus ("3-pathologist majority vote") label
# ---------------------------------------------------------------------------
df = pd.read_csv("data/risk_factors_cervical_cancer.csv")
df = df.replace("?", np.nan).apply(pd.to_numeric, errors="coerce")

screening_cols = ["Hinselmann", "Schiller", "Citology"]
df["consensus_positive"] = (df[screening_cols].sum(axis=1) >= 2).astype(int)

leakage_cols = screening_cols + [
    "Biopsy", "Dx:Cancer", "Dx:CIN", "Dx:HPV", "Dx", "consensus_positive",
]
feature_cols = [c for c in df.columns if c not in leakage_cols]

X = df[feature_cols]
y = df["consensus_positive"]

print(f"Dataset: {len(df)} patients, {len(feature_cols)} features")
print(f"Consensus-positive rate: {y.mean():.2%} ({y.sum()}/{len(y)})")
majority_baseline = max(y.mean(), 1 - y.mean())
print(f"Majority-class baseline accuracy: {majority_baseline:.2%}  <- any model must clear this to mean anything")

# ---------------------------------------------------------------------------
# 2. Impute missing values (many columns use '?' for "not reported")
# ---------------------------------------------------------------------------
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# 3. XGBoost - class-weighted
#
# IMPORTANT: an unweighted model on this data just predicts "negative" for
# every patient - it clears the 94% majority baseline on accuracy alone
# while catching zero positive cases (0% recall). That's a textbook
# accuracy-paradox trap, not a working classifier. Weighting trades some
# raw accuracy for the ability to actually flag at-risk patients, which is
# what matters here.
# ---------------------------------------------------------------------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
)
xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

print("\n=== XGBoost ===")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("ROC AUC:", roc_auc_score(y_test, xgb_proba))
print(classification_report(y_test, xgb_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, xgb_pred))

# ---------------------------------------------------------------------------
# 4. DNN - same features, scaled, with class weighting
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class_weight = {
    0: 1.0,
    1: (y_train == 0).sum() / (y_train == 1).sum(),
}

dnn = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid"),
])
dnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

dnn.fit(
    X_train_scaled, y_train,
    validation_split=0.15,
    epochs=60,
    batch_size=16,
    class_weight=class_weight,
    verbose=0,
)

dnn_proba = dnn.predict(X_test_scaled, verbose=0).ravel()
dnn_pred = (dnn_proba >= 0.5).astype(int)

print("\n=== DNN ===")
print("Accuracy:", accuracy_score(y_test, dnn_pred))
print("ROC AUC:", roc_auc_score(y_test, dnn_proba))
print(classification_report(y_test, dnn_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, dnn_pred))

# ---------------------------------------------------------------------------
# 5. SHAP - feature importance / interpretability for the XGBoost model
# ---------------------------------------------------------------------------
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

mean_abs_shap = pd.Series(
    np.abs(shap_values).mean(axis=0), index=feature_cols
).sort_values(ascending=False)

print("\n=== Top 10 features by mean |SHAP value| (XGBoost) ===")
print(mean_abs_shap.head(10))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150)
print("\nSaved SHAP summary plot to shap_summary.png")
