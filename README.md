# Cervical Cancer Risk Prediction

Compares an XGBoost classifier against a small DNN for early cervical cancer risk detection, with SHAP for interpretability — on the UCI "Cervical Cancer (Risk Factors)" dataset (858 patient records, 32 demographic/behavioral/medical-history features).

## Ground truth: a 3-reader majority vote

The dataset includes four independent diagnostic results per patient: `Hinselmann`, `Schiller`, `Citology`, and `Biopsy`. The first three are each a clinician's screening read of the same patient (two colposcopy methods + a cytology/Pap smear read) — no single one is perfectly reliable on its own, so the label used here is a **majority vote across those 3 readers** (≥2 of 3 positive). `Biopsy` (the histological gold standard) and the `Dx:*` prior-diagnosis columns are excluded from the feature set, since using them as predictors would leak the answer.

## A real finding, not just a result

The positive rate in this data is only **5.6%** (48/858 patients) — severe class imbalance. An unweighted model on this data learns to just predict "negative" for every patient: it clears the 94.4% majority-class baseline on raw accuracy while catching **zero** actual positive cases. That's a textbook accuracy-paradox trap, not a working classifier — worth knowing before trusting any accuracy number on this dataset.

The results below use class-weighted training (`scale_pos_weight` for XGBoost, `class_weight` for the DNN), trading some raw accuracy for the ability to actually flag at-risk patients.

## Results

| Model | Accuracy | ROC AUC | Recall (positive class) | Precision (positive class) |
|---|---|---|---|---|
| Majority-class baseline | 94.4% | — | 0% | — |
| XGBoost (class-weighted) | 86.0% | 0.66 | 30% | 15% |
| DNN (class-weighted) | 78.5% | 0.57 | 30% | 9% |

XGBoost is the stronger model here — better AUC and precision at the same recall, and it's the one SHAP is run against below. With only 48 positive cases total, a few points of difference run-to-run (seed, split) is normal; the qualitative result — weighting trades accuracy for actually catching positives, XGBoost beats the DNN here — is stable.

## Interpretability (SHAP)

![SHAP summary plot](shap_summary.png)

Top risk factors by mean |SHAP value|: years on hormonal contraceptives, number of pregnancies, age at first intercourse, age, and number of sexual partners — consistent with known cervical cancer risk factors in the clinical literature (HPV exposure accumulates with more partners and earlier sexual activity; long-term hormonal contraceptive use is a separately established risk factor).

## Setup

```bash
pip install -r requirements.txt
python train.py
```

Downloads nothing at runtime — `data/risk_factors_cervical_cancer.csv` is included (it's the full UCI dataset, ~100KB).

## Tech stack

XGBoost, TensorFlow/Keras, SHAP, scikit-learn, pandas

## Dataset

[UCI Machine Learning Repository: Cervical Cancer (Risk Factors)](https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors) — Fernandes, Cardoso & Fernandes, 2017.
