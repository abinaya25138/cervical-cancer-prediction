# Cervical Cancer Risk Prediction

Predicts a positive cervical biopsy from patients' demographic, behavioural and medical-history risk factors. It compares an XGBoost classifier with a small neural network, evaluates both with repeated cross-validation against a chance-level control, and uses SHAP values to show which risk factors the model relies on.

Data: the UCI [Cervical Cancer (Risk Factors)](https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors) dataset (Fernandes, Cardoso & Fernandes, 2017), 858 patients.

## Target and features

- **Target: the biopsy result**, the histological diagnostic standard. 55 of 858 patients (6.4%) are positive.
- **Features: 28 risk factors**, such as age, sexual history, pregnancies, smoking, contraceptive use and STD history.
- **Excluded:** the three screening tests (Hinselmann, Schiller, cytology) and the `Dx:*` diagnosis flags. These are other diagnostic results for the same patient, so using them as inputs could leak the answer. Including them makes it easy to report very high accuracy that would not hold for a real screening decision.

## How the evaluation works

With only 55 positive patients, a single 80/20 split leaves about 11 positives in the test set, so one split's score depends heavily on luck. Instead:

- **10× repeated stratified 5-fold cross-validation**, giving 50 test folds.
- **Imputation, scaling and class weighting are fitted on each training fold only**, so nothing from a test fold leaks into training.
- **A shuffled-label control:** the same XGBoost pipeline trained on randomly permuted labels shows what chance looks like.
- **Main metrics are ranking metrics**, ROC AUC and PR AUC, because accuracy is meaningless when 94% of patients are negative.

## Results

| Model (50 test folds) | ROC AUC | PR AUC | Recall | Precision |
|---|---|---|---|---|
| **XGBoost** | **0.61 ± 0.07** | **0.135 ± 0.05** | 34% | 11% |
| Neural network (32-16 MLP) | 0.48 ± 0.07 | 0.102 ± 0.04 | 14% | 10% |
| Shuffled labels (chance) | 0.50 ± 0.07 | 0.076 ± 0.02 | 29% | 6% |

PR AUC for a random ranking equals the positive rate, 0.064. Recall and precision use a 0.5 threshold.

- **XGBoost finds a real signal.** Its AUC is clearly above the chance control, and its PR AUC is about twice that of a random ranking.
- **The neural network does not.** With 28 tabular features and 55 positives, it performs at chance level. Gradient-boosted trees are the better fit for data this small.
- **The signal is modest.** These risk factors raise or lower the odds, but on their own they are far from a diagnostic test.

### How we got here

The first version predicted a majority vote of the three screening tests and was evaluated on one 80/20 split, where XGBoost scored ROC AUC 0.66. Repeated cross-validation showed that this was luck of the split: across 50 folds that setup averaged 0.47, no better than chance. Switching the target to the biopsy result is what produced a signal that holds up.

## Interpretability (SHAP)

![SHAP values for the XGBoost model](shap_summary.png)

SHAP values come from XGBoost's built-in TreeSHAP (`pred_contribs=True`) on the model refit to all patients. The largest effects:

- **First sexual intercourse:** a later age lowers predicted risk.
- **Hormonal contraceptives (years):** longer use raises predicted risk.
- **Age and number of pregnancies.**
- **Any STD history:** raises predicted risk.

These patterns are consistent with known risk factors in the clinical literature. SHAP describes what the model learned from this data; it does not establish cause.

## Run it

```bash
pip install -r requirements.txt
python train.py
```

This writes `results.json` and `shap_summary.png`. The full UCI dataset (about 100 KB) is included in `data/`.

## Tech stack

Python, XGBoost, scikit-learn, pandas, NumPy, matplotlib
