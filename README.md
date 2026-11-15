# Benchmarking Classifiers for Predictive Maintenance on Imbalanced Industrial Sensor Data

**Team Aliens** — Jacopo Ghinato, Giorgio Bianchini, Pratigya Paudel  
MAP6114 Machine Learning, Fall 2026 — University of West Florida

---

## Overview

This project benchmarks **11 classifier families** on the [AI4I 2020 Predictive Maintenance dataset](https://doi.org/10.24432/C5HS5C) (10,000 records, 3.4% failure rate), treating failure prediction as an imbalanced binary classification problem.

### Models

| Family | Models |
|---|---|
| Linear | Logistic Regression (L1, L2, ElasticNet), Linear SVM |
| Kernel / Instance | RBF SVM, k-Nearest Neighbours |
| Probabilistic | Gaussian Naive Bayes |
| Tree-based | Decision Tree, Random Forest, Gradient Boosting, XGBoost |

### Key Results

| Model | PR-AUC | ROC-AUC | Macro-F1 |
|---|---|---|---|
| **Gradient Boosting** | **0.801** | 0.972 | **0.868** |
| XGBoost | 0.798 | 0.975 | 0.855 |
| Random Forest | 0.751 | 0.966 | 0.835 |
| RBF SVM | 0.679 | 0.965 | 0.785 |
| Gaussian Naive Bayes | 0.181 | 0.886 | 0.616 |

Gradient Boosting and XGBoost are statistically tied (bootstrap p = 0.46) and significantly outperform all other models.

---

## Project Structure

```
├── main.py                  # Entry point — runs the full pipeline
├── src/
│   ├── config.py            # Constants, paths, feature names, seed
│   ├── data.py              # Load, clean, train/test split
│   ├── preprocessing.py     # Sklearn pipeline builders
│   ├── models.py            # Model specs and hyperparameter grids
│   ├── evaluation.py        # Tuning, thresholds, metrics, bootstrap tests
│   ├── analysis.py          # Ablation, SMOTE, feature importance, SHAP
│   └── plots.py             # All 17 figures
├── data/
│   └── ai4i2020.csv         # Downloaded from UCI (not tracked in git)
├── outputs/
│   ├── figures/             # 17 PNG figures (regenerated on each run)
│   │   ├── 01_class_distribution.png
│   │   ├── 02_feature_distributions.png
│   │   ├── 03_correlation_heatmap.png
│   │   ├── 04_pairwise_scatter.png
│   │   ├── 05_pca_variance.png
│   │   ├── 06_ablation_heatmap.png
│   │   ├── 07_pr_curves.png
│   │   ├── 08_roc_curves.png
│   │   ├── 09_confusion_matrices.png
│   │   ├── 10_threshold_tradeoff.png
│   │   ├── 11_calibration.png
│   │   ├── 12_logreg_coefficients.png
│   │   ├── 13_permutation_importance.png
│   │   ├── 14_shap_summary.png
│   │   ├── 15_shap_bar.png
│   │   ├── 16_model_ranking.png
│   │   └── 17_multi_metric_comparison.png
│   ├── test_metrics.csv
│   ├── ablation_results.csv
│   └── smote_comparison.csv
└── README.md
```

---

## Setup

**Requirements:** Python 3.10+

```bash
pip install scikit-learn numpy pandas matplotlib seaborn imbalanced-learn xgboost shap
```

**Download the dataset:**

```bash
mkdir -p data
curl -L -o data/ai4i2020.csv \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
```

---

## Usage

```bash
python main.py
```

The full pipeline takes ~10–15 minutes and runs through:

1. **Data loading & cleaning** — drops IDs and failure-mode flags (target leakage)
2. **EDA** — class distribution, feature distributions, correlations, scatter plots
3. **Stratified 80/20 split** — preserves 3.4% failure rate in both sets
4. **Preprocessing** — one-hot encoding, standardisation, pairwise interaction terms, PCA analysis
5. **Hyperparameter tuning** — GridSearchCV with stratified 5-fold CV, scored by PR-AUC
6. **Threshold selection** — maximises F1 on out-of-fold predictions (not default 0.5)
7. **Test-set evaluation** — PR-AUC, ROC-AUC, Macro-F1, confusion matrices
8. **Ablation study** — interactions vs PCA across 4 models
9. **SMOTE comparison** — SMOTE vs `class_weight='balanced'`
10. **Feature importance** — logistic regression coefficients, Gini importance, permutation importance, SHAP
11. **Statistical significance** — paired bootstrap test (3,000 resamples) on PR-AUC
12. **17 publication-ready figures** saved to `outputs/figures/`

---

## Research Questions

**RQ1: How do different classifier families compare on the rare failure class?**  
Tree ensembles (Gradient Boosting, XGBoost) achieve PR-AUC ~0.80, significantly outperforming linear models (~0.55) and Gaussian NB (0.18). ROC-AUC separates models poorly (0.89–0.97 for all), confirming PR-AUC as the right primary metric for imbalanced data.

**RQ2: Do interaction terms and PCA help simpler models?**  
Interaction terms boost Logistic Regression by +0.06 PR-AUC, partially closing the gap to ensembles. PCA consistently hurts all models (−0.15 to −0.22), discarding minority-class-discriminative variance.

**RQ3: Which features drive failure predictions?**  
All methods converge: **Torque**, **Rotational speed**, **Tool wear**, and their pairwise interactions are the primary drivers — matching the dataset's known physical failure rules (power = torque × speed, overstrain = tool wear × torque).

---

## Dataset

**AI4I 2020 Predictive Maintenance Dataset**  
Matzka, S. (2020). UCI Machine Learning Repository. [https://doi.org/10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C)

- 10,000 records from a simulated industrial milling process
- 6 features: product type (L/M/H), air temperature, process temperature, rotational speed, torque, tool wear
- Binary target: machine failure (3.39% positive rate)
- 5 failure modes (TWF, HDF, PWF, OSF, RNF) — dropped before training to avoid target leakage

---

## References

1. Zonta et al. (2020). Predictive maintenance in the Industry 4.0: A systematic literature review. *Computers & Industrial Engineering*, 150.
2. Carvalho et al. (2019). A systematic literature review of machine learning methods applied to predictive maintenance. *Computers & Industrial Engineering*, 137.
3. Matzka, S. (2020). Explainable AI for predictive maintenance applications. *IEEE AI4I Conference*.
4. Saito & Rehmsmeier (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, 10(3).
5. Pedregosa et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12.
