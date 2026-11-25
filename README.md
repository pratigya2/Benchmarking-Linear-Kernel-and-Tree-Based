# Benchmarking Classifiers for Predictive Maintenance on Imbalanced Industrial Sensor Data

**Team Aliens** — Jacopo Ghinato, Giorgio Bianchini, Pratigya Paudel  
MAP6114 Machine Learning, Fall 2026 — University of West Florida

---

## Overview

This project benchmarks **11 classifiers from 4 model families** on the [AI4I 2020 Predictive Maintenance dataset](https://doi.org/10.24432/C5HS5C) (10,000 records, 3.4% failure rate), treating failure prediction as an imbalanced binary classification problem.

### Models

| Family | Models |
|---|---|
| Linear | Logistic Regression (L1, L2, ElasticNet), Linear SVM |
| Kernel / Instance | RBF SVM, k-Nearest Neighbours |
| Probabilistic | Gaussian Naive Bayes |
| Tree-based | Decision Tree, Random Forest, Gradient Boosting, XGBoost |

### Key Results

Test set: 2,000 rows, 68 failures. PR-AUC with bootstrap 95% CI (2,000 resamples); TP/FP/FN at the threshold selected on out-of-fold training predictions. Source: `outputs/test_metrics.csv`, `outputs/significance.csv`.

| Model | PR-AUC [95% CI] | ROC-AUC | Macro-F1 | TP/FP/FN |
|---|---|---|---|---|
| **XGBoost** | **0.802** [0.708, 0.885] | 0.975 | 0.882 | 47/7/21 |
| Gradient Boosting | 0.798 [0.704, 0.879] | 0.961 | 0.883 | 53/16/15 |
| Random Forest | 0.763 [0.666, 0.851] | 0.962 | 0.842 | 50/26/18 |
| RBF SVM | 0.679 [0.567, 0.778] | 0.965 | 0.785 | 41/31/27 |
| Decision Tree | 0.654 [0.533, 0.765] | 0.909 | 0.825 | 40/13/28 |
| Linear SVM | 0.554 [0.439, 0.680] | 0.957 | 0.779 | 41/34/27 |
| LogReg ElasticNet | 0.547 [0.433, 0.670] | 0.959 | 0.788 | 49/48/19 |
| LogReg L1 | 0.546 [0.435, 0.673] | 0.959 | 0.786 | 49/49/19 |
| LogReg L2 | 0.545 [0.433, 0.671] | 0.959 | 0.788 | 49/48/19 |
| k-NN | 0.483 [0.367, 0.618] | 0.905 | 0.752 | 33/26/35 |
| Gaussian Naive Bayes | 0.181 [0.129, 0.263] | 0.886 | 0.616 | 33/137/35 |

Paired bootstrap test of XGBoost against each other model (one-sided, Holm correction over 10 comparisons): XGBoost and Gradient Boosting are statistically tied (Δ PR-AUC 0.003, Holm p = 0.41); Random Forest is not significantly worse after correction (Δ 0.039, p = 0.040 before and 0.080 after Holm); the other 8 models are significantly worse (Holm p ≤ 0.015).

---

## Project Structure

```
├── main.py                  # Entry point — runs the full pipeline
├── analyze.py               # Reloads outputs/_state.joblib and reruns the post-hoc analyses (no retuning)
├── requirements.txt         # Pinned dependency versions
├── src/
│   ├── config.py            # Constants, paths, feature names, seed
│   ├── data.py              # Load, clean, train/test split
│   ├── preprocessing.py     # Sklearn pipeline builders
│   ├── models.py            # Model specs and hyperparameter grids
│   ├── evaluation.py        # Tuning, thresholds, metrics, bootstrap tests
│   ├── analysis.py          # Ablation, SMOTE, coefficients, feature importance, SHAP
│   ├── error_analysis.py    # Errors by failure mode (flags joined after prediction)
│   ├── post_analyses.py     # Post-hoc analyses run from the saved state (main.py and analyze.py)
│   ├── reproducibility.py   # Run log, run metadata, ConvergenceWarning counting
│   └── plots.py             # All 26 figures (300 dpi)
├── data/
│   └── ai4i2020.csv         # Downloaded from UCI (not tracked in git)
├── outputs/
│   ├── figures/             # 26 PNG figures (regenerated on each run)
│   │   ├── 01_class_distribution.png
│   │   ├── 01b_failure_modes.png
│   │   ├── 02_feature_distributions.png
│   │   ├── 03_correlation_heatmap.png
│   │   ├── 04_pairwise_scatter.png
│   │   ├── 05_pca_variance.png
│   │   ├── 06_ablation_heatmap.png
│   │   ├── 07_pr_curves.png
│   │   ├── 08_roc_curves.png
│   │   ├── 09_confusion_matrices.png
│   │   ├── 09b_confusion_all.png
│   │   ├── 10_threshold_tradeoff.png
│   │   ├── 11_calibration.png
│   │   ├── 12_logreg_coefficients.png
│   │   ├── 12b_logreg_coef_comparison.png
│   │   ├── 12c_l1_path.png
│   │   ├── 13_permutation_importance.png
│   │   ├── 13b_gini_importance.png
│   │   ├── 14_shap_summary.png
│   │   ├── 15_shap_bar.png
│   │   ├── 16_model_ranking.png
│   │   ├── 17_multi_metric_comparison.png
│   │   ├── 18_recall_by_failure_mode.png
│   │   ├── 19_errors_on_physics.png
│   │   ├── 20_importance_by_family.png
│   │   └── 21_importance_agreement.png
│   ├── tables/              # Post-hoc analysis tables
│   │   ├── label_noise.csv
│   │   ├── logreg_coefficients.csv
│   │   ├── l1_path.csv, l1_path_coefficients.csv
│   │   ├── nb_diagnosis.csv
│   │   ├── error_by_failure_mode.csv
│   │   ├── best_model_false_negatives.csv   # XGBoost and Gradient Boosting
│   │   ├── permutation_raw.csv
│   │   ├── importance_spearman.csv, importance_agreement_summary.csv
│   │   ├── impurity_importance.csv
│   │   └── failure_rules_vs_importance.csv
│   ├── test_metrics.csv
│   ├── cv_results.csv       # Tuning summary per model (grid edge, convergence warnings)
│   ├── ablation_results.csv # Mean CV PR-AUC (+ _std.csv and _long.csv)
│   ├── smote_comparison.csv
│   ├── significance.csv     # Bootstrap CIs and Holm-corrected paired tests
│   ├── run_metadata.json    # Commit, versions, hardware, seed, runtime
│   ├── run.log              # Full console log of the run
│   └── _state.joblib        # Fitted models, thresholds, test scores
├── tests/
│   └── test_pipeline.py     # pytest suite (includes a --fast smoke run)
└── README.md
```

---

## Setup

**Requirements:** Python 3.12 (the version the reference results were produced with). Exact package versions are pinned in `requirements.txt`.

**macOS prerequisite:** XGBoost needs the OpenMP runtime, which is not bundled with the pip wheel. Without it, `import xgboost` fails with `Library not loaded: @rpath/libomp.dylib`.

```bash
brew install libomp
```

**Create the environment** (conda-forge channel; any Python 3.12 virtual environment works as well):

```bash
conda create -n aliens-ml -c conda-forge --override-channels python=3.12 -y
conda activate aliens-ml
pip install -r requirements.txt
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
python main.py                      # full run (writes to outputs/)
python main.py --fast               # smoke test, < 2 min; results not meaningful
python main.py --skip-ablation --skip-smote
python main.py --out-dir results/run_a
python analyze.py                   # rerun the post-hoc analyses from outputs/_state.joblib
python analyze.py --check-only      # only check that the saved models reproduce the test scores
```

| Option | Effect |
|---|---|
| `--fast` | 2,000-row stratified subsample, 3-fold CV, first grid value only, 200 bootstrap resamples, no SHAP |
| `--skip-ablation` | Skip the interactions/PCA ablation study |
| `--skip-smote` | Skip the SMOTE vs class-weight comparison |
| `--out-dir DIR` | Write all outputs to `DIR` (default `outputs/`) |

The full pipeline takes about 40 minutes on an Apple M3 and runs through:

1. **Data loading & cleaning** — drops IDs and failure-mode flags (target leakage)
2. **EDA** — class distribution, feature distributions, correlations, scatter plots
3. **Stratified 80/20 split** — preserves 3.4% failure rate in both sets
4. **Preprocessing** — one-hot encoding, standardisation, pairwise interaction terms, PCA analysis
5. **Hyperparameter tuning** — GridSearchCV with stratified 5-fold CV, scored by PR-AUC
6. **Threshold selection** — maximises F1 on out-of-fold predictions (not default 0.5)
7. **Test-set evaluation** — PR-AUC, ROC-AUC, Macro-F1, confusion matrices
8. **Ablation study** — interactions vs PCA for all 11 models (each configuration re-tuned; mean ± std over folds)
9. **SMOTE comparison** — SMOTE vs class weighting for LogReg L2, Random Forest, Gradient Boosting and XGBoost, with the same grids and CV folds as the main benchmark (SMOTE applied after scaling, before the interaction terms)
10. **Feature importance** — logistic regression coefficients, Gini importance, permutation importance, SHAP
11. **Statistical significance** — bootstrap 95% CIs of test PR-AUC and a paired bootstrap test of the best model vs every other model (2,000 resamples, one-sided, Holm-corrected); saved to `significance.csv`
12. **Post-hoc analyses** (also available from the saved state via `analyze.py`) — failure-mode label checks, logistic-regression coefficient comparison and L1 path, Naive Bayes diagnosis, confusion matrices for all models, error analysis by failure mode, permutation importance of the 6 raw variables for all models with cross-family agreement, built-in tree importances, failure rules vs importance; tables saved to `outputs/tables/`
13. **26 figures at 300 dpi** saved to `outputs/figures/`

---

## Research Questions

**RQ1: How do different classifier families compare on the rare failure class?**  
XGBoost (0.802) and Gradient Boosting (0.798) have the highest test PR-AUC, followed by Random Forest (0.763). The four linear models score 0.545–0.554 and Gaussian NB 0.181. ROC-AUC spans only 0.886–0.975 across all 11 models, while PR-AUC spans 0.181–0.802.

**RQ2: Do interaction terms and PCA help simpler models?**  
11-model ablation, CV PR-AUC, each configuration re-tuned (`outputs/ablation_results*.csv`):
- Pairwise interaction terms raise the four linear models by +0.062 to +0.067 (to 0.508–0.510) and XGBoost by +0.014. They lower the other six models, from −0.003 (Random Forest) to −0.112 (Gaussian NB). Fold standard deviations are 0.02–0.09.
- With interactions, the linear models (0.51) stay below the tree ensembles (0.74–0.80).
- PCA (95% variance) lowers CV PR-AUC for 10 of 11 models (−0.011 to −0.260 vs. no PCA). The exception is Gaussian NB (+0.061).

**RQ3: Which features drive failure predictions?**  
Permutation importance of the 6 raw input variables on the test set, for all 11 models (`outputs/tables/permutation_raw.csv`):
- Torque ranks first for k-NN, Gaussian NB and all four tree models. Air temperature ranks first for the four linear models and RBF SVM. Product type ranks last for 10 of 11 models.
- Rank agreement (Spearman over the 6 variables) is 0.89–1.00 among tree models and 0.83–1.00 among linear models, but 0.31–0.71 between the two groups.
- Torque, which enters the PWF and OSF rules, has median rank 1. Tool wear, which enters the OSF and TWF rules, has median rank 5.
- TWF is the least-detected failure mode: recall 0.0–0.3 for every model (`outputs/tables/error_by_failure_mode.csv`).

---

## Dataset

**AI4I 2020 Predictive Maintenance Dataset**  
Matzka, S. (2020). UCI Machine Learning Repository. [https://doi.org/10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C)

- 10,000 records from a simulated industrial milling process
- 6 features: product type (L/M/H), air temperature, process temperature, rotational speed, torque, tool wear
- Binary target: machine failure (3.39% positive rate)
- 5 failure modes (TWF, HDF, PWF, OSF, RNF) — dropped before training to avoid target leakage; used only after prediction, for the error analysis

---

## Known Limitations

- **ElasticNet does not converge.** With balanced class weights, the `saga` solver (the only scikit-learn solver for elastic net) reaches `max_iter=5000` in 55 of 75 CV fits and in the final refit. More iterations or a looser tolerance do not fix it. Results are reported as they are, and the counts are in `cv_results.csv`.
- **Grid edges.** These parameters were selected at the top of their grid:
  - Gradient Boosting `n_estimators=500` and `max_depth=7`, and XGBoost `max_depth=9`. Moving one step further changes CV PR-AUC by +0.001 to +0.007, below the fold standard deviation (0.035–0.048).
  - ElasticNet `C=100` and `l1_ratio=0.8`. The best CV PR-AUC at C = 1, 10 and 100 is 0.504, 0.503 and 0.508 (unconverged fits).
- **Gaussian NB ties.** Every `var_smoothing` value from 1e-15 to 1e-7 gives the same CV PR-AUC (0.1898), and 1e-5 gives 0.1898 as well. The first grid value is selected.
- **SMOTE.** It interpolates the one-hot `Type` columns, so synthetic rows can have fractional product-type values.
- **Label noise.**
  - 18 rows carry a failure-mode flag (all RNF) but have `Machine failure = 0`.
  - 9 failures carry no mode flag.
  - The UCI documentation lists 5 RNF cases; the CSV has 19 RNF flags.
- **Small test set.** Only 68 test failures, so the PR-AUC 95% CIs are about 0.18 wide (e.g. XGBoost [0.708, 0.885]).
- **Gradient Boosting weighting.** GB uses balanced sample weights for consistency with the other models. Adding them lowered GB's CV PR-AUC from 0.819 to 0.765.

---

## References

1. Zonta et al. (2020). Predictive maintenance in the Industry 4.0: A systematic literature review. *Computers & Industrial Engineering*, 150.
2. Carvalho et al. (2019). A systematic literature review of machine learning methods applied to predictive maintenance. *Computers & Industrial Engineering*, 137.
3. Matzka, S. (2020). Explainable AI for predictive maintenance applications. *IEEE AI4I Conference*.
4. Saito & Rehmsmeier (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, 10(3).
5. Pedregosa et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12.
