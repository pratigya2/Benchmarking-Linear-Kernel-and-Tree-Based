"""
Benchmarking Linear, Kernel and Tree-Based Classifiers
for Predictive Maintenance on Imbalanced Industrial Sensor Data

Team Aliens - MAP6114 Machine Learning, Fall 2026

Usage:
    python main.py
"""

import sys
import io
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np

from src.config import SEED, OUT_DIR, FIG_DIR, NUM_FEATURES, TARGET
from src.data import load_raw, clean, split
from src.preprocessing import build_preprocessor
from src.models import get_model_specs, get_smote_specs
from src.evaluation import (
    get_cv, tune_models, select_thresholds,
    evaluate_test, significance_table, print_classification_reports,
)
from src.analysis import (
    pca_variance_analysis, run_ablation, run_smote_comparison,
    logreg_coefficients, tree_importances, permutation_analysis,
    shap_analysis,
)
from src import plots

np.random.seed(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def section(num, title):
    print(f"\n{'=' * 70}")
    print(f"{num}. {title}")
    print("=" * 70)


def main():
    # ── 1. Load & clean ──────────────────────────────────────────────
    section(1, "DATA LOADING & CLEANING")
    df_raw = load_raw()
    df = clean(df_raw)
    print(f"Raw shape: {df_raw.shape}  ->  Cleaned shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")

    # ── 2. EDA ────────────────────────────────────────────────────────
    section(2, "EXPLORATORY DATA ANALYSIS")
    y = df[TARGET]
    print(f"Failure rate: {y.mean():.4f} ({y.sum()}/{len(y)})")
    print(f"\nDescriptive statistics:")
    print(df[NUM_FEATURES].describe().round(2).to_string())

    plots.plot_class_distribution(df, df_raw)
    plots.plot_feature_distributions(df)
    plots.plot_correlation_heatmap(df)
    plots.plot_pairwise_scatter(df)
    print("EDA figures saved.")

    # ── 3. Split ──────────────────────────────────────────────────────
    section(3, "TRAIN / TEST SPLIT")
    X_train, X_test, y_train, y_test = split(df)
    print(f"Train: {X_train.shape[0]} ({y_train.sum()} failures)")
    print(f"Test:  {X_test.shape[0]} ({y_test.sum()} failures)")

    # ── 4. Preprocessing ──────────────────────────────────────────────
    section(4, "PREPROCESSING")
    pca_model, cumvar, n95, n_feat = pca_variance_analysis(X_train)
    print(f"Features after interactions: {n_feat}")
    print(f"PCA components for 95% variance: {n95}")
    plots.plot_pca_variance(pca_model, cumvar, n95)

    # ── 5. Model definitions ──────────────────────────────────────────
    section(5, "MODEL DEFINITIONS")
    model_specs = get_model_specs(y_train)
    print(f"Models to benchmark: {len(model_specs)}")
    for name, spec in model_specs.items():
        grid_size = 1
        for v in spec["grid"].values():
            grid_size *= len(v)
        print(f"  {name}: {grid_size} configs x 5 folds")

    # ── 6. Hyperparameter tuning ──────────────────────────────────────
    section(6, "HYPERPARAMETER TUNING")
    cv = get_cv()
    preprocessor = build_preprocessor(with_interactions=True, with_pca=False)
    cv_results, estimators = tune_models(
        model_specs, preprocessor, X_train, y_train, cv)

    # ── 7. Threshold selection ────────────────────────────────────────
    section(7, "THRESHOLD SELECTION")
    thresholds = select_thresholds(estimators, X_train, y_train, cv)

    # ── 8. Test-set evaluation ────────────────────────────────────────
    section(8, "TEST-SET EVALUATION")
    metrics_df, probas = evaluate_test(estimators, thresholds, X_test, y_test)
    print("\n" + metrics_df.to_string())
    metrics_df.to_csv(OUT_DIR / "test_metrics.csv")

    best_name = metrics_df.index[0]

    # ── 9. Ablation ───────────────────────────────────────────────────
    section(9, "ABLATION STUDY")
    abl_pivot = run_ablation(model_specs, X_train, y_train, cv)
    print("\n" + abl_pivot.round(4).to_string())
    abl_pivot.to_csv(OUT_DIR / "ablation_results.csv")
    plots.plot_ablation_heatmap(abl_pivot)

    # ── 10. SMOTE comparison ──────────────────────────────────────────
    section(10, "SMOTE vs CLASS_WEIGHT")
    smote_specs = get_smote_specs()
    smote_df = run_smote_comparison(smote_specs, cv_results,
                                    X_train, y_train, cv)
    smote_df.to_csv(OUT_DIR / "smote_comparison.csv", index=False)

    # ── 11. Evaluation plots ──────────────────────────────────────────
    section(11, "EVALUATION PLOTS")
    plots.plot_pr_curves(metrics_df, probas, y_test)
    plots.plot_roc_curves(metrics_df, probas, y_test)
    plots.plot_confusion_matrices(metrics_df, probas, thresholds, y_test)
    plots.plot_threshold_tradeoff(best_name, probas, thresholds, y_test)
    print("Evaluation plots saved.")

    # ── 12. Calibration ───────────────────────────────────────────────
    section(12, "CALIBRATION ANALYSIS")
    plots.plot_calibration(probas, y_test)
    print("Calibration plot saved.")

    # ── 13. Feature importance ────────────────────────────────────────
    section(13, "FEATURE IMPORTANCE")

    coef_df = logreg_coefficients(estimators["LogReg_L2"])
    print("LogReg L2 coefficients (top 15):")
    print(coef_df.head(15).to_string(index=False))
    plots.plot_logreg_coefficients(coef_df)

    for name, imp_df in tree_importances(estimators).items():
        print(f"\n{name} - top 10 (Gini):")
        print(imp_df.head(10).to_string(index=False))

    perm_df = permutation_analysis(estimators[best_name], X_test, y_test)
    print(f"\nPermutation importance ({best_name}):")
    print(perm_df.head(15).to_string(index=False))
    plots.plot_permutation_importance(perm_df, best_name)

    try:
        shap_values, X_shap = shap_analysis(estimators[best_name], X_test)
        if shap_values is not None:
            plots.plot_shap(shap_values, X_shap, best_name)
            print("SHAP plots saved.")
    except Exception as e:
        print(f"SHAP skipped: {e}")

    # ── 14. Statistical significance ──────────────────────────────────
    section(14, "STATISTICAL SIGNIFICANCE")
    significance_table(metrics_df, probas, y_test)

    # ── 15. Final ranking ─────────────────────────────────────────────
    section(15, "FINAL MODEL RANKING")
    print(metrics_df[["PR-AUC", "ROC-AUC", "Macro-F1"]].round(4).to_string())
    plots.plot_model_ranking(metrics_df, y_test)

    # ── 16. Classification reports ────────────────────────────────────
    section(16, "CLASSIFICATION REPORTS - TOP 3")
    print_classification_reports(
        metrics_df, probas, thresholds, estimators, X_test, y_test)

    # ── Done ──────────────────────────────────────────────────────────
    section("", "DONE")
    n_figs = len(list(FIG_DIR.glob("*.png")))
    print(f"Outputs: {OUT_DIR}")
    print(f"Figures: {n_figs} PNGs in {FIG_DIR}")


if __name__ == "__main__":
    main()
