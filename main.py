"""
Benchmarking Linear, Kernel and Tree-Based Classifiers
for Predictive Maintenance on Imbalanced Industrial Sensor Data

Team Aliens - MAP6114 Machine Learning, Fall 2026

Usage:
    python main.py                 # full run
    python main.py --fast          # smoke test (< 2 min)
    python main.py --help          # all options
"""

import argparse
import sys
import io
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
# Warnings: show every distinct warning (including Deprecation/FutureWarnings
# raised by our own code) instead of silencing them globally. Only known
# third-party noise is filtered. ConvergenceWarnings are additionally counted
# per model (see src/reproducibility.py) and reported in cv_results.csv.
warnings.simplefilter("default")
# seaborn.heatmap calls Colormap.set_bad, which matplotlib plans to deprecate.
warnings.filterwarnings("ignore", category=PendingDeprecationWarning,
                        message=r"The set_bad function will be deprecated")

import joblib
import numpy as np

from sklearn.model_selection import train_test_split

from src import config
from src.config import SEED, CV_FOLDS, NUM_FEATURES, TARGET
from src.data import load_raw, clean, split
from src.preprocessing import build_preprocessor
from src.models import get_model_specs
from src.evaluation import (
    get_cv, tune_models, select_thresholds, cv_results_table,
    evaluate_test, significance_table, print_classification_reports,
)
from src.analysis import (
    pca_variance_analysis, run_ablation, run_smote_comparison,
    logreg_coefficients, tree_importances, permutation_analysis,
    shap_analysis,
)
from src import plots
from src.reproducibility import start_log, RunMetadata

# --fast smoke-test settings (never used for reported results)
FAST_N_ROWS = 2000
FAST_CV_FOLDS = 3
FAST_N_BOOT = 200


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark classifiers on the AI4I 2020 dataset.")
    parser.add_argument(
        "--fast", action="store_true",
        help=f"smoke test: {FAST_N_ROWS}-row stratified subsample, "
             f"{FAST_CV_FOLDS}-fold CV, one configuration per model, "
             f"{FAST_N_BOOT} bootstrap resamples, no SHAP")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="skip the interactions/PCA ablation study")
    parser.add_argument("--skip-smote", action="store_true",
                        help="skip the SMOTE vs class-weight comparison")
    parser.add_argument("--out-dir", default=str(config.OUT_DIR),
                        help="output directory (default: outputs/)")
    return parser.parse_args(argv)


def shrink_grids(specs):
    """Keep only the first value of every hyperparameter (--fast mode)."""
    for spec in specs.values():
        spec["grid"] = {k: v[:1] for k, v in spec["grid"].items()}
    return specs


def section(num, title):
    print(f"\n{'=' * 70}")
    print(f"{num}. {title}")
    print("=" * 70)


def main(argv=None):
    args = parse_args(argv)
    config.set_out_dir(args.out_dir)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    start_log(config.OUT_DIR / "run.log")
    n_folds = FAST_CV_FOLDS if args.fast else CV_FOLDS
    run_meta = RunMetadata(config.OUT_DIR / "run_metadata.json",
                           argv=sys.argv if argv is None else ["main.py"] + list(argv),
                           cv_folds=n_folds, fast=args.fast)
    if args.fast:
        print(f"*** FAST MODE: smoke test only, results are not meaningful ***")

    # ── 1. Load & clean ──────────────────────────────────────────────
    section(1, "DATA LOADING & CLEANING")
    df_raw = load_raw()
    df = clean(df_raw)
    if args.fast:
        df, _ = train_test_split(df, train_size=FAST_N_ROWS,
                                 stratify=df[TARGET], random_state=SEED)
        df_raw = df_raw.loc[df.index]
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
    if args.fast:
        shrink_grids(model_specs)
    print(f"Models to benchmark: {len(model_specs)}")
    for name, spec in model_specs.items():
        grid_size = 1
        for v in spec["grid"].values():
            grid_size *= len(v)
        print(f"  {name}: {grid_size} configs x {n_folds} folds")

    # ── 6. Hyperparameter tuning ──────────────────────────────────────
    section(6, "HYPERPARAMETER TUNING")
    cv = get_cv(n_folds)
    preprocessor = build_preprocessor(with_interactions=True, with_pca=False)
    cv_results, estimators = tune_models(
        model_specs, preprocessor, X_train, y_train, cv)

    # ── 7. Threshold selection ────────────────────────────────────────
    section(7, "THRESHOLD SELECTION")
    threshold_warnings = {}
    thresholds = select_thresholds(estimators, X_train, y_train, cv,
                                   warning_counts=threshold_warnings,
                                   model_specs=model_specs)

    cv_table = cv_results_table(cv_results, threshold_warnings)
    cv_table.to_csv(config.OUT_DIR / "cv_results.csv", index=False)
    on_edge = cv_table.loc[cv_table["best_on_grid_edge"], ["model", "edge_params"]]
    if len(on_edge):
        print("\nBest value on the grid edge (>= 3-value numeric grids):")
        print(on_edge.to_string(index=False))
    print("\nConvergenceWarnings per model "
          "(CV fits / refit / threshold-selection fits):")
    print(cv_table[["model", "convergence_warnings", "refit_convergence_warnings",
                    "threshold_convergence_warnings"]].to_string(index=False))

    # ── 8. Test-set evaluation ────────────────────────────────────────
    section(8, "TEST-SET EVALUATION")
    metrics_df, probas = evaluate_test(estimators, thresholds, X_test, y_test)
    print("\n" + metrics_df.to_string())
    metrics_df.to_csv(config.OUT_DIR / "test_metrics.csv")

    # Persist the fitted state so later analyses can run without retuning
    # (see analyze.py). Test rows are stored by index into the cleaned data.
    joblib.dump({
        "estimators": estimators,
        "thresholds": thresholds,
        "probas": probas,
        "metrics_df": metrics_df,
        "cv_results": cv_results,
        "cv_table": cv_table,
        "train_index": X_train.index.to_numpy(),
        "test_index": X_test.index.to_numpy(),
        "fast_mode": args.fast,
        "git_commit": run_meta.data["git_commit"],
    }, config.OUT_DIR / "_state.joblib")
    print(f"Fitted state saved to {config.OUT_DIR / '_state.joblib'}")

    best_name = metrics_df.index[0]
    # Models scored by decision_function (SVMs): no probabilities
    score_only = [n for n, e in estimators.items()
                  if not hasattr(e, "predict_proba")]

    # ── 9. Ablation ───────────────────────────────────────────────────
    section(9, "ABLATION STUDY")
    if args.skip_ablation:
        print("Skipped (--skip-ablation).")
    else:
        abl_pivot = run_ablation(model_specs, X_train, y_train, cv)
        print("\n" + abl_pivot.round(4).to_string())
        abl_pivot.to_csv(config.OUT_DIR / "ablation_results.csv")
        plots.plot_ablation_heatmap(abl_pivot)

    # ── 10. SMOTE comparison ──────────────────────────────────────────
    section(10, "SMOTE vs CLASS_WEIGHT")
    if args.skip_smote:
        print("Skipped (--skip-smote).")
    else:
        smote_df = run_smote_comparison(model_specs, cv_results,
                                        X_train, y_train, cv)
        smote_df.to_csv(config.OUT_DIR / "smote_comparison.csv", index=False)

    # ── 11. Evaluation plots ──────────────────────────────────────────
    section(11, "EVALUATION PLOTS")
    plots.plot_pr_curves(metrics_df, probas, y_test)
    plots.plot_roc_curves(metrics_df, probas, y_test)
    plots.plot_confusion_matrices(metrics_df, probas, thresholds, y_test)
    plots.plot_threshold_tradeoff(best_name, probas, thresholds, y_test,
                                  score_only=score_only)
    print("Evaluation plots saved.")

    # ── 12. Calibration ───────────────────────────────────────────────
    section(12, "CALIBRATION ANALYSIS")
    plots.plot_calibration(probas, y_test, score_only=score_only)
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

    if args.fast:
        print("SHAP skipped (--fast).")
    else:
        try:
            shap_values, X_shap = shap_analysis(estimators[best_name], X_test)
            if shap_values is not None:
                plots.plot_shap(shap_values, X_shap, best_name)
                print("SHAP plots saved.")
        except Exception as e:
            print(f"SHAP skipped: {e}")

    # ── 14. Statistical significance ──────────────────────────────────
    section(14, "STATISTICAL SIGNIFICANCE")
    significance_table(metrics_df, probas, y_test,
                       n_boot=FAST_N_BOOT if args.fast else 3000)

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
    n_figs = len(list(config.FIG_DIR.glob("*.png")))
    print(f"Outputs: {config.OUT_DIR}")
    print(f"Figures: {n_figs} PNGs in {config.FIG_DIR}")
    run_meta.finish()
    print(f"Wall-clock time: {run_meta.data['wall_clock_seconds']:.1f} s")


if __name__ == "__main__":
    main()
