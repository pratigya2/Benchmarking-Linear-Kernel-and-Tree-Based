"""Post-hoc analyses that run on a fitted pipeline state (Phase 3).

``run_all`` needs only the saved state (``_state.joblib``) and the data, so it
runs both at the end of ``main.py`` and from ``analyze.py`` without retuning.
Nothing here feeds back into model selection: test-set analyses are
descriptive and happen after the final evaluation.
"""

import numpy as np
import pandas as pd

from . import config, plots
from .analysis import (
    logreg_coefficient_comparison, l1_regularisation_path,
    raw_permutation_importance, importance_agreement, impurity_importances,
    MODEL_FAMILIES,
)
from .preprocessing import get_feature_names
from .config import FAILURE_MODES, TARGET
from .data import load_raw
from . import error_analysis as ea


def _save_table(df, name, index=False):
    config.TAB_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.TAB_DIR / name, index=index)


def _header(title):
    print(f"\n--- {title} ---")


# ── 3.1 Failure modes and label noise ─────────────────────────────────

def label_noise_table(df_raw):
    """Counts of each failure-mode flag by target label, plus consistency
    checks between the target and the flags."""
    y = df_raw[TARGET]
    flags = df_raw[FAILURE_MODES]
    any_flag = flags.sum(axis=1) > 0
    rows = []
    for m in FAILURE_MODES:
        f = flags[m] == 1
        rows.append({"group": "mode", "item": m, "n_rows": int(f.sum()),
                     "machine_failure_1": int((f & (y == 1)).sum()),
                     "machine_failure_0": int((f & (y == 0)).sum())})
    checks = {
        "any_mode_flag": any_flag,
        "failure_without_mode_flag": (y == 1) & ~any_flag,
        "mode_flag_without_failure": (y == 0) & any_flag,
        "multiple_mode_flags": flags.sum(axis=1) > 1,
    }
    for name, mask in checks.items():
        rows.append({"group": "consistency", "item": name,
                     "n_rows": int(mask.sum()),
                     "machine_failure_1": int((mask & (y == 1)).sum()),
                     "machine_failure_0": int((mask & (y == 0)).sum())})
    out = pd.DataFrame(rows)
    out.attrs["n_rows"] = len(df_raw)
    return out


# ── 3.2 Logistic-regression coefficients ───────────────────────────────

def logreg_analysis(state, X_train, y_train, fast=False):
    coef = logreg_coefficient_comparison(state["estimators"])
    _save_table(coef, "logreg_coefficients.csv", index=True)
    zeros = (coef == 0).sum()
    print("Exactly-zero coefficients at the selected C: "
          + ", ".join(f"{m} {int(z)}/{len(coef)}" for m, z in zeros.items()))
    plots.plot_logreg_coef_comparison(coef)

    Cs = np.logspace(-3, 2, 5 if fast else 20)
    path, summary = l1_regularisation_path(
        state["estimators"]["LogReg_L1"], X_train, y_train, Cs)
    _save_table(path, "l1_path_coefficients.csv", index=True)
    _save_table(summary, "l1_path.csv")
    print("L1 path (C, non-zero / total):")
    print(summary.to_string(index=False))
    plots.plot_l1_path(path, summary)


def score_only_models(state):
    """Models scored with decision_function (no predict_proba): the SVMs."""
    return [n for n, e in state["estimators"].items()
            if not hasattr(e, "predict_proba")]


# ── 3.5 Error analysis by failure mode ─────────────────────────────────

# XGBoost and GB are statistically tied on test PR-AUC (significance.csv),
# so the per-row error analysis covers both.
ERROR_ANALYSIS_MODELS = ["XGBoost", "GradientBoosting"]


def error_analysis(state, y_test, models=ERROR_ANALYSIS_MODELS):
    err = ea.error_by_failure_mode(state, y_test)
    _save_table(err, "error_by_failure_mode.csv")
    show = ["model", "TP", "FP", "FN", "fp_on_flagged_negatives"] + \
           [f"recall_{m}" for m in ea.MODES]
    print(err[show].round(2).to_string(index=False))
    print("Test failures per mode: " + ", ".join(
        f"{m} {int(err[f'n_{m}'].iloc[0])}" for m in ea.MODES))
    plots.plot_recall_by_failure_mode(err, ea.MODES)

    models = [m for m in models if m in state["probas"]]
    fn = ea.false_negatives(state, y_test, models)
    _save_table(fn, "best_model_false_negatives.csv")
    print("False negatives by mode:")
    print(ea.fn_counts_by_mode(fn).to_string())

    preds = ea.predictions(state)
    plots.plot_errors_on_physics(ea.raw_test_rows(y_test.index),
                                 {m: preds[m] for m in models},
                                 ea.OSF_STRAIN_LIMITS, ea.PWF_POWER_BAND)
    return err


# ── 3.3 Gaussian Naive Bayes diagnosis ─────────────────────────────────

def nb_diagnosis(state, X_train, y_train, err):
    """Correlation checks on the training set (NB assumes features are
    independent within each class) and NB's recall per failure mode."""
    air, proc = "Air temperature [K]", "Process temperature [K]"
    rows = [{"check": "pearson_r_air_vs_process_temp_train",
             "value": X_train[air].corr(X_train[proc]),
             "note": "raw training set, both classes"}]
    for c in (0, 1):
        sub = X_train[y_train == c]
        rows.append({"check": f"pearson_r_air_vs_process_temp_train_class{c}",
                     "value": sub[air].corr(sub[proc]),
                     "note": f"within class {c} (n = {len(sub)})"})

    # Correlations among the 27 features NB actually receives
    est = state["estimators"]["GaussianNB"]
    Xt = pd.DataFrame(est.named_steps["pre"].transform(X_train),
                      columns=get_feature_names(est.named_steps["pre"]))
    for c in (0, 1):
        r = Xt[(y_train == c).to_numpy()].corr().abs().to_numpy()
        upper = r[np.triu_indices_from(r, k=1)]
        rows += [
            {"check": f"max_abs_r_between_model_features_class{c}",
             "value": float(np.nanmax(upper)),
             "note": f"{Xt.shape[1]} pipeline features, within class {c}"},
            {"check": f"n_feature_pairs_abs_r_gt_0.9_class{c}",
             "value": int(np.sum(upper > 0.9)),
             "note": f"of {len(upper)} pairs, within class {c}"},
        ]

    nb = err.set_index("model").loc["GaussianNB"]
    for k in ["TP", "FP", "FN"]:
        rows.append({"check": f"test_{k}", "value": int(nb[k]),
                     "note": "GaussianNB at its CV-selected threshold"})
    for m in ea.MODES:
        n = int(nb[f"n_{m}"])
        rows.append({"check": f"test_recall_{m}", "value": nb[f"recall_{m}"],
                     "note": f"{n - int(nb[f'fn_{m}'])}/{n} test failures detected"})
    return pd.DataFrame(rows)


# ── 3.6 Feature-importance agreement across families ───────────────────

def importance_analysis(state, X_test, y_test, fast=False):
    perm = raw_permutation_importance(state["estimators"], X_test, y_test,
                                      n_repeats=2 if fast else 10)
    _save_table(perm, "permutation_raw.csv")
    ranks = perm.pivot(index="model", columns="variable", values="rank")
    ranks = ranks.loc[[m for m in MODEL_FAMILIES if m in ranks.index]]
    print("Rank of each raw variable (1 = most important):")
    print(ranks.to_string())

    rho, summary = importance_agreement(perm)
    _save_table(rho, "importance_spearman.csv", index=True)
    _save_table(summary, "importance_agreement_summary.csv")
    print("Spearman rank correlation by family pair:")
    print(summary.round(2).to_string(index=False))
    plots.plot_importance_by_family(perm)
    plots.plot_importance_agreement(rho, MODEL_FAMILIES)

    imp = impurity_importances(state["estimators"])
    _save_table(imp, "impurity_importance.csv")
    plots.plot_impurity_importance(imp)
    return perm


# ── Driver ────────────────────────────────────────────────────────────

def run_all(state, X_train, X_test, y_train, y_test, fast=False):
    """Run every Phase 3 analysis and write tables/ and figures/."""
    rows = X_train.index.union(X_test.index)
    df_raw = load_raw().loc[rows]

    _header("3.1 Failure modes and label noise")
    noise = label_noise_table(df_raw)
    _save_table(noise, "label_noise.csv")
    print(noise.to_string(index=False))
    plots.plot_failure_modes(noise)

    _header("3.2 Logistic-regression coefficients and L1 path")
    logreg_analysis(state, X_train, y_train, fast)

    _header("3.4 Confusion matrices of all models")
    plots.plot_confusion_all(state["metrics_df"], state["probas"],
                             state["thresholds"], y_test,
                             score_only=score_only_models(state))

    _header("3.5 Error analysis by failure mode")
    err = error_analysis(state, y_test)

    _header("3.3 Gaussian Naive Bayes diagnosis")
    nb = nb_diagnosis(state, X_train, y_train, err)
    _save_table(nb, "nb_diagnosis.csv")
    print(nb.to_string(index=False))

    _header("3.6 Feature-importance agreement across model families")
    importance_analysis(state, X_test, y_test, fast)
