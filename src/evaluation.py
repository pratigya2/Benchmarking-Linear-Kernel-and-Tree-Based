import json
import numbers
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_auc_score, f1_score, confusion_matrix, classification_report,
)

from .config import SEED, CV_FOLDS
from .models import fit_params
from .reproducibility import (
    CountingPipeline, scoring_with_convergence, total_convergence_warnings,
    count_convergence, reemit,
)


def get_cv(n_splits=CV_FOLDS):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def tune_models(model_specs, preprocessor, X_train, y_train, cv=None):
    cv = cv or get_cv()
    results = {}
    estimators = {}

    for name, spec in model_specs.items():
        print(f"\n--- {name} ---")
        t0 = time.time()

        pipe = CountingPipeline([("pre", preprocessor), ("clf", spec["clf"])])
        # All models are tuned on PR-AUC (average precision works with
        # decision_function scores too, e.g. LinearSVM).
        scoring = "average_precision"

        # Second (pseudo-)metric counts ConvergenceWarnings per fit;
        # refit="score" keeps model selection on the main metric.
        gs = GridSearchCV(
            pipe, spec["grid"], scoring=scoring_with_convergence(scoring),
            cv=cv, n_jobs=-1, refit="score", return_train_score=True,
        )
        gs.fit(X_train, y_train, **fit_params(spec, y_train))
        elapsed = time.time() - t0

        print(f"  Best params: {gs.best_params_}")
        print(f"  Best CV {scoring}: {gs.best_score_:.4f}")
        print(f"  Time: {elapsed:.1f}s")
        n_conv = total_convergence_warnings(gs.cv_results_, gs.n_splits_)
        n_conv_refit = gs.best_estimator_.n_convergence_warnings_
        print(f"  ConvergenceWarnings: {n_conv} in "
              f"{len(gs.cv_results_['params']) * gs.n_splits_} CV fits, "
              f"{n_conv_refit} in refit")

        results[name] = {
            "best_params": gs.best_params_,
            "cv_score": gs.best_score_,
            "scoring": scoring,
            "time": elapsed,
            "summary": _cv_summary(name, gs, spec["grid"], scoring, elapsed),
        }
        estimators[name] = gs.best_estimator_

    return results, estimators


def grid_edge_check(best_params, grid):
    """Flag hyperparameters whose selected value lies on the grid boundary.

    Only numeric parameters are checked (``None`` is treated as the largest
    value, e.g. ``max_depth=None`` = unlimited). Parameters with >= 3 grid
    values set the edge flag; parameters with exactly 2 values are always
    "on the boundary" by construction, so they are only reported.

    Returns (best_on_grid_edge, edge_params, two_value_params_at_bound).
    """
    edge, two_value = [], []
    for param, values in grid.items():
        numeric = [v for v in values if v is not None]
        if not all(isinstance(v, numbers.Number) and not isinstance(v, bool)
                   for v in numeric):
            continue  # categorical parameter: no ordering, no edge
        ordered = sorted(numeric) + ([None] if None in values else [])
        if len(ordered) < 2:
            continue
        best = best_params[param]
        if best == ordered[0] or best == ordered[-1]:
            (edge if len(ordered) >= 3 else two_value).append(param)
    return bool(edge), edge, two_value


def _cv_summary(name, gs, grid, scoring, elapsed):
    res = gs.cv_results_
    i = gs.best_index_
    n_splits = gs.n_splits_
    folds = [float(res[f"split{k}_test_score"][i]) for k in range(n_splits)]
    on_edge, edge_params, two_value = grid_edge_check(gs.best_params_, grid)
    n_configs = len(res["params"])
    return {
        "model": name,
        "scoring": scoring,
        "best_params": json.dumps(gs.best_params_),
        "best_cv_mean": float(res["mean_test_score"][i]),
        "best_cv_std": float(res["std_test_score"][i]),
        "fold_scores": json.dumps(folds),
        "n_configs": n_configs,
        "n_folds": n_splits,
        "n_fits": n_configs * n_splits,
        "tuning_time_s": round(elapsed, 2),
        "mean_fit_time_s": round(float(np.mean(res["mean_fit_time"])), 4),
        "best_on_grid_edge": on_edge,
        "edge_params": ";".join(edge_params),
        "two_value_params_at_bound": ";".join(two_value),
        "convergence_warnings": total_convergence_warnings(res, n_splits),
        "refit_convergence_warnings": gs.best_estimator_.n_convergence_warnings_,
    }


def cv_results_table(results, threshold_warnings=None):
    """One row per model, as saved to cv_results.csv."""
    df = pd.DataFrame([r["summary"] for r in results.values()])
    if threshold_warnings is not None:
        df["threshold_convergence_warnings"] = df["model"].map(threshold_warnings)
    return df


def select_thresholds(estimators, X_train, y_train, cv=None,
                      warning_counts=None, model_specs=None):
    """F1-optimal threshold on out-of-fold predictions.

    If ``warning_counts`` (a dict) is given, it is filled with the number of
    ConvergenceWarnings raised by the out-of-fold refits of each model.
    cross_val_predict runs in this process (n_jobs=None), so the warnings
    can be captured directly; they are re-emitted afterwards.
    """
    cv = cv or get_cv()
    thresholds = {}

    for name, est in estimators.items():
        if not (hasattr(est, "predict_proba") or hasattr(est, "decision_function")):
            thresholds[name] = 0.5
            continue
        params = fit_params((model_specs or {}).get(name, {}), y_train)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            if hasattr(est, "predict_proba"):
                oof = cross_val_predict(est, X_train, y_train, cv=cv,
                                        method="predict_proba",
                                        params=params)[:, 1]
            else:
                oof = cross_val_predict(est, X_train, y_train, cv=cv,
                                        method="decision_function",
                                        params=params)
        if warning_counts is not None:
            warning_counts[name] = count_convergence(records)
        reemit(records)

        prec, rec, thresh = precision_recall_curve(y_train, oof)
        f1s = 2 * prec * rec / (prec + rec + 1e-12)
        best_idx = np.argmax(f1s)
        thresholds[name] = float(thresh[min(best_idx, len(thresh) - 1)])
        print(f"  {name}: threshold={thresholds[name]:.4f}, "
              f"OOF-F1={f1s[best_idx]:.4f}")

    return thresholds


def evaluate_test(estimators, thresholds, X_test, y_test):
    metrics = {}
    probas = {}

    for name, est in estimators.items():
        if hasattr(est, "predict_proba"):
            proba = est.predict_proba(X_test)[:, 1]
        elif hasattr(est, "decision_function"):
            proba = est.decision_function(X_test)
        else:
            proba = None

        tau = thresholds.get(name, 0.5)

        if proba is not None:
            y_pred = (proba >= tau).astype(int)
            pr_auc = average_precision_score(y_test, proba)
            roc = roc_auc_score(y_test, proba)
            probas[name] = proba
        else:
            y_pred = est.predict(X_test)
            pr_auc = roc = np.nan

        macro_f1 = f1_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred)

        metrics[name] = {
            "PR-AUC": pr_auc, "ROC-AUC": roc, "Macro-F1": macro_f1,
            "Threshold": tau,
            "TP": int(cm[1, 1]), "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]), "TN": int(cm[0, 0]),
        }

    metrics_df = (pd.DataFrame(metrics).T
                  .sort_values("PR-AUC", ascending=False))
    return metrics_df, probas


def holm_adjust(pvalues):
    """Holm step-down adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted


def bootstrap_significance(metrics_df, probas, y_test, n_boot=2000,
                           seed=SEED, alpha=0.05):
    """Bootstrap CIs of test PR-AUC and paired tests of the best model.

    One shared set of ``n_boot`` resamples of the test set (seed ``seed``)
    is used for every model, so differences are paired. For each model:
    percentile 95% CI of PR-AUC. For the best model (highest test PR-AUC)
    vs each other model: one-sided p-value for H1 "best > other",
    p = (#{resamples with delta <= 0} + 1) / (n_boot + 1), then Holm
    correction over all comparisons.
    """
    y = np.asarray(y_test)
    n = len(y)
    models = [m for m in metrics_df.index if m in probas]
    best = models[0]

    rng = np.random.RandomState(seed)
    boot = {m: np.empty(n_boot) for m in models}
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        while y[idx].sum() == 0:  # PR-AUC undefined without positives
            idx = rng.choice(n, n, replace=True)
        for m in models:
            boot[m][b] = average_precision_score(y[idx], probas[m][idx])

    rows = []
    for m in models:
        point = average_precision_score(y, probas[m])
        lo, hi = np.percentile(boot[m], [2.5, 97.5])
        row = {"model": m, "pr_auc": point, "ci_low": lo, "ci_high": hi,
               "delta_vs_best": np.nan, "delta_ci_low": np.nan,
               "delta_ci_high": np.nan, "p_one_sided": np.nan}
        if m != best:
            deltas = boot[best] - boot[m]
            row["delta_vs_best"] = (average_precision_score(y, probas[best])
                                    - point)
            row["delta_ci_low"], row["delta_ci_high"] = np.percentile(
                deltas, [2.5, 97.5])
            row["p_one_sided"] = ((deltas <= 0).sum() + 1) / (n_boot + 1)
        rows.append(row)

    df = pd.DataFrame(rows)
    others = df["model"] != best
    df["p_holm"] = np.nan
    df.loc[others, "p_holm"] = holm_adjust(df.loc[others, "p_one_sided"])
    # empty for the best model itself (nothing to compare)
    df["significant_0.05"] = pd.array(
        [pd.NA if m == best else bool(ph < alpha)
         for m, ph in zip(df["model"], df["p_holm"])], dtype="boolean")
    df.attrs.update(best=best, n_boot=n_boot, seed=seed)
    return df


def significance_table(metrics_df, probas, y_test, n_boot=2000):
    """Print and return the bootstrap significance table."""
    df = bootstrap_significance(metrics_df, probas, y_test, n_boot=n_boot)
    best = df.attrs["best"]
    print(f"Test PR-AUC with bootstrap 95% CI ({n_boot} paired resamples, "
          f"seed {SEED}).")
    print(f"Paired test of {best} vs each other model: one-sided "
          f"(H1: {best} > other), Holm-corrected over {len(df) - 1} "
          f"comparisons.")
    for _, r in df.iterrows():
        line = (f"  {r['model']:20s} PR-AUC={r['pr_auc']:.4f} "
                f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
        if r["model"] != best:
            line += (f"  delta={r['delta_vs_best']:+.4f}  "
                     f"p(one-sided)={r['p_one_sided']:.4f}  "
                     f"p(Holm)={r['p_holm']:.4f}"
                     f"{'  *' if r['significant_0.05'] else ''}")
        print(line)
    return df


def print_classification_reports(metrics_df, probas, thresholds,
                                 estimators, X_test, y_test, top_n=3):
    for name in metrics_df.index[:top_n]:
        tau = thresholds.get(name, 0.5)
        if name in probas:
            y_pred = (probas[name] >= tau).astype(int)
        else:
            y_pred = estimators[name].predict(X_test)
        print(f"\n=== {name} (t={tau:.4f}) ===")
        print(classification_report(y_test, y_pred,
                                    target_names=["Normal", "Failure"]))
