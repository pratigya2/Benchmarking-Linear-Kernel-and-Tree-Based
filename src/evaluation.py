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
        gs.fit(X_train, y_train)
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
                      warning_counts=None):
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
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            if hasattr(est, "predict_proba"):
                oof = cross_val_predict(est, X_train, y_train, cv=cv,
                                        method="predict_proba")[:, 1]
            else:
                oof = cross_val_predict(est, X_train, y_train, cv=cv,
                                        method="decision_function")
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


def paired_bootstrap_test(y_true, proba_a, proba_b,
                          n_boot=3000, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    y_arr = y_true.values if hasattr(y_true, "values") else y_true

    delta = (average_precision_score(y_arr, proba_a) -
             average_precision_score(y_arr, proba_b))
    count = 0
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        diff = (average_precision_score(y_arr[idx], proba_a[idx]) -
                average_precision_score(y_arr[idx], proba_b[idx]))
        if diff <= 0:
            count += 1

    return delta, count / n_boot


def significance_table(metrics_df, probas, y_test, n_boot=3000):
    ranked = [n for n in metrics_df.index if n in probas]
    if len(ranked) < 2:
        return

    best = ranked[0]
    print(f"Comparing {best} vs. each other model (PR-AUC):")
    for other in ranked[1:]:
        delta, pval = paired_bootstrap_test(
            y_test, probas[best], probas[other], n_boot=n_boot)
        sig = ("***" if pval < 0.001 else
               "**" if pval < 0.01 else
               "*" if pval < 0.05 else "ns")
        print(f"  vs {other:20s}: delta={delta:+.4f}, p={pval:.4f} {sig}")


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
