import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_auc_score, f1_score, confusion_matrix, classification_report,
)

from .config import SEED, CV_FOLDS


def get_cv(n_splits=CV_FOLDS):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def tune_models(model_specs, preprocessor, X_train, y_train, cv=None):
    cv = cv or get_cv()
    results = {}
    estimators = {}

    for name, spec in model_specs.items():
        print(f"\n--- {name} ---")
        t0 = time.time()

        pipe = Pipeline([("pre", preprocessor), ("clf", spec["clf"])])
        scoring = "roc_auc" if spec.get("no_proba") else "average_precision"

        gs = GridSearchCV(
            pipe, spec["grid"], scoring=scoring,
            cv=cv, n_jobs=-1, refit=True, return_train_score=True,
        )
        gs.fit(X_train, y_train)
        elapsed = time.time() - t0

        print(f"  Best params: {gs.best_params_}")
        print(f"  Best CV {scoring}: {gs.best_score_:.4f}")
        print(f"  Time: {elapsed:.1f}s")

        results[name] = {
            "best_params": gs.best_params_,
            "cv_score": gs.best_score_,
            "scoring": scoring,
            "time": elapsed,
        }
        estimators[name] = gs.best_estimator_

    return results, estimators


def select_thresholds(estimators, X_train, y_train, cv=None):
    cv = cv or get_cv()
    thresholds = {}

    for name, est in estimators.items():
        if hasattr(est, "predict_proba"):
            oof = cross_val_predict(est, X_train, y_train, cv=cv,
                                    method="predict_proba")[:, 1]
        elif hasattr(est, "decision_function"):
            oof = cross_val_predict(est, X_train, y_train, cv=cv,
                                    method="decision_function")
        else:
            thresholds[name] = 0.5
            continue

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
