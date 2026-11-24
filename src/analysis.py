import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

from .config import SEED
from .preprocessing import (
    build_preprocessor, build_column_transformer, get_feature_names,
    interaction_steps,
)
from .evaluation import get_cv
from .models import fit_params
from .reproducibility import (
    CountingPipeline, CountingImbPipeline, scoring_with_convergence,
    total_convergence_warnings, best_index,
)


def _best_score_and_warnings(gs):
    """Best mean CV score (GridSearchCV's own rule) and ConvergenceWarnings."""
    res = gs.cv_results_
    score = float(res["mean_test_score"][best_index(res)])
    return score, total_convergence_warnings(res, gs.n_splits_)


def pca_variance_analysis(X_train):
    pre = build_preprocessor(with_interactions=True, with_pca=False)
    X_transformed = pre.fit_transform(X_train)
    pca = PCA(random_state=SEED).fit(X_transformed)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n95 = int(np.searchsorted(cumvar, 0.95) + 1)
    return pca, cumvar, n95, X_transformed.shape[1]


def run_ablation(model_specs, X_train, y_train, cv=None):
    cv = cv or get_cv()
    configs = [
        ("baseline",          False, False),
        ("interactions",      True,  False),
        ("PCA_only",          False, True),
        ("interactions+PCA",  True,  True),
    ]

    subset_keys = ["LogReg_L2", "RBF_SVM", "RandomForest", "XGBoost"]
    subset = {k: model_specs[k] for k in subset_keys if k in model_specs}

    rows = []
    for config_name, use_int, use_pca in configs:
        pre = build_preprocessor(with_interactions=use_int,
                                 with_pca=use_pca, n_components=0.95)
        for name, spec in subset.items():
            pipe = CountingPipeline([("pre", pre), ("clf", spec["clf"])])
            scoring = "average_precision"
            gs = GridSearchCV(pipe, spec["grid"],
                              scoring=scoring_with_convergence(scoring),
                              cv=cv, n_jobs=-1, refit=False)
            gs.fit(X_train, y_train, **fit_params(spec, y_train))
            best_score, n_conv = _best_score_and_warnings(gs)
            rows.append({
                "config": config_name, "model": name,
                "cv_score": best_score, "metric": scoring,
            })
            print(f"  {config_name:20s} | {name:15s} | "
                  f"{scoring}={best_score:.4f} | "
                  f"ConvergenceWarnings={n_conv}")

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="config", columns="model", values="cv_score")
    pivot = pivot.loc[["baseline", "interactions",
                       "PCA_only", "interactions+PCA"]]
    return pivot


SMOTE_MODELS = ["LogReg_L2", "RandomForest", "GradientBoosting", "XGBoost"]


def _without_class_weighting(clf):
    """Copy of a classifier with its built-in class weighting switched off."""
    clf = clone(clf)
    params = clf.get_params()
    if "class_weight" in params:
        clf.set_params(class_weight=None)
        return clf, "class_weight=None"
    if "scale_pos_weight" in params:
        clf.set_params(scale_pos_weight=1)
        return clf, "scale_pos_weight=1"
    return clf, "no sample_weight"


def run_smote_comparison(model_specs, cv_results, X_train, y_train, cv=None,
                         models=SMOTE_MODELS):
    """SMOTE vs the class weighting used in the main benchmark.

    Fair comparison: same grids and CV folds as the tuned models; the only
    change is SMOTE oversampling (placed after scaling and before the
    interaction terms, so synthetic rows have products consistent with their
    factors) instead of class weights (class_weight=None, XGBoost
    scale_pos_weight=1, Gradient Boosting without sample weights).
    """
    cv = cv or get_cv()

    rows = []
    for name in models:
        spec = model_specs[name]
        clf, weighting_off = _without_class_weighting(spec["clf"])
        pipe = CountingImbPipeline([
            ("encode_scale", build_column_transformer()),
            ("smote", SMOTE(random_state=SEED)),
            *interaction_steps(),
            ("clf", clf),
        ])
        gs = GridSearchCV(pipe, spec["grid"],
                          scoring=scoring_with_convergence("average_precision"),
                          cv=cv, n_jobs=-1, refit=False)
        gs.fit(X_train, y_train)
        res = gs.cv_results_
        i = best_index(res)
        smote_score = float(res["mean_test_score"][i])
        n_conv = total_convergence_warnings(res, gs.n_splits_)

        cw = cv_results[name]["cv_score"]
        cw_std = cv_results[name]["summary"]["best_cv_std"]
        rows.append({
            "model": name, "class_weight_CV": cw,
            "SMOTE_CV": smote_score,
            "diff": smote_score - cw,
            "class_weight_CV_std": cw_std,
            "SMOTE_CV_std": float(res["std_test_score"][i]),
            "class_weight_params": json.dumps(cv_results[name]["best_params"]),
            "SMOTE_params": json.dumps(res["params"][i]),
            "SMOTE_setting": weighting_off,
            "SMOTE_convergence_warnings": n_conv,
        })
        print(f"  {name:17s} | class_weight={cw:.4f} | "
              f"SMOTE={smote_score:.4f} | "
              f"diff={smote_score - cw:+.4f} | "
              f"ConvergenceWarnings={n_conv}")

    return pd.DataFrame(rows)


def logreg_coefficients(estimator):
    feature_names = get_feature_names(estimator.named_steps["pre"])
    coefs = estimator.named_steps["clf"].coef_[0]
    df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    df["abs_coef"] = df["coef"].abs()
    return df.sort_values("abs_coef", ascending=False)


def tree_importances(estimators, model_names=None):
    model_names = model_names or ["RandomForest", "GradientBoosting", "XGBoost"]
    results = {}
    for name in model_names:
        est = estimators[name]
        feature_names = get_feature_names(est.named_steps["pre"])
        imp = est.named_steps["clf"].feature_importances_
        df = pd.DataFrame({"feature": feature_names, "importance": imp})
        results[name] = df.sort_values("importance", ascending=False)
    return results


def permutation_analysis(estimator, X_test, y_test):
    pre = estimator.named_steps["pre"]
    feature_names = get_feature_names(pre)
    X_pre = pre.transform(X_test)

    result = permutation_importance(
        estimator.named_steps["clf"], X_pre, y_test,
        n_repeats=20, random_state=SEED,
        scoring="average_precision", n_jobs=-1,
    )

    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })
    return df.sort_values("importance_mean", ascending=False)


def shap_analysis(estimator, X_test):
    import shap

    pre = estimator.named_steps["pre"]
    clf = estimator.named_steps["clf"]

    if not hasattr(clf, "feature_importances_"):
        return None, None

    feature_names = get_feature_names(pre)
    X_pre = pd.DataFrame(pre.transform(X_test), columns=feature_names)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_pre)
    return shap_values, X_pre
