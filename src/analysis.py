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


def pca_variance_analysis(X_train):
    pre = build_preprocessor(with_interactions=True, with_pca=False)
    X_transformed = pre.fit_transform(X_train)
    pca = PCA(random_state=SEED).fit(X_transformed)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n95 = int(np.searchsorted(cumvar, 0.95) + 1)
    return pca, cumvar, n95, X_transformed.shape[1]


ABLATION_CONFIGS = ["baseline", "interactions", "PCA_only", "interactions+PCA"]


def run_ablation(model_specs, X_train, y_train, cv=None):
    """Interactions / PCA ablation for every model (each config re-tuned).

    Returns (mean_pivot, std_pivot, long_df): best mean CV PR-AUC and its
    std over folds per config x model, and one row per config/model with
    the selected parameters and ConvergenceWarnings.
    """
    cv = cv or get_cv()
    configs = [
        ("baseline",          False, False),
        ("interactions",      True,  False),
        ("PCA_only",          False, True),
        ("interactions+PCA",  True,  True),
    ]

    rows = []
    for config_name, use_int, use_pca in configs:
        pre = build_preprocessor(with_interactions=use_int,
                                 with_pca=use_pca, n_components=0.95)
        for name, spec in model_specs.items():
            pipe = CountingPipeline([("pre", pre), ("clf", spec["clf"])])
            scoring = "average_precision"
            gs = GridSearchCV(pipe, spec["grid"],
                              scoring=scoring_with_convergence(scoring),
                              cv=cv, n_jobs=-1, refit=False)
            gs.fit(X_train, y_train, **fit_params(spec, y_train))
            res = gs.cv_results_
            i = best_index(res)
            mean, std = float(res["mean_test_score"][i]), float(res["std_test_score"][i])
            n_conv = total_convergence_warnings(res, gs.n_splits_)
            rows.append({
                "config": config_name, "model": name,
                "cv_mean": mean, "cv_std": std, "metric": scoring,
                "best_params": json.dumps(res["params"][i]),
                "convergence_warnings": n_conv,
            })
            print(f"  {config_name:20s} | {name:17s} | "
                  f"{scoring}={mean:.4f} +/- {std:.4f} | "
                  f"ConvergenceWarnings={n_conv}")

    long_df = pd.DataFrame(rows)
    models = list(model_specs)
    pivots = [long_df.pivot(index="config", columns="model", values=v)
              .loc[ABLATION_CONFIGS, models] for v in ("cv_mean", "cv_std")]
    return pivots[0], pivots[1], long_df


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

    # The models were fitted on arrays; pass an array too. (XGBoost rejects
    # DataFrame column names containing "[", "]" or "<", e.g. "Torque [Nm]".)
    # Feature names are kept in X_pre for the plots.
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_pre.to_numpy())
    return shap_values, X_pre


# ── Phase 3.2: logistic-regression coefficients ───────────────────────

LOGREG_MODELS = ["LogReg_L2", "LogReg_L1", "LogReg_ElasticNet"]


def logreg_coefficient_comparison(estimators, models=LOGREG_MODELS):
    """Selected coefficients of the three logistic regressions side by side.

    Inputs are the pipeline features: standardised numeric variables, their
    pairwise products and the 0/1 Type dummies. Rows are sorted by |L2 coef|.
    """
    cols = {}
    for name in models:
        est = estimators[name]
        cols[name] = pd.Series(est.named_steps["clf"].coef_[0],
                               index=get_feature_names(est.named_steps["pre"]))
    df = pd.DataFrame(cols)
    df = df.loc[df[models[0]].abs().sort_values(ascending=False).index]
    df.index.name = "feature"
    return df


def l1_regularisation_path(estimator, X_train, y_train, Cs):
    """Refit the selected L1 model over ``Cs`` (all other settings fixed).

    Uses the fitted preprocessor of the selected pipeline (fitted on the
    training set). Returns (coef_df: C x feature, summary_df).
    """
    pre = estimator.named_steps["pre"]
    names = get_feature_names(pre)
    Xt = pre.transform(X_train)
    selected_C = estimator.named_steps["clf"].C
    Cs = np.unique(np.append(np.asarray(Cs, dtype=float), selected_C))
    coefs, rows = [], []
    for C in Cs:
        clf = clone(estimator.named_steps["clf"]).set_params(C=C).fit(Xt, y_train)
        coef = clf.coef_[0]
        coefs.append(coef)
        rows.append({"C": C, "n_features": len(coef),
                     "n_nonzero": int(np.sum(coef != 0)),
                     "n_exactly_zero": int(np.sum(coef == 0)),
                     "n_iter": int(np.max(clf.n_iter_)),
                     "cv_selected": bool(np.isclose(C, selected_C))})
    coef_df = pd.DataFrame(coefs, index=pd.Index(Cs, name="C"), columns=names)
    return coef_df, pd.DataFrame(rows)
