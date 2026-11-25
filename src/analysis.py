import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from .config import SEED, CAT_COL, NUM_FEATURES
from .preprocessing import build_preprocessor, get_feature_names
from .evaluation import get_cv
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
            scoring = ("roc_auc" if spec.get("no_proba")
                       else "average_precision")
            gs = GridSearchCV(pipe, spec["grid"],
                              scoring=scoring_with_convergence(scoring),
                              cv=cv, n_jobs=-1, refit=False)
            gs.fit(X_train, y_train)
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


def run_smote_comparison(smote_specs, cv_results, X_train, y_train, cv=None):
    cv = cv or get_cv()

    rows = []
    for name, spec in smote_specs.items():
        ct = ColumnTransformer([
            ("cat", OneHotEncoder(drop="first", sparse_output=False),
             [CAT_COL]),
            ("num", StandardScaler(), NUM_FEATURES),
        ])
        pipe = CountingImbPipeline([
            ("encode_scale", ct),
            ("interactions", PolynomialFeatures(
                degree=2, interaction_only=True, include_bias=False)),
            ("smote", SMOTE(random_state=SEED)),
            ("clf", spec["clf"]),
        ])
        gs = GridSearchCV(pipe, spec["grid"],
                          scoring=scoring_with_convergence("average_precision"),
                          cv=cv, n_jobs=-1, refit=False)
        gs.fit(X_train, y_train)
        smote_score, n_conv = _best_score_and_warnings(gs)

        cw = cv_results[name]["cv_score"]
        rows.append({
            "model": name, "class_weight_CV": cw,
            "SMOTE_CV": smote_score,
            "diff": smote_score - cw,
        })
        print(f"  {name:15s} | class_weight={cw:.4f} | "
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
