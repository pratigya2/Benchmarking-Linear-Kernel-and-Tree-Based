import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

from .config import SEED


def get_model_specs(y_train=None):
    """Models and grids. LogisticRegression uses the l1_ratio API
    (scikit-learn >= 1.8): 0 = L2, 1 = L1, in between = elastic net.
    The former `penalty=` argument is deprecated and removed in 1.10."""
    pos_weight = 1.0
    if y_train is not None:
        n_neg = (y_train == 0).sum()
        n_pos = max((y_train == 1).sum(), 1)
        pos_weight = n_neg / n_pos

    return {
        "LogReg_L2": {
            "clf": LogisticRegression(
                l1_ratio=0.0, solver="lbfgs", class_weight="balanced",
                max_iter=5000, random_state=SEED),
            "grid": {"clf__C": [0.01, 0.1, 1, 10, 100]},
        },
        "LogReg_L1": {
            # liblinear converges in < 60 iterations for every C in the grid
            # (saga hit max_iter=5000 in most fits). liblinear also penalises
            # the intercept; intercept_scaling=10 makes that penalty small.
            "clf": LogisticRegression(
                l1_ratio=1.0, solver="liblinear", intercept_scaling=10,
                class_weight="balanced", max_iter=5000, random_state=SEED),
            "grid": {"clf__C": [0.01, 0.1, 1, 10, 100]},
        },
        "LogReg_ElasticNet": {
            # l1_ratio is tuned in the grid (0 < l1_ratio < 1 = elastic net).
            # Known limitation: saga (the only solver supporting elastic net)
            # does not converge with balanced class weights, even with
            # max_iter=200000 or tol=1e-3 (it converges in ~1,700 iterations
            # without class weights). max_iter stays at 5000; the warnings are
            # counted in cv_results.csv.
            "clf": LogisticRegression(
                solver="saga", class_weight="balanced",
                max_iter=5000, random_state=SEED),
            "grid": {"clf__C": [0.01, 0.1, 1, 10, 100],
                     "clf__l1_ratio": [0.2, 0.5, 0.8]},
        },
        "LinearSVM": {
            "clf": LinearSVC(
                class_weight="balanced", dual="auto",
                max_iter=10000, random_state=SEED),
            "grid": {"clf__C": [0.01, 0.1, 1, 10]},
        },
        "RBF_SVM": {
            "clf": SVC(
                kernel="rbf", class_weight="balanced",
                probability=True, random_state=SEED),
            "grid": {"clf__C": [0.1, 1, 10, 100],
                     "clf__gamma": ["scale", 0.01, 0.1]},
        },
        "KNN": {
            "clf": KNeighborsClassifier(),
            "grid": {"clf__n_neighbors": [3, 5, 7, 11, 15, 21, 31, 45],
                     "clf__weights": ["uniform", "distance"]},
        },
        "GaussianNB": {
            "clf": GaussianNB(),
            "grid": {"clf__var_smoothing": [1e-15, 1e-13, 1e-11, 1e-9, 1e-7, 1e-5]},
        },
        "DecisionTree": {
            "clf": DecisionTreeClassifier(
                class_weight="balanced", random_state=SEED),
            # max_depth=None (unlimited) is the top of the range and cannot
            # be widened further.
            "grid": {"clf__max_depth": [3, 5, 7, 10, None],
                     "clf__min_samples_leaf": [1, 5, 10, 20, 50]},
        },
        "RandomForest": {
            "clf": RandomForestClassifier(
                class_weight="balanced", random_state=SEED, n_jobs=-1),
            "grid": {"clf__n_estimators": [100, 300, 500],
                     "clf__max_depth": [5, 10, None],
                     "clf__min_samples_leaf": [1, 5]},
        },
        "GradientBoosting": {
            # No class_weight argument: balanced weights are passed as
            # sample_weight at fit time (see fit_params).
            "clf": GradientBoostingClassifier(random_state=SEED),
            "sample_weight": "balanced",
            "grid": {"clf__n_estimators": [100, 300, 500],
                     "clf__max_depth": [3, 5, 7],
                     "clf__learning_rate": [0.05, 0.1],
                     "clf__subsample": [0.8, 1.0]},
        },
        "XGBoost": {
            # scale_pos_weight = n_neg / n_pos gives positives the same
            # relative weight as class_weight="balanced" (equivalent mechanism).
            "clf": xgb.XGBClassifier(
                eval_metric="logloss", scale_pos_weight=pos_weight,
                random_state=SEED, n_jobs=-1, verbosity=0),
            "grid": {"clf__n_estimators": [100, 300, 500],
                     "clf__max_depth": [3, 5, 7, 9],
                     "clf__learning_rate": [0.05, 0.1],
                     "clf__subsample": [0.8, 1.0]},
        },
    }


def fit_params(spec, y):
    """Extra fit() arguments for a model spec inside a Pipeline("clf").

    Models without a class_weight argument (Gradient Boosting) get balanced
    sample weights computed from the labels they are fitted on. Pass the
    result to GridSearchCV.fit / cross_val_predict(params=...); both slice
    it per fold.
    """
    if spec.get("sample_weight") == "balanced":
        return {"clf__sample_weight": compute_sample_weight("balanced", y)}
    return {}


def get_smote_specs():
    return {
        "LogReg_L2": {
            "clf": LogisticRegression(
                l1_ratio=0.0, solver="lbfgs",
                max_iter=5000, random_state=SEED),
            "grid": {"clf__C": [0.1, 1, 10]},
        },
        "RandomForest": {
            "clf": RandomForestClassifier(random_state=SEED, n_jobs=-1),
            "grid": {"clf__n_estimators": [100, 300],
                     "clf__max_depth": [5, 10]},
        },
        "XGBoost": {
            "clf": xgb.XGBClassifier(
                eval_metric="logloss",
                random_state=SEED, n_jobs=-1, verbosity=0),
            "grid": {"clf__n_estimators": [100, 300],
                     "clf__max_depth": [3, 5],
                     "clf__learning_rate": [0.05, 0.1]},
        },
    }
