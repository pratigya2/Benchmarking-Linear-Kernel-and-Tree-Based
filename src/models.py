import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

from .config import SEED


def get_model_specs(y_train=None):
    pos_weight = 1.0
    if y_train is not None:
        n_neg = (y_train == 0).sum()
        n_pos = max((y_train == 1).sum(), 1)
        pos_weight = n_neg / n_pos

    return {
        "LogReg_L2": {
            "clf": LogisticRegression(
                penalty="l2", solver="lbfgs", class_weight="balanced",
                max_iter=5000, random_state=SEED),
            "grid": {"clf__C": [0.01, 0.1, 1, 10, 100]},
        },
        "LogReg_L1": {
            "clf": LogisticRegression(
                penalty="l1", solver="saga", class_weight="balanced",
                max_iter=5000, random_state=SEED),
            "grid": {"clf__C": [0.01, 0.1, 1, 10, 100]},
        },
        "LogReg_ElasticNet": {
            "clf": LogisticRegression(
                penalty="elasticnet", solver="saga", class_weight="balanced",
                max_iter=5000, random_state=SEED),
            "grid": {"clf__C": [0.01, 0.1, 1, 10],
                     "clf__l1_ratio": [0.2, 0.5, 0.8]},
        },
        "LinearSVM": {
            "clf": LinearSVC(
                class_weight="balanced", dual="auto",
                max_iter=10000, random_state=SEED),
            "grid": {"clf__C": [0.01, 0.1, 1, 10]},
            "no_proba": True,
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
            "grid": {"clf__n_neighbors": [3, 5, 7, 11, 15],
                     "clf__weights": ["uniform", "distance"]},
        },
        "GaussianNB": {
            "clf": GaussianNB(),
            "grid": {"clf__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]},
        },
        "DecisionTree": {
            "clf": DecisionTreeClassifier(
                class_weight="balanced", random_state=SEED),
            "grid": {"clf__max_depth": [3, 5, 7, 10, None],
                     "clf__min_samples_leaf": [1, 5, 10]},
        },
        "RandomForest": {
            "clf": RandomForestClassifier(
                class_weight="balanced", random_state=SEED, n_jobs=-1),
            "grid": {"clf__n_estimators": [100, 300],
                     "clf__max_depth": [5, 10, None],
                     "clf__min_samples_leaf": [1, 5]},
        },
        "GradientBoosting": {
            "clf": GradientBoostingClassifier(random_state=SEED),
            "grid": {"clf__n_estimators": [100, 300],
                     "clf__max_depth": [3, 5, 7],
                     "clf__learning_rate": [0.05, 0.1],
                     "clf__subsample": [0.8, 1.0]},
        },
        "XGBoost": {
            "clf": xgb.XGBClassifier(
                eval_metric="logloss", scale_pos_weight=pos_weight,
                random_state=SEED, n_jobs=-1, verbosity=0),
            "grid": {"clf__n_estimators": [100, 300],
                     "clf__max_depth": [3, 5, 7],
                     "clf__learning_rate": [0.05, 0.1],
                     "clf__subsample": [0.8, 1.0]},
        },
    }


def get_smote_specs():
    return {
        "LogReg_L2": {
            "clf": LogisticRegression(
                penalty="l2", solver="lbfgs",
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
