"""
Re-run analyses from a saved pipeline state, without retuning.

main.py saves the fitted estimators, thresholds, test-set probabilities and
metrics to <out-dir>/_state.joblib. This script loads that state, rebuilds
the exact test set from the saved row indices, checks that the stored
probabilities are reproduced by the loaded models, and prints the results.
Later analyses (error analysis, importances, ...) can build on load_state()
and rebuild_split().

Usage:
    python analyze.py                 # uses outputs/
    python analyze.py --out-dir DIR
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.config import TARGET
from src.data import load_raw, clean


def load_state(out_dir=None):
    path = Path(out_dir or config.OUT_DIR) / "_state.joblib"
    if not path.exists():
        sys.exit(f"No saved state at {path}. Run main.py first.")
    return joblib.load(path)


def rebuild_split(state):
    """Return X_train, X_test, y_train, y_test exactly as used by main.py."""
    df = clean(load_raw())
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    tr, te = state["train_index"], state["test_index"]
    return X.loc[tr], X.loc[te], y.loc[tr], y.loc[te]


def check_probas(state, X_test):
    """Max absolute difference between stored and recomputed scores."""
    diffs = {}
    for name, est in state["estimators"].items():
        if name not in state["probas"]:
            continue
        if hasattr(est, "predict_proba"):
            fresh = est.predict_proba(X_test)[:, 1]
        else:
            fresh = est.decision_function(X_test)
        diffs[name] = float(np.max(np.abs(fresh - state["probas"][name])))
    return diffs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", default=str(config.OUT_DIR))
    args = parser.parse_args(argv)

    state = load_state(args.out_dir)
    print(f"State from commit {state.get('git_commit')} "
          f"(fast mode: {state.get('fast_mode')})")

    X_train, X_test, y_train, y_test = rebuild_split(state)
    print(f"Train: {len(X_train)} rows ({int(y_train.sum())} failures), "
          f"test: {len(X_test)} rows ({int(y_test.sum())} failures)")

    diffs = check_probas(state, X_test)
    worst = max(diffs.values())
    print(f"Stored vs recomputed test scores: max |diff| = {worst:.2e} "
          f"({'OK' if worst < 1e-9 else 'MISMATCH'})")

    with pd.option_context("display.width", 200):
        print("\nTest metrics:")
        print(state["metrics_df"].to_string())
        print("\nCV summary:")
        print(state["cv_table"][["model", "best_cv_mean", "best_cv_std",
                                 "best_on_grid_edge", "convergence_warnings"]]
              .to_string(index=False))
    return 0 if worst < 1e-9 else 1


if __name__ == "__main__":
    sys.exit(main())
