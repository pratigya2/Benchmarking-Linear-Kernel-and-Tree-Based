"""
Re-run analyses from a saved pipeline state, without retuning.

main.py saves the fitted estimators, thresholds, test-set probabilities and
metrics to <out-dir>/_state.joblib. This script loads that state, rebuilds
the exact test set from the saved row indices, checks that the stored
probabilities are reproduced by the loaded models, and prints the results.
It then runs the post-hoc analyses of src/post_analyses.py (the same
functions main.py calls at the end of a full run) and writes their tables
and figures to <out-dir>/tables and <out-dir>/figures.

Usage:
    python analyze.py                 # uses outputs/
    python analyze.py --out-dir DIR
    python analyze.py --check-only    # only verify the saved state
"""

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Same warning policy as main.py: show each distinct warning, filter only
# known third-party noise (seaborn's Colormap.set_bad call).
warnings.simplefilter("default")
warnings.filterwarnings("ignore", category=PendingDeprecationWarning,
                        message=r"The set_bad function will be deprecated")

from src import config
from src.config import TARGET
from src.data import load_raw, clean
from src.post_analyses import run_all


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
    parser.add_argument("--check-only", action="store_true",
                        help="only check the saved state; skip the analyses")
    args = parser.parse_args(argv)
    config.set_out_dir(args.out_dir)

    state = load_state()
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
    if worst >= 1e-9:
        return 1
    if not args.check_only:
        print("\nPost-hoc analyses:")
        run_all(state, X_train, X_test, y_train, y_test,
                fast=bool(state.get("fast_mode")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
