"""Post-hoc analyses that run on a fitted pipeline state (Phase 3).

``run_all`` needs only the saved state (``_state.joblib``) and the data, so it
runs both at the end of ``main.py`` and from ``analyze.py`` without retuning.
Nothing here feeds back into model selection: test-set analyses are
descriptive and happen after the final evaluation.
"""

import pandas as pd

from . import config, plots
from .config import FAILURE_MODES, TARGET
from .data import load_raw


def _save_table(df, name, index=False):
    config.TAB_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.TAB_DIR / name, index=index)


def _header(title):
    print(f"\n--- {title} ---")


# ── 3.1 Failure modes and label noise ─────────────────────────────────

def label_noise_table(df_raw):
    """Counts of each failure-mode flag by target label, plus consistency
    checks between the target and the flags."""
    y = df_raw[TARGET]
    flags = df_raw[FAILURE_MODES]
    any_flag = flags.sum(axis=1) > 0
    rows = []
    for m in FAILURE_MODES:
        f = flags[m] == 1
        rows.append({"group": "mode", "item": m, "n_rows": int(f.sum()),
                     "machine_failure_1": int((f & (y == 1)).sum()),
                     "machine_failure_0": int((f & (y == 0)).sum())})
    checks = {
        "any_mode_flag": any_flag,
        "failure_without_mode_flag": (y == 1) & ~any_flag,
        "mode_flag_without_failure": (y == 0) & any_flag,
        "multiple_mode_flags": flags.sum(axis=1) > 1,
    }
    for name, mask in checks.items():
        rows.append({"group": "consistency", "item": name,
                     "n_rows": int(mask.sum()),
                     "machine_failure_1": int((mask & (y == 1)).sum()),
                     "machine_failure_0": int((mask & (y == 0)).sum())})
    out = pd.DataFrame(rows)
    out.attrs["n_rows"] = len(df_raw)
    return out


# ── Driver ────────────────────────────────────────────────────────────

def run_all(state, X_train, X_test, y_train, y_test, fast=False):
    """Run every Phase 3 analysis and write tables/ and figures/."""
    rows = X_train.index.union(X_test.index)
    df_raw = load_raw().loc[rows]

    _header("3.1 Failure modes and label noise")
    noise = label_noise_table(df_raw)
    _save_table(noise, "label_noise.csv")
    print(noise.to_string(index=False))
    plots.plot_failure_modes(noise)
