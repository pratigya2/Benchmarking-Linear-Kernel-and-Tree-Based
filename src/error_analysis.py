"""Error analysis by failure mode (Phase 3.5).

The failure-mode flags (TWF, HDF, PWF, OSF, RNF) are never model inputs.
Here they are joined back to the test rows *after* prediction, by row index
into the raw CSV, to see which kinds of failure each model detects.

Conventions:
- Recall for a mode = detected / failures (``Machine failure = 1``) carrying
  that flag. A row with several flags counts in each of its modes.
- "no_flag" = failures with no mode flag.
- Rows with a mode flag but ``Machine failure = 0`` (label noise; all RNF in
  this dataset) are negatives; predicting them positive is counted in
  ``fp_on_flagged_negatives``.
"""

import numpy as np
import pandas as pd

from .config import FAILURE_MODES, TARGET
from .data import load_raw

MODES = FAILURE_MODES + ["no_flag"]

# Generating rules of the synthetic dataset (UCI documentation; Matzka 2020)
OSF_STRAIN_LIMITS = {"L": 11_000, "M": 12_000, "H": 13_000}   # min * Nm
PWF_POWER_BAND = (3_500, 9_000)                                 # W


# Variables entering each failure mode's generating rule
FAILURE_RULES = {
    "HDF": ("process - air temperature < 8.6 K and rotational speed < 1,380 rpm",
            ["Air temperature [K]", "Process temperature [K]",
             "Rotational speed [rpm]"]),
    "PWF": ("power = torque x speed (rad/s) outside 3,500-9,000 W",
            ["Torque [Nm]", "Rotational speed [rpm]"]),
    "OSF": ("tool wear x torque > 11,000 / 12,000 / 13,000 min*Nm for L / M / H",
            ["Tool wear [min]", "Torque [Nm]", "Type"]),
    "TWF": ("tool replaced or failed at a random tool wear between 200 and 240 min",
            ["Tool wear [min]"]),
    "RNF": ("random failure with probability 0.1%, independent of the inputs", []),
}


def raw_test_rows(index):
    """Raw CSV rows (features, target, IDs and mode flags) for ``index``."""
    return load_raw().loc[index]


def mode_masks(raw):
    """Boolean mask per mode, restricted to actual failures."""
    y = raw[TARGET] == 1
    flags = raw[FAILURE_MODES] == 1
    masks = {m: y & flags[m] for m in FAILURE_MODES}
    masks["no_flag"] = y & ~flags.any(axis=1)
    return masks


def predictions(state):
    """0/1 test predictions of every model at its CV-selected threshold."""
    return {name: (np.asarray(p) >= state["thresholds"][name]).astype(int)
            for name, p in state["probas"].items()}


def error_by_failure_mode(state, y_test):
    """One row per model: TP/FP/FN, FP on flagged negatives, and per mode
    the number of failures, the number missed (FN) and the recall."""
    raw = raw_test_rows(y_test.index)
    y = raw[TARGET].to_numpy()
    masks = mode_masks(raw)
    flagged_neg = ((raw[FAILURE_MODES] == 1).any(axis=1) & (raw[TARGET] == 0)).to_numpy()
    preds = predictions(state)
    rows = []
    for name in state["metrics_df"].index:
        pred = preds[name]
        row = {"model": name,
               "TP": int(((pred == 1) & (y == 1)).sum()),
               "FP": int(((pred == 1) & (y == 0)).sum()),
               "FN": int(((pred == 0) & (y == 1)).sum()),
               "n_flagged_negatives": int(flagged_neg.sum()),
               "fp_on_flagged_negatives": int((pred[flagged_neg] == 1).sum())}
        for m in MODES:
            mk = masks[m].to_numpy()
            n = int(mk.sum())
            hit = int(pred[mk].sum())
            row[f"n_{m}"] = n
            row[f"fn_{m}"] = n - hit
            row[f"recall_{m}"] = hit / n if n else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def add_physics_quantities(df):
    """Derived quantities used by the failure rules (for reading only)."""
    out = df.copy()
    omega = out["Rotational speed [rpm]"] * 2 * np.pi / 60          # rad/s
    out["power [W]"] = out["Torque [Nm]"] * omega
    out["strain [min*Nm]"] = out["Tool wear [min]"] * out["Torque [Nm]"]
    out["temp_diff [K]"] = (out["Process temperature [K]"]
                            - out["Air temperature [K]"])
    out["osf_strain_limit [min*Nm]"] = out["Type"].map(OSF_STRAIN_LIMITS)
    return out


def false_negatives(state, y_test, models):
    """All missed failures of ``models``: raw features, mode flags, score,
    threshold and the derived physics quantities."""
    raw = add_physics_quantities(raw_test_rows(y_test.index))
    preds = predictions(state)
    parts = []
    for name in models:
        fn = (raw[TARGET] == 1).to_numpy() & (preds[name] == 0)
        part = raw.loc[fn].copy()
        part.insert(0, "model", name)
        part.insert(1, "score", np.asarray(state["probas"][name])[fn])
        part.insert(2, "threshold", state["thresholds"][name])
        flags = part[FAILURE_MODES]
        part["modes"] = flags.apply(
            lambda r: "+".join(m for m in FAILURE_MODES if r[m] == 1) or "no_flag",
            axis=1)
        parts.append(part)
    return pd.concat(parts).rename_axis("row_index").reset_index()


def fn_counts_by_mode(fn_df):
    """Number of false negatives per model and mode (multi-flag rows count
    in each of their modes)."""
    out = {}
    for name, g in fn_df.groupby("model", sort=False):
        counts = {m: int((g[m] == 1).sum()) for m in FAILURE_MODES}
        counts["no_flag"] = int((g["modes"] == "no_flag").sum())
        counts["total_FN"] = len(g)
        out[name] = counts
    return pd.DataFrame(out).T
