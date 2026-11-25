"""Fast checks of the data handling, the split and the leakage guarantees.

Run with ``pytest`` from the repository root. All tests except the smoke
runs take a few seconds; the smoke run (``main.py --fast``, shared by the
last two tests) takes ~25 s.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_PATH, ID_COLS, LEAK_COLS, SEED, TARGET  # noqa: E402
from src.data import load_raw, clean, split  # noqa: E402
from src.preprocessing import build_preprocessor  # noqa: E402
from src.error_analysis import mode_masks  # noqa: E402

pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(), reason=f"dataset not found at {DATA_PATH}")


@pytest.fixture(scope="module")
def df():
    return clean(load_raw())


def test_clean_drops_ids_and_failure_mode_flags(df):
    dropped = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
    assert set(dropped) == set(ID_COLS + LEAK_COLS)
    assert not set(dropped) & set(df.columns)
    assert TARGET in df.columns


def test_split_sizes_and_stratification(df):
    X_train, X_test, y_train, y_test = split(df)
    assert (len(X_train), len(X_test)) == (8000, 2000)
    assert (int(y_train.sum()), int(y_test.sum())) == (271, 68)
    assert not set(X_train.index) & set(X_test.index)
    assert TARGET not in X_train.columns


def test_preprocessor_fitted_on_train_only(df):
    # Tiny split: the preprocessor must learn its statistics from the
    # training rows only and must not refit when transforming test rows.
    small, _ = train_test_split(df, train_size=300, stratify=df[TARGET],
                                random_state=SEED)
    X, y = small.drop(columns=[TARGET]), small[TARGET]
    X_tr, X_te, _, _ = train_test_split(X, y, test_size=0.3, stratify=y,
                                        random_state=SEED)
    pre = build_preprocessor(with_interactions=True)
    with pytest.raises(NotFittedError):
        check_is_fitted(pre)

    pre.fit(X_tr)
    scaler = pre.named_steps["encode_scale"].named_transformers_["num"]
    num_cols = list(scaler.feature_names_in_)
    np.testing.assert_allclose(scaler.mean_, X_tr[num_cols].mean().to_numpy())
    mean_before = scaler.mean_.copy()

    pre.transform(X_te)
    np.testing.assert_array_equal(scaler.mean_, mean_before)
    assert not np.allclose(scaler.mean_, X_te[num_cols].mean().to_numpy())


def test_no_constant_columns_after_preprocessing(df):
    X_train, _, _, _ = split(df)
    pre = build_preprocessor(with_interactions=True).fit(X_train)
    Xt = pre.transform(X_train)
    assert np.all(Xt.std(axis=0) > 0)
    names = pre.get_feature_names_out()
    assert len(names) == Xt.shape[1] == 27  # 7 base + 21 products - 1 constant
    assert "Type_L Type_M" not in names


def test_failure_mode_masks_match_label_noise_facts():
    masks = mode_masks(load_raw())
    counts = {m: int(v.sum()) for m, v in masks.items()}
    # failures per mode (multi-mode rows count in each); RNF: 1 of 19 flags
    assert counts == {"TWF": 46, "HDF": 115, "PWF": 95, "OSF": 98,
                      "RNF": 1, "no_flag": 9}


PHASE3_TABLES = [
    "label_noise.csv", "logreg_coefficients.csv", "l1_path.csv",
    "nb_diagnosis.csv", "error_by_failure_mode.csv",
    "best_model_false_negatives.csv", "permutation_raw.csv",
    "importance_spearman.csv", "failure_rules_vs_importance.csv",
]


@pytest.fixture(scope="module")
def fast_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("fast")
    result = subprocess.run(
        [sys.executable, "main.py", "--fast", "--out-dir", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    return out, result


def test_fast_run_exits_cleanly(fast_run):
    out, result = fast_run
    assert result.returncode == 0, result.stderr[-2000:]
    for name in ["test_metrics.csv", "cv_results.csv", "run_metadata.json",
                 "run.log", "_state.joblib"]:
        assert (out / name).exists(), name
    for name in PHASE3_TABLES:
        assert (out / "tables" / name).exists(), name


def test_analyze_reruns_post_hoc_analyses_from_state(fast_run):
    out, result = fast_run
    assert result.returncode == 0, result.stderr[-2000:]
    for f in (out / "tables").glob("*.csv"):
        f.unlink()
    res = subprocess.run(
        [sys.executable, "analyze.py", "--out-dir", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr[-2000:]
    for name in PHASE3_TABLES:
        assert (out / "tables" / name).exists(), name
