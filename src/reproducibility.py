"""Run bookkeeping: log teeing, run metadata, warning accounting."""

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from importlib import metadata as _importlib_metadata

from imblearn.pipeline import Pipeline as _ImbPipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline as _SkPipeline

from . import config



class _Tee:
    """File-like object that writes to a console stream and a log file."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file

    def write(self, text):
        self._stream.write(text)
        self._log.write(text)
        return len(text)

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def __getattr__(self, name):
        # isatty, fileno, encoding, ... come from the console stream
        return getattr(self._stream, name)


def start_log(log_path):
    """Duplicate everything printed to stdout and stderr into ``log_path``.

    The log is line-buffered so it can be followed live (``tail -f``).
    Output of joblib worker processes goes straight to the inherited file
    descriptors and is therefore not captured here.
    """
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    return log_file


# ── Run metadata ──────────────────────────────────────────────────────

_LIBRARIES = [
    "numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn",
    "xgboost", "shap", "matplotlib", "seaborn", "joblib",
]


def _run(cmd):
    try:
        out = subprocess.run(cmd, cwd=config.ROOT, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _git_info():
    status = _run(["git", "status", "--porcelain"])
    return {
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        # True if tracked or untracked (non-ignored) files differ from HEAD
        "git_dirty": None if status is None else bool(status),
    }


def _hardware_info():
    system = platform.system()
    cpu_model, physical_cores, ram_bytes = None, None, None
    if system == "Darwin":
        cpu_model = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        physical_cores = _run(["sysctl", "-n", "hw.physicalcpu"])
        ram_bytes = _run(["sysctl", "-n", "hw.memsize"])
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_model = line.split(":", 1)[1].strip()
                        break
            ram_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (OSError, ValueError):
            pass
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model or platform.processor() or None,
        "cpu_physical_cores": int(physical_cores) if physical_cores else None,
        "cpu_logical_cores": os.cpu_count(),
        "ram_gb": round(int(ram_bytes) / 1024**3, 1) if ram_bytes else None,
    }


def _library_versions():
    versions = {}
    for lib in _LIBRARIES:
        try:
            versions[lib] = _importlib_metadata.version(lib)
        except _importlib_metadata.PackageNotFoundError:
            versions[lib] = None
    return versions


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class RunMetadata:
    """Collects run metadata and writes it to <out_dir>/run_metadata.json.

    The file is written once at start (status "running") and rewritten at
    the end, so a crashed run still leaves a record of what was attempted.
    """

    def __init__(self, path, argv, cv_folds, fast):
        self.path = path
        self._t0 = time.perf_counter()
        self.data = {
            "status": "running",
            "start_time": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "end_time": None,
            "wall_clock_seconds": None,
            "command_line": [sys.executable] + list(argv),
            **_git_info(),
            "seed": config.SEED,
            "test_size": config.TEST_SIZE,
            "cv_folds": cv_folds,
            "fast_mode": fast,
            "data_file": str(config.DATA_PATH.relative_to(config.ROOT)),
            "data_md5": _md5(config.DATA_PATH) if config.DATA_PATH.exists() else None,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "libraries": _library_versions(),
            **_hardware_info(),
        }
        self.write()

    def update(self, **fields):
        self.data.update(fields)

    def finish(self):
        self.data["status"] = "completed"
        self.data["end_time"] = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
        self.data["wall_clock_seconds"] = round(time.perf_counter() - self._t0, 1)
        self.write()

    def write(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
            f.write("\n")


# ── Convergence-warning accounting ────────────────────────────────────
#
# GridSearchCV fits run in joblib worker processes, so warnings raised there
# never reach the main process. The pipelines below record the
# ConvergenceWarnings raised by their own fit, store the count on the fitted
# object, and re-emit every captured warning (so nothing is silenced). A
# scorer then reads the count back, which makes it available in
# GridSearchCV.cv_results_ for every (configuration, fold) fit.

# Shared registry: with the "default" action each distinct warning is shown
# once per process (until the warning filters are next modified).
_REEMIT_REGISTRY = {}


def reemit(records):
    """Re-issue warnings captured with ``catch_warnings(record=True)``."""
    for w in records:
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno,
                               registry=_REEMIT_REGISTRY)


def count_convergence(records):
    return sum(issubclass(w.category, ConvergenceWarning) for w in records)


class _ConvergenceCountingMixin:
    """Pipeline whose fit() also sets ``n_convergence_warnings_``.

    fit() delegates unchanged to the parent pipeline, so fitted models and
    their predictions are identical to a plain Pipeline.
    """

    def fit(self, X, y=None, **params):
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            super().fit(X, y, **params)
        self.n_convergence_warnings_ = count_convergence(records)
        reemit(records)
        return self


class CountingPipeline(_ConvergenceCountingMixin, _SkPipeline):
    pass


class CountingImbPipeline(_ConvergenceCountingMixin, _ImbPipeline):
    pass


def convergence_count_scorer(estimator, X, y):
    """Pseudo-scorer: number of ConvergenceWarnings raised while fitting."""
    return float(getattr(estimator, "n_convergence_warnings_", 0))


CONV_KEY = "n_conv"


def scoring_with_convergence(scoring):
    """Multi-metric scoring dict; use with ``refit="score"`` (or False).

    The main metric keeps the key "score", so cv_results_ still has
    mean_test_score / rank_test_score / splitK_test_score and model
    selection is the same as with ``scoring=scoring``.
    """
    return {"score": scoring, CONV_KEY: convergence_count_scorer}


def total_convergence_warnings(cv_results, n_splits):
    """Sum of ConvergenceWarnings over all (configuration, fold) fits."""
    return int(round(sum(cv_results[f"split{k}_test_{CONV_KEY}"].sum()
                         for k in range(n_splits))))


def best_index(cv_results):
    """Same rule GridSearchCV uses for best_index_ (lowest rank, first)."""
    return int(cv_results["rank_test_score"].argmin())
