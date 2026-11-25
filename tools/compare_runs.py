"""Compare two pipeline output folders and write a markdown report.

Compares, when present in the folders: test_metrics.csv, cv_results.csv,
ablation_results.csv (+ _std.csv), smote_comparison.csv, significance.csv.
If a folder has no significance.csv but has _state.joblib, the test PR-AUC
95% CIs are computed from the saved test scores with the same bootstrap as
main.py (2,000 shared resamples, seed 42), so older runs get CIs too.

Usage:
    python tools/compare_runs.py OLD_DIR NEW_DIR [--out report.md]
        [--labels v1 v2]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── helpers ───────────────────────────────────────────────────────────

def md_table(df, floatfmt="{:.4f}"):
    """Render a DataFrame as a GitHub markdown table (no extra deps)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return "" if v is None else str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def read_csv(folder, name, **kw):
    path = Path(folder) / name
    return pd.read_csv(path, **kw) if path.exists() else None


def run_info(folder):
    meta = Path(folder) / "run_metadata.json"
    if not meta.exists():
        return "no run_metadata.json"
    d = json.loads(meta.read_text())
    commit = (d.get("git_commit") or "?")[:7]
    return (f"commit {commit}, dirty={d.get('git_dirty')}, "
            f"runtime {d.get('wall_clock_seconds')} s, "
            f"fast={d.get('fast_mode')}")


def pr_auc_ci(folder):
    """Model -> (ci_low, ci_high, source)."""
    sig = read_csv(folder, "significance.csv")
    if sig is not None:
        return {r.model: (r.ci_low, r.ci_high, "significance.csv")
                for r in sig.itertuples()}
    state_path = Path(folder) / "_state.joblib"
    if not state_path.exists():
        return {}
    import joblib
    from src.config import TARGET
    from src.data import load_raw, clean
    from src.evaluation import bootstrap_significance
    state = joblib.load(state_path)
    y_test = clean(load_raw())[TARGET].loc[state["test_index"]]
    sig = bootstrap_significance(state["metrics_df"], state["probas"], y_test)
    return {r.model: (r.ci_low, r.ci_high, "_state.joblib bootstrap")
            for r in sig.itertuples()}


# ── sections ──────────────────────────────────────────────────────────

def compare_test_metrics(a_dir, b_dir, la, lb):
    a = read_csv(a_dir, "test_metrics.csv", index_col=0)
    b = read_csv(b_dir, "test_metrics.csv", index_col=0)
    if a is None or b is None:
        return "test_metrics.csv missing in one folder.\n"
    ci_a, ci_b = pr_auc_ci(a_dir), pr_auc_ci(b_dir)
    cv_a = read_csv(a_dir, "cv_results.csv")
    cv_b = read_csv(b_dir, "cv_results.csv")

    def cv(df, m):
        if df is None or m not in set(df["model"]):
            return np.nan, ""
        r = df.set_index("model").loc[m]
        if r["scoring"] != "average_precision":
            return np.nan, f"tuned on {r['scoring']}"
        return float(r["best_cv_mean"]), ""

    def ci(d, m):
        return f"[{d[m][0]:.4f}, {d[m][1]:.4f}]" if m in d else ""

    def cm(df, m):
        r = df.loc[m]
        return f"{int(r['TP'])}/{int(r['FP'])}/{int(r['FN'])}"

    rows = []
    for m in b.index:
        cva, note_a = cv(cv_a, m)
        cvb, _ = cv(cv_b, m)
        rows.append({
            "Model": m,
            f"CV PR-AUC {la}": cva if not note_a else note_a,
            f"CV PR-AUC {lb}": cvb,
            f"Test PR-AUC {la}": a.loc[m, "PR-AUC"],
            f"95% CI {la}": ci(ci_a, m),
            f"Test PR-AUC {lb}": b.loc[m, "PR-AUC"],
            f"95% CI {lb}": ci(ci_b, m),
            "Δ PR-AUC": b.loc[m, "PR-AUC"] - a.loc[m, "PR-AUC"],
            f"ROC-AUC {la}": a.loc[m, "ROC-AUC"],
            f"ROC-AUC {lb}": b.loc[m, "ROC-AUC"],
            f"Macro-F1 {la}": a.loc[m, "Macro-F1"],
            f"Macro-F1 {lb}": b.loc[m, "Macro-F1"],
            f"Threshold {la}": a.loc[m, "Threshold"],
            f"Threshold {lb}": b.loc[m, "Threshold"],
            f"TP/FP/FN {la}": cm(a, m),
            f"TP/FP/FN {lb}": cm(b, m),
        })
    src = {s for d in (ci_a, ci_b) for (_, _, s) in d.values()}
    out = ["## Test metrics (ordered by " + lb + " test PR-AUC)", "",
           md_table(pd.DataFrame(rows)), "",
           f"CI source: {', '.join(sorted(src)) or 'none'} "
           "(percentile, 2,000 shared resamples, seed 42).", ""]
    rank_a = list(a.index)
    out.append(f"Ranking {la}: " + " > ".join(rank_a))
    out.append("")
    out.append(f"Ranking {lb}: " + " > ".join(b.index))
    return "\n".join(out) + "\n"


def compare_cv_params(a_dir, b_dir, la, lb):
    a = read_csv(a_dir, "cv_results.csv")
    b = read_csv(b_dir, "cv_results.csv")
    if a is None or b is None:
        return ""
    a, b = a.set_index("model"), b.set_index("model")
    rows = [{"Model": m, f"Scoring {la}": a.loc[m, "scoring"] if m in a.index else "",
             f"Best params {la}": a.loc[m, "best_params"] if m in a.index else "",
             f"Best params {lb}": b.loc[m, "best_params"],
             f"Conv. warnings (CV) {la}": a.loc[m].get("convergence_warnings", "") if m in a.index else "",
             f"Conv. warnings (CV) {lb}": b.loc[m].get("convergence_warnings", "")}
            for m in b.index]
    return "## Selected hyperparameters\n\n" + md_table(pd.DataFrame(rows)) + "\n"


def compare_ablation(a_dir, b_dir, la, lb):
    a = read_csv(a_dir, "ablation_results.csv", index_col=0)
    b = read_csv(b_dir, "ablation_results.csv", index_col=0)
    if a is None or b is None:
        return ""
    b_std = read_csv(b_dir, "ablation_results_std.csv", index_col=0)
    rows = []
    for m in b.columns:
        for cfg in b.index:
            va = a.loc[cfg, m] if m in a.columns and cfg in a.index else np.nan
            vb = b.loc[cfg, m]
            rows.append({"Model": m, "Config": cfg, la: va,
                         lb: vb,
                         f"{lb} std": b_std.loc[cfg, m] if b_std is not None else np.nan,
                         "Δ": vb - va})
    only_b = [m for m in b.columns if m not in a.columns]
    note = (f"\nModels only in {lb}: {', '.join(only_b)}.\n" if only_b else "")
    return ("## Ablation (CV PR-AUC)\n\n" + md_table(pd.DataFrame(rows)) +
            "\n" + note)


def compare_smote(a_dir, b_dir, la, lb):
    a = read_csv(a_dir, "smote_comparison.csv")
    b = read_csv(b_dir, "smote_comparison.csv")
    if a is None or b is None:
        return ""
    a, b = a.set_index("model"), b.set_index("model")
    rows = []
    for m in b.index:
        ina = m in a.index
        rows.append({"Model": m,
                     f"class weight {la}": a.loc[m, "class_weight_CV"] if ina else np.nan,
                     f"SMOTE {la}": a.loc[m, "SMOTE_CV"] if ina else np.nan,
                     f"diff {la}": a.loc[m, "diff"] if ina else np.nan,
                     f"class weight {lb}": b.loc[m, "class_weight_CV"],
                     f"SMOTE {lb}": b.loc[m, "SMOTE_CV"],
                     f"diff {lb}": b.loc[m, "diff"]})
    return "## SMOTE vs class weighting (CV PR-AUC)\n\n" + md_table(pd.DataFrame(rows)) + "\n"


def significance_section(folder, label):
    sig = read_csv(folder, "significance.csv")
    if sig is None:
        return ""
    cols = ["model", "pr_auc", "ci_low", "ci_high", "delta_vs_best",
            "p_one_sided", "p_holm", "significant_0.05"]
    return (f"## Significance ({label})\n\n" + md_table(sig[cols]) + "\n")


def build_report(a_dir, b_dir, la="old", lb="new"):
    parts = [f"# Run comparison: {la} vs {lb}", "",
             f"- {la}: `{a_dir}` ({run_info(a_dir)})",
             f"- {lb}: `{b_dir}` ({run_info(b_dir)})", "",
             compare_test_metrics(a_dir, b_dir, la, lb),
             compare_cv_params(a_dir, b_dir, la, lb),
             compare_ablation(a_dir, b_dir, la, lb),
             compare_smote(a_dir, b_dir, la, lb),
             significance_section(a_dir, la),
             significance_section(b_dir, lb)]
    return "\n".join(p for p in parts if p is not None)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--labels", nargs=2, default=["old", "new"])
    p.add_argument("--out", help="markdown file (default: print)")
    args = p.parse_args(argv)
    report = build_report(args.old, args.new, *args.labels)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"Report written to {args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
