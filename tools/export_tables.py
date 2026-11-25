"""Export report-ready tables (CSV + LaTeX) and a figure index from outputs/.

Everything is read from a finished run's output folder; nothing is typed
by hand. Writes:
    <dest>/tables/{main_results, ablation, smote, error_by_mode,
                   reproducibility}.{csv,tex}
    <dest>/figures/README.md   (one line per figure: file -> content)

The LaTeX tables use booktabs (\\toprule, \\midrule, \\bottomrule).

Usage:
    python tools/export_tables.py [--src outputs] [--dest results]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

MODES = ["TWF", "HDF", "PWF", "OSF", "RNF", "no_flag"]

# File name -> factual description of what the figure shows.
FIGURES = {
    "01_class_distribution.png": "Target class counts and failure rate by product type (whole dataset).",
    "01b_failure_modes.png": "Failure-mode flag counts split by target label, and label-consistency counts (failures without a flag, flags without a failure, rows with several flags).",
    "02_feature_distributions.png": "Density histograms of the five numeric input variables by target class (whole dataset).",
    "03_correlation_heatmap.png": "Pearson correlation matrix (lower triangle) of the five numeric input variables and the target (whole dataset).",
    "04_pairwise_scatter.png": "Scatter plots of torque vs rotational speed, tool wear vs torque and air vs process temperature, coloured by target class.",
    "05_pca_variance.png": "Individual and cumulative explained variance of PCA on the training features with interaction terms, with the number of components for 95%.",
    "06_ablation_heatmap.png": "CV PR-AUC (mean ± std over 5 folds) of all 11 models under the four feature configurations (baseline, interactions, PCA only, interactions + PCA).",
    "07_pr_curves.png": "Test-set precision-recall curves of all models.",
    "08_roc_curves.png": "Test-set ROC curves of all models.",
    "09_confusion_matrices.png": "Test-set confusion matrices of the top 4 models by test PR-AUC at their selected thresholds.",
    "09b_confusion_all.png": "Test-set confusion matrices of all 11 models at their selected thresholds (counts and row-normalised rates).",
    "10_threshold_tradeoff.png": "Precision, recall and F1 vs probability threshold for the best model on the test set, with the CV-selected threshold marked.",
    "11_calibration.png": "Reliability diagram (test set, 10 bins) for LogReg L2, Random Forest, Gradient Boosting and XGBoost; SVMs excluded.",
    "12_logreg_coefficients.png": "Top 15 coefficients of the L2 logistic regression (standardised features).",
    "12b_logreg_coef_comparison.png": "Coefficients of the L1, L2 and ElasticNet logistic regressions at their selected C, with counts of exactly-zero coefficients.",
    "12c_l1_path.png": "L1 logistic-regression coefficients and number of non-zero coefficients vs C, with the CV-selected C marked.",
    "13_permutation_importance.png": "Test-set permutation importance of the 27 pipeline features for the best model.",
    "13b_gini_importance.png": "Built-in importances (top 15 pipeline features) of Decision Tree, Random Forest, Gradient Boosting (mean impurity decrease) and XGBoost (total gain).",
    "14_shap_summary.png": "SHAP summary (beeswarm) plot of the best model on the test set.",
    "15_shap_bar.png": "Mean absolute SHAP value per feature for the best model on the test set.",
    "16_model_ranking.png": "Test PR-AUC of all models, sorted, with the test positive-class rate (random baseline) as a vertical line.",
    "17_multi_metric_comparison.png": "Test PR-AUC, ROC-AUC and Macro-F1 of all models side by side.",
    "18_recall_by_failure_mode.png": "Test-set recall of all 11 models for each failure mode (TWF, HDF, PWF, OSF, RNF, no flag), with counts.",
    "19_errors_on_physics.png": "Test rows of XGBoost and Gradient Boosting on the tool wear-torque plane (OSF limits) and the speed-torque plane (PWF power band), with TP/FP/FN marked.",
    "20_importance_by_family.png": "Normalised test-set permutation importance of the 6 raw input variables, by model family, with individual models as dots.",
    "21_importance_agreement.png": "Spearman rank correlation between the raw-variable importance rankings of all 11 models.",
}


# ── LaTeX ─────────────────────────────────────────────────────────────

def _tex_escape(s):
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#"), ("±", r"$\pm$"),
                 ("Δ", r"$\Delta$"), ("×", r"$\times$"), ("≤", r"$\leq$")]:
        s = s.replace(a, b)
    return s


def to_latex(df, caption, label, align=None):
    align = align or "l" + "r" * (len(df.columns) - 1)
    lines = [r"\begin{table}[ht]", r"\centering", r"\small",
             rf"\caption{{{_tex_escape(caption)}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(_tex_escape(c) for c in df.columns) + r" \\",
             r"\midrule"]
    for _, row in df.iterrows():
        lines.append(" & ".join(_tex_escape(v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def save(df, dest, name, caption, align=None):
    (dest / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(dest / "tables" / f"{name}.csv", index=False)
    (dest / "tables" / f"{name}.tex").write_text(
        to_latex(df, caption, f"tab:{name}", align), encoding="utf-8")


# ── tables ────────────────────────────────────────────────────────────

def main_results(src):
    m = pd.read_csv(src / "test_metrics.csv", index_col=0)
    sig = pd.read_csv(src / "significance.csv").set_index("model")
    rows = []
    for name, r in m.iterrows():
        s = sig.loc[name]
        rows.append({
            "Model": name,
            "PR-AUC [95% CI]": f"{r['PR-AUC']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}]",
            "ROC-AUC": f"{r['ROC-AUC']:.3f}",
            "Macro-F1": f"{r['Macro-F1']:.3f}",
            "TP": int(r["TP"]), "FP": int(r["FP"]), "FN": int(r["FN"]),
            "Holm p vs best": "" if pd.isna(s["p_holm"]) else f"{s['p_holm']:.3f}",
        })
    return pd.DataFrame(rows)


def ablation(src):
    mean = pd.read_csv(src / "ablation_results.csv", index_col=0)
    std = pd.read_csv(src / "ablation_results_std.csv", index_col=0)
    out = pd.DataFrame({"Model": mean.columns})
    for cfg in mean.index:
        out[cfg] = [f"{mean.loc[cfg, m]:.3f} ± {std.loc[cfg, m]:.3f}"
                    for m in mean.columns]
    return out


def smote(src):
    s = pd.read_csv(src / "smote_comparison.csv")
    return pd.DataFrame({
        "Model": s["model"],
        "Class weighting": [f"{a:.3f} ± {b:.3f}" for a, b in
                            zip(s["class_weight_CV"], s["class_weight_CV_std"])],
        "SMOTE": [f"{a:.3f} ± {b:.3f}" for a, b in
                  zip(s["SMOTE_CV"], s["SMOTE_CV_std"])],
        "Δ (SMOTE − weighting)": [f"{d:+.3f}" for d in s["diff"]],
        "SMOTE run with": s["SMOTE_setting"],
    })


def error_by_mode(src):
    e = pd.read_csv(src / "tables" / "error_by_failure_mode.csv")
    out = pd.DataFrame({"Model": e["model"]})
    for m in MODES:
        n = e[f"n_{m}"]
        hit = n - e[f"fn_{m}"]
        label = f"{m} (n={int(n.iloc[0])})"
        out[label] = ["n/a" if k == 0 else f"{h / k:.2f} ({h}/{k})"
                      for h, k in zip(hit, n)]
    out["FP"] = e["FP"]
    out["FN"] = e["FN"]
    return out


def reproducibility(src):
    meta = json.loads((src / "run_metadata.json").read_text())
    cv = pd.read_csv(src / "cv_results.csv")
    rows = [
        ("Code commit", (meta.get("git_commit") or "")[:7]),
        ("Python", meta["python_version"]),
        *[(lib, ver) for lib, ver in meta["libraries"].items()],
        ("Random seed", meta["seed"]),
        ("Test size", f"{meta['test_size']:.0%} (stratified)"),
        ("CV folds", f"{meta['cv_folds']} (stratified, shuffled, seed {meta['seed']})"),
        ("Tuning metric", "average precision (PR-AUC)"),
        ("Data file MD5", meta["data_md5"]),
        ("Hardware", f"{meta['cpu_model']}, {meta['ram_gb']:.0f} GB RAM, "
                     f"{meta['cpu_physical_cores']} cores"),
        ("Platform", meta["platform"]),
        ("Full-run wall-clock time", f"{meta['wall_clock_seconds']:.0f} s "
                                     f"({meta['wall_clock_seconds'] / 60:.1f} min)"),
    ]
    for r in cv.itertuples():
        params = json.loads(r.best_params)
        txt = ", ".join(f"{k.replace('clf__', '')}={v}" for k, v in params.items())
        rows.append((f"Selected hyperparameters: {r.model}", txt))
    return pd.DataFrame(rows, columns=["Setting", "Value"])


def figure_index(src, dest):
    figs = sorted(p.name for p in (src / "figures").glob("*.png"))
    missing = [f for f in figs if f not in FIGURES]
    lines = ["# Figures", "",
             f"Generated by `tools/export_tables.py` from `{src.name}/figures/` "
             "(produced by `main.py`). One line per figure: file name → content.", ""]
    lines += [f"- `{f}` → {FIGURES.get(f, '(no description)')}" for f in figs]
    (dest / "figures").mkdir(parents=True, exist_ok=True)
    (dest / "figures" / "README.md").write_text("\n".join(lines) + "\n",
                                                encoding="utf-8")
    return figs, missing


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--src", default=str(ROOT / "outputs"))
    p.add_argument("--dest", default=str(ROOT / "results"))
    args = p.parse_args(argv)
    src, dest = Path(args.src), Path(args.dest)

    meta = json.loads((src / "run_metadata.json").read_text())
    if meta.get("fast_mode"):
        sys.exit("Refusing to export tables from a --fast run.")

    save(main_results(src), dest, "main_results",
         "Test-set results (2,000 rows, 68 failures). PR-AUC with bootstrap 95% CI; "
         "TP/FP/FN at the CV-selected threshold; Holm-corrected one-sided p-value vs the best model.",
         align="lrrrrrrr")
    save(ablation(src), dest, "ablation",
         "Ablation: CV PR-AUC (mean ± std over 5 folds), each configuration re-tuned.")
    save(smote(src), dest, "smote",
         "SMOTE vs class weighting: CV PR-AUC (mean ± std over 5 folds), same grids and folds.",
         align="lrrrl")
    save(error_by_mode(src), dest, "error_by_mode",
         "Test-set recall by failure mode (detected/failures with that flag) at the CV-selected threshold.")
    save(reproducibility(src), dest, "reproducibility",
         "Reproducibility: software, settings, hardware, runtime and selected hyperparameters.",
         align="lp{9cm}")
    figs, missing = figure_index(src, dest)
    print(f"Tables written to {dest / 'tables'}; figure index lists {len(figs)} figures.")
    if missing:
        print("Figures without a description: " + ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
