import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve

from . import config
from .config import TARGET, NUM_FEATURES, PHYSICS_PAIRS

DPI = 150
SAVE_KW = dict(dpi=DPI, bbox_inches="tight")


def _save(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIG_DIR / name, **SAVE_KW)
    plt.close(fig)


def plot_class_distribution(df_clean, df_raw):
    y = df_clean[TARGET]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    counts = y.value_counts().sort_index()
    axes[0].bar(["No failure", "Failure"], counts.values,
                color=["#4C72B0", "#DD8452"], edgecolor="black")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 100, str(v), ha="center", fontweight="bold")
    axes[0].set_ylabel("Number of samples")
    axes[0].set_title("Target class distribution")

    tf = df_raw.groupby("Type")[TARGET].agg(["sum", "count"])
    tf["rate"] = tf["sum"] / tf["count"]
    axes[1].bar(tf.index, tf["rate"],
                color=["#55A868", "#C44E52", "#8172B2"], edgecolor="black")
    axes[1].set_ylabel("Failure rate")
    axes[1].set_title("Failure rate by product type")
    for i, (_, row) in enumerate(tf.iterrows()):
        axes[1].text(i, row["rate"] + 0.002,
                     f'{row["rate"]:.3f}\n({int(row["sum"])}/{int(row["count"])})',
                     ha="center", fontsize=9)

    plt.tight_layout()
    _save(fig, "01_class_distribution.png")


def plot_feature_distributions(df_clean):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, feat in enumerate(NUM_FEATURES):
        for label, color in [(0, "#4C72B0"), (1, "#DD8452")]:
            subset = df_clean[df_clean[TARGET] == label][feat]
            axes[i].hist(subset, bins=40, alpha=0.6, color=color,
                         label="Failure" if label else "Normal", density=True)
        axes[i].set_title(feat.replace(" [", "\n["), fontsize=10)
        axes[i].legend(fontsize=8)
    plt.suptitle("Feature distributions by class", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, "02_feature_distributions.png")


def plot_correlation_heatmap(df_clean):
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df_clean[NUM_FEATURES + [TARGET]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    short = [c.split(" [")[0] for c in corr.columns]
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                mask=mask, ax=ax, square=True,
                xticklabels=short, yticklabels=short)
    ax.set_title("Pearson correlation matrix")
    plt.tight_layout()
    _save(fig, "03_correlation_heatmap.png")


def plot_pairwise_scatter(df_clean):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for i, (x, y_col) in enumerate(PHYSICS_PAIRS):
        for label, color, zo in [(0, "#4C72B0", 1), (1, "#DD8452", 2)]:
            m = df_clean[TARGET] == label
            axes[i].scatter(
                df_clean.loc[m, x], df_clean.loc[m, y_col],
                alpha=0.3 if label == 0 else 0.8,
                s=5 if label == 0 else 20,
                color=color, zorder=zo,
                label="Failure" if label else "Normal")
        axes[i].set_xlabel(x)
        axes[i].set_ylabel(y_col)
        axes[i].legend(fontsize=8, markerscale=3)
    plt.suptitle("Key feature pairs", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, "04_pairwise_scatter.png")


def plot_pca_variance(pca_model, cumvar, n95):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, len(cumvar) + 1),
           pca_model.explained_variance_ratio_, alpha=0.6, label="Individual")
    ax.step(range(1, len(cumvar) + 1), cumvar, where="mid",
            color="#DD8452", linewidth=2, label="Cumulative")
    ax.axhline(0.95, ls="--", color="grey", label="95% threshold")
    ax.axvline(n95, ls=":", color="red", label=f"k={n95} components")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA variance analysis (after interactions)")
    ax.legend()
    plt.tight_layout()
    _save(fig, "05_pca_variance.png")


def plot_ablation_heatmap(pivot_df):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap="YlGnBu", ax=ax)
    ax.set_title("Ablation study: CV score by preprocessing config and model")
    ax.set_ylabel("Preprocessing config")
    plt.tight_layout()
    _save(fig, "06_ablation_heatmap.png")


def plot_pr_curves(metrics_df, probas, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in metrics_df.index:
        if name not in probas:
            continue
        prec, rec, _ = precision_recall_curve(y_test, probas[name])
        ap = metrics_df.loc[name, "PR-AUC"]
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})", linewidth=1.5)
    ax.axhline(y_test.mean(), ls="--", color="grey", label="Random baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (test set)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    _save(fig, "07_pr_curves.png")


def plot_roc_curves(metrics_df, probas, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in metrics_df.index:
        if name not in probas:
            continue
        fpr, tpr, _ = roc_curve(y_test, probas[name])
        auc_val = metrics_df.loc[name, "ROC-AUC"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", linewidth=1.5)
    ax.plot([0, 1], [0, 1], ls="--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves (test set)")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    _save(fig, "08_roc_curves.png")


def plot_confusion_matrices(metrics_df, probas, thresholds, y_test, top_n=4):
    names = [n for n in metrics_df.index[:top_n]]
    fig, axes = plt.subplots(1, top_n, figsize=(4.5 * top_n, 4))
    for i, name in enumerate(names):
        p = probas.get(name, np.zeros(len(y_test)))
        tau = thresholds.get(name, 0.5)
        cm = confusion_matrix(y_test, (p >= tau).astype(int))
        ConfusionMatrixDisplay(cm, display_labels=["Normal", "Failure"]).plot(
            ax=axes[i], cmap="Blues", colorbar=False)
        axes[i].set_title(name, fontsize=10)
    plt.suptitle("Confusion matrices (test set)", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, "09_confusion_matrices.png")


def plot_threshold_tradeoff(best_name, probas, thresholds, y_test,
                            score_only=()):
    if best_name not in probas:
        return
    if best_name in score_only:
        print(f"Threshold trade-off plot skipped: {best_name} outputs "
              f"decision_function scores, not probabilities in [0, 1].")
        return
    proba = probas[best_name]
    threshs = np.linspace(0.01, 0.99, 200)
    f1s, precs, recs = [], [], []
    for t in threshs:
        yp = (proba >= t).astype(int)
        tp = ((yp == 1) & (y_test == 1)).sum()
        fp = ((yp == 1) & (y_test == 0)).sum()
        fn = ((yp == 0) & (y_test == 1)).sum()
        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        f1s.append(2 * p * r / (p + r + 1e-12))
        precs.append(p)
        recs.append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(threshs, f1s, label="F1", linewidth=2)
    ax.plot(threshs, precs, label="Precision", linewidth=1.5, ls="--")
    ax.plot(threshs, recs, label="Recall", linewidth=1.5, ls="--")
    bt = thresholds[best_name]
    ax.axvline(bt, ls=":", color="red", label=f"Selected t={bt:.3f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold tradeoff - {best_name}")
    ax.legend()
    plt.tight_layout()
    _save(fig, "10_threshold_tradeoff.png")


def plot_calibration(probas, y_test,
                     models=("LogReg_L2", "RandomForest",
                             "GradientBoosting", "XGBoost"),
                     score_only=()):
    """Reliability diagram. Models in ``score_only`` output decision_function
    scores (not probabilities) and are excluded."""
    excluded = [m for m in score_only if m in probas]
    if excluded:
        print(f"Calibration plot excludes {', '.join(excluded)}: "
              f"decision_function scores are not probabilities.")
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in models:
        if name not in probas or name in score_only:
            continue
        frac, mean_pred = calibration_curve(
            y_test, probas[name], n_bins=10, strategy="uniform")
        ax.plot(mean_pred, frac, "s-", label=name)
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curves (reliability diagram)")
    ax.legend()
    plt.tight_layout()
    _save(fig, "11_calibration.png")


def plot_logreg_coefficients(coef_df, top_n=15):
    top = coef_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#DD8452" if c > 0 else "#4C72B0" for c in top["coef"]]
    ax.barh(range(len(top)), top["coef"].values, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient value (standardised features)")
    ax.set_title("Logistic Regression (L2) - Top 15 coefficients")
    plt.tight_layout()
    _save(fig, "12_logreg_coefficients.png")


def plot_permutation_importance(perm_df, best_name, top_n=15):
    top = perm_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top)), top["importance_mean"].values,
            xerr=top["importance_std"].values,
            color="#55A868", edgecolor="black", alpha=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean decrease in PR-AUC")
    ax.set_title(f"Permutation importance - {best_name} (test set)")
    plt.tight_layout()
    _save(fig, "13_permutation_importance.png")


def plot_shap(shap_values, X_df, best_name):
    import shap

    fig, _ = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_df, max_display=15,
                      show=False, plot_size=None)
    plt.title(f"SHAP summary - {best_name}")
    plt.tight_layout()
    _save(fig, "14_shap_summary.png")

    fig, _ = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_values, X_df, max_display=15,
                      plot_type="bar", show=False, plot_size=None)
    plt.title(f"SHAP mean |value| - {best_name}")
    plt.tight_layout()
    _save(fig, "15_shap_bar.png")


def plot_model_ranking(metrics_df, y_test):
    plot_df = metrics_df.dropna(subset=["PR-AUC"]).sort_values("PR-AUC")
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(plot_df)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(plot_df.index, plot_df["PR-AUC"], color=colors, edgecolor="black")
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(row["PR-AUC"] + 0.005, i, f'{row["PR-AUC"]:.3f}',
                va="center", fontsize=9)
    ax.set_xlabel("Test PR-AUC (Average Precision)")
    ax.set_title("Model ranking by PR-AUC")
    ax.axvline(y_test.mean(), ls="--", color="red", label="Random baseline")
    ax.legend()
    plt.tight_layout()
    _save(fig, "16_model_ranking.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, metric in enumerate(["PR-AUC", "ROC-AUC", "Macro-F1"]):
        sdf = metrics_df.dropna(subset=[metric]).sort_values(metric)
        axes[i].barh(sdf.index, sdf[metric],
                     color=colors[:len(sdf)], edgecolor="black")
        axes[i].set_xlabel(metric)
        axes[i].set_title(metric)
        for j, (_, row) in enumerate(sdf.iterrows()):
            axes[i].text(row[metric] + 0.005, j, f'{row[metric]:.3f}',
                         va="center", fontsize=8)
    plt.suptitle("Test set - multi-metric comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, "17_multi_metric_comparison.png")
