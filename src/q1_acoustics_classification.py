"""Problem 1: Acoustic characteristics, nonparametric tests, and call-type classification."""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from statsmodels.stats.multitest import multipletests

from config import FIGURES_DIR, RANDOM_SEED, TABLES_DIR, resolve_data_dir

COMMON_FEATURES = ["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75"]
ALL_FEATURES = COMMON_FEATURES + ["F02", "PulseRate"]


def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, std, median, and IQR for each acoustic feature by CallType."""
    rows = []
    for call_type, grp in df.groupby("CallType"):
        for col in ALL_FEATURES:
            s = grp[col].dropna()
            rows.append(
                {
                    "CallType": call_type,
                    "Feature": col,
                    "N": len(s),
                    "Mean": s.mean() if len(s) else np.nan,
                    "Std": s.std() if len(s) > 1 else np.nan,
                    "Median": s.median() if len(s) else np.nan,
                    "IQR": (s.quantile(0.75) - s.quantile(0.25)) if len(s) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def run_nonparametric_tests(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Kruskal-Wallis H, Holm-corrected Mann-Whitney U, and Holm-corrected KS tests."""
    call_types = sorted(df["CallType"].unique())
    pairs = list(combinations(call_types, 2))
    rows = []

    for col in COMMON_FEATURES:
        groups = [df.loc[df["CallType"] == ct, col].dropna().values for ct in call_types]
        kw_stat, kw_p = stats.kruskal(*groups)

        mwu_raw, ks_raw = [], []
        for c1, c2 in pairs:
            g1 = df.loc[df["CallType"] == c1, col].dropna().values
            g2 = df.loc[df["CallType"] == c2, col].dropna().values
            _, p_mwu = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            _, p_ks = stats.ks_2samp(g1, g2)
            mwu_raw.append(p_mwu)
            ks_raw.append(p_ks)

        _, mwu_adj, _, _ = multipletests(mwu_raw, method="holm")
        _, ks_adj, _, _ = multipletests(ks_raw, method="holm")

        row = {"Feature": col, "KW_H": kw_stat, "KW_p": kw_p}
        for (c1, c2), p_m, p_k in zip(pairs, mwu_adj, ks_adj):
            row[f"MWU_{c1}_vs_{c2}_holm_p"] = p_m
            row[f"KS_{c1}_vs_{c2}_holm_p"] = p_k
        rows.append(row)

    # Also test PulseRate between mod and tv (flat has N=0)
    pr_mod = df.loc[df["CallType"] == "mod", "PulseRate"].dropna().values
    pr_tv = df.loc[df["CallType"] == "tv", "PulseRate"].dropna().values
    u_pr, p_pr_mwu = stats.mannwhitneyu(pr_mod, pr_tv, alternative="two-sided")
    d_pr, p_pr_ks = stats.ks_2samp(pr_mod, pr_tv)
    pr_df = pd.DataFrame(
        [
            {
                "Feature": "PulseRate (mod vs tv)",
                "N_mod": len(pr_mod),
                "N_tv": len(pr_tv),
                "MWU_U": u_pr,
                "MWU_p": p_pr_mwu,
                "KS_D": d_pr,
                "KS_p": p_pr_ks,
            }
        ]
    )
    return pd.DataFrame(rows), pr_df


def evaluate_classifiers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Compare Track A (6 common features) and Track B (structural + all features) via 5-fold CV."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    y = df["CallType"].values

    # Track A: 6 common features
    X_common = df[COMMON_FEATURES].values
    track_a_models = {
        "TrackA_LogReg_Common6": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        ),
        "TrackA_SVM_RBF_Common6": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", random_state=RANDOM_SEED)
        ),
        "TrackA_RF_Common6": RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED),
    }

    # Track B: include structural missingness indicators + native-NaN HistGB
    df_b = df[ALL_FEATURES].copy()
    df_b["has_F02"] = df["F02"].notna().astype(int)
    df_b["has_PulseRate"] = df["PulseRate"].notna().astype(int)
    X_full = df_b.values

    track_b_models = {
        "TrackB_HistGB_AllFeatures": HistGradientBoostingClassifier(
            max_depth=4, random_state=RANDOM_SEED
        ),
    }

    results = []
    oof_preds = {}

    for name, model in track_a_models.items():
        scores = cross_validate(model, X_common, y, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=-1)
        results.append(
            {
                "Model": name,
                "FeatureSet": "Common6",
                "CV_Accuracy_Mean": scores["test_accuracy"].mean(),
                "CV_Accuracy_Std": scores["test_accuracy"].std(),
                "CV_MacroF1_Mean": scores["test_f1_macro"].mean(),
                "CV_MacroF1_Std": scores["test_f1_macro"].std(),
            }
        )
        if name == "TrackA_RF_Common6":
            oof_preds[name] = cross_val_predict(model, X_common, y, cv=cv, n_jobs=-1)

    for name, model in track_b_models.items():
        scores = cross_validate(model, X_full, y, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=-1)
        results.append(
            {
                "Model": name,
                "FeatureSet": "All8+Indicators",
                "CV_Accuracy_Mean": scores["test_accuracy"].mean(),
                "CV_Accuracy_Std": scores["test_accuracy"].std(),
                "CV_MacroF1_Mean": scores["test_f1_macro"].mean(),
                "CV_MacroF1_Std": scores["test_f1_macro"].std(),
            }
        )
        oof_preds[name] = cross_val_predict(model, X_full, y, cv=cv, n_jobs=-1)

    rf = track_a_models["TrackA_RF_Common6"].fit(X_common, y)
    importances = pd.DataFrame(
        {"Feature": COMMON_FEATURES, "RF_Importance": rf.feature_importances_}
    ).sort_values("RF_Importance", ascending=False)
    return pd.DataFrame(results), importances, oof_preds


def plot_all_visualizations(df: pd.DataFrame, oof_preds: dict) -> None:
    """Generate 8-feature boxplots, PCA biplot, and confusion matrices."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)

    # 1. 2x4 distributions for all 8 acoustic measures
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for ax, col in zip(axes.flat, ALL_FEATURES):
        sub = df[["CallType", col]].dropna()
        sns.boxplot(data=sub, x="CallType", y=col, hue="CallType", legend=False, ax=ax)
        sns.stripplot(data=sub, x="CallType", y=col, color="0.25", size=2.5, alpha=0.5, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "q1_feature_boxplot.png", dpi=300)
    plt.close(fig)

    # 2. PCA biplot on standardized 6 common features
    X_std = StandardScaler().fit_transform(df[COMMON_FEATURES].values)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(X_std)
    pca_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "CallType": df["CallType"]})

    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="CallType", style="CallType", s=55, alpha=0.85, ax=ax)
    scale = 2.8
    for i, feat in enumerate(COMMON_FEATURES):
        vx, vy = pca.components_[0, i] * scale, pca.components_[1, i] * scale
        ax.arrow(0, 0, vx, vy, color="dimgray", alpha=0.75, head_width=0.08, length_includes_head=True)
        ax.text(vx * 1.12, vy * 1.12, feat, fontsize=9, color="black", ha="center", va="center")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA Biplot of 6 Common Acoustic Features")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "q1_pca_biplot.png", dpi=300)
    plt.close(fig)

    # 3. Confusion matrices (Track A vs Track B)
    labels = ["flat", "mod", "tv"]
    y_true = df["CallType"].values
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for ax, (name, y_pred) in zip(axes, oof_preds.items()):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "q1_confusion_matrices.png", dpi=300)
    plt.close(fig)


def main() -> None:
    data_dir = resolve_data_dir()
    df = pd.read_csv(data_dir / "Data01_AfricanPenguins_seasurface_calls_acoustics.csv")

    desc_df = compute_descriptive_stats(df)
    tests_df, pr_df = run_nonparametric_tests(df)
    cv_df, imp_df, oof_preds = evaluate_classifiers(df)

    desc_df.to_csv(TABLES_DIR / "q1_descriptive_stats.csv", index=False)
    tests_df.to_csv(TABLES_DIR / "q1_nonparametric_tests.csv", index=False)
    pr_df.to_csv(TABLES_DIR / "q1_pulserate_tests.csv", index=False)
    cv_df.to_csv(TABLES_DIR / "q1_classifier_cv.csv", index=False)
    imp_df.to_csv(TABLES_DIR / "q1_rf_importances.csv", index=False)
    plot_all_visualizations(df, oof_preds)

    print("=== Classifier 5-Fold CV ===")
    print(cv_df.to_string(index=False))
    print("\n=== PulseRate (mod vs tv) ===")
    print(pr_df.to_string(index=False))


if __name__ == "__main__":
    main()
