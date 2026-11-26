"""Problem 1: Acoustic characteristics, nonparametric tests, and call-type classification."""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
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


def run_nonparametric_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run Kruskal-Wallis H tests and Holm-corrected pairwise Mann-Whitney U tests."""
    call_types = sorted(df["CallType"].unique())
    rows = []
    for col in COMMON_FEATURES:
        groups = [df.loc[df["CallType"] == ct, col].dropna().values for ct in call_types]
        kw_stat, kw_p = stats.kruskal(*groups)

        pairs = list(combinations(call_types, 2))
        raw_pvals = []
        u_stats = []
        for c1, c2 in pairs:
            g1 = df.loc[df["CallType"] == c1, col].dropna().values
            g2 = df.loc[df["CallType"] == c2, col].dropna().values
            u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            u_stats.append(u_stat)
            raw_pvals.append(p_val)

        _, adj_pvals, _, _ = multipletests(raw_pvals, method="holm")
        row = {"Feature": col, "KW_H": kw_stat, "KW_p": kw_p}
        for (c1, c2), adj_p in zip(pairs, adj_pvals):
            row[f"MWU_{c1}_vs_{c2}_holm_p"] = adj_p
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_classifiers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare classifiers on the 6 common features via Stratified 5-Fold CV."""
    X = df[COMMON_FEATURES].values
    y = df["CallType"].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    models = {
        "LogisticRegression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        ),
        "SVM_RBF": make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=RANDOM_SEED)),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED),
    }

    results = []
    for name, model in models.items():
        scores = cross_validate(
            model, X, y, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=-1
        )
        results.append(
            {
                "Model": name,
                "CV_Accuracy_Mean": scores["test_accuracy"].mean(),
                "CV_Accuracy_Std": scores["test_accuracy"].std(),
                "CV_MacroF1_Mean": scores["test_f1_macro"].mean(),
                "CV_MacroF1_Std": scores["test_f1_macro"].std(),
            }
        )

    rf = models["RandomForest"].fit(X, y)
    importances = pd.DataFrame(
        {"Feature": COMMON_FEATURES, "RF_Importance": rf.feature_importances_}
    ).sort_values("RF_Importance", ascending=False)
    return pd.DataFrame(results), importances


def plot_feature_distributions(df: pd.DataFrame) -> None:
    """Save a 2x3 boxplot panel for the 6 common acoustic features."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, col in zip(axes.flat, COMMON_FEATURES):
        sns.boxplot(data=df, x="CallType", y=col, hue="CallType", legend=False, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "q1_feature_boxplot.png", dpi=300)
    plt.close(fig)


def main() -> None:
    data_dir = resolve_data_dir()
    df = pd.read_csv(data_dir / "Data01_AfricanPenguins_seasurface_calls_acoustics.csv")

    desc_df = compute_descriptive_stats(df)
    tests_df = run_nonparametric_tests(df)
    cv_df, imp_df = evaluate_classifiers(df)

    desc_df.to_csv(TABLES_DIR / "q1_descriptive_stats.csv", index=False)
    tests_df.to_csv(TABLES_DIR / "q1_nonparametric_tests.csv", index=False)
    cv_df.to_csv(TABLES_DIR / "q1_classifier_cv.csv", index=False)
    imp_df.to_csv(TABLES_DIR / "q1_rf_importances.csv", index=False)
    plot_feature_distributions(df)

    print("Q1 outputs saved to:", TABLES_DIR, "and", FIGURES_DIR)
    print(cv_df.to_string(index=False))


if __name__ == "__main__":
    main()
