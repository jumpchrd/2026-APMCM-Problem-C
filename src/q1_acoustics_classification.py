"""Problem 1: Acoustic characteristics, nonparametric tests, and call-type classification.

Revision notes (v2)
-------------------
* Cross-validation is now GROUPED BY RECORDING DATE.  Call types recorded on the
  same day may come from the same individuals, so a random split would leak
  identity information between train and test.
* Added a majority-class baseline and a label-permutation test, so the reported
  classifier performance has a reference point.
* Reporting now includes balanced accuracy and per-class recall alongside
  macro-F1 (the classes are imbalanced: flat 44 / mod 40 / tv 24).
* Nonparametric tests now report effect sizes (epsilon-squared for Kruskal-Wallis,
  rank-biserial correlation for Mann-Whitney) and a GLOBAL Holm correction across
  all 18 pairwise tests (in addition to the per-feature correction).
* The structural key (F02 present -> tv; PulseRate present -> mod; otherwise flat)
  is evaluated explicitly and reported as a diagnostic key, NOT as a learned
  classifier.  The HistGB model that uses the missingness indicators is relabelled
  as a leakage upper bound, because those indicators are deterministic functions
  of the class label.
"""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from config import FIGURES_DIR, RANDOM_SEED, TABLES_DIR, resolve_data_dir

COMMON_FEATURES = ["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75"]
ALL_FEATURES = COMMON_FEATURES + ["F02", "PulseRate"]
CLASSES = ["flat", "mod", "tv"]
N_PERMUTATIONS = 150


# --------------------------------------------------------------------------- #
# 1. Descriptive statistics
# --------------------------------------------------------------------------- #
def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
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


# --------------------------------------------------------------------------- #
# 2. Nonparametric tests with effect sizes and global correction
# --------------------------------------------------------------------------- #
def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-biserial correlation from the Mann-Whitney U statistic."""
    u, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    return float(2.0 * u / (len(x) * len(y)) - 1.0)


def run_nonparametric_tests(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    call_types = sorted(df["CallType"].unique())
    pairs = list(combinations(call_types, 2))
    rows = []
    raw_p_records = []  # for the global Holm correction

    for col in COMMON_FEATURES:
        groups = [df.loc[df["CallType"] == ct, col].dropna().values for ct in call_types]
        kw_stat, kw_p = stats.kruskal(*groups)
        n_total = sum(len(g) for g in groups)
        k_groups = len(groups)
        eps2 = max(0.0, (kw_stat - k_groups + 1) / (n_total - k_groups))

        row = {"Feature": col, "KW_H": kw_stat, "KW_p": kw_p, "KW_epsilon2": eps2}

        pair_ps_mwu, pair_ps_ks = [], []
        pair_details = []
        for c1, c2 in pairs:
            g1 = df.loc[df["CallType"] == c1, col].dropna().values
            g2 = df.loc[df["CallType"] == c2, col].dropna().values
            u, p_mwu = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            d_ks, p_ks = stats.ks_2samp(g1, g2)
            pair_ps_mwu.append(p_mwu)
            pair_ps_ks.append(p_ks)
            pair_details.append((c1, c2, u, p_mwu, d_ks, p_ks, rank_biserial(g1, g2)))
            raw_p_records.append((col, c1, c2, p_mwu))

        _, mwu_adj, _, _ = multipletests(pair_ps_mwu, method="holm")
        _, ks_adj, _, _ = multipletests(pair_ps_ks, method="holm")

        for (c1, c2, u, p_mwu, d_ks, p_ks, rb), p_m, p_k in zip(pair_details, mwu_adj, ks_adj):
            row[f"MWU_{c1}_vs_{c2}_U"] = u
            row[f"MWU_{c1}_vs_{c2}_p_raw"] = p_mwu
            row[f"MWU_{c1}_vs_{c2}_holm_p"] = p_m
            row[f"MWU_{c1}_vs_{c2}_rank_biserial"] = rb
            row[f"KS_{c1}_vs_{c2}_D"] = d_ks
            row[f"KS_{c1}_vs_{c2}_holm_p"] = p_k
        rows.append(row)

    tests_df = pd.DataFrame(rows)

    # Global Holm across all feature x pair tests (18 tests)
    raw_ps = [r[3] for r in raw_p_records]
    _, global_adj, _, _ = multipletests(raw_ps, method="holm")
    global_map = {(r[0], r[1], r[2]): p for r, p in zip(raw_p_records, global_adj)}
    for c1, c2 in pairs:
        tests_df[f"MWU_{c1}_vs_{c2}_global_holm_p"] = [
            global_map[(f, c1, c2)] for f in COMMON_FEATURES
        ]

    # PulseRate: only mod vs tv (flat has no measurable pulse rate)
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
                "MWU_rank_biserial": rank_biserial(pr_mod, pr_tv),
                "KS_D": d_pr,
                "KS_p": p_pr_ks,
                "Note": "flat excluded: no measurable pulse rate",
            }
        ]
    )
    return tests_df, pr_df


# --------------------------------------------------------------------------- #
# 3. Models: baseline, scalar-feature classifiers, structural key, leakage bound
# --------------------------------------------------------------------------- #
def make_models() -> dict:
    return {
        "Baseline_Majority": DummyClassifier(strategy="most_frequent"),
        "LogReg_Common6": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
        ),
        "RF_Common6": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1
        ),
    }


def oof_metrics(y_true, y_pred) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "BalancedAccuracy": balanced_accuracy_score(y_true, y_pred),
        "MacroF1": f1_score(y_true, y_pred, average="macro"),
        "Recall_flat": recall_score(y_true, y_pred, labels=["flat"], average="macro", zero_division=0),
        "Recall_mod": recall_score(y_true, y_pred, labels=["mod"], average="macro", zero_division=0),
        "Recall_tv": recall_score(y_true, y_pred, labels=["tv"], average="macro", zero_division=0),
    }


def evaluate_classifiers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Grouped-by-date CV for scalar-feature models, plus baseline and leakage bound."""
    groups = df["Date"].values
    y = df["CallType"].values
    cv = GroupKFold(n_splits=5)
    X_common = df[COMMON_FEATURES].values

    rows, oof_preds = [], {}

    # --- global majority reference: predict the overall most frequent class ---
    majority = pd.Series(y).value_counts().idxmax()
    global_pred = np.array([majority] * len(y))
    rows.append(
        {
            "Model": "Baseline_GlobalMajority",
            "FeatureSet": "none",
            "CV": "in-sample reference",
            **oof_metrics(y, global_pred),
            "Fold_Accuracy_Mean": np.nan,
            "Fold_Accuracy_Std": np.nan,
            "Fold_MacroF1_Mean": np.nan,
            "Fold_MacroF1_Std": np.nan,
            "Note": f"always predict '{majority}'",
        }
    )

    # --- baseline + scalar models ---
    for name, model in make_models().items():
        scores = cross_validate(
            model, X_common, y, cv=cv, groups=groups,
            scoring=["accuracy", "f1_macro"], n_jobs=-1,
        )
        pred = cross_val_predict(model, X_common, y, cv=cv, groups=groups)
        metrics = oof_metrics(y, pred)
        rows.append(
            {
                "Model": name,
                "FeatureSet": "Common6",
                "CV": "GroupKFold(5) by Date",
                **metrics,
                "Fold_Accuracy_Mean": scores["test_accuracy"].mean(),
                "Fold_Accuracy_Std": scores["test_accuracy"].std(),
                "Fold_MacroF1_Mean": scores["test_f1_macro"].mean(),
                "Fold_MacroF1_Std": scores["test_f1_macro"].std(),
                "Note": "reference" if name.startswith("Baseline") else "scalar features only",
            }
        )
        if name == "RF_Common6":
            oof_preds[name] = pred

    # --- leakage upper bound: indicators are deterministic functions of the label ---
    df_b = df[ALL_FEATURES].copy()
    df_b["has_F02"] = df["F02"].notna().astype(int)
    df_b["has_PulseRate"] = df["PulseRate"].notna().astype(int)
    X_leak = df_b.values
    leak_model = HistGradientBoostingClassifier(max_depth=4, random_state=RANDOM_SEED)
    leak_pred = cross_val_predict(leak_model, X_leak, y, cv=cv, groups=groups)
    leak_metrics = oof_metrics(y, leak_pred)
    rows.append(
        {
            "Model": "HistGB_All8_Indicators",
            "FeatureSet": "All8+MissingnessIndicators",
            "CV": "GroupKFold(5) by Date",
            **leak_metrics,
            "Fold_Accuracy_Mean": np.nan,
            "Fold_Accuracy_Std": np.nan,
            "Fold_MacroF1_Mean": np.nan,
            "Fold_MacroF1_Std": np.nan,
            "Note": "LEAKAGE UPPER BOUND: has_F02 == is_tv and has_PulseRate == is_flat by definition",
        }
    )
    oof_preds["HistGB_All8_Indicators"] = leak_pred

    # --- structural diagnostic key (not a learned model) ---
    key_pred = np.where(
        df["F02"].notna(), "tv", np.where(df["PulseRate"].notna(), "mod", "flat")
    )
    key_metrics = oof_metrics(y, key_pred)
    rows.append(
        {
            "Model": "StructuralKey",
            "FeatureSet": "F02 presence / PulseRate presence",
            "CV": "none (deterministic rule)",
            **key_metrics,
            "Fold_Accuracy_Mean": np.nan,
            "Fold_Accuracy_Std": np.nan,
            "Fold_MacroF1_Mean": np.nan,
            "Fold_MacroF1_Std": np.nan,
            "Note": "diagnostic key based on call definition, not statistical learning",
        }
    )
    oof_preds["StructuralKey"] = key_pred

    # Feature importance from the RF on the six scalar features
    rf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_common, y)
    imp_df = pd.DataFrame(
        {"Feature": COMMON_FEATURES, "RF_Importance": rf.feature_importances_}
    ).sort_values("RF_Importance", ascending=False)

    # Permutation test for the scalar-feature RF (is 0.42 better than chance?)
    perm = permutation_test_rf(X_common, y, groups)
    pd.DataFrame(
        [{"Observed_MacroF1": perm["observed"], "Permutation_p": perm["p_value"], "N_permutations": N_PERMUTATIONS}]
    ).to_csv(TABLES_DIR / "q1_permutation_test.csv", index=False)

    cv_df = pd.DataFrame(rows)
    cv_df["Permutation_p_RF"] = np.where(
        cv_df["Model"] == "RF_Common6", perm["p_value"], np.nan
    )
    return cv_df, imp_df, oof_preds


def permutation_test_rf(X, y, groups) -> dict:
    cv = GroupKFold(n_splits=5)
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    observed = f1_score(y, cross_val_predict(model, X, y, cv=cv, groups=groups), average="macro")

    rng = np.random.default_rng(RANDOM_SEED)
    null = []
    for _ in range(N_PERMUTATIONS):
        y_perm = rng.permutation(y)
        pred = cross_val_predict(model, X, y_perm, cv=cv, groups=groups)
        null.append(f1_score(y_perm, pred, average="macro"))
    null = np.array(null)
    p_value = float((np.sum(null >= observed) + 1) / (len(null) + 1))
    return {"observed": float(observed), "p_value": p_value, "null": null}


def evaluate_single_voice_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Flat vs mod among single-voice calls (F02 absent, n=84, balanced 44/40).

    This is the fair test of whether scalar features carry usable information:
    the two classes are nearly balanced here, unlike the 50-row ambiguous subset.
    """
    sub = df[df["F02"].isna()].copy()
    y = sub["CallType"].values
    groups = sub["Date"].values
    X = sub[COMMON_FEATURES].values
    cv = GroupKFold(n_splits=5)

    rows = []
    for name, model in {
        "LogReg_Common6": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
        ),
        "RF_Common6": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
    }.items():
        pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
        rows.append(
            {
                "Model": name,
                "Task": "flat vs mod (single-voice)",
                "N": len(sub),
                "N_flat": int((y == "flat").sum()),
                "N_mod": int((y == "mod").sum()),
                "Accuracy": accuracy_score(y, pred),
                "BalancedAccuracy": balanced_accuracy_score(y, pred),
                "MacroF1": f1_score(y, pred, average="macro"),
                "MajorityAccuracy": max((y == "flat").mean(), (y == "mod").mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_ambiguous_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Among samples with neither F02 nor PulseRate, can scalar features split flat vs mod?"""
    sub = df[df["F02"].isna() & df["PulseRate"].isna()].copy()
    y = sub["CallType"].values
    groups = sub["Date"].values
    X = sub[COMMON_FEATURES].values
    cv = GroupKFold(n_splits=5)

    rows = []
    for name, model in {
        "LogReg_Common6": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
        ),
        "RF_Common6": RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1),
    }.items():
        pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
        rows.append(
            {
                "Model": name,
                "N": len(sub),
                "N_flat": int((y == "flat").sum()),
                "N_mod": int((y == "mod").sum()),
                "Accuracy": accuracy_score(y, pred),
                "MacroF1": f1_score(y, pred, average="macro"),
                "MajorityAccuracy": max((y == "flat").mean(), (y == "mod").mean()),
                "Note": "only 6 mod samples are ambiguous; results are unstable",
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. Figures
# --------------------------------------------------------------------------- #
def plot_all_visualizations(df: pd.DataFrame, oof_preds: dict) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)

    # 2x4 distributions
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

    # PCA biplot on standardized common features
    X_std = StandardScaler().fit_transform(df[COMMON_FEATURES].values)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(X_std)
    pca_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "CallType": df["CallType"]})

    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.scatterplot(
        data=pca_df, x="PC1", y="PC2", hue="CallType", style="CallType",
        s=55, alpha=0.85, ax=ax,
    )
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

    # Confusion matrices: scalar RF, structural key, leakage upper bound
    y_true = df["CallType"].values
    panels = [
        ("RF (6 scalar features)", "RF_Common6"),
        ("Structural key", "StructuralKey"),
        ("Leakage upper bound", "HistGB_All8_Indicators"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (title, key) in zip(axes, panels):
        cm = confusion_matrix(y_true, oof_preds[key], labels=CLASSES)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES).plot(
            ax=ax, cmap="Blues", colorbar=False
        )
        ax.set_title(title)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "q1_confusion_matrices.png", dpi=300)
    plt.close(fig)


def main() -> None:
    data_dir = resolve_data_dir()
    df = pd.read_csv(data_dir / "Data01_AfricanPenguins_seasurface_calls_acoustics.csv")

    desc_df = compute_descriptive_stats(df)
    tests_df, pr_df = run_nonparametric_tests(df)
    cv_df, imp_df, oof_preds = evaluate_classifiers(df)
    amb_df = evaluate_ambiguous_subset(df)
    sv_df = evaluate_single_voice_subset(df)

    desc_df.to_csv(TABLES_DIR / "q1_descriptive_stats.csv", index=False)
    tests_df.to_csv(TABLES_DIR / "q1_nonparametric_tests.csv", index=False)
    pr_df.to_csv(TABLES_DIR / "q1_pulserate_tests.csv", index=False)
    cv_df.to_csv(TABLES_DIR / "q1_classifier_cv.csv", index=False)
    imp_df.to_csv(TABLES_DIR / "q1_rf_importances.csv", index=False)
    amb_df.to_csv(TABLES_DIR / "q1_ambiguous_subset.csv", index=False)
    sv_df.to_csv(TABLES_DIR / "q1_single_voice_subset.csv", index=False)

    # Long-format confusion matrix for the structural key (used in the paper)
    key_cm = confusion_matrix(df["CallType"].values, oof_preds["StructuralKey"], labels=CLASSES)
    pd.DataFrame(key_cm, index=CLASSES, columns=CLASSES).to_csv(
        TABLES_DIR / "q1_structural_key_confusion.csv"
    )

    plot_all_visualizations(df, oof_preds)

    print("=== Grouped CV (by recording date) ===")
    cols = ["Model", "Accuracy", "BalancedAccuracy", "MacroF1", "Recall_flat", "Recall_mod", "Recall_tv", "Note"]
    print(cv_df[cols].round(3).to_string(index=False))
    print("\n=== Single-voice subset (flat vs mod, F02 absent) ===")
    print(sv_df.round(3).to_string(index=False))
    print("\n=== Ambiguous subset (no F02, no PulseRate) ===")
    print(amb_df.round(3).to_string(index=False))
    print("\n=== RF permutation test ===")
    print(f"observed macro-F1 = {cv_df.loc[cv_df.Model=='RF_Common6','MacroF1'].iloc[0]:.3f}")
    print(pd.read_csv(TABLES_DIR / "q1_permutation_test.csv").round(4).to_string(index=False))
    print("\n=== KW tests ===")
    print(tests_df[["Feature", "KW_H", "KW_p", "KW_epsilon2"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
