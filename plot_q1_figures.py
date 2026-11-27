"""Regenerate the three Problem-1 figures as standalone PNGs.

Run from the repository root:
    python plot_q1_figures.py

Requires: pandas, numpy, matplotlib, seaborn, scikit-learn.
The input file is data/Data01_AfricanPenguins_seasurface_calls_acoustics.csv
(or the path in the APMCM_C_DATA_DIR environment variable).
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
FIGURES_DIR = BASE / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
COMMON_FEATURES = ["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75"]
ALL_FEATURES = COMMON_FEATURES + ["F02", "PulseRate"]
CLASSES = ["flat", "mod", "tv"]


def find_data_file() -> Path:
    env = os.environ.get("APMCM_C_DATA_DIR")
    candidates = []
    if env:
        candidates.append(Path(env) / "Data01_AfricanPenguins_seasurface_calls_acoustics.csv")
    candidates.append(BASE / "data" / "Data01_AfricanPenguins_seasurface_calls_acoustics.csv")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Cannot find Data01_AfricanPenguins_seasurface_calls_acoustics.csv. "
        "Put it in ./data or set APMCM_C_DATA_DIR."
    )


def prepare_predictions(df: pd.DataFrame) -> dict:
    """Out-of-fold predictions from grouped cross-validation."""
    y = df["CallType"].values
    groups = df["Date"].values
    cv = GroupKFold(n_splits=5)

    X_common = df[COMMON_FEATURES].values
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    rf_pred = cross_val_predict(rf, X_common, y, cv=cv, groups=groups)

    key_pred = np.where(
        df["F02"].notna(), "tv", np.where(df["PulseRate"].notna(), "mod", "flat")
    )

    df_b = df[ALL_FEATURES].copy()
    df_b["has_F02"] = df["F02"].notna().astype(int)
    df_b["has_PulseRate"] = df["PulseRate"].notna().astype(int)
    leak = HistGradientBoostingClassifier(max_depth=4, random_state=RANDOM_SEED)
    leak_pred = cross_val_predict(leak, df_b.values, y, cv=cv, groups=groups)

    return {
        "RF_Common6": rf_pred,
        "StructuralKey": key_pred,
        "HistGB_All8_Indicators": leak_pred,
    }


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
    data_path = find_data_file()
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} calls from {data_path}")
    oof_preds = prepare_predictions(df)
    plot_all_visualizations(df, oof_preds)
    print(f"Saved 3 figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
