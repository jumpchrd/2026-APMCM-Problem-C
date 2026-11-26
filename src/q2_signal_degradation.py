"""Problem 2: Signal propagation degradation, model comparison, and effective range."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import brentq, curve_fit
from sklearn.decomposition import PCA

from config import FIGURES_DIR, RANDOM_SEED, TABLES_DIR, resolve_data_dir

DISTANCES = np.array([4, 8, 16, 32, 64, 128, 256], dtype=float)
DIST_COLS = ["4m", "8m", "16m", "32m", "64m", "128m", "256m"]
CALL_TYPES = ["flat", "mod", "tv"]
DIM_NAMES = ["Spectrogram", "AmplitudeSpectrum", "Envelope"]

FILE_MAP = {
    "flat": (
        "Data03_correlation1m_flat_call_spectrogram.csv",
        "Data06_correlation1m_flat_call_amplitude_spectrum.csv",
        "Data09_correlation1m_flat_call_envelope.csv",
    ),
    "mod": (
        "Data04_correlation1m_modulated_call_spectrogram.csv",
        "Data07_correlation1m_modulated_call_amplitude_spectrum.csv",
        "Data10_correlation1m_modulated_call_envelope.csv",
    ),
    "tv": (
        "Data05_correlation1m_twovoice_call_spectrogram.csv",
        "Data08_correlation1m_twovoice_call_amplitude_spectrum.csv",
        "Data11_correlation1m_twovoice_call_envelope.csv",
    ),
}

THRESHOLDS = {
    "potential_0.50": 0.50,
    "recognition_0.65": 0.65,
    "reliable_0.75": 0.75,
}


def load_raw_dimensions(data_dir, call_type: str) -> list[np.ndarray]:
    """Load the 3 correlation matrices (spectrogram, amplitude spectrum, envelope)."""
    return [pd.read_csv(data_dir / f)[DIST_COLS].values for f in FILE_MAP[call_type]]


def compute_objective_weights(data_dir) -> pd.DataFrame:
    """Compute PCA-1 and Entropy weights pooled across all call types and distances."""
    stacked = []
    for ct in CALL_TYPES:
        mats = load_raw_dimensions(data_dir, ct)
        # Flatten (300 x 7) -> (2100 x 3)
        stacked.append(np.column_stack([m.ravel() for m in mats]))
    all_obs = np.vstack(stacked)
    valid = all_obs[~np.isnan(all_obs).any(axis=1)]

    # PCA first component weights
    pca = PCA(n_components=1, random_state=RANDOM_SEED).fit(valid)
    w_pca = np.abs(pca.components_[0])
    w_pca = w_pca / w_pca.sum()

    # Entropy Weight Method (EWM)
    eps = 1e-12
    col_min = valid.min(axis=0)
    col_max = valid.max(axis=0)
    normed = (valid - col_min) / (col_max - col_min + eps)
    p = normed / (normed.sum(axis=0) + eps)
    k = 1.0 / np.log(len(valid))
    entropy = -k * np.sum(p * np.log(p + eps), axis=0)
    div = 1.0 - entropy
    w_ewm = div / div.sum()

    return pd.DataFrame(
        {
            "Dimension": DIM_NAMES,
            "Equal_Weight": [1 / 3, 1 / 3, 1 / 3],
            "PCA_Weight": w_pca,
            "Entropy_Weight": w_ewm,
        }
    )


def combine_fidelity(mats: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Weighted average across 3 correlation matrices, ignoring NaNs safely."""
    stacked = np.stack(mats, axis=-1)  # (N, 7, 3)
    w = np.asarray(weights, dtype=float)
    mask = ~np.isnan(stacked)
    w_masked = np.where(mask, w[None, None, :], 0.0)
    num = np.nansum(stacked * w[None, None, :], axis=-1)
    den = np.sum(w_masked, axis=-1)
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


# --- Candidate degradation models ---
def exp_decay(d: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.exp(-b * d) + c


def log_linear_decay(d: np.ndarray, a: float, b: float, alpha: float) -> np.ndarray:
    return a - b * np.log(d) - alpha * d


def logistic_4pl(d: np.ndarray, d_min: float, d_max: float, d0: float, p: float) -> np.ndarray:
    return d_min + (d_max - d_min) / (1.0 + (d / d0) ** p)


MODEL_SPECS = {
    "Exponential": (exp_decay, [0.6, 0.01, 0.2], ([0, 0, 0], [2, 1, 1])),
    "LogLinear": (log_linear_decay, [1.0, 0.05, 0.001], ([0, 0, 0], [2, 1, 0.1])),
    "Logistic4PL": (logistic_4pl, [0.1, 0.9, 60.0, 1.2], ([0, 0.5, 1.0, 0.1], [0.6, 1.2, 500.0, 10.0])),
}


def fit_model(model_name: str, y_mean: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Fit a candidate model and return (params, R2, RMSE, AIC)."""
    fn, p0, bounds = MODEL_SPECS[model_name]
    popt, _ = curve_fit(fn, DISTANCES, y_mean, p0=p0, bounds=bounds, maxfev=10000)
    y_hat = fn(DISTANCES, *popt)
    resid = y_mean - y_hat
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y_mean - y_mean.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    rmse = float(np.sqrt(sse / len(DISTANCES)))
    n = len(DISTANCES)
    k = len(popt)
    aic = n * np.log(sse / n + 1e-12) + 2 * k
    return popt, r2, rmse, aic


def invert_model(model_name: str, popt: np.ndarray, tau: float) -> float:
    """Numerically solve model(d) = tau on [1, 1000] meters."""
    fn = MODEL_SPECS[model_name][0]
    f_lo = fn(np.array([1.0]), *popt)[0] - tau
    f_hi = fn(np.array([1000.0]), *popt)[0] - tau
    if f_lo * f_hi > 0:
        return np.nan
    return float(brentq(lambda x: fn(np.array([x]), *popt)[0] - tau, 1.0, 1000.0))


def bootstrap_effective_range(
    fidelity_mat: np.ndarray, model_name: str, n_boot: int = 500
) -> tuple[np.ndarray, dict[str, tuple[float, float, float]]]:
    """Fit model and compute Bootstrap 95% CIs for effective distances."""
    mean_y = np.nanmean(fidelity_mat, axis=0)
    popt, _, _, _ = fit_model(model_name, mean_y)

    rng = np.random.default_rng(RANDOM_SEED)
    boot_dists = {k: [] for k in THRESHOLDS}
    n_rows = fidelity_mat.shape[0]
    fn, _, bounds = MODEL_SPECS[model_name]

    for _ in range(n_boot):
        idx = rng.integers(0, n_rows, size=n_rows)
        y_b = np.nanmean(fidelity_mat[idx], axis=0)
        try:
            p_b, _ = curve_fit(fn, DISTANCES, y_b, p0=popt, bounds=bounds, maxfev=4000)
            for k, tau in THRESHOLDS.items():
                boot_dists[k].append(invert_model(model_name, p_b, tau))
        except RuntimeError:
            continue

    summary = {}
    for k, tau in THRESHOLDS.items():
        pt = invert_model(model_name, popt, tau)
        vals = np.array([v for v in boot_dists[k] if not np.isnan(v)])
        lo = float(np.percentile(vals, 2.5)) if len(vals) else np.nan
        hi = float(np.percentile(vals, 97.5)) if len(vals) else np.nan
        summary[k] = (pt, lo, hi)
    return popt, summary


def main() -> None:
    data_dir = resolve_data_dir()
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)

    # 1. Objective weights
    weights_df = compute_objective_weights(data_dir)
    weights_df.to_csv(TABLES_DIR / "q2_fidelity_weights.csv", index=False)
    pca_weights = weights_df["PCA_Weight"].values

    # 2. Model comparison (using PCA-weighted fidelity)
    comp_rows = []
    fidelity_by_ct = {}
    for ct in CALL_TYPES:
        mats = load_raw_dimensions(data_dir, ct)
        f_mat = combine_fidelity(mats, pca_weights)
        fidelity_by_ct[ct] = f_mat
        y_mean = np.nanmean(f_mat, axis=0)
        for m_name in MODEL_SPECS:
            try:
                _, r2, rmse, aic = fit_model(m_name, y_mean)
            except RuntimeError:
                r2, rmse, aic = np.nan, np.nan, np.nan
            comp_rows.append(
                {"CallType": ct, "Model": m_name, "R2": r2, "RMSE": rmse, "AIC": aic}
            )
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(TABLES_DIR / "q2_model_comparison.csv", index=False)

    # Select best overall model by lowest mean AIC
    best_model = comp_df.groupby("Model")["AIC"].mean().idxmin()

    # 3. Effective range + Bootstrap 95% CI using best_model
    range_rows = []
    fitted_params = {}
    for ct in CALL_TYPES:
        popt, summary = bootstrap_effective_range(fidelity_by_ct[ct], best_model)
        fitted_params[ct] = popt
        for k, (pt, lo, hi) in summary.items():
            range_rows.append(
                {
                    "CallType": ct,
                    "Model": best_model,
                    "Criterion": k,
                    "d_eff_m": pt,
                    "CI_2.5_m": lo,
                    "CI_97.5_m": hi,
                }
            )
    range_df = pd.DataFrame(range_rows)
    range_df.to_csv(TABLES_DIR / "q2_effective_range.csv", index=False)

    # 4. Sensitivity analysis (threshold sweep + weight scheme sweep)
    sens_rows = []
    tau_grid = np.arange(0.45, 0.81, 0.05)
    for ct in CALL_TYPES:
        for tau in tau_grid:
            d_val = invert_model(best_model, fitted_params[ct], float(tau))
            sens_rows.append(
                {"Analysis": "ThresholdSweep", "CallType": ct, "Setting": f"{tau:.2f}", "d_eff_m": d_val}
            )

    weight_schemes = {
        "PCA": weights_df["PCA_Weight"].values,
        "Equal": weights_df["Equal_Weight"].values,
        "Entropy": weights_df["Entropy_Weight"].values,
        "SpecOnly": np.array([1.0, 0.0, 0.0]),
        "AmpOnly": np.array([0.0, 1.0, 0.0]),
        "EnvOnly": np.array([0.0, 0.0, 1.0]),
    }
    for ct in CALL_TYPES:
        mats = load_raw_dimensions(data_dir, ct)
        for s_name, w_vec in weight_schemes.items():
            f_mat = combine_fidelity(mats, w_vec)
            popt, _, _, _ = fit_model(best_model, np.nanmean(f_mat, axis=0))
            d_pot = invert_model(best_model, popt, 0.50)
            d_rel = invert_model(best_model, popt, 0.75)
            sens_rows.append(
                {"Analysis": "WeightScheme_0.50", "CallType": ct, "Setting": s_name, "d_eff_m": d_pot}
            )
            sens_rows.append(
                {"Analysis": "WeightScheme_0.75", "CallType": ct, "Setting": s_name, "d_eff_m": d_rel}
            )
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(TABLES_DIR / "q2_sensitivity_analysis.csv", index=False)

    # 5. Plots
    # Plot A: Degradation curves
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    d_grid = np.linspace(4, 256, 300)
    palette = {"flat": "#1f77b4", "mod": "#ff7f0e", "tv": "#2ca02c"}
    fn = MODEL_SPECS[best_model][0]
    for ct in CALL_TYPES:
        mat = fidelity_by_ct[ct]
        mean_y = np.nanmean(mat, axis=0)
        std_y = np.nanstd(mat, axis=0)
        ax.errorbar(DISTANCES, mean_y, yerr=std_y, fmt="o", capsize=3, color=palette[ct], alpha=0.75)
        ax.plot(d_grid, fn(d_grid, *fitted_params[ct]), color=palette[ct], lw=2.2, label=f"{ct} ({best_model})")
    for k, tau in THRESHOLDS.items():
        ax.axhline(tau, ls="--", lw=1.1, color="gray", alpha=0.7)
        ax.text(258, tau, k, va="center", fontsize=8.5, color="dimgray")
    ax.set_xlim(0, 295)
    ax.set_xlabel("Propagation Distance (m)")
    ax.set_ylabel("Composite Acoustic Fidelity F(d)")
    ax.set_title(f"Signal Propagation Degradation ({best_model} Fit, PCA-Weighted)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "q2_degradation_curves.png", dpi=300)
    plt.close(fig)

    # Plot B: Effective range with 95% CI
    focus_df = range_df[range_df["Criterion"].isin(["potential_0.50", "reliable_0.75"])].copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=focus_df, x="Criterion", y="d_eff_m", hue="CallType", palette=palette, ax=ax)
    for bar, (_, row) in zip(ax.patches, focus_df.sort_values(["CallType", "Criterion"]).iterrows()):
        pass
    # Overlay exact CI error bars
    x_positions = {"potential_0.50": 0, "reliable_0.75": 1}
    offsets = {"flat": -0.27, "mod": 0.0, "tv": 0.27}
    for _, r in focus_df.iterrows():
        xp = x_positions[r["Criterion"]] + offsets[r["CallType"]]
        yerr = [[r["d_eff_m"] - r["CI_2.5_m"]], [r["CI_97.5_m"] - r["d_eff_m"]]]
        ax.errorbar(xp, r["d_eff_m"], yerr=yerr, fmt="none", ecolor="black", capsize=4, lw=1.3)
    ax.set_ylabel("Effective Communication Range (m)")
    ax.set_title("Effective Communication Range with 95% Bootstrap CI")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "q2_effective_range_ci.png", dpi=300)
    plt.close(fig)

    # Plot C: Sensitivity analysis
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    t_df = sens_df[sens_df["Analysis"] == "ThresholdSweep"].copy()
    t_df["Threshold"] = t_df["Setting"].astype(float)
    sns.lineplot(data=t_df, x="Threshold", y="d_eff_m", hue="CallType", marker="o", palette=palette, ax=axes[0])
    axes[0].set_title("Sensitivity to Recognition Threshold")
    axes[0].set_ylabel("Effective Range (m)")

    w_df = sens_df[sens_df["Analysis"] == "WeightScheme_0.50"]
    sns.barplot(data=w_df, x="Setting", y="d_eff_m", hue="CallType", palette=palette, ax=axes[1])
    axes[1].set_title("Sensitivity to Weighting Scheme (Threshold = 0.50)")
    axes[1].set_ylabel("Effective Range (m)")
    axes[1].tick_params(axis="x", rotation=25)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "q2_sensitivity_analysis.png", dpi=300)
    plt.close(fig)

    print("=== Objective Weights ===")
    print(weights_df.to_string(index=False))
    print("\n=== Model Comparison ===")
    print(comp_df.to_string(index=False))
    print(f"\n=== Best Model: {best_model} | Effective Range ===")
    print(range_df.to_string(index=False))


if __name__ == "__main__":
    main()
