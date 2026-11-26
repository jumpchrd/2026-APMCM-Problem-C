"""Problem 2: Signal propagation degradation and effective communication range."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from config import FIGURES_DIR, RANDOM_SEED, TABLES_DIR, resolve_data_dir

DISTANCES = np.array([4, 8, 16, 32, 64, 128, 256], dtype=float)
DIST_COLS = ["4m", "8m", "16m", "32m", "64m", "128m", "256m"]

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

THRESHOLDS = {"potential_0.50": 0.50, "moderate_0.65": 0.65, "reliable_0.75": 0.75}


def exp_decay(d: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Three-parameter exponential decay curve."""
    return a * np.exp(-b * d) + c


def invert_exp_decay(tau: float, a: float, b: float, c: float) -> float:
    """Solve a * exp(-b * d) + c = tau for distance d."""
    if tau <= c or a <= 0 or b <= 0 or (tau - c) >= a:
        return np.nan
    return float(-np.log((tau - c) / a) / b)


def load_composite_fidelity(
    data_dir, call_type: str, weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
) -> np.ndarray:
    """Compute composite acoustic fidelity matrix (n_samples x n_distances)."""
    files = FILE_MAP[call_type]
    mats = [pd.read_csv(data_dir / f)[DIST_COLS].values for f in files]
    w1, w2, w3 = weights
    return w1 * mats[0] + w2 * mats[1] + w3 * mats[2]


def fit_and_bootstrap(
    fidelity_mat: np.ndarray, n_boot: int = 500
) -> tuple[np.ndarray, dict[str, tuple[float, float, float]]]:
    """Fit exponential decay to mean fidelity and estimate Bootstrap 95% CIs for d_eff."""
    mean_y = np.nanmean(fidelity_mat, axis=0)
    popt, _ = curve_fit(
        exp_decay, DISTANCES, mean_y, p0=[0.6, 0.01, 0.2], bounds=([0, 0, 0], [2, 1, 1]), maxfev=5000
    )

    rng = np.random.default_rng(RANDOM_SEED)
    boot_dists = {k: [] for k in THRESHOLDS}
    n_rows = fidelity_mat.shape[0]
    for _ in range(n_boot):
        idx = rng.integers(0, n_rows, size=n_rows)
        y_b = np.nanmean(fidelity_mat[idx], axis=0)
        try:
            p_b, _ = curve_fit(
                exp_decay, DISTANCES, y_b, p0=popt, bounds=([0, 0, 0], [2, 1, 1]), maxfev=3000
            )
            for k, tau in THRESHOLDS.items():
                boot_dists[k].append(invert_exp_decay(tau, *p_b))
        except RuntimeError:
            continue

    summary = {}
    for k, tau in THRESHOLDS.items():
        pt = invert_exp_decay(tau, *popt)
        vals = np.array([v for v in boot_dists[k] if not np.isnan(v)])
        lo = float(np.percentile(vals, 2.5)) if len(vals) else np.nan
        hi = float(np.percentile(vals, 97.5)) if len(vals) else np.nan
        summary[k] = (pt, lo, hi)
    return popt, summary


def main() -> None:
    data_dir = resolve_data_dir()
    rows = []

    plt.figure(figsize=(8, 5))
    d_grid = np.linspace(4, 256, 300)

    for ct in ["flat", "mod", "tv"]:
        mat = load_composite_fidelity(data_dir, ct)
        popt, summary = fit_and_bootstrap(mat)
        for k, (pt, lo, hi) in summary.items():
            rows.append(
                {
                    "CallType": ct,
                    "Criterion": k,
                    "d_eff_m": pt,
                    "CI_2.5_m": lo,
                    "CI_97.5_m": hi,
                    "a": popt[0],
                    "b": popt[1],
                    "c": popt[2],
                }
            )
        plt.scatter(DISTANCES, np.nanmean(mat, axis=0), label=f"{ct} (mean)")
        plt.plot(d_grid, exp_decay(d_grid, *popt), label=f"{ct} (fit)")

    for k, tau in THRESHOLDS.items():
        plt.axhline(tau, linestyle="--", alpha=0.5, label=f"Threshold {tau:.2f}")

    plt.xlabel("Propagation Distance (m)")
    plt.ylabel("Composite Acoustic Fidelity F(d)")
    plt.title("Signal Propagation Degradation by Call Type")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "q2_degradation_curves.png", dpi=300)
    plt.close()

    res_df = pd.DataFrame(rows)
    res_df.to_csv(TABLES_DIR / "q2_effective_range.csv", index=False)
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    main()
