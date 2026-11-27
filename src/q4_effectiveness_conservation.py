"""Problem 4: Acoustic communication effectiveness index and conservation scenarios."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.optimize import brentq

from config import FIGURES_DIR, TABLES_DIR, resolve_data_dir
from q2_signal_degradation import (
    CALL_TYPES,
    combine_fidelity,
    compute_objective_weights,
    fit_model,
    load_raw_dimensions,
    logistic_4pl,
)
from q3_playback_responses import TIER_NAMES, fit_main_gee, predict_gee, prepare_playback_df

# Ecological utility weights for the 5 response tiers (0=None .. 4=ApproachInvolved)
UTILITY_WEIGHTS = {
    0: 0.00,  # None
    1: 0.25,  # HeadUpOnly (vigilance / detection)
    2: 0.50,  # VocalOnly (acoustic contact)
    3: 0.75,  # HeadUp+Vocal (attentive vocal reply)
    4: 1.00,  # ApproachInvolved (cohesion / foraging recruitment)
}


def fit_q2_fidelity_curves(data_dir) -> dict[str, np.ndarray]:
    """Fit PCA-weighted Logistic4PL fidelity curves for flat, mod, tv."""
    weights_df = compute_objective_weights(data_dir)
    pca_w = weights_df["PCA_Weight"].values
    params = {}
    for ct in CALL_TYPES:
        mats = load_raw_dimensions(data_dir, ct)
        f_mat = combine_fidelity(mats, pca_w)
        popt, _, _, _ = fit_model("Logistic4PL", np.nanmean(f_mat, axis=0))
        params[ct] = popt
    return params


def fit_q3_mnlogit(df_play: pd.DataFrame):
    """Fit unified MNLogit model on 5 response tiers for Q4 integration."""
    mn_df = df_play.copy()
    mn_df["IsFlat"] = (mn_df["CallType"] == "flat").astype(float)
    mn_df["IsMod"] = (mn_df["CallType"] == "mod").astype(float)
    mn_df["IsTv"] = (mn_df["CallType"] == "tv").astype(float)
    mn_df["CtxSearching"] = (mn_df["Context"] == "searching").astype(float)
    mn_df["CtxCommuting"] = (mn_df["Context"] == "commuting").astype(float)
    mn_df["log2Distance"] = np.log2(mn_df["Distance"].clip(lower=1))

    cols = ["IsFlat", "IsMod", "IsTv", "CtxSearching", "CtxCommuting", "log2Distance", "LogNpenguins"]
    X = sm.add_constant(mn_df[cols], has_constant="add")
    y = mn_df["ResponseTier"]
    return sm.MNLogit(y, X).fit(method="bfgs", maxiter=500, disp=False), cols


def predict_tier_probs(mn_res, cols: list[str], call_type: str, context: str, dist_m: float, n_penguins: float) -> np.ndarray:
    """Return predicted 5-tier probability vector [P0, P1, P2, P3, P4]."""
    row = {
        "const": 1.0,
        "IsFlat": 1.0 if call_type == "flat" else 0.0,
        "IsMod": 1.0 if call_type == "mod" else 0.0,
        "IsTv": 1.0 if call_type == "tv" else 0.0,
        "CtxSearching": 1.0 if context == "searching" else 0.0,
        "CtxCommuting": 1.0 if context == "commuting" else 0.0,
        "log2Distance": float(np.log2(max(dist_m, 1.0))),
        "LogNpenguins": np.log(max(n_penguins, 1.0)),
    }
    X_new = pd.DataFrame([row])[["const"] + cols]
    return mn_res.predict(X_new).values[0]


def solve_radius(fn, d_min: float = 1.0, d_max: float = 400.0) -> float:
    """Find root of fn(d) = 0 on [d_min, d_max], returning boundary if monotonic above/below."""
    v_lo, v_hi = fn(d_min), fn(d_max)
    if v_lo <= 0:
        return 0.0
    if v_hi >= 0:
        return d_max
    return float(brentq(fn, d_min, d_max))


def main() -> None:
    data_dir = resolve_data_dir()
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)

    q2_params = fit_q2_fidelity_curves(data_dir)
    df_play = prepare_playback_df(data_dir)
    mn_res, mn_cols = fit_q3_mnlogit(df_play)
    gee_res, _ = fit_main_gee(df_play)

    med_n = float(df_play["Npenguins"].median())
    u_vec = np.array([UTILITY_WEIGHTS[i] for i in range(5)])
    contexts = ["searching", "commuting", "after"]
    ctx_weights = df_play["Context"].value_counts(normalize=True).to_dict()

    # 1. Effectiveness Index E(d) across distances & contexts
    d_grid = np.linspace(10, 250, 49)
    eff_rows = []
    for ct in CALL_TYPES:
        for d in d_grid:
            f_val = float(logistic_4pl(np.array([d]), *q2_params[ct])[0])
            # Context-weighted behavioral utility and overall response probability
            u_beh = 0.0
            p_resp = 0.0
            p_mn = 0.0
            for ctx in contexts:
                probs = predict_tier_probs(mn_res, mn_cols, ct, ctx, float(d), med_n)
                p_none = float(probs[0])
                # Composition across response tiers, conditional on responding
                cond = probs[1:] / (1.0 - p_none) if p_none < 1.0 else np.zeros(len(u_vec) - 1)
                # Overall response probability comes from the Problem-3 GEE
                p_g = float(predict_gee(gee_res, ct, ctx, float(d), med_n)[0])
                u_beh += ctx_weights[ctx] * p_g * float(np.dot(cond, u_vec[1:]))
                p_resp += ctx_weights[ctx] * p_g
                p_mn += ctx_weights[ctx] * (1.0 - p_none)
            e_idx = float(np.sqrt(max(f_val, 0.0) * max(u_beh, 0.0)))
            eff_rows.append(
                {
                    "CallType": ct,
                    "Distance_m": float(d),
                    "Fidelity_F": f_val,
                    "BehavioralUtility_U": u_beh,
                    "ResponseProb": p_resp,
                    "ResponseProb_MN": p_mn,
                    "Effectiveness_E": e_idx,
                }
            )
    eff_df = pd.DataFrame(eff_rows)
    eff_df.to_csv(TABLES_DIR / "q4_effectiveness_index.csv", index=False)
    cons = (eff_df["ResponseProb"] - eff_df["ResponseProb_MN"]).abs()
    pd.DataFrame([{"N": len(cons), "MeanAbsDiff": cons.mean(), "MaxAbsDiff": cons.max(),
                   "GEE_Model": "q3 main GEE", "MN_Model": "5-tier multinomial"}]).to_csv(
        TABLES_DIR / "q4_model_consistency.csv", index=False)

    # 2. Expected responding individuals & sensitivity to individual response proportion rho
    sens_rows = []
    for ct in CALL_TYPES:
        for n_grp in [4, int(med_n), 16]:
            for rho in [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
                # Evaluate at 50m, 100m, 150m (searching context)
                for dist in [50.0, 100.0, 150.0]:
                    p_trial_resp = float(predict_gee(gee_res, ct, "searching", dist, n_grp)[0])
                    # Convert the group-level "at least one responds" probability into
                    # an individual-level probability under homogeneity, then scale by rho.
                    p_ind = 1.0 - (1.0 - p_trial_resp) ** (1.0 / n_grp) if p_trial_resp < 1 else 1.0
                    exp_n = n_grp * rho * p_ind
                    sens_rows.append(
                        {
                            "CallType": ct,
                            "Npenguins": n_grp,
                            "IndividualProp_rho": rho,
                            "Distance_m": dist,
                            "TrialRespProb": p_trial_resp,
                            "ExpectedResponders": exp_n,
                        }
                    )
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(TABLES_DIR / "q4_expected_responders_sensitivity.csv", index=False)

    # 3. Noise / Adverse Sea Scenarios
    # Scenario A: Recognition threshold raised 0.50 -> 0.60 -> 0.70
    scen_rows = []
    for ct in CALL_TYPES:
        for tau in [0.50, 0.60, 0.70]:
            r_eff = solve_radius(lambda d: float(logistic_4pl(np.array([d]), *q2_params[ct])[0]) - tau)
            p_resp_at_r = float(predict_gee(gee_res, ct, "searching", max(r_eff, 1.0), med_n)[0])
            p_ind_at_r = 1.0 - (1.0 - p_resp_at_r) ** (1.0 / med_n) if p_resp_at_r < 1.0 else 1.0
            scen_rows.append(
                {
                    "Scenario": "ThresholdElevation",
                    "CallType": ct,
                    "Parameter": f"tau={tau:.2f}",
                    "EffectiveRadius_m": r_eff,
                    "RespProbAtRadius": p_resp_at_r,
                    "ExpectedResponders_rho0.5": med_n * 0.5 * p_ind_at_r,
                }
            )

    # Scenario B: Signal correlation reduced by 0%, 5%, 10%, 15%, 20% across all distances (at tau=0.50)
    for ct in CALL_TYPES:
        for delta in [0.0, 0.05, 0.10, 0.15, 0.20]:
            r_eff = solve_radius(
                lambda d: (1.0 - delta) * float(logistic_4pl(np.array([d]), *q2_params[ct])[0]) - 0.50
            )
            p_resp_at_r = float(predict_gee(gee_res, ct, "searching", max(r_eff, 1.0), med_n)[0])
            p_ind_at_r = 1.0 - (1.0 - p_resp_at_r) ** (1.0 / med_n) if p_resp_at_r < 1.0 else 1.0
            scen_rows.append(
                {
                    "Scenario": "CorrelationReduction",
                    "CallType": ct,
                    "Parameter": f"drop={int(delta*100)}%",
                    "EffectiveRadius_m": r_eff,
                    "RespProbAtRadius": p_resp_at_r,
                    "ExpectedResponders_rho0.5": med_n * 0.5 * p_ind_at_r,
                }
            )
    # Scenario C: ABSOLUTE reduction of the correlation (+/- 0.05, 0.10, 0.15, 0.20)
    # The problem statement is ambiguous ("5%-20% reduction"); both readings are reported.
    for ct in CALL_TYPES:
        for delta in [0.0, 0.05, 0.10, 0.15, 0.20]:
            r_eff = solve_radius(
                lambda d: float(logistic_4pl(np.array([d]), *q2_params[ct])[0]) - delta - 0.50
            )
            p_resp_at_r = float(predict_gee(gee_res, ct, "searching", max(r_eff, 1.0), med_n)[0])
            p_ind_at_r = 1.0 - (1.0 - p_resp_at_r) ** (1.0 / med_n) if p_resp_at_r < 1.0 else 1.0
            scen_rows.append(
                {
                    "Scenario": "CorrelationReductionAbsolute",
                    "CallType": ct,
                    "Parameter": f"drop={delta:.2f}",
                    "EffectiveRadius_m": r_eff,
                    "RespProbAtRadius": p_resp_at_r,
                    "ExpectedResponders_rho0.5": med_n * 0.5 * p_ind_at_r,
                }
            )

    scen_df = pd.DataFrame(scen_rows)
    scen_df.to_csv(TABLES_DIR / "q4_noise_scenarios.csv", index=False)

    # 4. Visualizations
    palette = {"flat": "#1f77b4", "mod": "#ff7f0e", "tv": "#2ca02c"}

    # Plot A: Effectiveness Index E(d)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    sns.lineplot(data=eff_df, x="Distance_m", y="Effectiveness_E", hue="CallType", palette=palette, lw=2.2, ax=ax)
    ax.axhline(0.30, ls="--", color="gray", alpha=0.7, label="Reference E = 0.30")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Acoustic Communication Effectiveness E(d)")
    ax.set_title("Integrated Communication Effectiveness Index at Sea (Npenguins = 8)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "q4_effectiveness_curves.png", dpi=300)
    plt.close(fig)

    # Plot B: Noise scenarios
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    s_a = scen_df[scen_df["Scenario"] == "ThresholdElevation"]
    sns.barplot(data=s_a, x="Parameter", y="EffectiveRadius_m", hue="CallType", palette=palette, ax=axes[0])
    axes[0].set_title("Scenario A: Recognition Threshold Elevation")
    axes[0].set_ylabel("Effective Communication Radius (m)")

    s_b = scen_df[scen_df["Scenario"] == "CorrelationReduction"]
    sns.barplot(data=s_b, x="Parameter", y="EffectiveRadius_m", hue="CallType", palette=palette, ax=axes[1])
    axes[1].set_title("Scenario B: Overall Signal Correlation Reduction (tau = 0.50)")
    axes[1].set_ylabel("Effective Communication Radius (m)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "q4_noise_scenarios.png", dpi=300)
    plt.close(fig)

    print("=== Q4 Noise Scenarios Summary ===")
    print(scen_df.to_string(index=False))


if __name__ == "__main__":
    main()
