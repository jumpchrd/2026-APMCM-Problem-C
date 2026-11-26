"""Problem 3: Behavioral responses, 8-state combinations, and censored latency modeling."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import optimize, stats

from config import FIGURES_DIR, RANDOM_SEED, TABLES_DIR, resolve_data_dir

COMBO_LABELS = {
    "000": "None (000)",
    "100": "HeadUp (100)",
    "010": "Vocal (010)",
    "001": "Approach (001)",
    "110": "HeadUp+Vocal (110)",
    "101": "HeadUp+Approach (101)",
    "011": "Vocal+Approach (011)",
    "111": "AllThree (111)",
}

TIER_MAP = {
    "000": 0,  # None
    "100": 1,  # HeadUpOnly
    "010": 2,  # VocalOnly
    "110": 3,  # HeadUp+Vocal
    "001": 4,  # ApproachInvolved
    "101": 4,
    "011": 4,
    "111": 4,
}
TIER_NAMES = {
    0: "None",
    1: "HeadUpOnly",
    2: "VocalOnly",
    3: "HeadUp+Vocal",
    4: "ApproachInvolved",
}


def prepare_playback_df(data_dir) -> pd.DataFrame:
    """Load Data02 and derive 8-state combo, 5-tier state, and initiation latency."""
    df = pd.read_csv(data_dir / "Data02_AfricanPenguins_playbacks_data.csv")
    df["ComboCode"] = (
        df["HeadsUpYesNo"].astype(int).astype(str)
        + df["VocalYesNo"].astype(int).astype(str)
        + df["ApproachYesNo"].astype(int).astype(str)
    )
    df["ComboLabel"] = df["ComboCode"].map(COMBO_LABELS)
    df["ResponseTier"] = df["ComboCode"].map(TIER_MAP)
    df["LogNpenguins"] = np.log(df["Npenguins"].clip(lower=1))
    df["IsCall"] = (df["Type"] == "call").astype(int)
    df["InitLatency"] = df[["HeadsUpLatency", "VocalLatency", "ApproachLatency"]].min(axis=1)
    return df


def analyze_overall_responses(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare overall response rates (call vs control) and fit GEE logistic model."""
    ct = pd.crosstab(df["Type"], df["ResponseYesNo"])
    chi2, p_chi, _, _ = stats.chi2_contingency(ct)
    _, p_fisher = stats.fisher_exact(ct.values)

    summary_rows = []
    for grp_col in ["Type", "Context", "CallType"]:
        for val, sub in df.groupby(grp_col):
            summary_rows.append(
                {
                    "GroupBy": grp_col,
                    "Level": val,
                    "N": len(sub),
                    "Responded": int(sub["ResponseYesNo"].sum()),
                    "ResponseRate": sub["ResponseYesNo"].mean(),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_df["Call_vs_Control_Chi2_p"] = p_chi
    summary_df["Call_vs_Control_Fisher_p"] = p_fisher

    # GEE Logistic Regression clustered by expID (calls vs control unified)
    gee_df = df.dropna(subset=["Context", "CallType", "Distance", "LogNpenguins"]).copy()
    gee = smf.gee(
        "ResponseYesNo ~ C(CallType, Treatment('control')) + C(Context) + Distance + LogNpenguins",
        groups="expID",
        data=gee_df,
        family=sm.families.Binomial(),
    ).fit()

    coef_df = pd.DataFrame(
        {
            "Term": gee.params.index,
            "Estimate": gee.params.values,
            "StdErr": gee.bse.values,
            "z_value": gee.tvalues.values,
            "p_value": gee.pvalues.values,
            "OddsRatio": np.exp(gee.params.values),
        }
    )
    return summary_df, coef_df


def analyze_combinations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze 8-state response distributions and fit Multinomial Logit on 5 tiers."""
    # Empirical 8-state distribution
    dist_rows = []
    for sub_name, sub in [("control", df[df["Type"] == "control"]), ("call_all", df[df["Type"] == "call"])] + [
        (f"call_{ct}", df[(df["Type"] == "call") & (df["CallType"] == ct)]) for ct in ["flat", "mod", "tv"]
    ]:
        counts = sub["ComboCode"].value_counts()
        n = len(sub)
        for code, label in COMBO_LABELS.items():
            c = int(counts.get(code, 0))
            dist_rows.append(
                {"Group": sub_name, "ComboCode": code, "ComboLabel": label, "Count": c, "Prob": c / n if n else 0.0}
            )
    dist_df = pd.DataFrame(dist_rows)

    # Chi-square test on 8-state distribution: call vs control
    ct_8 = pd.crosstab(df["Type"], df["ComboCode"])
    chi2_8, p_8, _, _ = stats.chi2_contingency(ct_8)
    dist_df["Call_vs_Control_8State_Chi2"] = chi2_8
    dist_df["Call_vs_Control_8State_p"] = p_8

    # Unified Multinomial Logit on 5 tiers (with IsCall indicator + CallType dummies among calls)
    mn_df = df.copy()
    mn_df["IsFlat"] = (mn_df["CallType"] == "flat").astype(float)
    mn_df["IsMod"] = (mn_df["CallType"] == "mod").astype(float)
    mn_df["IsTv"] = (mn_df["CallType"] == "tv").astype(float)
    mn_df["CtxSearching"] = (mn_df["Context"] == "searching").astype(float)
    mn_df["CtxCommuting"] = (mn_df["Context"] == "commuting").astype(float)
    mn_df["Dist100m"] = mn_df["Distance"] / 100.0

    X = sm.add_constant(
        mn_df[["IsFlat", "IsMod", "IsTv", "CtxSearching", "CtxCommuting", "Dist100m", "LogNpenguins"]]
    )
    y = mn_df["ResponseTier"]
    mn_res = sm.MNLogit(y, X).fit(method="bfgs", maxiter=500, disp=False)

    mn_rows = []
    params = mn_res.params
    pvals = mn_res.pvalues
    for col in params.columns:
        tier_id = int(col) + 1  # MNLogit omits base category 0
        for term in params.index:
            mn_rows.append(
                {
                    "Tier": TIER_NAMES[tier_id],
                    "Term": term,
                    "Coef": params.loc[term, col],
                    "p_value": pvals.loc[term, col],
                    "RelativeRiskRatio": np.exp(params.loc[term, col]),
                }
            )
    return dist_df, pd.DataFrame(mn_rows)


def fit_censored_lognormal_aft(
    df_calls: pd.DataFrame, latency_col: str, event_col: str, censor_time: float = 60.0
) -> dict:
    """Fit right-censored Log-Normal AFT model via MLE on call playback trials."""
    sub = df_calls.copy()
    y_time = np.where(sub[event_col] == 1, np.maximum(sub[latency_col].fillna(0.5), 0.25), censor_time)
    event = sub[event_col].values.astype(float)

    X = np.column_stack(
        [
            np.ones(len(sub)),
            (sub["CallType"] == "mod").astype(float),
            (sub["CallType"] == "tv").astype(float),
            (sub["Context"] == "searching").astype(float),
            (sub["Context"] == "commuting").astype(float),
            sub["Distance"].values / 100.0,
            sub["LogNpenguins"].values,
        ]
    )
    log_t = np.log(y_time)

    def neg_loglik(theta):
        beta = theta[:-1]
        sigma = np.exp(theta[-1])
        mu = X @ beta
        z = (log_t - mu) / sigma
        ll_obs = -np.log(sigma) + stats.norm.logpdf(z)
        ll_cens = stats.norm.logsf(z)
        return -np.sum(event * ll_obs + (1.0 - event) * ll_cens)

    theta0 = np.zeros(X.shape[1] + 1)
    theta0[0] = np.log(10.0)
    theta0[-1] = np.log(1.0)
    res = optimize.minimize(neg_loglik, theta0, method="BFGS")
    return {"beta": res.x[:-1], "sigma": float(np.exp(res.x[-1])), "success": res.success}


def predict_representative_latencies(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit censored AFT models and predict at Npenguins=median (8) and Distances 50, 100, 150m."""
    calls = df[df["Type"] == "call"].copy()
    med_n = float(df["Npenguins"].median())

    # Response probability model among calls for representative conditions
    prob_mod = smf.glm(
        "ResponseYesNo ~ C(CallType, Treatment('flat')) + C(Context) + Distance + LogNpenguins",
        data=calls,
        family=sm.families.Binomial(),
    ).fit()

    aft_init = fit_censored_lognormal_aft(calls, "InitLatency", "ResponseYesNo")
    aft_head = fit_censored_lognormal_aft(calls, "HeadsUpLatency", "HeadsUpYesNo")
    aft_vocal = fit_censored_lognormal_aft(calls, "VocalLatency", "VocalYesNo")
    aft_app = fit_censored_lognormal_aft(calls, "ApproachLatency", "ApproachYesNo")

    # Intensity summary among responders
    int_rows = []
    for ct, grp in calls.groupby("CallType"):
        for col in ["VocalNcalls", "HeadsUpDuration", "ApproachDuration"]:
            pos = grp.loc[grp[col] > 0, col]
            int_rows.append(
                {
                    "CallType": ct,
                    "IntensityMeasure": col,
                    "N_active": len(pos),
                    "Mean_active": pos.mean() if len(pos) else 0.0,
                    "Median_active": pos.median() if len(pos) else 0.0,
                }
            )
    intensity_df = pd.DataFrame(int_rows)

    # Representative predictions at Npenguins = median, averaging/fixing Context = searching (foraging)
    pred_rows = []
    for ct in ["flat", "mod", "tv"]:
        for dist in [50.0, 100.0, 150.0]:
            x_prob = pd.DataFrame(
                [
                    {
                        "CallType": ct,
                        "Context": "searching",
                        "Distance": dist,
                        "LogNpenguins": np.log(med_n),
                    }
                ]
            )
            p_resp = float(prob_mod.predict(x_prob).iloc[0])

            x_aft = np.array(
                [
                    1.0,
                    1.0 if ct == "mod" else 0.0,
                    1.0 if ct == "tv" else 0.0,
                    1.0,  # searching
                    0.0,  # commuting
                    dist / 100.0,
                    np.log(med_n),
                ]
            )

            # Median latency conditional on perception (exp(mu))
            def cond_median(aft_fit):
                return float(np.exp(x_aft @ aft_fit["beta"]))

            # Check observed cell availability near this distance (+-25m)
            obs_near = calls[(calls["CallType"] == ct) & ((calls["Distance"] - dist).abs() <= 25)]
            obs_resp_rate = obs_near["ResponseYesNo"].mean() if len(obs_near) else np.nan
            status = "No response" if p_resp < 0.50 else "Response expected"

            pred_rows.append(
                {
                    "CallType": ct,
                    "Context": "searching",
                    "Npenguins_median": med_n,
                    "Distance_m": dist,
                    "Pred_ResponseProb": p_resp,
                    "Status": status,
                    "AFT_InitLatency_s": np.nan if status == "No response" else cond_median(aft_init),
                    "AFT_HeadsUpLatency_s": np.nan if status == "No response" else cond_median(aft_head),
                    "AFT_VocalLatency_s": np.nan if status == "No response" else cond_median(aft_vocal),
                    "AFT_ApproachLatency_s": np.nan if status == "No response" else cond_median(aft_app),
                    "Obs_Near_N": len(obs_near),
                    "Obs_Near_RespRate": obs_resp_rate,
                }
            )
    return pd.DataFrame(pred_rows), intensity_df


def plot_q3_figures(df: pd.DataFrame, dist_df: pd.DataFrame, pred_df: pd.DataFrame) -> None:
    """Save Q3 behavioral combination distribution and predicted response/latency plots."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)

    # 1. 8-state combination probability distribution
    sub_dist = dist_df[dist_df["Group"].isin(["control", "call_flat", "call_mod", "call_tv"])].copy()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    sns.barplot(data=sub_dist, x="ComboLabel", y="Prob", hue="Group", ax=ax)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Response Behavior Combination (HeadsUp, Vocal, Approach)")
    ax.set_title("Probability Distribution of 8 Behavioral Response Combinations")
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "q3_response_combinations.png", dpi=300)
    plt.close(fig)

    # 2. Response rate by Distance & CallType + Predicted Init Latency
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.lineplot(
        data=pred_df, x="Distance_m", y="Pred_ResponseProb", hue="CallType", marker="o", ax=axes[0]
    )
    axes[0].axhline(0.5, ls="--", color="gray", alpha=0.7, label="50% Threshold")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("Predicted Response Probability (Npenguins = 8, Searching)")
    axes[0].set_ylabel("Response Probability")

    plot_lat = pred_df.copy()
    sns.barplot(data=plot_lat, x="Distance_m", y="AFT_InitLatency_s", hue="CallType", ax=axes[1])
    axes[1].set_title("Predicted Initiation Latency (Censored AFT; Blank = No Response)")
    axes[1].set_ylabel("Predicted Initiation Latency (s)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "q3_latency_predictions.png", dpi=300)
    plt.close(fig)


def main() -> None:
    data_dir = resolve_data_dir()
    df = prepare_playback_df(data_dir)

    summary_df, gee_df = analyze_overall_responses(df)
    dist_df, mn_df = analyze_combinations(df)
    pred_df, intensity_df = predict_representative_latencies(df)

    summary_df.to_csv(TABLES_DIR / "q3_overall_response_rates.csv", index=False)
    gee_df.to_csv(TABLES_DIR / "q3_gee_logistic_model.csv", index=False)
    dist_df.to_csv(TABLES_DIR / "q3_8state_distribution.csv", index=False)
    mn_df.to_csv(TABLES_DIR / "q3_multinomial_logit.csv", index=False)
    pred_df.to_csv(TABLES_DIR / "q3_representative_latency_table.csv", index=False)
    intensity_df.to_csv(TABLES_DIR / "q3_response_intensity.csv", index=False)

    plot_q3_figures(df, dist_df, pred_df)

    print("=== Q3 GEE Logistic Model (CallType vs Control) ===")
    print(gee_df[["Term", "Estimate", "p_value", "OddsRatio"]].to_string(index=False))
    print("\n=== Q3 Representative Predictions (Npenguins=8, Searching) ===")
    print(
        pred_df[
            [
                "CallType",
                "Distance_m",
                "Pred_ResponseProb",
                "Status",
                "AFT_InitLatency_s",
                "AFT_VocalLatency_s",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
