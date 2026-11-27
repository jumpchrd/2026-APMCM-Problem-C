"""Problem 3: Behavioral responses, 8-state combinations, and censored latency modeling.

Revision notes (v2)
-------------------
* Latency models are now fitted on BOTH natural calls and control playbacks, so the
  "common modeling framework" requirement of the problem is met.
* Distance enters as log2(Distance): coefficients are interpreted per doubling of
  distance, and the extrapolation is far better behaved than a raw linear term.
* The MAIN latency model is now a hurdle: part 1 is the GEE response probability,
  part 2 is a log-normal model fitted to responders only.  This matches the
  problem's instruction that "no response" is not a long latency.  Predicted
  medians are bounded by the 60 s observation window.
* A censored log-normal AFT (non-response = censoring at 60 s) is kept as a
  sensitivity analysis.  It produces medians beyond the window and is reported
  with explicit "not estimable (>60 s)" flags rather than absurd numbers.
* Response probabilities and latencies now come from ONE consistent framework,
  with 95% CIs.
* Added: McNemar test on matched call/control pairs, calibration-by-distance table,
  permutation test for the 8-state distribution, and a vocal-event consistency fix.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import optimize, stats
from statsmodels.stats.contingency_tables import mcnemar

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

OBS_WINDOW_S = 60.0
CALL_TYPES = ["flat", "mod", "tv"]
ALL_PLAYBACK = CALL_TYPES + ["control"]
CONTEXTS = ["searching", "commuting", "after"]


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

    # Vocal consistency fix: 4 rows have VocalYesNo=0 but VocalNcalls>0.
    # Define the vocal event as "yes OR at least one call recorded".
    df["VocalEvent"] = ((df["VocalYesNo"] == 1) | (df["VocalNcalls"] > 0)).astype(int)

    df["LogNpenguins"] = np.log(df["Npenguins"].clip(lower=1))
    df["log2Distance"] = np.log2(df["Distance"].clip(lower=1))
    df["IsCall"] = (df["Type"] == "call").astype(int)
    df["InitLatency"] = df[["HeadsUpLatency", "VocalLatency", "ApproachLatency"]].min(axis=1)
    return df


# --------------------------------------------------------------------------- #
# 1. Overall response rates, paired tests, and the main GEE model
# --------------------------------------------------------------------------- #
def append_gee_design(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log2Distance"] = np.log2(out["Distance"].clip(lower=1))
    out["LogNpenguins"] = np.log(out["Npenguins"].clip(lower=1))
    return out


def fit_main_gee(df: pd.DataFrame):
    """The single response-probability model shared by Problem 3 and Problem 4."""
    gee_df = df.dropna(subset=["Context", "CallType", "Distance", "LogNpenguins"]).copy()
    gee = smf.gee(
        "ResponseYesNo ~ C(CallType, Treatment('control')) + C(Context) + log2Distance + LogNpenguins",
        groups="expID",
        data=gee_df,
        family=sm.families.Binomial(),
    ).fit()
    return gee, gee_df


def analyze_overall_responses(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare overall response rates (call vs control), paired McNemar, and GEE."""
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

    # --- McNemar test on the 139 matched call/control pairs ---
    wide = (
        df.pivot_table(index="expID", columns="Type", values="ResponseYesNo", aggfunc="first")
        .dropna(subset=["call", "control"])
    )
    table = pd.crosstab(wide["call"], wide["control"]).reindex(index=[0, 1], columns=[0, 1]).fillna(0)
    mcn = mcnemar(table.values.astype(int), exact=True)
    pair_rows = [
        {
            "Analysis": "AllPairs",
            "N_pairs": int(table.values.sum()),
            "BothRespond": int(table.loc[1, 1]),
            "CallOnly": int(table.loc[1, 0]),
            "ControlOnly": int(table.loc[0, 1]),
            "Neither": int(table.loc[0, 0]),
            "McNemar_p": mcn.pvalue,
        }
    ]

    # --- McNemar restricted to pairs played at exactly the same distance ---
    dist_wide = df.pivot_table(
        index="expID", columns="Type", values="Distance", aggfunc="first"
    ).dropna(subset=["call", "control"])
    matched_ids = dist_wide.index[dist_wide["call"] == dist_wide["control"]]
    matched = wide.loc[wide.index.isin(matched_ids)]
    t2 = pd.crosstab(matched["call"], matched["control"]).reindex(index=[0, 1], columns=[0, 1]).fillna(0)
    mcn2 = mcnemar(t2.values.astype(int), exact=True)
    pair_rows.append(
        {
            "Analysis": "MatchedDistance",
            "N_pairs": int(t2.values.sum()),
            "BothRespond": int(t2.loc[1, 1]),
            "CallOnly": int(t2.loc[1, 0]),
            "ControlOnly": int(t2.loc[0, 1]),
            "Neither": int(t2.loc[0, 0]),
            "McNemar_p": mcn2.pvalue,
        }
    )
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(TABLES_DIR / "q3_paired_mcnemar.csv", index=False)

    # --- Main response model: GEE logistic, clustered by experiment ---
    gee, gee_df = fit_main_gee(df)

    coef_df = pd.DataFrame(
        {
            "Term": gee.params.index,
            "Estimate": gee.params.values,
            "StdErr": gee.bse.values,
            "z_value": gee.tvalues.values,
            "p_value": gee.pvalues.values,
            "OddsRatio": np.exp(gee.params.values),
            "OR_CI_low": np.exp(gee.params.values - 1.96 * gee.bse.values),
            "OR_CI_high": np.exp(gee.params.values + 1.96 * gee.bse.values),
        }
    )
    return summary_df, coef_df, pair_df, gee, gee_df


def design_row(gee_res, call_type: str, context: str, distance: float, n_penguins: float) -> pd.Series:
    """Build one design row keyed by the fitted model's parameter names."""
    vals = {}
    for name in gee_res.params.index:
        if name == "Intercept":
            v = 1.0
        elif "CallType" in name and "[T.flat]" in name:
            v = float(call_type == "flat")
        elif "CallType" in name and "[T.mod]" in name:
            v = float(call_type == "mod")
        elif "CallType" in name and "[T.tv]" in name:
            v = float(call_type == "tv")
        elif "Context" in name and "[T.commuting]" in name:
            v = float(context == "commuting")
        elif "Context" in name and "[T.searching]" in name:
            v = float(context == "searching")
        elif name == "log2Distance":
            v = float(np.log2(max(distance, 1.0)))
        elif name == "LogNpenguins":
            v = float(np.log(max(n_penguins, 1.0)))
        else:
            v = 0.0
        vals[name] = v
    return pd.Series(vals).reindex(gee_res.params.index).fillna(0.0)


def predict_gee(gee_res, call_type, context, distance, n_penguins) -> tuple[float, float, float]:
    """Return (probability, lower, upper) from the GEE model via the delta method."""
    x = design_row(gee_res, call_type, context, distance, n_penguins).values
    beta = gee_res.params.values
    cov = np.asarray(gee_res.cov_params())
    eta = float(x @ beta)
    se = float(np.sqrt(max(x @ cov @ x, 0.0)))
    p = 1.0 / (1.0 + np.exp(-eta))
    lo = 1.0 / (1.0 + np.exp(-(eta - 1.96 * se)))
    hi = 1.0 / (1.0 + np.exp(-(eta + 1.96 * se)))
    return p, lo, hi


# --------------------------------------------------------------------------- #
# 2. Behavioral combinations (8 states) and tier-level multinomial model
# --------------------------------------------------------------------------- #
def analyze_combinations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Empirical 8-state distributions, permutation test, and multinomial logit."""
    dist_rows = []
    groups = [("control", df[df["Type"] == "control"]), ("call_all", df[df["Type"] == "call"])]
    groups += [
        (f"call_{ct}", df[(df["Type"] == "call") & (df["CallType"] == ct)]) for ct in CALL_TYPES
    ]
    for sub_name, sub in groups:
        counts = sub["ComboCode"].value_counts()
        n = len(sub)
        for code, label in COMBO_LABELS.items():
            c = int(counts.get(code, 0))
            dist_rows.append(
                {"Group": sub_name, "ComboCode": code, "ComboLabel": label, "Count": c, "Prob": c / n if n else 0.0}
            )
    dist_df = pd.DataFrame(dist_rows)

    # Chi-square + permutation test (expected counts are too small for the asymptotic p)
    ct_8 = pd.crosstab(df["Type"], df["ComboCode"])
    chi2_8, p_asym, dof, expected = stats.chi2_contingency(ct_8)
    rng = np.random.default_rng(RANDOM_SEED)
    labels = df["Type"].values
    combo = df["ComboCode"].values
    perm_stats = []
    for _ in range(5000):
        perm = rng.permutation(labels)
        tab = pd.crosstab(perm, combo)
        try:
            s, _ = stats.chi2_contingency(tab, correction=False)[:2]
        except ValueError:
            continue
        perm_stats.append(s)
    perm_stats = np.array(perm_stats)
    p_perm = float((np.sum(perm_stats >= chi2_8) + 1) / (len(perm_stats) + 1))
    dist_df["Call_vs_Control_8State_Chi2"] = chi2_8
    dist_df["Call_vs_Control_8State_p_asymptotic"] = p_asym
    dist_df["Call_vs_Control_8State_p_permutation"] = p_perm
    dist_df["MinExpectedCount"] = float(expected.min())

    # Unified multinomial logit on 5 tiers (control = reference)
    mn_df = df.copy()
    mn_df["IsFlat"] = (mn_df["CallType"] == "flat").astype(float)
    mn_df["IsMod"] = (mn_df["CallType"] == "mod").astype(float)
    mn_df["IsTv"] = (mn_df["CallType"] == "tv").astype(float)
    mn_df["CtxSearching"] = (mn_df["Context"] == "searching").astype(float)
    mn_df["CtxCommuting"] = (mn_df["Context"] == "commuting").astype(float)

    X = sm.add_constant(
        mn_df[["IsFlat", "IsMod", "IsTv", "CtxSearching", "CtxCommuting", "log2Distance", "LogNpenguins"]]
    )
    y = mn_df["ResponseTier"]
    mn_res = sm.MNLogit(y, X).fit(method="bfgs", maxiter=500, disp=False)

    mn_rows = []
    params = mn_res.params
    pvals = mn_res.pvalues
    for col in params.columns:
        tier_id = int(col) + 1
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
    mn_df_out = pd.DataFrame(mn_rows)
    mn_df_out["PseudoR2"] = mn_res.prsquared
    mn_df_out["N"] = len(y)
    return dist_df, mn_df_out, mn_res


# --------------------------------------------------------------------------- #
# 3. Censored log-normal AFT (calls + control), with 60 s bound
# --------------------------------------------------------------------------- #
AFT_TERMS = ["Intercept", "IsFlat", "IsMod", "IsTv", "CtxSearching", "CtxCommuting", "log2Distance", "LogN"]


def aft_design_row(call_type: str, context: str, distance: float, n_penguins: float) -> np.ndarray:
    return np.array(
        [
            1.0,
            float(call_type == "flat"),
            float(call_type == "mod"),
            float(call_type == "tv"),
            float(context == "searching"),
            float(context == "commuting"),
            float(np.log2(max(distance, 1.0))),
            float(np.log(max(n_penguins, 1.0))),
        ]
    )


def fit_censored_lognormal_aft(
    df: pd.DataFrame, latency_col: str, event_col: str, cluster_col: str = "expID"
) -> dict:
    """Fit a right-censored log-normal AFT via MLE on all playback rows."""
    sub = df.copy()
    event = sub[event_col].astype(int).values
    lat = pd.to_numeric(sub[latency_col], errors="coerce").values

    # Event observed but latency missing -> treat as censored (defensive; none in current data)
    missing_latency = (event == 1) & np.isnan(lat)
    event = event.copy()
    event[missing_latency] = 0

    y_time = np.where(event == 1, np.maximum(lat, 0.25), OBS_WINDOW_S)
    y_time = np.where(np.isnan(y_time), OBS_WINDOW_S, y_time)

    X = np.column_stack(
        [
            np.ones(len(sub)),
            (sub["CallType"] == "flat").astype(float),
            (sub["CallType"] == "mod").astype(float),
            (sub["CallType"] == "tv").astype(float),
            (sub["Context"] == "searching").astype(float),
            (sub["Context"] == "commuting").astype(float),
            np.log2(sub["Distance"].clip(lower=1)),
            np.log(sub["Npenguins"].clip(lower=1)),
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
    theta0[0] = np.log(5.0)
    theta0[-1] = np.log(1.0)
    res = optimize.minimize(neg_loglik, theta0, method="BFGS", options={"maxiter": 2000})
    theta = res.x
    beta = theta[:-1]
    sigma = float(np.exp(theta[-1]))

    # Covariance from the inverse Hessian (fall back to a numerical Hessian)
    try:
        hess_inv = np.asarray(res.hess_inv)
        cov = hess_inv[:-1, :-1] if hess_inv.shape[0] == len(theta) else np.linalg.pinv(_num_hess(neg_loglik, theta))
    except Exception:
        cov = np.linalg.pinv(_num_hess(neg_loglik, theta))

    # Note: standard errors come from the Hessian; they are not cluster-robust.
    # With two rows per experiment this is a mild under-estimate and is flagged
    # as a limitation in the paper.
    return {
        "beta": beta,
        "sigma": sigma,
        "cov": cov,
        "success": bool(res.success),
        "n": len(sub),
        "n_events": int(event.sum()),
        "n_censored": int((1 - event).sum()),
        "n_missing_latency_treated_censored": int(missing_latency.sum()),
    }


def _num_hess(f, theta):
    """Numerical Hessian of a scalar function."""
    n = len(theta)
    h = 1e-4
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n); ej = np.zeros(n)
            ei[i] = h; ej[j] = h
            H[i, j] = (
                f(theta + ei + ej) - f(theta + ei - ej) - f(theta - ei + ej) + f(theta - ei - ej)
            ) / (4 * h * h)
    return H


def predict_aft_median(aft: dict, call_type: str, context: str, distance: float, n_penguins: float) -> tuple[float, float, float, str]:
    """Return (median, low, high, status) with a hard 60 s observation-window bound."""
    x = aft_design_row(call_type, context, distance, n_penguins)
    mu = float(x @ aft["beta"])
    se = float(np.sqrt(max(x @ aft["cov"] @ x, 0.0)))
    med = float(np.exp(mu))
    lo = float(np.exp(mu - 1.96 * se))
    hi = float(np.exp(mu + 1.96 * se))
    if med > OBS_WINDOW_S or not np.isfinite(med):
        return np.nan, np.nan, np.nan, "not estimable (>60 s)"
    return med, lo, hi, "estimated"


def fit_hurdle_lognormal(df: pd.DataFrame, latency_col: str, event_col: str) -> dict:
    """Part 2 of the hurdle model: log-normal latency AMONG responders only.

    Non-responders are handled by the GEE response-probability model (part 1), so
    they are excluded here rather than being treated as an unobserved long latency.
    """
    sub = df[(df[event_col] == 1) & df[latency_col].notna()].copy()
    X = np.column_stack(
        [
            np.ones(len(sub)),
            (sub["CallType"] == "flat").astype(float),
            (sub["CallType"] == "mod").astype(float),
            (sub["CallType"] == "tv").astype(float),
            (sub["Context"] == "searching").astype(float),
            (sub["Context"] == "commuting").astype(float),
            np.log2(sub["Distance"].clip(lower=1)),
            np.log(sub["Npenguins"].clip(lower=1)),
        ]
    )
    y = np.log(np.maximum(pd.to_numeric(sub[latency_col], errors="coerce").values, 0.25))
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = float(resid @ resid / dof)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    return {
        "beta": beta,
        "sigma": float(np.sqrt(sigma2)),
        "cov": cov,
        "n": len(sub),
    }


def predict_hurdle_median(hurdle: dict, call_type: str, context: str, distance: float, n_penguins: float):
    """Median latency conditional on responding, with a delta-method 95% CI."""
    x = aft_design_row(call_type, context, distance, n_penguins)
    mu = float(x @ hurdle["beta"])
    se = float(np.sqrt(max(x @ hurdle["cov"] @ x, 0.0)))
    med = float(np.exp(mu))
    lo = float(np.exp(mu - 1.96 * se))
    hi = float(np.exp(mu + 1.96 * se))
    if med > OBS_WINDOW_S or not np.isfinite(med):
        return np.nan, np.nan, np.nan, "not estimable (>60 s)"
    return med, lo, hi, "estimated"


def predict_representative_latencies(
    df: pd.DataFrame, gee_res, gee_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the hurdle-based representative table plus a censored-AFT sensitivity table."""
    med_n = float(df["Npenguins"].median())

    # Main approach: hurdle (response probability x latency among responders)
    hurdle_init = fit_hurdle_lognormal(df, "InitLatency", "ResponseYesNo")
    hurdle_head = fit_hurdle_lognormal(df, "HeadsUpLatency", "HeadsUpYesNo")
    hurdle_vocal = fit_hurdle_lognormal(df, "VocalLatency", "VocalEvent")
    hurdle_app = fit_hurdle_lognormal(df, "ApproachLatency", "ApproachYesNo")

    # Sensitivity approach: censored AFT treating non-response as censoring at 60 s
    aft_init = fit_censored_lognormal_aft(df, "InitLatency", "ResponseYesNo")
    aft_head = fit_censored_lognormal_aft(df, "HeadsUpLatency", "HeadsUpYesNo")
    aft_vocal = fit_censored_lognormal_aft(df, "VocalLatency", "VocalEvent")
    aft_app = fit_censored_lognormal_aft(df, "ApproachLatency", "ApproachYesNo")

    models = {
        "Init": aft_init,
        "HeadsUp": aft_head,
        "Vocal": aft_vocal,
        "Approach": aft_app,
    }
    hurdle_models = {
        "Init": hurdle_init,
        "HeadsUp": hurdle_head,
        "Vocal": hurdle_vocal,
        "Approach": hurdle_app,
    }

    # Censored-AFT parameters (sensitivity analysis)
    fit_rows = []
    for name, m in models.items():
        row = {
            "Behavior": name,
            "N": m["n"],
            "N_events": m["n_events"],
            "N_censored": m["n_censored"],
            "CensoringRate": m["n_censored"] / m["n"],
            "Sigma": m["sigma"],
            "Converged": m["success"],
        }
        for term, b in zip(AFT_TERMS, m["beta"]):
            row[f"beta_{term}"] = b
        fit_rows.append(row)
    pd.DataFrame(fit_rows).to_csv(TABLES_DIR / "q3_aft_parameters.csv", index=False)

    # Hurdle part-2 parameters (main approach: responders only)
    hurdle_rows = []
    for name, m in hurdle_models.items():
        row = {"Behavior": name, "N_responders": m["n"], "ResidualSigma": m["sigma"]}
        for term, b in zip(AFT_TERMS, m["beta"]):
            row[f"beta_{term}"] = b
        hurdle_rows.append(row)
    pd.DataFrame(hurdle_rows).to_csv(TABLES_DIR / "q3_hurdle_parameters.csv", index=False)

    # Intensity summary among responders (calls only, as in the original analysis)
    calls = df[df["Type"] == "call"]
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
    intensity_df.to_csv(TABLES_DIR / "q3_response_intensity.csv", index=False)

    # Observed cells near each representative distance (+-15 m) for calibration checks
    def observed_near(ct, context, dist):
        sub = df[
            (df["CallType"] == ct)
            & (df["Context"] == context)
            & ((df["Distance"] - dist).abs() <= 15)
        ]
        return len(sub), (sub["ResponseYesNo"].mean() if len(sub) else np.nan)

    pred_rows = []
    for ct in ALL_PLAYBACK:
        for dist in [50.0, 100.0, 150.0]:
            p, p_lo, p_hi = predict_gee(gee_res, ct, "searching", dist, med_n)
            row = {
                "CallType": ct,
                "Context": "searching",
                "Npenguins_median": med_n,
                "Distance_m": dist,
                "Pred_ResponseProb": p,
                "RespProb_CI_low": p_lo,
                "RespProb_CI_high": p_hi,
            }
            for key, m in hurdle_models.items():
                med, lo, hi, status = predict_hurdle_median(m, ct, "searching", dist, med_n)
                row[f"Median_{key}Latency_s"] = med
                row[f"{key}Latency_CI_low"] = lo
                row[f"{key}Latency_CI_high"] = hi
                row[f"{key}Latency_status"] = status
            n_obs, obs_rate = observed_near(ct, "searching", dist)
            row["Obs_Near_N"] = n_obs
            row["Obs_Near_RespRate"] = obs_rate
            pred_rows.append(row)
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(TABLES_DIR / "q3_representative_latency_table.csv", index=False)

    # Sensitivity: censored AFT (treats non-response as censoring at 60 s)
    sens_rows = []
    for ct in ALL_PLAYBACK:
        for dist in [50.0, 100.0, 150.0]:
            row = {"CallType": ct, "Distance_m": dist}
            for key, m in models.items():
                med, lo, hi, status = predict_aft_median(m, ct, "searching", dist, med_n)
                row[f"AFT_Median_{key}Latency_s"] = med
                row[f"AFT_{key}_CI_low"] = lo
                row[f"AFT_{key}_CI_high"] = hi
                row[f"AFT_{key}_status"] = status
            sens_rows.append(row)
    pd.DataFrame(sens_rows).to_csv(TABLES_DIR / "q3_aft_sensitivity.csv", index=False)

    # Supplementary table: every context
    ctx_rows = []
    for ct in ALL_PLAYBACK:
        for ctx in CONTEXTS:
            for dist in [50.0, 100.0, 150.0]:
                p, p_lo, p_hi = predict_gee(gee_res, ct, ctx, dist, med_n)
                med, lo, hi, status = predict_hurdle_median(hurdle_init, ct, ctx, dist, med_n)
                ctx_rows.append(
                    {
                        "CallType": ct,
                        "Context": ctx,
                        "Npenguins_median": med_n,
                        "Distance_m": dist,
                        "Pred_ResponseProb": p,
                        "RespProb_CI_low": p_lo,
                        "RespProb_CI_high": p_hi,
                        "Median_InitLatency_s": med,
                        "InitLatency_CI_low": lo,
                        "InitLatency_CI_high": hi,
                        "InitLatency_status": status,
                    }
                )
    ctx_df = pd.DataFrame(ctx_rows)
    ctx_df.to_csv(TABLES_DIR / "q3_representative_all_contexts.csv", index=False)

    return pred_df, intensity_df, ctx_df


# --------------------------------------------------------------------------- #
# 3b. Response-intensity models (counts and censored durations)
# --------------------------------------------------------------------------- #
FULL_TERMS = ["Intercept", "IsFlat", "IsMod", "IsTv", "CtxSearching", "CtxCommuting", "log2Distance", "LogN"]
REDUCED_TERMS = ["Intercept", "IsFlat", "IsMod", "IsTv"]


def _design_from_terms(terms, call_type, context, distance, n_penguins) -> np.ndarray:
    vals = {
        "Intercept": 1.0,
        "IsFlat": float(call_type == "flat"),
        "IsMod": float(call_type == "mod"),
        "IsTv": float(call_type == "tv"),
        "CtxSearching": float(context == "searching"),
        "CtxCommuting": float(context == "commuting"),
        "log2Distance": float(np.log2(max(distance, 1.0))),
        "LogN": float(np.log(max(n_penguins, 1.0))),
    }
    return np.array([vals[t] for t in terms], dtype=float)


def fit_event_gee(df: pd.DataFrame, event_col: str):
    """GEE logistic model for the probability that a specific behaviour occurs."""
    return smf.gee(
        f"{event_col} ~ C(CallType, Treatment('control')) + C(Context) + log2Distance + LogNpenguins",
        groups="expID",
        data=df,
        family=sm.families.Binomial(),
    ).fit()


def fit_nb_count(df: pd.DataFrame, col: str, terms: list[str]) -> dict:
    """Negative-binomial count model among trials with at least one call."""
    sub = df[df[col] > 0].copy()
    X = np.vstack([_design_from_terms(terms, r.CallType, r.Context, r.Distance, r.Npenguins)
                   for r in sub.itertuples()])
    y = sub[col].astype(float).values
    res = sm.NegativeBinomial(y, X).fit(disp=False, maxiter=500)
    return {
        "terms": terms,
        "params": np.asarray(res.params, dtype=float),
        "bse": np.asarray(res.bse, dtype=float),
        "pvalues": np.asarray(res.pvalues, dtype=float),
        "alpha": float(res.params[-1]) if len(res.params) > len(terms) else float(res.params[-1]),
        "n": len(sub),
        "success": bool(res.mle_retvals.get("converged", True)) if hasattr(res, "mle_retvals") else True,
    }


def fit_lognormal_censored_positive(df: pd.DataFrame, col: str, terms: list[str], cap: float = OBS_WINDOW_S) -> dict:
    """Log-normal model for duration conditional on the behaviour occurring.

    Durations equal to the 60 s observation cap are treated as right-censored.
    """
    sub = df[df[col] > 0].copy()
    x_rows = np.vstack([_design_from_terms(terms, r.CallType, r.Context, r.Distance, r.Npenguins)
                        for r in sub.itertuples()])
    event = (sub[col].values < cap).astype(float)
    y_time = np.where(event == 1, sub[col].values, cap)
    log_t = np.log(np.maximum(y_time, 0.25))

    def neg_loglik(theta):
        beta = theta[:-1]
        sigma = np.exp(theta[-1])
        mu = x_rows @ beta
        z = (log_t - mu) / sigma
        ll_obs = -np.log(sigma) + stats.norm.logpdf(z)
        ll_cens = stats.norm.logsf(z)
        return -np.sum(event * ll_obs + (1.0 - event) * ll_cens)

    theta0 = np.zeros(x_rows.shape[1] + 1)
    theta0[0] = np.log(np.median(sub[col]))
    theta0[-1] = np.log(1.0)
    res = optimize.minimize(neg_loglik, theta0, method="BFGS", options={"maxiter": 2000})
    if not res.success:
        res = optimize.minimize(neg_loglik, theta0, method="Nelder-Mead", options={"maxiter": 5000})

    beta = res.x[:-1]
    sigma = float(np.exp(res.x[-1]))
    hess_inv = getattr(res, "hess_inv", None)
    if isinstance(hess_inv, np.ndarray) and hess_inv.ndim == 2 and hess_inv.shape[0] >= len(beta):
        cov = hess_inv[: len(beta), : len(beta)]
    else:
        cov = np.linalg.pinv(_num_hess(neg_loglik, res.x))[: len(beta), : len(beta)]
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    pvals = 2.0 * (1.0 - stats.norm.cdf(np.abs(beta / np.where(se > 0, se, np.nan))))
    return {
        "terms": terms,
        "beta": beta,
        "bse": se,
        "pvalues": pvals,
        "sigma": sigma,
        "cov": cov,
        "n": len(sub),
        "n_censored": int((event == 0).sum()),
        "success": bool(res.success),
    }


def analyze_intensity(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two-part models for vocalisation count, head-up duration and approach duration."""
    specs = [
        ("VocalNcalls", "VocalEvent", "count", FULL_TERMS),
        ("HeadsUpDuration", "HeadsUpYesNo", "duration", FULL_TERMS),
        ("ApproachDuration", "ApproachYesNo", "duration", REDUCED_TERMS),
    ]
    med_n = float(df["Npenguins"].median())
    coef_rows, pred_rows = [], []

    for measure, event_col, kind, terms in specs:
        gee_ev = fit_event_gee(df, event_col)
        if kind == "count":
            fit = fit_nb_count(df, measure, terms)
            model_name = "NegativeBinomial(part2)"
            extra = {"alpha": fit["alpha"], "n_censored": np.nan, "sigma": np.nan}
        else:
            fit = fit_lognormal_censored_positive(df, measure, terms)
            model_name = "LognormalCensored(part2)"
            extra = {"alpha": np.nan, "n_censored": fit["n_censored"], "sigma": fit["sigma"]}

        # coefficient table
        if kind == "count":
            for term, b, se, p in zip(terms, fit["params"], fit["bse"], fit["pvalues"]):
                coef_rows.append({"Measure": measure, "Part2Model": model_name, "Term": term,
                                  "Estimate": b, "StdErr": se, "p_value": p,
                                  "N_positive": fit["n"], **extra})
        else:
            for term, b, se, pv in zip(terms, fit["beta"], fit["bse"], fit["pvalues"]):
                coef_rows.append({"Measure": measure, "Part2Model": model_name, "Term": term,
                                  "Estimate": b, "StdErr": se, "p_value": pv,
                                  "N_positive": fit["n"], **extra})

        # predictions at representative conditions
        for ct in CALL_TYPES:
            for dist in [50.0, 100.0, 150.0]:
                p_event, p_lo, p_hi = predict_gee(gee_ev, ct, "searching", dist, med_n)
                x = _design_from_terms(terms, ct, "searching", dist, med_n)
                if kind == "count":
                    cond_mean = float(np.exp(x @ fit["params"][: len(terms)]))
                    cond_median = np.nan
                else:
                    mu = float(x @ fit["beta"])
                    cond_median = float(np.exp(mu))
                    cond_mean = float(np.exp(mu + 0.5 * fit["sigma"] ** 2))
                central = cond_median if kind == "duration" else cond_mean
                pred_rows.append({
                    "Measure": measure,
                    "Part2Model": model_name,
                    "CallType": ct,
                    "Context": "searching",
                    "Distance_m": dist,
                    "Npenguins": med_n,
                    "P_behavior": p_event,
                    "P_CI_low": p_lo,
                    "P_CI_high": p_hi,
                    "Cond_Median": cond_median,
                    "Cond_Mean": cond_mean,
                    "Expected_Value": p_event * central,
                    "Expected_Basis": "P x median" if kind == "duration" else "P x mean",
                })
    return pd.DataFrame(coef_rows), pd.DataFrame(pred_rows)


# --------------------------------------------------------------------------- #
# 4. Calibration by distance (observed vs GEE-predicted)
# --------------------------------------------------------------------------- #
def calibration_by_distance(df: pd.DataFrame, gee_res) -> pd.DataFrame:
    bins = [(0, 50), (50, 100), (100, 150), (150, 201)]
    rows = []
    for lo, hi in bins:
        sub = df[(df["Distance"] > lo) & (df["Distance"] <= hi)]
        if not len(sub):
            continue
        preds = [
            predict_gee(gee_res, r.CallType, r.Context, r.Distance, r.Npenguins)[0]
            for r in sub.itertuples()
        ]
        rows.append(
            {
                "Distance_bin": f"({lo},{hi}]",
                "N": len(sub),
                "ObservedRate": sub["ResponseYesNo"].mean(),
                "PredictedRate": float(np.mean(preds)),
                "CalibrationGap": float(np.mean(preds) - sub["ResponseYesNo"].mean()),
            }
        )
    cal_df = pd.DataFrame(rows)
    cal_df.to_csv(TABLES_DIR / "q3_calibration_by_distance.csv", index=False)
    return cal_df


# --------------------------------------------------------------------------- #
# 5. Figures
# --------------------------------------------------------------------------- #
def plot_q3_figures(df: pd.DataFrame, dist_df: pd.DataFrame, pred_df: pd.DataFrame) -> None:
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

    # 2. Predicted response probability and initiation latency (bounded)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.lineplot(
        data=pred_df[pred_df["CallType"] != "control"],
        x="Distance_m",
        y="Pred_ResponseProb",
        hue="CallType",
        marker="o",
        ax=axes[0],
    )
    axes[0].axhline(0.5, ls="--", color="gray", alpha=0.7, label="50% reference")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("Predicted Response Probability (Npenguins = 8, Searching)")
    axes[0].set_ylabel("Response Probability")

    plot_lat = pred_df[pred_df["CallType"] != "control"].copy()
    sns.barplot(data=plot_lat, x="Distance_m", y="Median_InitLatency_s", hue="CallType", ax=axes[1])
    y_top = float(np.nanmax(plot_lat["Median_InitLatency_s"])) * 1.35
    axes[1].set_ylim(0, max(y_top, 5.0))
    axes[1].set_title("Median Initiation Latency Given a Response (Hurdle Model)")
    axes[1].set_ylabel("Median Initiation Latency (s)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "q3_latency_predictions.png", dpi=300)
    plt.close(fig)


def main() -> None:
    data_dir = resolve_data_dir()
    df = prepare_playback_df(data_dir)
    df = append_gee_design(df)

    summary_df, gee_df_out, pair_df, gee_res, gee_data = analyze_overall_responses(df)
    dist_df, mn_df, _ = analyze_combinations(df)
    pred_df, intensity_df, ctx_df = predict_representative_latencies(df, gee_res, gee_data)
    int_coef_df, int_pred_df = analyze_intensity(df)
    cal_df = calibration_by_distance(gee_data, gee_res)

    summary_df.to_csv(TABLES_DIR / "q3_overall_response_rates.csv", index=False)
    gee_df_out.to_csv(TABLES_DIR / "q3_gee_logistic_model.csv", index=False)
    dist_df.to_csv(TABLES_DIR / "q3_8state_distribution.csv", index=False)
    mn_df.to_csv(TABLES_DIR / "q3_multinomial_logit.csv", index=False)
    int_coef_df.to_csv(TABLES_DIR / "q3_intensity_models.csv", index=False)
    int_pred_df.to_csv(TABLES_DIR / "q3_intensity_predictions.csv", index=False)

    plot_q3_figures(df, dist_df, pred_df)

    print("=== Q3 Main GEE (control = reference) ===")
    print(gee_df_out[["Term", "Estimate", "p_value", "OddsRatio", "OR_CI_low", "OR_CI_high"]].to_string(index=False))
    print("\n=== Paired McNemar ===")
    print(pair_df.to_string(index=False))
    print("\n=== 8-state test ===")
    print(
        f"chi2={dist_df['Call_vs_Control_8State_Chi2'].iloc[0]:.2f}, "
        f"p(asymptotic)={dist_df['Call_vs_Control_8State_p_asymptotic'].iloc[0]:.3g}, "
        f"p(permutation)={dist_df['Call_vs_Control_8State_p_permutation'].iloc[0]:.3g}, "
        f"min expected={dist_df['MinExpectedCount'].iloc[0]:.2f}"
    )
    print("\n=== Calibration by distance ===")
    print(cal_df.to_string(index=False))
    print("\n=== Representative table (Npenguins = median, Searching) ===")
    cols = [
        "CallType",
        "Distance_m",
        "Pred_ResponseProb",
        "Median_InitLatency_s",
        "InitLatency_status",
        "Median_HeadsUpLatency_s",
        "HeadsUpLatency_status",
        "Obs_Near_N",
        "Obs_Near_RespRate",
    ]
    print(pred_df[cols].to_string(index=False))
    print("\n=== Response intensity at 100 m (Npenguins = median, Searching) ===")
    sub = int_pred_df[int_pred_df["Distance_m"] == 100.0]
    print(
        sub[["Measure", "CallType", "P_behavior", "Cond_Median", "Cond_Mean", "Expected_Value"]]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
