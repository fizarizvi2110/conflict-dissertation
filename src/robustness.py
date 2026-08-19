"""
robustness.py
=============
Robustness checks for the main specifications.

  1. Robust vs clustered SEs (with only 18 clusters, clustered SEs are
     unreliable per Cameron, Gelbach & Miller 2008 -> report robust SEs).
  2. Extend the distributed lag to 12 months (H1 frequency, H2 success rate).
  3. Subsample checks (12-lag), run for BOTH hypotheses:
        - drop Nineveh          (is the effect driven by one province?)
        - drop 2016 from lags   (restrict to 2018+ so no lag reaches into 2016)
        - ISIS-attributed only  (does the effect hold for ISIS alone?)

Input: data/panel_iraq_2016_2020.csv  (produced by build_panel.py)
NOTE:  the H2 ISIS-only check needs the `n_isis_success` column -- make sure
       build_panel.py creates it and the panel is rebuilt before running.

Run:
    python src/robustness.py
"""

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

PANEL_PATH = "data/panel_iraq_2016_2020.csv"
LAG_VARS_6 = [f"log_strikes_lag{k}" for k in range(1, 7)]
LAG_VARS_12 = [f"log_strikes_lag{k}" for k in range(1, 13)]


def load_panel(path=PANEL_PATH):
    return pd.read_csv(path, parse_dates=["year_month"])


def add_lags_7_12(panel):
    """The saved panel carries lags 1-6; extend to 12 for the robustness models."""
    panel = panel.sort_values(["province", "year_month"]).copy()
    for lag in range(7, 13):
        panel[f"log_strikes_lag{lag}"] = (
            panel.groupby("province")["log_strikes"].shift(lag)
        )
    return panel


def cumulative(result, lags):
    return sum(result.params[f"log_strikes_lag{k}"] for k in lags)


# ---------------------------------------------------------------------------
# 1. Robust vs clustered SEs (6-lag H1)
# ---------------------------------------------------------------------------
def compare_ses(panel):
    est = panel.dropna(subset=["log_strikes_lag6"]).set_index(["province", "year_month"])
    formula = (
        "n_attacks ~ " + " + ".join(LAG_VARS_6)
        + " + log_turkish_strikes + EntityEffects + TimeEffects"
    )
    model = PanelOLS.from_formula(formula, data=est)
    clustered = model.fit(cov_type="clustered", cluster_entity=True)
    robust = model.fit(cov_type="robust")

    print("[H1] SE comparison (clustered vs robust):")
    for k in range(1, 7):
        var = f"log_strikes_lag{k}"
        sc, sr = clustered.std_errors[var], robust.std_errors[var]
        print(f"  {var}: clustered={sc:.4f}  robust={sr:.4f}  ratio={sc / sr:.2f}")

    wald = robust.wald_test(formula=" + ".join(LAG_VARS_6) + " = 0")
    print(f"\n[H1 robust] cumulative: {cumulative(robust, range(1, 7)):.4f}  "
          f"Wald p: {wald.pval:.6f}")


# ---------------------------------------------------------------------------
# 2. Main 12-lag models
# ---------------------------------------------------------------------------
def h1_12lag(panel):
    est = panel.dropna(subset=["log_strikes_lag12"]).set_index(["province", "year_month"])
    formula = (
        "n_attacks ~ " + " + ".join(LAG_VARS_12)
        + " + log_turkish_strikes + EntityEffects + TimeEffects"
    )
    result = PanelOLS.from_formula(formula, data=est).fit(cov_type="robust")
    wald = result.wald_test(formula=" + ".join(LAG_VARS_12) + " = 0")
    print(f"\n[H1 12-lag] N={len(est)}")
    print(f"  cumulative 1-12: {cumulative(result, range(1, 13)):.4f}  "
          f"Wald p: {wald.pval:.6f}")
    print(f"  cumulative 1-6 : {cumulative(result, range(1, 7)):.4f}")
    print(f"  cumulative 7-12: {cumulative(result, range(7, 13)):.4f}")
    return result


def h2_12lag(panel):
    est = panel.dropna(subset=["log_strikes_lag12"]).copy()
    est = est[est["n_attacks"] > 0]
    est["pct_success"] = est["n_success"] / est["n_attacks"]
    est = est.set_index(["province", "year_month"])
    result = PanelOLS.from_formula(
        "pct_success ~ " + " + ".join(LAG_VARS_12)
        + " + log_turkish_strikes + EntityEffects + TimeEffects",
        data=est,
    ).fit(cov_type="robust")
    wald = result.wald_test(formula=" + ".join(LAG_VARS_12) + " = 0")
    print(f"\n[H2 12-lag] N={len(est)}")
    print(f"  cumulative 1-12: {cumulative(result, range(1, 13)):.4f}  "
          f"Wald p: {wald.pval:.6f}")
    print(f"  cumulative 1-6 : {cumulative(result, range(1, 7)):.4f}")
    print(f"  cumulative 7-12: {cumulative(result, range(7, 13)):.4f}")
    return result


# ---------------------------------------------------------------------------
# 3. Parametrized subsample checks (12-lag) -- shared by H1 and H2
# ---------------------------------------------------------------------------
def _prep_h1(panel, outcome="n_attacks"):
    """H1-style: count outcome on the full panel (zeros kept)."""
    est = panel.dropna(subset=["log_strikes_lag12"]).copy()
    return est, outcome


def _prep_h2(panel, num="n_success", den="n_attacks", rate="pct_success"):
    """H2-style: a rate outcome, defined only where the denominator is > 0."""
    est = panel.dropna(subset=["log_strikes_lag12"]).copy()
    est = est[est[den] > 0].copy()
    est[rate] = est[num] / est[den]
    return est, rate


def run_spec(panel, label, outcome, subset=None):
    """Fit the 12-lag TWFE model for a given outcome on an optional subset."""
    est = panel if subset is None else subset(panel)
    est = est.set_index(["province", "year_month"])
    result = PanelOLS.from_formula(
        f"{outcome} ~ " + " + ".join(LAG_VARS_12)
        + " + log_turkish_strikes + EntityEffects + TimeEffects",
        data=est,
    ).fit(cov_type="robust")
    wald = result.wald_test(formula=" + ".join(LAG_VARS_12) + " = 0")
    print(f"[{label}] N={len(est)}  "
          f"cumulative: {cumulative(result, range(1, 13)):.4f}  Wald p: {wald.pval:.6f}")
    return result


def robustness_suite(panel):
    # Subset filters (province / time), applied before outcome prep
    drop_nin = lambda df: df[df["province"] != "Nineveh"]
    drop_16 = lambda df: df[df["year_month"] >= "2018-01-01"]

    print("\n-- H1 (attack frequency) --")
    est_h1, _ = _prep_h1(panel)
    run_spec(est_h1, "Drop Nineveh - H1", "n_attacks", subset=drop_nin)
    run_spec(est_h1, "Drop 2016 - H1", "n_attacks", subset=drop_16)
    run_spec(est_h1, "ISIS-only - H1", "n_isis_attacks")

    print("\n-- H2 (attack success rate) --")
    est_h2, rate = _prep_h2(panel)
    run_spec(est_h2, "Drop Nineveh - H2", rate, subset=drop_nin)
    run_spec(est_h2, "Drop 2016 - H2", rate, subset=drop_16)
    # ISIS success rate: needs n_isis_success (added in build_panel.py)
    est_h2i, rate_i = _prep_h2(
        panel, num="n_isis_success", den="n_isis_attacks", rate="pct_isis_success"
    )
    run_spec(est_h2i, "ISIS-only - H2", rate_i)


def main():
    panel = add_lags_7_12(load_panel())
    compare_ses(panel)
    h1_12lag(panel)
    h2_12lag(panel)
    print("\n" + "=" * 70)
    print("SUBSAMPLE CHECKS (12-lag, H1 + H2)")
    print("=" * 70)
    robustness_suite(panel)


if __name__ == "__main__":
    main()
