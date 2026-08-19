"""
models.py
=========
Main two-way fixed-effects distributed-lag specifications (lags 1-6).

  H1 (grievance / retaliation): effect of strikes on attack FREQUENCY (n_attacks)
  H2 (capacity degradation):    effect of strikes on attack SUCCESS RATE (pct_success)

Both models include province + month fixed effects, Turkish strikes as an
exogenous control, and a distributed lag of log strikes. Also runs a
decomposition across count and composition outcomes and saves the H1/H2 lag
profile figure to figures/.

Input : data/panel_iraq_2016_2020.csv  (produced by build_panel.py)
Output: figures/lag_profiles_h1_h2.png

Run:
    python src/models.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS

PANEL_PATH = "data/panel_iraq_2016_2020.csv"
LAG_VARS = [f"log_strikes_lag{k}" for k in range(1, 7)]
WALD_CUMULATIVE = " + ".join(LAG_VARS) + " = 0"


def load_panel(path=PANEL_PATH):
    panel = pd.read_csv(path, parse_dates=["year_month"])
    return panel


def cumulative(result):
    """Sum of the six lag coefficients."""
    return sum(result.params[f"log_strikes_lag{k}"] for k in range(1, 7))


# ---------------------------------------------------------------------------
# H1: attack frequency
# ---------------------------------------------------------------------------
def estimate_h1(panel):
    est = panel.dropna(subset=["log_strikes_lag6"]).copy()
    est = est.set_index(["province", "year_month"])

    formula = (
        "n_attacks ~ " + " + ".join(LAG_VARS)
        + " + log_turkish_strikes + EntityEffects + TimeEffects"
    )
    print(f"[H1] estimation sample: {len(est)} province-months")

    model = PanelOLS.from_formula(formula, data=est)
    result = model.fit(cov_type="clustered", cluster_entity=True)
    print(result.summary)

    wald = result.wald_test(formula=WALD_CUMULATIVE)
    print(f"\n[H1] cumulative effect (sum b1-b6): {cumulative(result):.4f}")
    print(f"[H1] Wald (cumulative = 0): {wald}")
    return result


# ---------------------------------------------------------------------------
# H2: attack success rate
# ---------------------------------------------------------------------------
def add_composition(df):
    """Composition shares, defined only where attacks occurred."""
    df = df.copy()
    df["pct_bombing"] = df["n_bombing"] / df["n_attacks"]
    df["pct_suicide"] = df["n_suicide"] / df["n_attacks"]
    df["pct_success"] = df["n_success"] / df["n_attacks"]
    df["pct_soft_target"] = (
        (df["n_targ_civilian"] + df["n_targ_business"]) / df["n_attacks"]
    )
    return df


def estimate_h2(panel):
    est3 = panel.dropna(subset=["log_strikes_lag6"]).copy()
    est3 = est3[est3["n_attacks"] > 0]
    est3 = add_composition(est3)
    est3 = est3.set_index(["province", "year_month"])

    formula = (
        "pct_success ~ " + " + ".join(LAG_VARS)
        + " + log_turkish_strikes + EntityEffects + TimeEffects"
    )
    print(f"\n[H2] estimation sample: {len(est3)} province-months (attacks > 0)")

    result = PanelOLS.from_formula(formula, data=est3).fit(
        cov_type="clustered", cluster_entity=True
    )
    print(result.summary)

    wald = result.wald_test(formula=WALD_CUMULATIVE)
    print(f"\n[H2] cumulative effect: {cumulative(result):.4f}")
    print(f"[H2] Wald p-value: {wald.pval:.6f}")
    return result, est3


# ---------------------------------------------------------------------------
# Decompositions
# ---------------------------------------------------------------------------
def decompose_counts(panel):
    """H1 decomposition across count outcomes (full panel, zeros included)."""
    est = panel.dropna(subset=["log_strikes_lag6"]).copy()
    est = est.set_index(["province", "year_month"])

    count_outcomes = {
        "n_attacks": "Total attacks (H1 main)",
        "n_isis_attacks": "ISIS-attributed attacks",
        "n_bombing": "Bombings",
        "n_armed_assault": "Armed assaults",
        "n_assassination": "Assassinations",
        "n_kidnapping": "Kidnappings",
        "n_suicide": "Suicide attacks",
        "n_success": "Successful attacks",
        "n_failure": "Failed attacks",
        "n_targ_civilian": "Attacks on civilians",
        "n_targ_military": "Attacks on military",
        "n_targ_police": "Attacks on police",
    }

    print("\n" + "=" * 70)
    print("DECOMPOSITION: COUNT OUTCOMES (full panel)")
    print("=" * 70)
    for var, label in count_outcomes.items():
        result = PanelOLS.from_formula(
            f"{var} ~ " + " + ".join(LAG_VARS)
            + " + log_turkish_strikes + EntityEffects + TimeEffects",
            data=est,
        ).fit(cov_type="clustered", cluster_entity=True)
        cumul = cumulative(result)
        wald = result.wald_test(formula=WALD_CUMULATIVE)
        print(f"\n{label}: cumul {cumul:+.4f}  Wald p {wald.pval:.4f}")


def decompose_composition(est3):
    """H2 decomposition across composition-share outcomes."""
    est = est3.copy()
    outcomes = {
        "pct_bombing": "Bombing share (complexity)",
        "pct_suicide": "Suicide share (org capacity)",
        "pct_success": "Success rate (planning quality)",
        "pct_soft_target": "Soft-target share (capability)",
    }
    print("\n" + "=" * 70)
    print("DECOMPOSITION: COMPOSITION SHARES (attacks > 0)")
    print("=" * 70)
    for var, label in outcomes.items():
        result = PanelOLS.from_formula(
            f"{var} ~ " + " + ".join(LAG_VARS)
            + " + log_turkish_strikes + EntityEffects + TimeEffects",
            data=est,
        ).fit(cov_type="clustered", cluster_entity=True)
        cumul = cumulative(result)
        wald = result.wald_test(formula=WALD_CUMULATIVE)
        print(f"\n{label}: cumul {cumul:+.4f}  Wald p {wald.pval:.4f}")


# ---------------------------------------------------------------------------
# Figure: lag profiles
# ---------------------------------------------------------------------------
def plot_lag_profiles(h1_result, h2_result, out="figures/lag_profiles_h1_h2.png"):
    lags = range(1, 7)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, result, color, title in [
        (axes[0], h1_result, "steelblue",
         "H1: Effect on attack frequency\n(Grievance channel)"),
        (axes[1], h2_result, "indianred",
         "H2: Effect on attack success rate\n(Degradation channel)"),
    ]:
        coefs = [result.params[f"log_strikes_lag{k}"] for k in lags]
        ci = result.conf_int()
        lo = [ci.loc[f"log_strikes_lag{k}", "lower"] for k in lags]
        hi = [ci.loc[f"log_strikes_lag{k}", "upper"] for k in lags]
        ax.bar(lags, coefs, color=color, alpha=0.7, zorder=2)
        ax.errorbar(
            lags, coefs,
            yerr=[np.array(coefs) - np.array(lo), np.array(hi) - np.array(coefs)],
            fmt="none", color="black", capsize=4, zorder=3,
        )
        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Lag (months)")
        ax.set_ylabel("Coefficient")
        ax.set_title(title)
        ax.set_xticks(list(lags))

    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nSaved figure -> {out}")


def main():
    panel = load_panel()
    h1_result = estimate_h1(panel)
    h2_result, est3 = estimate_h2(panel)
    decompose_counts(panel)
    decompose_composition(est3)
    plot_lag_profiles(h1_result, h2_result)


if __name__ == "__main__":
    main()
