"""
build_panel.py
==============
Constructs the province-month panel for the Iraq airstrikes study (2016-2020).

Pipeline:
  1. Load ACLED strike data + GTD attack data.
  2. Filter ACLED to Coalition-vs-ISIL strikes; harmonise province names across
     the two independent sources.
  3. Aggregate each source to a province-month (PM) panel.
  4. Merge onto a full 18-province x 60-month skeleton (Turkish strikes added as
     an exogenous control).
  5. Add log transforms and distributed-lag terms (lags 1-6).
  6. Save to data/panel_iraq_2016_2020.csv.

Inputs  : data/ACLED.csv, data/GTD.xlsx
Output  : data/panel_iraq_2016_2020.csv

Run:
    python src/build_panel.py
"""

import numpy as np
import pandas as pd

STUDY_START = "2016-01-01"
STUDY_END = "2020-12-31"

# ACLED admin1 -> GTD provstate name harmonisation
ACLED_TO_GTD = {
    "Ninewa": "Nineveh",
    "Salah Al Din": "Saladin",
    "Erbil": "Arbil",
    "Al Qadissiya": "Al Qadisiyah",
    "Al Sulaymaniyah": "Sulaymaniyah",
    "Kerbala": "Karbala",
    # Already matching: Al Anbar, Babil, Baghdad, Diyala, Kirkuk
}

TURKISH_PROV_MAP = {
    "Ninewa": "Nineveh",
    "Duhok": "Dihok",
    "Erbil": "Arbil",
    "Al Sulaymaniyah": "Sulaymaniyah",
}


# ---------------------------------------------------------------------------
# Load & clean
# ---------------------------------------------------------------------------
def load_acled(path="data/ACLED.csv"):
    """Load ACLED and return the Coalition-vs-ISIL strike subset."""
    acled = pd.read_csv(path)

    coalition_isis = acled[
        (acled["actor1"] == "Global Coalition Against Daesh")
        & (acled["actor2"] == "Islamic State in Iraq and the Levant (ISIL)")
    ].copy()

    coalition_isis["province"] = coalition_isis["admin1"].replace(ACLED_TO_GTD)
    coalition_isis["event_date"] = pd.to_datetime(coalition_isis["event_date"])
    coalition_isis["year_month"] = (
        coalition_isis["event_date"].dt.to_period("M").dt.to_timestamp()
    )
    print(f"Coalition-vs-ISIL strikes: {len(coalition_isis)}")
    return acled, coalition_isis


def load_gtd(path="data/GTD.xlsx"):
    """Load GTD, filter to Iraq 2016-2021, harmonise provinces, derive indicators."""
    gtd = pd.read_excel(path)
    gtd_iq = gtd[
        (gtd["country_txt"] == "Iraq") & (gtd["iyear"].between(2016, 2021))
    ].copy()

    # Province name fixes (sub-district -> parent province)
    gtd_iq["provstate"] = gtd_iq["provstate"].replace(
        {"Balad Ruz": "Diyala", "Halabja": "Sulaymaniyah", "Rutbah": "Al Anbar"}
    )

    # Drop attacks with no assignable province: an "Unknown"/NaN province can't
    # be placed in a province-month cell. This restores the 18-province panel
    # (and the 8,689-attack total).
    gtd_iq = gtd_iq[
        gtd_iq["provstate"].notna() & (gtd_iq["provstate"] != "Unknown")
    ].copy()

    # Date + casualties
    gtd_iq["date"] = pd.to_datetime(
        dict(
            year=gtd_iq["iyear"],
            month=gtd_iq["imonth"],
            day=gtd_iq["iday"].replace(0, 1),
        ),
        errors="coerce",
    )
    gtd_iq["year_month"] = gtd_iq["date"].dt.to_period("M").dt.to_timestamp()
    gtd_iq["casualties"] = gtd_iq["nkill"] + gtd_iq["nwound"]

    # ---- Base indicators ----------------------------------------------------
    # NOTE: the original notebook assumed is_isis / is_suicide / is_success
    # already existed on the frame. They are derived explicitly here so the
    # script runs end-to-end. VERIFY these definitions match your original
    # coding before trusting downstream numbers.
    gtd_iq["is_isis"] = (
        gtd_iq["gname"].str.contains("Islamic State|ISIL", case=False, na=False)
    ).astype(int)
    gtd_iq["is_suicide"] = (gtd_iq["suicide"] == 1).astype(int)
    gtd_iq["is_success"] = (gtd_iq["success"] == 1).astype(int)
    # Successful ISIS attacks — numerator for the H2 ISIS-only success rate
    gtd_iq["is_isis_success"] = (
        (gtd_iq["is_isis"] == 1) & (gtd_iq["success"] == 1)
    ).astype(int)

    # ---- Composition indicators --------------------------------------------
    gtd_iq["is_failure"] = (gtd_iq["success"] == 0).astype(int)
    gtd_iq["is_not_suicide"] = (gtd_iq["suicide"] == 0).astype(int)
    gtd_iq["is_bombing"] = (gtd_iq["attacktype1_txt"] == "Bombing/Explosion").astype(int)
    gtd_iq["is_armed_assault"] = (gtd_iq["attacktype1_txt"] == "Armed Assault").astype(int)
    gtd_iq["is_assassination"] = (gtd_iq["attacktype1_txt"] == "Assassination").astype(int)
    gtd_iq["is_kidnapping"] = (
        gtd_iq["attacktype1_txt"].str.contains("Hostage|Kidnapping", na=False)
    ).astype(int)
    gtd_iq["is_targ_civilian"] = (
        gtd_iq["targtype1_txt"] == "Private Citizens & Property"
    ).astype(int)
    gtd_iq["is_targ_military"] = (gtd_iq["targtype1_txt"] == "Military").astype(int)
    gtd_iq["is_targ_police"] = (gtd_iq["targtype1_txt"] == "Police").astype(int)
    gtd_iq["is_targ_govt"] = (gtd_iq["targtype1_txt"] == "Government (General)").astype(int)
    gtd_iq["is_targ_business"] = (gtd_iq["targtype1_txt"] == "Business").astype(int)

    print(f"GTD Iraq 2016-2021 attacks: {len(gtd_iq)}")
    return gtd_iq


# ---------------------------------------------------------------------------
# Province-month aggregation
# ---------------------------------------------------------------------------
def build_gtd_pm(gtd_iq):
    """Aggregate GTD to a province-month outcome panel."""
    gtd_pm = (
        gtd_iq.groupby(["provstate", "year_month"])
        .agg(
            n_attacks=("eventid", "count"),
            n_isis_attacks=("is_isis", "sum"),
            n_isis_success=("is_isis_success", "sum"),
            n_killed=("nkill", "sum"),
            n_wounded=("nwound", "sum"),
            n_casualties=("casualties", "sum"),
            n_attacks_with_casualties=("casualties", "count"),
            n_suicide=("is_suicide", "sum"),
            n_not_suicide=("is_not_suicide", "sum"),
            n_success=("is_success", "sum"),
            n_failure=("is_failure", "sum"),
            n_bombing=("is_bombing", "sum"),
            n_armed_assault=("is_armed_assault", "sum"),
            n_assassination=("is_assassination", "sum"),
            n_kidnapping=("is_kidnapping", "sum"),
            n_targ_civilian=("is_targ_civilian", "sum"),
            n_targ_military=("is_targ_military", "sum"),
            n_targ_police=("is_targ_police", "sum"),
            n_targ_govt=("is_targ_govt", "sum"),
            n_targ_business=("is_targ_business", "sum"),
        )
        .reset_index()
        .rename(columns={"provstate": "province"})
    )
    gtd_pm["casualties_per_attack"] = (
        gtd_pm["n_casualties"] / gtd_pm["n_attacks_with_casualties"]
    )
    gtd_pm["killed_per_attack"] = (
        gtd_pm["n_killed"] / gtd_pm["n_attacks_with_casualties"]
    )
    print(f"GTD province-months: {len(gtd_pm)}")
    return gtd_pm


def build_acled_pm(coalition_isis):
    """Aggregate Coalition-vs-ISIL strikes to a province-month treatment panel."""
    acled_pm = (
        coalition_isis.groupby(["province", "year_month"])
        .agg(
            n_strikes=("event_id_cnty", "count"),
            strike_fatalities=("fatalities", "sum"),
        )
        .reset_index()
    )
    print(f"ACLED province-months: {len(acled_pm)}")
    return acled_pm


def build_turkish_pm(acled):
    """Aggregate Turkish strikes (exogenous control) to province-month."""
    turkish = acled[acled["actor1"].str.contains("Turkey", na=False)].copy()
    turkish["province"] = turkish["admin1"].replace(TURKISH_PROV_MAP)
    turkish["event_date"] = pd.to_datetime(turkish["event_date"])
    turkish["year_month"] = turkish["event_date"].dt.to_period("M").dt.to_timestamp()
    turkish = turkish[
        (turkish["year_month"] >= STUDY_START) & (turkish["year_month"] <= STUDY_END)
    ]
    turkish_pm = (
        turkish.groupby(["province", "year_month"])
        .agg(n_turkish_strikes=("event_id_cnty", "count"))
        .reset_index()
    )
    print(f"Turkish strike province-months: {len(turkish_pm)}")
    return turkish_pm


# ---------------------------------------------------------------------------
# Merge + panel prep
# ---------------------------------------------------------------------------
def assemble_panel(gtd_iq, gtd_pm, acled_pm, turkish_pm):
    """Merge onto a full skeleton, fill zeros, add logs and lags 1-6."""
    all_provinces = sorted(gtd_iq["provstate"].unique())
    months = pd.date_range(STUDY_START, STUDY_END, freq="MS")
    print(f"Panel size: {len(all_provinces)} x {len(months)} = "
          f"{len(all_provinces) * len(months)}")

    skeleton = pd.DataFrame(
        [{"province": p, "year_month": m} for p in all_provinces for m in months]
    )

    panel = skeleton.merge(gtd_pm, on=["province", "year_month"], how="left")
    panel = panel.merge(acled_pm, on=["province", "year_month"], how="left")
    panel = panel.merge(turkish_pm, on=["province", "year_month"], how="left")

    # Count columns: NaN means a genuine zero
    count_cols = [
        "n_attacks", "n_isis_attacks", "n_killed", "n_wounded",
        "n_casualties", "n_attacks_with_casualties",
        "n_suicide", "n_not_suicide", "n_success", "n_failure",
        "n_bombing", "n_armed_assault", "n_assassination", "n_kidnapping",
        "n_targ_civilian", "n_targ_military", "n_targ_police",
        "n_targ_govt", "n_targ_business",
        "n_strikes", "strike_fatalities", "n_turkish_strikes",
    ]
    for col in count_cols:
        panel[col] = panel[col].fillna(0).astype(int)
    # casualties_per_attack / killed_per_attack left NaN where no attacks occurred

    # Panel identifiers + logs
    panel = panel.sort_values(["province", "year_month"]).reset_index(drop=True)
    panel["province_id"] = panel["province"].astype("category").cat.codes
    panel["time_index"] = panel.groupby("year_month").ngroup()
    panel["log_strikes"] = np.log1p(panel["n_strikes"])
    panel["log_turkish_strikes"] = np.log1p(panel["n_turkish_strikes"])

    # Distributed-lag terms 1-6 (7-12 added in robustness.py)
    for lag in range(1, 7):
        panel[f"log_strikes_lag{lag}"] = (
            panel.groupby("province")["log_strikes"].shift(lag)
        )

    n_usable = panel["log_strikes_lag6"].notna().sum()
    print(f"Final panel: {len(panel)} rows | usable after 6 lags: {n_usable}")
    return panel


def main():
    acled, coalition_isis = load_acled()
    gtd_iq = load_gtd()

    gtd_pm = build_gtd_pm(gtd_iq)
    acled_pm = build_acled_pm(coalition_isis)
    turkish_pm = build_turkish_pm(acled)

    panel = assemble_panel(gtd_iq, gtd_pm, acled_pm, turkish_pm)

    out = "data/panel_iraq_2016_2020.csv"
    panel.to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
