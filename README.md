# More Frequent, Less Effective: Aerial Strikes and Insurgent Violence in Iraq (2016–2020)

> MSc dissertation — Applied Social Data Science, London School of Economics (MY400)

Estimating the causal effect of Global Coalition against Daesh airstrikes on insurgent violence during the anti-ISIL campaign in Iraq. Using a province-month panel and a distributed-lag design, this project separates two competing mechanisms: whether strikes **provoke** retaliatory violence, and whether they **degrade** insurgent operational capacity.

---

## Research question

Do aerial strikes against an insurgency reduce the violence it produces, or do they intensify it? The campaign against ISIL in Iraq offers a setting to test both possibilities at once, distinguishing:

- **H1 — Grievance / retaliation:** strikes increase the *frequency* of subsequent attacks.
- **H2 — Capacity degradation:** strikes reduce the *success rate* of attacks (attacks that still occur are less lethal or less likely to achieve their aim).

These are not mutually exclusive — a strike campaign can inflame and degrade simultaneously, and the design is built to detect both.

---

## Data

The design deliberately draws **treatment and outcome from independent sources** to avoid circularity (a strike and an attack being coded from the same underlying reporting).

| Role | Source | Scope |
|------|--------|-------|
| **Treatment** — coalition airstrikes | ACLED (air/drone strike events, coalition vs. ISIL) | 3,924 events |
| **Outcome** — insurgent attacks | GTD (Global Terrorism Database) | 8,689 events |
| **Control** — exogenous strike activity | Turkish airstrikes | included as covariate |

**Panel:** 18 Iraqi provinces × 60 months (2016–2020) = 1,080 province-month cells.
Assembled panel: `panel_iraq_2016_2020.csv`.

---

## Method

- **Distributed-lag model** with 12 monthly lags, capturing how a strike's effect accumulates and decays over the following year.
- **Two-way fixed effects:** province fixed effects (absorbing time-invariant local characteristics) and month fixed effects (absorbing nationwide shocks and trends).
- **Robust standard errors.** With only 18 clusters, clustered SEs are unreliable (Cameron & Miller); robust SEs are reported instead.
- **Turkish strikes** enter as an exogenous control to separate coalition activity from other aerial operations.

---

## Key findings

**H1 — Strikes increase attack frequency (supported, robust).**
Cumulative effect **+11.49** (p < 0.001). The result holds across specifications:

| Specification | Cumulative effect |
|---------------|-------------------|
| Full sample | +11.49 |
| Drop Nineveh | +7.23 |
| Drop 2016 | +5.20 |
| ISIS-only attacks | +7.25 |

**H2 — Strikes reduce attack success rate (supported, but conditional).**
Cumulative effect **−0.049** (p = 0.011). This effect depends on sustained bombardment — dropping Nineveh weakens it substantially (p = 0.22), suggesting capacity degradation emerges only where strike pressure is heavy and continuous.

**Interpretation:** the campaign appears to have provoked *more* attacks while making the average attack *less* successful — a boomerang on frequency alongside genuine capacity attrition where bombardment was concentrated.

---

## Repository structure

```
.
├── data/
│   └── panel_iraq_2016_2020.csv     # assembled panel (built by build_panel.py)
│                                    # raw ACLED.csv / GTD.xlsx go here too (git-ignored)
├── src/
│   ├── build_panel.py               # loads ACLED + GTD -> province-month panel
│   ├── models.py                    # main 6-lag TWFE models (H1, H2) + figure
│   └── robustness.py                # robust SEs, 12-lag, subsample checks
├── notebooks/
│   ├── 01_build_and_clean.ipynb     # build the panel (calls build_panel.py)
│   ├── 02_twfe.ipynb                # main H1/H2 models (calls models.py)
│   ├── 03_robustness.ipynb          # robustness checks (calls robustness.py)
│   └── eda.ipynb                    # exploratory notebook with figures
├── figures/                         # generated figures (lag profiles)
├── requirements.txt
└── README.md
```
---

**Note on derived indicators:** `build_panel.py` derives the `is_isis`,
`is_suicide`, and `is_success` flags from GTD fields (`gname`, `suicide`,
`success`). The original notebook assumed these already existed — verify the
definitions match your coding before relying on the numbers.

---

## Extensions / future work

- **Wind instrumental variable:** an IV strategy using ERA5 wind data (to isolate quasi-random variation in strike targeting/accuracy) is a planned extension for this period, not part of the main specification.

---

## Data sources & acknowledgements

- **ACLED** — Armed Conflict Location & Event Data Project.
- **GTD** — Global Terrorism Database, University of Maryland (START).
- Supervised by Dr. La Lova, LSE Department of Methodology.

*Data used under the terms of each provider; see their respective licences for redistribution conditions.*
