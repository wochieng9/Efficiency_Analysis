# Modular DEA/SFA Efficiency Tool

This is a modularized version of the single-file Streamlit DEA/SFA application.
The refactor separates numerical model logic from Streamlit UI code so DEA, SFA,
bootstrap, result-building, and export behavior can be tested and maintained independently.

## Layout

```text
efficiency_tool_modularized/
├── run_app.py
├── requirements.txt
├── efficiency_tool/
│   ├── app_cache.py
│   ├── config.py
│   ├── exports.py
│   ├── dea/
│   │   ├── core.py
│   │   ├── bootstrap.py
│   │   └── malmquist.py
│   ├── sfa/
│   │   ├── core.py
│   │   ├── bootstrap.py
│   │   └── inference.py
│   ├── results/
│   │   ├── tables.py
│   │   └── time_trends.py
│   ├── ui/
│   │   ├── app.py
│   │   ├── sidebar.py
│   │   ├── tabs.py
│   │   ├── charts.py
│   │   ├── widgets.py
│   │   └── sfa_details.py
│   └── utils/
│       ├── data.py
│       └── stats.py
└── tests/
    └── test_core_smoke.py
```

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run run_app.py
```

## Design notes

- `dea/` and `sfa/` contain Streamlit-free numerical code, including adjacent-period DEA Malmquist productivity indices.
- `app_cache.py` is the only place where Streamlit caching wraps compute functions.
- `ui/` contains Streamlit layout, tabs, widgets, and chart rendering.
- `results/tables.py` centralizes result table construction.
- `results/time_trends.py` builds period-level and unit-level improvement summaries when a time column has more than one period.
- `utils/` contains data validation/cleaning and statistical helper functions.

For production, add unit tests around the core numerical modules before changing solver logic.


## Panel / time-period data

When both a DMU identifier column and a time column are selected in the sidebar,
the app treats each row as a DMU-period observation and builds result keys as
`<DMU> | <time>`. For example, Facility A in 2022 and Facility A in 2023 are
kept as separate analysis units rather than flagged as duplicate DMU IDs. If the
same DMU appears more than once in the same selected period, row numbers are
appended only to those duplicate DMU-period labels.


## Improvement over time

When a time column contains more than one period, the app adds a `Time Trends`
tab. The tab compares the same base unit across ordered periods and shows:

- average DEA, SFA, and scale-efficiency scores by period;
- first-to-last unit-level changes;
- top improvers and decliners for the selected score;
- optional score trajectories for top improvers; and
- export sheets for period summaries, change summaries, unit changes, and long score trajectories.

For standard DEA/SFA/scale scores, higher values are treated as improvement. If
the Stata/FRONTIER cost-ratio convention is selected for SFA, lower SFA values
are treated as improvement because 1 is the best cost-ratio value.

These time-trend summaries describe changes in the already-estimated scores. They
are separate from the DEA Malmquist tab and they do not convert the pooled SFA
model into a panel SFA estimator.

## Malmquist productivity index

When both a DMU identifier and a time column are selected, the sidebar offers
`Run Malmquist productivity index`. The app then adds a `Malmquist` tab whenever
finite adjacent-period indices can be calculated. The implementation estimates
four DEA distance-style scores for each unit present in two adjacent periods:

- `e00`: period-0 observation against the period-0 frontier;
- `e01`: period-0 observation against the period-1 frontier;
- `e10`: period-1 observation against the period-0 frontier; and
- `e11`: period-1 observation against the period-1 frontier.

It reports:

```text
Efficiency change, EC = e11 / e00
Technical change,  TC = sqrt((e10 / e11) * (e00 / e01))
Malmquist MPI,     MPI = EC * TC
```

Values above 1 indicate productivity improvement. The default reference-set
behavior uses all available units in each period's frontier, matching the usual
`Benchmarking::malmquist(..., SAMEREF = FALSE)` style. The optional
`Malmquist: use only common units as reference` checkbox restricts both adjacent
frontiers to units observed in both periods.

For output-oriented DEA, the Malmquist calculation uses the reciprocal distance
`1 / phi` internally so the interpretation remains `MPI > 1 = productivity
improvement`, while the standard DEA results table still exports the output
expansion factor separately for R/Stata comparisons.

## Matching R and Stata results

This version adds several compatibility controls that matter when comparing the
Streamlit output to native R/Stata libraries.

- DEA defaults in the sidebar are now input orientation and VRS returns, which
  match the common `Benchmarking::dea()` defaults in R.
- For output-oriented DEA, `DEA_Efficiency` is still the conventional 0-1 score
  `1 / phi`, but exports also include `DEA_Radial_Factor` and
  `DEA_R_Benchmarking_Style_Score`. Use those columns when matching packages
  that report the output expansion factor `phi` directly.
- DEA can be run either as one pooled DMU-period frontier or as separate
  frontiers within each selected time period. The separate-period mode is useful
  when matching year-by-year DEA runs.
- SFA variables can now be specified separately from DEA variables. This lets you
  use positive level variables for DEA and either raw or already logged variables
  for SFA.
- When `SFA selected variables are already logged` is checked, the app does not
  take logs again. This is the option to use when matching R/Stata formulas like
  `frontier ln_y ln_x1 ln_x2, ...`.
- For cost frontiers, the default SFA efficiency convention is the bounded
  `E[exp(-u)|epsilon]` measure. A Stata/FRONTIER-style cost ratio convention
  `E[exp(u)|epsilon]` is also available for comparisons where 1 is best and
  values above 1 indicate excess cost.

The SFA estimator is still a pooled cross-sectional half-normal model. It does
not implement Stata `xtfrontier`, panel time effects, truncated-normal
inefficiency, or efficiency-effects models.
