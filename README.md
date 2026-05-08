# 🧠 BRAIN BO Alpha Lab

A human-in-the-loop Bayesian optimisation toolkit for structured WorldQuant BRAIN alpha research.

This project explores how Bayesian optimisation can support manual alpha experimentation on WorldQuant BRAIN. Python suggests candidate alpha expressions and simulation settings; the user manually runs the simulation on BRAIN; the resulting TRAIN / TEST / IS metrics are recorded back into a resume-safe CSV workflow.

The project does not automate BRAIN, scrape platform data, or submit alphas automatically. Its purpose is to provide a reproducible research workflow for noisy, manual, black-box alpha experimentation.

---

## 🎯 Why This Project Matters

Alpha research on BRAIN is naturally a noisy experimental optimisation problem: each expression is a hypothesis, each simulation gives imperfect feedback, and the search space contains mixed continuous, discrete, and categorical variables.

This project treats alpha discovery as a structured human-in-the-loop research process rather than ad hoc trial and error. The emphasis is on reproducibility, experiment logging, and systematic iteration.

The workflow is deliberately **human-in-the-loop**:

1. 🧪 Python suggests one candidate alpha and simulation setting.
2. 🧑‍💻 The user manually runs the simulation on WorldQuant BRAIN.
3. 📥 The user enters the resulting metrics back into Python.
4. 💾 The result is saved immediately to CSV.
5. 🔁 Future suggestions are conditioned on the accumulated results.

---

## 📦 Repository Structure

```python
brain-bo-alpha-lab/
├── docs/
│   ├── github_terminal_workflow.md
│   ├── quick_start.md
│   ├── brain_alpha_parameters_and_checks.md
│   └── brain_operator_playbook.md
├── .gitignore
├── alpha_bo.py
├── alpha.ipynb
├── csv_combiner.py
├── combine_csv.ipynb
├── data_pool_filter.py
├── filter_data_pool.ipynb
├── backfill_period_metrics.py
├── backfill_period_metrics.ipynb
├── LICENSE
└── README.md
```

---

## 📚 Documentation

For detailed setup and usage instructions, see:

- [`docs/quick_start.md`](docs/quick_start.md) — beginner-friendly guide for running the workflow.
- [`docs/github_terminal_workflow.md`](docs/github_terminal_workflow.md) — practical Git/GitHub terminal commands for collaboration.
- [`docs/brain_alpha_parameters_and_checks.md`](docs/brain_alpha_parameters_and_checks.md) — explains BRAIN metrics, ratings, checks, and how to record simulation results.
- [`docs/brain_operator_playbook.md`](docs/brain_operator_playbook.md) — practical guide to BRAIN operators used for alpha-template design.

---

## 🚀 Quick Usage

The main interface is `run_one_trial(...)`:

```python
from alpha_bo import run_one_trial

new_result = run_one_trial(user="Angze", universe="TOP3000")
```

To suggest and enter multiple manual simulations in one notebook run, pass `batch`:

```python
new_results = run_one_trial(user="Angze", universe="TOP3000", batch=3)
```

This will:

1. load the existing CSV log for the selected user and universe,
2. suggest one candidate alpha with BRAIN-ready name, category, description, and settings,
3. ask the user to paste the TRAIN Aggregate Data metrics,
4. append the completed result to the correct CSV file.

Each user/universe pair has its own independent log file. For example:

```text
brain_bo_usa_top3000_angze.csv
brain_bo_usa_top1000_alice.csv
```

---

## 🔍 Current Search Space

The optimiser currently searches over:

```python
[
    n,                 # primary lookback window
    m,                 # secondary / normalisation lookback window
    decay,             # BRAIN decay setting
    truncation,        # maximum single-stock weight
    template_type,     # structured alpha expression family
    price_field,       # close / open / high / low / vwap
    transform,         # rank / zscore / scale
    neutralisation,    # None / Market / Sector / Industry / Subindustry
    pasteurisation,    # On / Off
    nan_handling,      # On / Off
    direction,         # 1 / -1 signal direction before transform
    smoothing_window,  # 3 / 5 / 10 for smoothed templates
]
```

The region is currently fixed to `USA`, Delay is currently fixed to `1`, and each universe is treated as a separate campaign.

Current template families include price momentum/reversion, low volatility, volume ratio/surprise, inverse volume ratio, time-series scale/rank/z-score, average-difference signals, short/long trend, volatility change, price-volume momentum/reversal/correlation, smoothed price and volume variants, high/low momentum spread, close-to-VWAP momentum, intraday position, argmax/argmin recency, and hump-smoothed momentum. Future suggestions avoid unsupported `ts_min` and `ts_max`; `ts_scale` and `ts_rank` are used because they are confirmed available in our current BRAIN setup.

---

## 💾 Resume-Safe Logging

Completed trials are saved immediately after the user enters the metrics.

This means the workflow can be stopped and resumed safely. If the notebook or Python process is closed after 10 completed trials, the next run will automatically reload those 10 trials from the relevant CSV file.

Pressing Enter on an empty metric prompt cancels the current trial without writing a CSV row.

CSV logs are excluded from version control because they may contain private alpha expressions and performance results.

---

## 🧾 Period Metrics And BO Score

New rows store BRAIN Aggregate Data metrics by visible period:

- `train_*` columns for the default TRAIN block used during normal BO runs
- optional `test_*` columns for validation metrics
- optional `is_*` columns for visible In-Sample sanity checks

The original pasted Aggregate Data text is also stored for each period when available: `train_aggregate_data`, `test_aggregate_data`, and `is_aggregate_data`. This keeps an audit trail next to the parsed numeric columns.

Each period can have its own score: `train_score`, `test_score`, and `is_score`.

The active BO target is `bo_score`. If TEST metrics exist, `bo_score` combines TRAIN and TEST with a penalty when TEST is much weaker than TRAIN. If TEST is missing, `bo_score` falls back to `train_score`.

Older CSV rows with generic metric columns such as `sharpe`, `fitness`, and `turnover_pct` are treated as TRAIN metrics when loaded. The old `score` column is preserved for compatibility and used only as a fallback when `bo_score` is unavailable.

Use `backfill_period_metrics.ipynb` to manually add TEST or IS metrics to promising old runs. This is a maintenance workflow, not an analysis notebook.

---

## 🧹 Data Pool Filtering

After raw logs are combined into a master data pool, use `filter_data_pool.ipynb` to create focused subsets for later analysis.

Typical data flow:

```text
Raw personal logs
    ↓
combine_csv.ipynb creates date-stamped combined master data pools
    ↓
filter_data_pool.ipynb creates timestamped filtered subset CSVs
    ↓
analysis notebooks use those filtered subsets
```

Example use cases:

- Market-neutralised runs only
- TOP3000 runs only
- one template family only
- Average-or-better BRAIN ratings only

Generated subset files are timestamped, for example:

```text
subset_neutralisation_market_2026-05-06_203512.csv
```

These generated subset files are ignored by Git and should not be committed unless intentionally shared.

---

## ⚠️ Important Notes

This project does **not** automate WorldQuant BRAIN submissions, scrape platform data, or bypass any manual platform workflow.

It is intended as a local research assistant for organising and guiding manual alpha experiments. The user remains responsible for manually entering expressions, choosing the suggested settings, running simulations, and recording the resulting metrics.

---

## 🧪 Project Status

This project is currently archived as a workflow prototype.

The focus of this version is the research infrastructure rather than validated alpha performance. It demonstrates candidate generation, batch execution support, resume-safe CSV logging, period-specific metric recording, data-pool combining, filtering, backfilling, and teammate-facing documentation.

`alpha_bo.py` currently keeps the core workflow in one file to minimise friction before the Round 1 deadline. After the deadline, a useful refactor would be to split it into smaller modules for candidate generation, alpha-template construction, metric parsing, scoring, and CSV logging.

---

## 👥Author

**Angze Li** 🦑

Imperial College London
