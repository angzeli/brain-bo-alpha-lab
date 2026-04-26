# 🧠 BRAIN BO Alpha Lab

A human-in-the-loop Bayesian optimisation workflow for systematic alpha research on WorldQuant BRAIN.

This project uses **BoTorch** and **GPyTorch** to suggest candidate alpha configurations, while keeping the actual WorldQuant BRAIN simulation step manual. The goal is not to automate submissions, but to make alpha research more structured, reproducible, and data-driven.

---

## 🎯 Motivation

Alpha research on BRAIN is naturally an experimental optimisation problem:

- each alpha expression is a hypothesis,
- each backtest gives noisy feedback,
- the design space is large and mixed continuous/categorical,
- and good research requires balancing exploration with exploitation.

This repository explores how Bayesian optimisation can be used as a lightweight assistant for this process.

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
├── README.md
├── alpha_bo.py
├── alpha.ipynb
├── docs/
│   ├── github_terminal_workflow.md
│   └── quick_start.md 
├── LICENSE
└── .gitignore
```

---

## 📚 Documentation

For detailed setup and usage instructions, see:

- [`docs/quick_start.md`](docs/quick_start.md) — beginner-friendly guide for running the workflow.
- [`docs/github_terminal_workflow.md`](docs/github_terminal_workflow.md) — practical Git/GitHub terminal commands for collaboration.

---

## 🚀 Quick Usage

The main interface is `run_one_trial(...)`:

```python
from alpha_bo import run_one_trial

new_result = run_one_trial(user="Angze", universe="TOP3000")
```

This will:

1. load the existing CSV log for the selected user and universe,
2. suggest one candidate alpha and BRAIN setting,
3. ask the user to enter the resulting BRAIN metrics,
4. append the completed result to the correct CSV file.

Each user/universe pair has its own independent log file. For example:

```text
brain_bo_usa_top3000_angze.csv
brain_bo_usa_top1000_angze.csv
brain_bo_usa_top3000_teammate.csv
```

---

## 🔍 Current Search Space

The optimiser currently searches over:

```python
[
    n,                 # primary lookback window
    m,                 # secondary / normalisation lookback window
    delay,             # Delay0 or Delay1
    decay,             # BRAIN decay setting
    truncation,        # maximum single-stock weight
    signal_type,       # momentum / reversal / volume / volatility
    price_field,       # close / open / high / low / vwap
    transform,         # rank / zscore / scale
    neutralisation,    # None / Market / Sector / Industry / Subindustry
    pasteurisation,    # On / Off
    nan_handling,      # On / Off
]
```

The region is currently fixed to `USA`, and each universe is treated as a separate campaign.

---

## 💾 Resume-Safe Logging

Completed trials are saved immediately after the user enters the metrics.

This means the workflow can be stopped and resumed safely. If the notebook or Python process is closed after 10 completed trials, the next run will automatically reload those 10 trials from the relevant CSV file.

CSV logs are excluded from version control because they may contain private alpha expressions and performance results.

---

## ⚠️ Important Notes

This project does **not** automate WorldQuant BRAIN submissions, scrape platform data, or bypass any manual platform workflow.

It is intended as a local research assistant for organising and guiding manual alpha experiments. The user remains responsible for manually entering expressions, choosing the suggested settings, running simulations, and recording the resulting metrics.

---

## 🧪 Project Status

Early experimental prototype.

Current priorities:

- improve the objective function after collecting real trials,
- add cleaner diagnostics for completed BO campaigns,
- improve handling of categorical variables,
- add more alpha expression templates,
- and compare BO-guided search against random/manual baselines.
