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
