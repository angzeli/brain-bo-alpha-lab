# 🚀 Quick Start

You do not need to understand Bayesian optimisation before using this project.

The workflow is:

1. Run one Python command.
2. Copy the suggested alpha name, category, description, expression, and settings into WorldQuant BRAIN.
3. Run the simulation manually on BRAIN.
4. Copy the resulting metrics back into Python.
5. The result is saved automatically.

## 1. Open the notebook

Open:

    alpha.ipynb

Then run the first cell:

    new_result = run_one_trial(user="YourName", universe="TOP3000")

For example:

    new_result = run_one_trial(user="Alice", universe="TOP3000")

To suggest several candidates at once, set `BATCH` in the notebook or pass:

    new_results = run_one_trial(user="Alice", universe="TOP3000", batch=3)

This creates or updates a personal CSV file such as:

    brain_bo_usa_top3000_alice.csv

## 2. Copy the suggested alpha into BRAIN

Python will print something like:

    Suggested alpha name:
    mom_close_20_volnorm_60

    Category:
    price momentum

    Description:
    Uses a 20-day close price change normalised by 60-day close volatility, transformed with rank. ...

    Alpha expression:
    rank(ts_delta(close, 20) / ts_std_dev(close, 60))

It will also print the suggested settings, for example:

    Region=USA, Universe=TOP3000, Delay=1, Neutralisation=Sector, Decay=5, ...

Copy these into BRAIN and run the simulation manually.

## 3. Enter the BRAIN results back into Python

After the BRAIN simulation finishes, Python will ask for:

    Sharpe:
    Turnover (%):
    Fitness:
    Returns (%):
    Drawdown (%):
    Margin (‱):
    BRAIN rating:

Copy these values from BRAIN.

For the rating, enter the label shown by BRAIN, for example:

    Good

or:

    Needs Improvement

If you press Enter on an empty metric prompt, the current trial is cancelled and no CSV row is saved.

## 4. Repeat

Run the same notebook cell again to get the next suggestion.

The project remembers previous results automatically by reading your CSV file.

## Important

Do not edit or delete other people’s CSV files.

Each teammate should use their own name:

    run_one_trial(user="Angze", universe="TOP3000")
    run_one_trial(user="Alice", universe="TOP3000")
    run_one_trial(user="Bob", universe="TOP3000")
