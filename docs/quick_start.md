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

    Paste TRAIN Aggregate Data block. Type DONE on a new line when finished.

Copy the BRAIN TRAIN Aggregate Data block and paste it into Python, then type:

    DONE

Python will parse Sharpe, Turnover, Fitness, Returns, Drawdown, and Margin from the pasted block. It will show the parsed values and ask you to confirm them.

The saved row uses explicit TRAIN columns such as `train_sharpe`, `train_fitness`, and `train_score`. It also stores the original pasted block in `train_aggregate_data`. The active BO target is `bo_score`; for normal TRAIN-only rows, `bo_score` equals `train_score`.

You can optionally add TEST or IS metrics later with `backfill_period_metrics.ipynb`. Those pasted blocks are saved as `test_aggregate_data` and `is_aggregate_data`. Old generic metric columns are treated as TRAIN metrics when logs are loaded.

After that, enter the separate BRAIN rating prompt:

    BRAIN rating:

For the rating, enter the label shown by BRAIN, for example:

    Good

or:

    Needs Improvement

If you type `cancel`, or finish an empty block with `DONE`, the current trial is cancelled and no CSV row is saved.

## 4. Repeat

Run the same notebook cell again to get the next suggestion.

The project remembers previous results automatically by reading your CSV file.

## Important

Do not edit or delete other people’s CSV files.

Each teammate should use their own name:

    run_one_trial(user="Angze", universe="TOP3000")
    run_one_trial(user="Alice", universe="TOP3000")
    run_one_trial(user="Bob", universe="TOP3000")
