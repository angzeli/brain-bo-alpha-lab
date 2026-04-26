import re
from pathlib import Path

import pandas as pd


RAW_LOG_PATTERN = re.compile(
    r"^brain_bo_(?P<region>[a-z]+)_(?P<universe>top\d+)_(?P<user>[a-z0-9_]+)\.csv$"
)


def find_log_files(
    directory=".",
    region=None,
    universe=None,
    pattern="brain_bo_*.csv",
):
    """
    Find raw BO log CSV files that follow the current team filename convention.

    Expected raw-log filename format:

        brain_bo_{region}_{universe}_{user}.csv

    Examples
    --------
    find_log_files()
    find_log_files(region="usa")
    find_log_files(region="usa", universe="top3000")

    Notes
    -----
    This deliberately ignores legacy or generated files such as:

    - brain_bo_log.csv
    - combined_brain_bo_*.csv
    - top_trials_*.csv
    """
    directory = Path(directory)
    files = []

    for file in directory.glob(pattern):
        if not file.is_file():
            continue

        match = RAW_LOG_PATTERN.match(file.name.lower())
        if match is None:
            continue

        file_region = match.group("region")
        file_universe = match.group("universe")

        if region is not None and file_region != str(region).lower():
            continue

        if universe is not None and file_universe != str(universe).lower():
            continue

        files.append(file)

    return sorted(files)


def load_and_combine_logs(files):
    """
    Load multiple BO CSV logs and combine them into one DataFrame.

    Adds a `source_file` column so each row can be traced back to its original log.
    """
    if not files:
        raise FileNotFoundError("No CSV log files were found.")

    dfs = []

    for file in files:
        file = Path(file)
        df = pd.read_csv(file)

        match = RAW_LOG_PATTERN.match(file.name.lower())
        if match is not None:
            df["source_region"] = match.group("region")
            df["source_universe"] = match.group("universe")
            df["source_user"] = match.group("user")

        df["source_file"] = file.name
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    return combined


def combine_logs(
    directory=".",
    region=None,
    universe=None,
    output_path=None,
):
    """
    Find, load, combine, and optionally save BO log files.
    """
    files = find_log_files(
        directory=directory,
        region=region,
        universe=universe,
    )

    if not files:
        filters = []
        if region is not None:
            filters.append(f"region={region}")
        if universe is not None:
            filters.append(f"universe={universe}")

        filter_text = f" for {', '.join(filters)}" if filters else ""
        raise FileNotFoundError(f"No raw BO log files found{filter_text} in {Path(directory).resolve()}.")

    combined = load_and_combine_logs(files)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)

    return combined


def summarize_combined_logs(df):
    """
    Produce a simple summary of combined BO results.
    """
    summary = {
        "n_trials": len(df),
        "n_users": df["user"].nunique() if "user" in df.columns else (
            df["source_user"].nunique() if "source_user" in df.columns else None
        ),
        "users": sorted(df["user"].dropna().unique()) if "user" in df.columns else (
            sorted(df["source_user"].dropna().unique()) if "source_user" in df.columns else None
        ),
        "mean_score": df["score"].mean() if "score" in df.columns else None,
        "best_score": df["score"].max() if "score" in df.columns else None,
        "mean_fitness": df["fitness"].mean() if "fitness" in df.columns else None,
        "mean_sharpe": df["sharpe"].mean() if "sharpe" in df.columns else None,
        "mean_turnover": df["turnover"].mean() if "turnover" in df.columns else None,
        "pass_rate": df["passed"].mean() if "passed" in df.columns else None,
    }

    return pd.Series(summary)


def top_trials(df, n=10, sort_by="score"):
    """
    Return the top n trials sorted by a chosen metric.
    """
    if sort_by not in df.columns:
        raise ValueError(f"`{sort_by}` is not a column in the DataFrame.")

    return df.sort_values(sort_by, ascending=False).head(n)