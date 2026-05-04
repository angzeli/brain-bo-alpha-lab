import re
from pathlib import Path

import pandas as pd


SUPPORTED_UNIVERSES = ("top3000", "top1000", "top500", "top200")
RAW_LOG_PATTERN = re.compile(
    r"^brain_bo_(?P<region>[a-z]+)_(?P<universe>top(?:3000|1000|500|200))_(?P<user>[a-z0-9_]+)\.csv$"
)


def parse_raw_log_filename(file):
    """Return source metadata for a current-format raw BO log filename."""
    file = Path(file)
    match = RAW_LOG_PATTERN.match(file.name.lower())
    if match is None:
        raise ValueError(
            f"{file.name} is not a current raw BO log. Expected "
            "brain_bo_{region}_{universe}_{user}.csv."
        )

    return match.groupdict()


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
    region = None if region is None else str(region).lower()
    universe = None if universe is None else str(universe).lower()

    if universe is not None and universe not in SUPPORTED_UNIVERSES:
        raise ValueError(f"Unsupported universe: {universe}. Choose from {SUPPORTED_UNIVERSES}.")

    for file in directory.glob(pattern):
        if not file.is_file():
            continue

        try:
            metadata = parse_raw_log_filename(file)
        except ValueError:
            continue

        if region is not None and metadata["region"] != region:
            continue

        if universe is not None and metadata["universe"] != universe:
            continue

        files.append(file)

    return sorted(files)


def load_and_combine_logs(files):
    """
    Load multiple BO CSV logs and combine them into one DataFrame.

    Adds source metadata columns so each row can be traced back to its raw log.
    """
    if not files:
        raise FileNotFoundError("No CSV log files were found.")

    dfs = []

    for file in files:
        file = Path(file)
        metadata = parse_raw_log_filename(file)
        df = pd.read_csv(file)

        df["source_region"] = metadata["region"]
        df["source_universe"] = metadata["universe"]
        df["source_user"] = metadata["user"]
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
