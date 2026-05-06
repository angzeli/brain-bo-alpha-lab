from datetime import datetime
from pathlib import Path
import shutil

import pandas as pd

from alpha_bo import (
    LEGACY_METRIC_COLUMNS,
    PERIODS,
    ensure_period_metric_columns,
    order_result_columns,
    parse_aggregate_data_block,
    period_aggregate_data_column,
    period_metric_column,
    recompute_scores_for_row as _recompute_scores_for_row,
)


def load_log_csv(path):
    """Load a raw BO log and add period metric columns in memory."""
    return ensure_period_metric_columns(pd.read_csv(Path(path)))


def save_log_csv(df, path, backup=True, backup_path=None):
    """Save an updated raw BO log, optionally creating a timestamped backup first."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    resolved_backup_path = None
    if backup and path.exists():
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            backup_path = path.with_name(f"{path.stem}.backup_{timestamp}{path.suffix}")
        resolved_backup_path = Path(backup_path)
        resolved_backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, resolved_backup_path)

    order_result_columns(df).to_csv(path, index=False)
    return resolved_backup_path


def rows_missing_period_metrics(df, period="test"):
    """Return rows missing one or more metric columns for the requested period."""
    period = _normalise_period(period)
    df = ensure_period_metric_columns(df)
    metric_columns = [period_metric_column(period, metric) for metric in LEGACY_METRIC_COLUMNS]
    missing_mask = df[metric_columns].isna().any(axis=1)
    return df.loc[missing_mask].copy()


def update_row_period_metrics(df, row_index, period, metrics, aggregate_data=None):
    """Fill one row's period metrics and recompute derived score columns."""
    period = _normalise_period(period)
    updated = ensure_period_metric_columns(df)

    if row_index not in updated.index:
        raise KeyError(f"Row index {row_index} was not found.")

    for metric in LEGACY_METRIC_COLUMNS:
        if metric not in metrics:
            raise ValueError(f"Missing parsed metric: {metric}")
        updated.loc[row_index, period_metric_column(period, metric)] = metrics[metric]

    if aggregate_data is not None:
        updated.loc[row_index, period_aggregate_data_column(period)] = str(aggregate_data).strip()

    updated.loc[row_index] = recompute_scores_for_row(updated.loc[row_index])
    return updated


def update_row_period_metrics_from_block(df, row_index, period, text):
    """Parse a pasted Aggregate Data block and update one row."""
    metrics = parse_aggregate_data_block(text)
    return update_row_period_metrics(df, row_index, period, metrics, aggregate_data=text)


def recompute_scores_for_row(row):
    """Recompute period scores and bo_score for a row after backfill edits."""
    return _recompute_scores_for_row(row, overwrite=True)


def recompute_scores_for_dataframe(df):
    """Recompute period scores and bo_score for every row."""
    updated = ensure_period_metric_columns(df)
    rows = [recompute_scores_for_row(row) for _, row in updated.iterrows()]
    return order_result_columns(pd.DataFrame(rows, columns=updated.columns))


def row_display_columns(df):
    """Return useful columns for identifying rows during manual backfill."""
    preferred = [
        "run_id",
        "run_timestamp",
        "batch_candidate_id",
        "user",
        "universe",
        "alpha_name",
        "alpha",
        "settings",
        "train_sharpe",
        "train_turnover_pct",
        "train_fitness",
        "train_returns_pct",
        "train_drawdown_pct",
        "train_margin_permyriad",
        "train_score",
        "test_score",
        "is_score",
        "bo_score",
        "brain_rating",
    ]
    return [column for column in preferred if column in df.columns]


def _normalise_period(period):
    period = str(period).strip().lower()
    if period not in PERIODS:
        raise ValueError(f"Unsupported period: {period}. Choose from {PERIODS}.")
    return period
