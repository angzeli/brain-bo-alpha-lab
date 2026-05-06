import ast
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


PARAM_COLUMNS = [
    "n",
    "m",
    "decay",
    "truncation",
    "template_type",
    "price_field",
    "transform",
    "neutralisation",
    "pasteurisation",
    "nan_handling",
]

LEGACY_TEMPLATE_MAP = {
    "momentum": "price_momentum",
    "reversal": "price_reversion",
    "volume": "volume_ratio",
    "volatility": "low_volatility",
}


def load_data_pool(input_csv):
    """Load a combined data-pool CSV."""
    return pd.read_csv(Path(input_csv))


def _parse_params_value(value):
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        value = ast.literal_eval(value)
    elif not isinstance(value, (list, tuple)):
        if pd.isna(value):
            return None

    if not isinstance(value, (list, tuple)):
        return None

    return list(value)


def _canonical_template_type(template_type):
    template_type = str(template_type).strip()
    return LEGACY_TEMPLATE_MAP.get(template_type.lower(), template_type)


def _normalise_neutralisation(neutralisation):
    neutralisation = str(neutralisation).strip()
    if neutralisation.lower() == "none":
        return "None"
    return neutralisation.capitalize()


def _params_to_columns(value):
    """Parse supported old/current params schemas into explicit filter columns."""
    try:
        params = _parse_params_value(value)
    except (ValueError, SyntaxError, TypeError):
        return {}

    if params is None:
        return {}

    try:
        # Old 3-parameter schema: [n, m, signal/template_type]
        if len(params) == 3:
            values = [
                int(params[0]),
                int(params[1]),
                pd.NA,
                pd.NA,
                _canonical_template_type(params[2]),
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
            ]
            return dict(zip(PARAM_COLUMNS, values))

        # Previous 9-parameter schema:
        # [n, m, decay, truncation, signal/template_type, price_field, transform,
        #  neutralisation, universe]
        if len(params) == 9:
            values = [
                int(params[0]),
                int(params[1]),
                int(params[2]),
                float(params[3]),
                _canonical_template_type(params[4]),
                str(params[5]),
                str(params[6]),
                _normalise_neutralisation(params[7]),
                "On",
                "Off",
            ]
            return dict(zip(PARAM_COLUMNS, values))

        # Current 10-parameter schema without delay.
        if len(params) == 10:
            values = [
                int(params[0]),
                int(params[1]),
                int(params[2]),
                float(params[3]),
                _canonical_template_type(params[4]),
                str(params[5]),
                str(params[6]),
                str(params[7]),
                str(params[8]),
                str(params[9]),
            ]
            return dict(zip(PARAM_COLUMNS, values))

        # Previous 11-parameter schema with optimised delay:
        # [n, m, delay, decay, truncation, template_type, price_field, transform,
        #  neutralisation, pasteurisation, nan_handling]
        if len(params) == 11:
            values = [
                int(params[0]),
                int(params[1]),
                int(params[3]),
                float(params[4]),
                _canonical_template_type(params[5]),
                str(params[6]),
                str(params[7]),
                str(params[8]),
                str(params[9]),
                str(params[10]),
            ]
            return dict(zip(PARAM_COLUMNS, values))

        # Previous 13-parameter schema:
        # [n, m, decay, truncation, template_type, price_field, transform,
        #  neutralisation, universe, pasteurisation, nan_handling,
        #  test_years, test_months]
        if len(params) == 13:
            values = [
                int(params[0]),
                int(params[1]),
                int(params[2]),
                float(params[3]),
                _canonical_template_type(params[4]),
                str(params[5]),
                str(params[6]),
                str(params[7]),
                str(params[9]),
                str(params[10]),
            ]
            return dict(zip(PARAM_COLUMNS, values))
    except (ValueError, TypeError):
        return {}

    return {}


def expand_params_columns(df):
    """
    Return a copy of df with params expanded into explicit filter columns.

    Existing columns are preserved. Rows with missing or malformed params simply
    receive missing values in any newly created params-derived columns.
    """
    expanded = df.copy()

    if "params" not in expanded.columns:
        return expanded

    parsed_rows = expanded["params"].apply(_params_to_columns)

    for column in PARAM_COLUMNS:
        if column in expanded.columns:
            continue
        expanded[column] = parsed_rows.apply(lambda values: values.get(column, pd.NA))

    return expanded


def _is_collection_filter(value):
    return isinstance(value, (list, tuple, set, frozenset))


def _normalise_for_match(value):
    return str(value).strip().casefold()


def apply_filters(df, filters, case_insensitive=True):
    """Apply scalar or list-style equality filters to a DataFrame."""
    filtered = df.copy()
    filters = {} if filters is None else dict(filters)

    for column, wanted in filters.items():
        if column not in filtered.columns:
            available = ", ".join(filtered.columns)
            raise ValueError(f"Filter column '{column}' was not found. Available columns: {available}")

        series = filtered[column]

        if _is_collection_filter(wanted):
            wanted_values = list(wanted)
            if case_insensitive and any(isinstance(value, str) for value in wanted_values):
                wanted_set = {_normalise_for_match(value) for value in wanted_values}
                mask = series.notna() & series.map(_normalise_for_match).isin(wanted_set)
            else:
                mask = series.isin(wanted_values)
        else:
            if case_insensitive and isinstance(wanted, str):
                wanted_value = _normalise_for_match(wanted)
                mask = series.notna() & (series.map(_normalise_for_match) == wanted_value)
            else:
                mask = series == wanted

        filtered = filtered.loc[mask].copy()

    return filtered


def _safe_filename_part(value):
    value = str(value).strip().lower().replace(" ", "_")
    value = re.sub(r"[^a-z0-9_.-]+", "", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "value"


def current_timestamp():
    """Return a filename-safe local timestamp."""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def build_subset_filename(filters, run_date=None, timestamp=None, prefix="subset"):
    """Build a readable timestamped CSV filename for a filtered subset."""
    if timestamp is None:
        if run_date is None:
            timestamp = current_timestamp()
        else:
            timestamp = f"{run_date}_{datetime.now().strftime('%H%M%S')}"

    filters = {} if filters is None else dict(filters)

    parts = [_safe_filename_part(prefix)]
    if filters:
        for column, value in filters.items():
            parts.append(_safe_filename_part(column))
            if _is_collection_filter(value):
                parts.append("-".join(_safe_filename_part(item) for item in value))
            else:
                parts.append(_safe_filename_part(value))
    else:
        parts.append("all")

    parts.append(_safe_filename_part(timestamp))
    return "_".join(parts) + ".csv"


def filter_data_pool(input_csv, filters, output_csv=None, expand_params=True):
    """Load, optionally expand params, filter, save, and return the filtered data pool."""
    input_csv = Path(input_csv)
    df = load_data_pool(input_csv)

    if expand_params:
        df = expand_params_columns(df)

    df_filtered = apply_filters(df, filters)

    if output_csv is None:
        output_csv = input_csv.with_name(build_subset_filename(filters))
    else:
        output_csv = Path(output_csv)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(output_csv, index=False)
    df_filtered.attrs["output_csv"] = str(output_csv)

    return df_filtered
