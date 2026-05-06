import ast
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.transforms import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

FIXED_REGION = "USA"
FIXED_UNIVERSE = "TOP3000"
FIXED_UNIT_HANDLING = "Verify"
FIXED_TEST_YEARS = 1
FIXED_TEST_MONTHS = 0
FIXED_DELAY = 1

UNIVERSES = ["TOP3000", "TOP1000", "TOP500", "TOP200"]
TEMPLATE_TYPES = [
    "price_momentum",
    "price_reversion",
    "low_volatility",
    "volume_ratio",
    "volume_ratio_inverse",
    "range_position",
    "time_series_rank",
    "short_long_trend",
    "volume_surprise",
    "price_volume_momentum",
    "price_volume_reversal",
    "smoothed_price_momentum",
    "smoothed_price_volume_reversal",
    "high_low_momentum_spread",
    "close_to_vwap_momentum",
    "intraday_position",
]
PRICE_FIELDS = ["close", "open", "high", "low", "vwap"]
TRANSFORMS = ["rank", "zscore", "scale"]
NEUTRALISATIONS = ["None", "Market", "Sector", "Industry", "Subindustry"]
BOOLEAN_SETTINGS = ["Off", "On"]
DIRECTIONS = [1, -1]
SMOOTHING_WINDOWS = [3, 5, 10]

CSV_COLUMN_ORDER = [
    "run_id",
    "run_timestamp",
    "user",
    "region",
    "universe",
    "batch_candidate_id",

    "alpha_name",
    "alpha_category",
    "alpha_description",
    "alpha",
    "settings",
    "params",

    "train_sharpe",
    "train_turnover_pct",
    "train_fitness",
    "train_returns_pct",
    "train_drawdown_pct",
    "train_margin_permyriad",
    "train_aggregate_data",
    "train_score",

    "test_sharpe",
    "test_turnover_pct",
    "test_fitness",
    "test_returns_pct",
    "test_drawdown_pct",
    "test_margin_permyriad",
    "test_aggregate_data",
    "test_score",

    "is_sharpe",
    "is_turnover_pct",
    "is_fitness",
    "is_returns_pct",
    "is_drawdown_pct",
    "is_margin_permyriad",
    "is_aggregate_data",
    "is_score",

    "bo_score",
    "score",
    "brain_rating",

    "latent_x",
    "source_file",
    "source_region",
    "source_universe",
    "source_user",
]

TEMPLATE_CATEGORY_MAP = {
    "price_momentum": "price momentum",
    "price_reversion": "price reversion",
    "low_volatility": "price reversion",
    "volume_ratio": "volume",
    "volume_ratio_inverse": "volume",
    "range_position": "price momentum",
    "time_series_rank": "price momentum",
    "short_long_trend": "price momentum",
    "volume_surprise": "volume",
    "price_volume_momentum": "price volume",
    "price_volume_reversal": "price volume",
    "smoothed_price_momentum": "price momentum",
    "smoothed_price_volume_reversal": "price volume",
    "high_low_momentum_spread": "price momentum",
    "close_to_vwap_momentum": "price momentum",
    "intraday_position": "price reversion",
}

LEGACY_TEMPLATE_MAP = {
    "momentum": "price_momentum",
    "reversal": "price_reversion",
    "volume": "volume_ratio",
    "volatility": "low_volatility",
}

BRAIN_RATINGS = [
    "Spectacular",
    "Excellent",
    "Good",
    "Average",
    "Needs Improvement",
]

BRAIN_RATING_ALIASES = {
    "spectacular": "Spectacular",
    "excellent": "Excellent",
    "good": "Good",
    "average": "Average",
    "avg": "Average",
    "needs improvement": "Needs Improvement",
    "need improvement": "Needs Improvement",
    "needs_improvement": "Needs Improvement",
    "need_improvement": "Needs Improvement",
    "needs-improvement": "Needs Improvement",
    "need-improvement": "Needs Improvement",
    "ni": "Needs Improvement",
}

STOP_BATCH = object()
MAX_CANDIDATE_RETRIES = 20
LATENT_DIM = 12
AGGREGATE_DATA_FIELDS = {
    "sharpe": "Sharpe",
    "turnover_pct": "Turnover",
    "fitness": "Fitness",
    "returns_pct": "Returns",
    "drawdown_pct": "Drawdown",
    "margin_permyriad": "Margin",
}
PERIODS = ("train", "test", "is")
PERIOD_SCORE_COLUMNS = {
    "train": "train_score",
    "test": "test_score",
    "is": "is_score",
}
PERIOD_AGGREGATE_DATA_COLUMNS = {
    "train": "train_aggregate_data",
    "test": "test_aggregate_data",
    "is": "is_aggregate_data",
}
LEGACY_METRIC_COLUMNS = list(AGGREGATE_DATA_FIELDS)
NUMBER_PATTERN = r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)"

LOOKBACK_MIN = 3
LOOKBACK_MAX = 120
DECAY_MIN = 1
DECAY_MAX = 60
TRUNCATION_MIN = 0.001
TRUNCATION_MAX = 0.20


def normalise_user_name(user):
    """Convert a user name into a safe lowercase filename component."""
    cleaned = str(user).strip().lower().replace(" ", "_")
    cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch == "_")

    if not cleaned:
        raise ValueError("User name must contain at least one letter or number.")

    return cleaned


def normalise_universe(universe):
    """Return the canonical universe name, accepting case-insensitive input."""
    cleaned = str(universe).strip().upper()
    if cleaned not in UNIVERSES:
        raise ValueError(f"Unsupported universe: {universe}. Choose from {UNIVERSES}.")
    return cleaned


def canonicalise_template_type(template_type):
    """Return the current template name, accepting legacy saved names."""
    cleaned = str(template_type).strip().lower()
    cleaned = LEGACY_TEMPLATE_MAP.get(cleaned, cleaned)
    if cleaned not in TEMPLATE_TYPES:
        raise ValueError(f"Unsupported template_type: {template_type}. Choose from {TEMPLATE_TYPES}.")
    return cleaned


def canonicalise_direction(direction):
    """Return direction as 1 or -1, accepting compact text labels."""
    if isinstance(direction, str):
        cleaned = direction.strip().lower()
        if cleaned in {"1", "+1", "positive", "pos", "long"}:
            return 1
        if cleaned in {"-1", "negative", "neg", "short", "inverse"}:
            return -1

    direction = int(direction)
    if direction not in DIRECTIONS:
        raise ValueError(f"Unsupported direction: {direction}. Choose from {DIRECTIONS}.")
    return direction


def canonicalise_smoothing_window(smoothing_window):
    """Return one of the configured smoothing windows."""
    window = int(smoothing_window)
    if window in SMOOTHING_WINDOWS:
        return window
    return min(SMOOTHING_WINDOWS, key=lambda candidate: abs(candidate - window))


def normalise_brain_rating(value):
    """Return the canonical BRAIN rating label for flexible user input."""
    cleaned = str(value).strip().lower()
    cleaned = " ".join(cleaned.replace("_", " ").replace("-", " ").split())
    return BRAIN_RATING_ALIASES.get(cleaned)


def get_log_path(user, universe=FIXED_UNIVERSE, region=FIXED_REGION):
    """Return a separate CSV log path for each user/region/universe campaign."""
    user_slug = normalise_user_name(user)
    region_slug = str(region).strip().lower()
    universe_slug = normalise_universe(universe).lower()

    return Path(f"brain_bo_{region_slug}_{universe_slug}_{user_slug}.csv")


def order_result_columns(df):
    """Apply a stable CSV column order while preserving any extra columns."""
    ordered = [column for column in CSV_COLUMN_ORDER if column in df.columns]
    extras = [column for column in df.columns if column not in ordered]
    return df.reindex(columns=ordered + extras)


torch.set_default_dtype(torch.double)

# Latent search dimensions, all in [0, 1]:
# x[0] = n_raw, primary lookback window
# x[1] = m_raw, secondary / normalisation lookback window
# x[2] = decay_raw
# x[3] = truncation_raw
# x[4] = template_type_raw
# x[5] = price_field_raw
# x[6] = transform_raw
# x[7] = neutralisation_raw
# x[8] = pasteurisation_raw
# x[9] = nan_handling_raw
# x[10] = direction_raw
# x[11] = smoothing_window_raw
# Region, universe, delay, unit handling, and test period are fixed to keep scores comparable across trials.
LATENT_BOUNDS = torch.tensor(
    [[0.0] * LATENT_DIM, [1.0] * LATENT_DIM],
    dtype=torch.double,
)


def canonicalise_params(params):
    """Normalise old/new saved params into the current 12-parameter schema.

    Current schema:
    [n, m, decay, truncation, template_type, price_field, transform, neutralisation,
     pasteurisation, nan_handling, direction, smoothing_window]

    Region, universe, delay, unit handling, and test period are intentionally fixed
    outside the optimised params so the BO objective is comparable across trials.
    """
    if isinstance(params, str):
        params = ast.literal_eval(params)

    if isinstance(params, dict):
        return [
            int(params["n"]),
            int(params["m"]),
            int(params["decay"]),
            float(params["truncation"]),
            canonicalise_template_type(params["template_type"]),
            str(params["price_field"]),
            str(params["transform"]),
            str(params["neutralisation"]),
            str(params["pasteurisation"]),
            str(params["nan_handling"]),
            canonicalise_direction(params.get("direction", 1)),
            canonicalise_smoothing_window(params.get("smoothing_window", 5)),
        ]

    # Old 3-parameter schema: [n, m, signal/template_type]
    if len(params) == 3:
        return [
            int(params[0]), int(params[1]), 1, 0.01,
            canonicalise_template_type(params[2]), "close", "rank", "None", "On", "Off",
            1, 5,
        ]

    # Previous 9-parameter schema:
    # [n, m, decay, truncation, signal/template_type, price_field, transform,
    #  neutralisation, universe]
    if len(params) == 9:
        neutralisation = str(params[7])
        neutralisation = "None" if neutralisation.lower() == "none" else neutralisation.capitalize()

        return [
            int(params[0]), int(params[1]), int(params[2]), float(params[3]),
            canonicalise_template_type(params[4]), str(params[5]), str(params[6]),
            neutralisation, "On", "Off", 1, 5,
        ]

    # Previous 13-parameter schema:
    # [n, m, decay, truncation, signal/template_type, price_field, transform,
    #  neutralisation, universe, pasteurisation, nan_handling,
    #  test_years, test_months]
    if len(params) == 13:
        return [
            int(params[0]), int(params[1]), int(params[2]), float(params[3]),
            canonicalise_template_type(params[4]), str(params[5]), str(params[6]), str(params[7]),
            str(params[9]), str(params[10]), 1, 5,
        ]

    # Previous/current 10-parameter schema without delay, direction, or smoothing.
    if len(params) == 10:
        return [
            int(params[0]), int(params[1]), int(params[2]), float(params[3]),
            canonicalise_template_type(params[4]), str(params[5]), str(params[6]), str(params[7]),
            str(params[8]), str(params[9]), 1, 5,
        ]

    # Previous 11-parameter schema with optimised delay. Drop saved delay and use FIXED_DELAY.
    if len(params) == 11:
        return [
            int(params[0]), int(params[1]), int(params[3]), float(params[4]),
            canonicalise_template_type(params[5]), str(params[6]), str(params[7]), str(params[8]),
            str(params[9]), str(params[10]), 1, 5,
        ]

    # Current 12-parameter schema.
    if len(params) == 12:
        return [
            int(params[0]), int(params[1]), int(params[2]), float(params[3]),
            canonicalise_template_type(params[4]), str(params[5]), str(params[6]), str(params[7]),
            str(params[8]), str(params[9]),
            canonicalise_direction(params[10]), canonicalise_smoothing_window(params[11]),
        ]

    raise ValueError(f"Unsupported params format with length {len(params)}: {params}")


def clip_unit(raw_value):
    return min(max(float(raw_value), 0.0), 1.0)


def decode_float(raw_value, lower, upper):
    return lower + clip_unit(raw_value) * (upper - lower)


def decode_int(raw_value, lower, upper):
    return int(round(decode_float(raw_value, lower, upper)))


def decode_category(raw_value, categories):
    if not categories:
        raise ValueError("categories must not be empty.")
    idx = min(int(clip_unit(raw_value) * len(categories)), len(categories) - 1)
    return categories[idx]


def encode_float(value, lower, upper):
    if upper == lower:
        return 0.0
    return clip_unit((float(value) - lower) / (upper - lower))


def encode_category(value, categories):
    if not categories:
        raise ValueError("categories must not be empty.")
    idx = categories.index(value)
    return (idx + 0.5) / len(categories)


def repair_candidate(decoded):
    """Apply minimal decoded-candidate cleanup before building an alpha."""
    repaired = dict(decoded)
    repair_flags = []

    rounded_truncation = round(float(repaired["truncation"]), 4)
    if rounded_truncation != repaired["truncation"]:
        repair_flags.append("rounded_truncation")
    repaired["truncation"] = rounded_truncation

    return repaired, repair_flags


def params_from_decoded_candidate(decoded):
    return [
        decoded["n"],
        decoded["m"],
        decoded["decay"],
        decoded["truncation"],
        decoded["template_type"],
        decoded["price_field"],
        decoded["transform"],
        decoded["neutralisation"],
        decoded["pasteurisation"],
        decoded["nan_handling"],
        decoded["direction"],
        decoded["smoothing_window"],
    ]


def decode_single_candidate(x_raw):
    """Decode a latent [0, 1]^d vector into repaired BRAIN-ready candidate values."""
    values = [float(value) for value in x_raw]
    if len(values) != LATENT_DIM:
        raise ValueError(f"Expected {LATENT_DIM} latent values, got {len(values)}.")

    decoded = {
        "n": decode_int(values[0], LOOKBACK_MIN, LOOKBACK_MAX),
        "m": decode_int(values[1], LOOKBACK_MIN, LOOKBACK_MAX),
        "decay": decode_int(values[2], DECAY_MIN, DECAY_MAX),
        "truncation": decode_float(values[3], TRUNCATION_MIN, TRUNCATION_MAX),
        "template_type": decode_category(values[4], TEMPLATE_TYPES),
        "price_field": decode_category(values[5], PRICE_FIELDS),
        "transform": decode_category(values[6], TRANSFORMS),
        "neutralisation": decode_category(values[7], NEUTRALISATIONS),
        "pasteurisation": decode_category(values[8], BOOLEAN_SETTINGS),
        "nan_handling": decode_category(values[9], BOOLEAN_SETTINGS),
        "direction": decode_category(values[10], DIRECTIONS),
        "smoothing_window": decode_category(values[11], SMOOTHING_WINDOWS),
    }

    repaired, _ = repair_candidate(decoded)
    return repaired


def decode_params(x_raw):
    """Convert a latent BoTorch candidate into the stored decoded params schema."""
    return params_from_decoded_candidate(decode_single_candidate(x_raw))


def encode_params_to_latent(params):
    """Convert stored decoded params into approximate latent [0, 1] coordinates."""
    (
        n, m, decay, truncation, template_type, price_field, transform,
        neutralisation, pasteurisation, nan_handling, direction, smoothing_window,
    ) = canonicalise_params(params)

    return [
        encode_float(n, LOOKBACK_MIN, LOOKBACK_MAX),
        encode_float(m, LOOKBACK_MIN, LOOKBACK_MAX),
        encode_float(decay, DECAY_MIN, DECAY_MAX),
        encode_float(truncation, TRUNCATION_MIN, TRUNCATION_MAX),
        encode_category(template_type, TEMPLATE_TYPES),
        encode_category(price_field, PRICE_FIELDS),
        encode_category(transform, TRANSFORMS),
        encode_category(neutralisation, NEUTRALISATIONS),
        encode_category(pasteurisation, BOOLEAN_SETTINGS),
        encode_category(nan_handling, BOOLEAN_SETTINGS),
        encode_category(direction, DIRECTIONS),
        encode_category(smoothing_window, SMOOTHING_WINDOWS),
    ]


def encode_params(params):
    """Backward-compatible alias for latent BO coordinates."""
    return encode_params_to_latent(params)


def candidate_key(params):
    return tuple(canonicalise_params(params))


def apply_transform(expression, transform):
    if transform == "rank":
        return f"rank({expression})"
    if transform == "zscore":
        return f"zscore({expression})"
    if transform == "scale":
        return f"scale({expression})"
    raise ValueError(f"Unknown transform: {transform}")


def apply_direction(expression, direction):
    if direction == 1:
        return expression
    if direction == -1:
        return f"-({expression})"
    raise ValueError(f"Unknown direction: {direction}")


def trend_windows(n, m):
    """Return short/long windows for templates that need n < m."""
    if n < m:
        return n, m
    if m < n:
        return m, n
    if n < 120:
        return n, n + 1
    return n - 1, n


def build_base_expression(params):
    """Build the untransformed Fast Expression body for the chosen template."""
    (
        n, m, decay, truncation, template_type, price_field, transform,
        neutralisation, pasteurisation, nan_handling, direction, smoothing_window,
    ) = canonicalise_params(params)

    if template_type == "price_momentum":
        return f"ts_delta({price_field}, {n}) / ts_std_dev({price_field}, {m})"
    if template_type == "price_reversion":
        return f"-ts_delta({price_field}, {n}) / ts_std_dev({price_field}, {m})"
    if template_type == "low_volatility":
        return f"-ts_std_dev({price_field}, {n})"
    if template_type == "volume_ratio":
        return f"ts_mean(volume, {n}) / ts_mean(volume, {m})"
    if template_type == "volume_ratio_inverse":
        return f"ts_mean(volume, {m}) / ts_mean(volume, {n})"
    if template_type == "range_position":
        return f"ts_scale({price_field}, {n})"
    if template_type == "time_series_rank":
        return f"ts_rank({price_field}, {n})"
    if template_type == "short_long_trend":
        short_window, long_window = trend_windows(n, m)
        return f"ts_mean({price_field}, {short_window}) / ts_mean({price_field}, {long_window}) - 1"
    if template_type == "volume_surprise":
        return f"(volume - ts_mean(volume, {n})) / ts_std_dev(volume, {m})"
    if template_type == "price_volume_momentum":
        return f"ts_delta({price_field}, {n}) * (volume / ts_mean(volume, {m}))"
    if template_type == "price_volume_reversal":
        return f"-ts_delta({price_field}, {n}) * (volume / ts_mean(volume, {m}))"
    if template_type == "smoothed_price_momentum":
        return f"ts_mean(ts_delta({price_field}, {n}), {smoothing_window}) / ts_std_dev({price_field}, {m})"
    if template_type == "smoothed_price_volume_reversal":
        return f"-ts_mean(ts_delta({price_field}, {n}), {smoothing_window}) * (volume / ts_mean(volume, {m}))"
    if template_type == "high_low_momentum_spread":
        return f"ts_delta(high, {n}) - ts_delta(low, {n})"
    if template_type == "close_to_vwap_momentum":
        return f"ts_delta(close, {n}) - ts_delta(vwap, {n})"
    if template_type == "intraday_position":
        return "(close - open) / (high - low)"

    raise ValueError(f"Unknown template_type: {template_type}")


def make_alpha(params, universe=FIXED_UNIVERSE):
    universe = normalise_universe(universe)
    (
        n, m, decay, truncation, template_type, price_field, transform,
        neutralisation, pasteurisation, nan_handling, direction, smoothing_window,
    ) = canonicalise_params(params)

    base_expression = build_base_expression(params)
    directed_expression = apply_direction(base_expression, direction)
    transformed_expression = apply_transform(directed_expression, transform)
    alpha = transformed_expression

    settings = (
        f"Region={FIXED_REGION}, Universe={universe}, Delay={FIXED_DELAY}\n"
        f"Neutralisation={neutralisation}, Decay={decay}, Truncation={truncation}\n\n"
        f"Pasteurisation={pasteurisation}, Unit handling={FIXED_UNIT_HANDLING}\n"
        f"NaN handling={nan_handling}, Test period={FIXED_TEST_YEARS}y {FIXED_TEST_MONTHS}m"
    )

    return alpha, settings


def build_alpha_metadata(params, universe=FIXED_UNIVERSE):
    """Build BRAIN-ready alpha name/category/description for a candidate."""
    universe = normalise_universe(universe)
    (
        n, m, decay, truncation, template_type, price_field, transform,
        neutralisation, pasteurisation, nan_handling, direction, smoothing_window,
    ) = canonicalise_params(params)

    alpha_category = TEMPLATE_CATEGORY_MAP[template_type]
    short_window, long_window = trend_windows(n, m)
    direction_prefix = "neg_" if direction == -1 else ""
    direction_note = "" if direction == 1 else " Negative direction is applied before the cross-sectional transform."

    if template_type == "price_momentum":
        alpha_name = f"{direction_prefix}mom_{price_field}_{n}_volnorm_{m}"
        uses = f"Uses a {n}-day {price_field} price change normalised by {m}-day {price_field} volatility"
    elif template_type == "price_reversion":
        alpha_name = f"{direction_prefix}rev_{price_field}_{n}_volnorm_{m}"
        uses = f"Uses a negative {n}-day {price_field} price change normalised by {m}-day {price_field} volatility"
    elif template_type == "low_volatility":
        alpha_name = f"{direction_prefix}low_vol_{price_field}_{n}_{transform}"
        uses = f"Uses negative {n}-day {price_field} volatility"
    elif template_type == "volume_ratio":
        alpha_name = f"{direction_prefix}vol_ratio_{n}_{m}_{transform}"
        uses = f"Uses the ratio of {n}-day average volume to {m}-day average volume"
    elif template_type == "volume_ratio_inverse":
        alpha_name = f"{direction_prefix}vol_ratio_inv_{m}_{n}_{transform}"
        uses = f"Uses the inverse ratio of {m}-day average volume to {n}-day average volume"
    elif template_type == "range_position":
        alpha_name = f"{direction_prefix}range_scale_{price_field}_{n}_{transform}"
        uses = f"Uses the time-series scaled position of {price_field} over a {n}-day lookback"
    elif template_type == "time_series_rank":
        alpha_name = f"{direction_prefix}ts_rank_{price_field}_{n}_{transform}"
        uses = f"Uses the {n}-day time-series rank of {price_field}"
    elif template_type == "short_long_trend":
        alpha_name = f"{direction_prefix}trend_{price_field}_{short_window}_{long_window}_{transform}"
        uses = f"Uses the ratio of {short_window}-day to {long_window}-day average {price_field} levels"
    elif template_type == "volume_surprise":
        alpha_name = f"{direction_prefix}vol_surprise_{n}_{m}_{transform}"
        uses = f"Uses current volume versus its {n}-day average, normalised by {m}-day volume volatility"
    elif template_type == "price_volume_momentum":
        alpha_name = f"{direction_prefix}pv_mom_{price_field}_{n}_{m}_{transform}"
        uses = f"Uses {n}-day {price_field} price change scaled by relative volume versus its {m}-day average"
    elif template_type == "price_volume_reversal":
        alpha_name = f"{direction_prefix}pv_rev_{price_field}_{n}_{m}_{transform}"
        uses = f"Uses negative {n}-day {price_field} price change scaled by relative volume versus its {m}-day average"
    elif template_type == "smoothed_price_momentum":
        alpha_name = f"{direction_prefix}mom_smooth_{price_field}_{n}_{m}_s{smoothing_window}_{transform}"
        uses = f"Uses a {smoothing_window}-day smoothed {n}-day {price_field} price change normalised by {m}-day {price_field} volatility"
    elif template_type == "smoothed_price_volume_reversal":
        alpha_name = f"{direction_prefix}pv_rev_smooth_{price_field}_{n}_{m}_s{smoothing_window}_{transform}"
        uses = f"Uses a {smoothing_window}-day smoothed negative {n}-day {price_field} price-change signal scaled by relative volume versus its {m}-day average"
    elif template_type == "high_low_momentum_spread":
        alpha_name = f"{direction_prefix}hl_spread_{n}_{transform}"
        uses = f"Uses the spread between high-price momentum and low-price momentum over a {n}-day lookback"
    elif template_type == "close_to_vwap_momentum":
        alpha_name = f"{direction_prefix}close_vwap_mom_{n}_{transform}"
        uses = f"Uses the difference between close-price momentum and VWAP momentum over a {n}-day lookback"
    elif template_type == "intraday_position":
        alpha_name = f"{direction_prefix}intraday_pos_{transform}"
        uses = "Uses where the close sits relative to the open inside the daily high-low range"
    else:
        raise ValueError(f"Unknown template_type: {template_type}")

    alpha_description = (
        f"{uses}, transformed with {transform}.{direction_note} "
        f"Tested with Delay{FIXED_DELAY}, {neutralisation} neutralisation, Decay {decay}, "
        f"Truncation {truncation}, Pasteurisation {pasteurisation}, "
        f"NaN handling {nan_handling}, Universe {universe}."
    )

    return {
        "alpha_name": alpha_name,
        "alpha_category": alpha_category,
        "alpha_description": alpha_description,
    }


def compute_period_score(sharpe, turnover_pct, fitness, returns_pct, drawdown_pct, margin_permyriad):
    sharpe = max(min(sharpe, 4.0), -4.0)
    fitness = max(min(fitness, 5.0), -2.0)
    returns_pct = max(min(returns_pct, 80.0), -80.0)
    turnover_pct = max(min(turnover_pct, 200.0), 0.0)
    drawdown_abs = max(min(abs(drawdown_pct), 80.0), 0.0)
    margin_permyriad = max(min(margin_permyriad, 50.0), -50.0)

    score = (
        1.00 * fitness
        + 0.50 * sharpe
        + 0.03 * returns_pct
        + 0.05 * margin_permyriad
        - 0.003 * turnover_pct
        - 0.03 * drawdown_abs
    )

    return float(score)


def compute_score(sharpe, turnover_pct, fitness, returns_pct, drawdown_pct, margin_permyriad):
    """Backward-compatible alias for the period-level score."""
    return compute_period_score(
        sharpe=sharpe,
        turnover_pct=turnover_pct,
        fitness=fitness,
        returns_pct=returns_pct,
        drawdown_pct=drawdown_pct,
        margin_permyriad=margin_permyriad,
    )


def compute_bo_score(train_score=None, test_score=None):
    """Return the active BO target from TRAIN/TEST scores."""
    train_score = _float_or_none(train_score)
    test_score = _float_or_none(test_score)

    if train_score is None:
        return None

    if test_score is None:
        return float(train_score)

    collapse_penalty = 0.5 * max(0.0, train_score - test_score)
    return float(0.4 * train_score + 0.6 * test_score - collapse_penalty)


def _float_or_none(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def period_metric_column(period, metric):
    return f"{period}_{metric}"


def period_aggregate_data_column(period):
    period = str(period).strip().lower()
    if period not in PERIODS:
        raise ValueError(f"Unsupported period: {period}. Choose from {PERIODS}.")
    return PERIOD_AGGREGATE_DATA_COLUMNS[period]


def add_period_prefix(metrics, period):
    return {
        period_metric_column(period, metric): value
        for metric, value in metrics.items()
    }


def period_metrics_from_row(row, period):
    metrics = {}
    for metric in LEGACY_METRIC_COLUMNS:
        value = _float_or_none(row.get(period_metric_column(period, metric)))
        if value is None:
            return None
        metrics[metric] = value
    return metrics


def compute_period_score_from_row(row, period):
    metrics = period_metrics_from_row(row, period)
    if metrics is None:
        return None
    return compute_period_score(**metrics)


def recompute_scores_for_row(row, overwrite=False):
    """Return a copy of row with missing period scores and bo_score filled."""
    updated = row.copy()

    for period, score_column in PERIOD_SCORE_COLUMNS.items():
        if overwrite or _float_or_none(updated.get(score_column)) is None:
            period_score = compute_period_score_from_row(updated, period)
            if period_score is not None:
                updated[score_column] = period_score

    if overwrite or _float_or_none(updated.get("bo_score")) is None:
        train_score = _float_or_none(updated.get("train_score"))
        test_score = _float_or_none(updated.get("test_score"))
        legacy_score = _float_or_none(updated.get("score"))

        if train_score is not None and test_score is not None:
            updated["bo_score"] = compute_bo_score(train_score=train_score, test_score=test_score)
        elif not overwrite and legacy_score is not None:
            updated["bo_score"] = float(updated["score"])
        elif train_score is not None:
            updated["bo_score"] = compute_bo_score(train_score=train_score, test_score=None)
        elif legacy_score is not None:
            updated["bo_score"] = float(updated["score"])

    return updated


def ensure_period_metric_columns(df):
    """
    Return a copy with legacy generic metrics treated as TRAIN metrics.

    This helper does not overwrite old `score`; it fills explicit period columns
    and derived scores only where they are missing.
    """
    df = df.copy()

    for metric in LEGACY_METRIC_COLUMNS:
        train_column = period_metric_column("train", metric)
        if train_column not in df.columns and metric in df.columns:
            df[train_column] = df[metric]
        elif train_column not in df.columns:
            df[train_column] = pd.NA
        elif metric in df.columns:
            df[train_column] = df[train_column].combine_first(df[metric])

    for period in PERIODS:
        for metric in LEGACY_METRIC_COLUMNS:
            column = period_metric_column(period, metric)
            if column not in df.columns:
                df[column] = pd.NA
        aggregate_data_column = period_aggregate_data_column(period)
        if aggregate_data_column not in df.columns:
            df[aggregate_data_column] = pd.NA
        score_column = PERIOD_SCORE_COLUMNS[period]
        if score_column not in df.columns:
            df[score_column] = pd.NA

    if "bo_score" not in df.columns:
        df["bo_score"] = pd.NA

    updated_rows = [recompute_scores_for_row(row) for _, row in df.iterrows()]
    return pd.DataFrame(updated_rows, columns=df.columns)


def get_bo_training_score(row):
    """Return the preferred saved target for BO training."""
    for column in ("bo_score", "score", "train_score"):
        value = _float_or_none(row.get(column))
        if value is not None:
            return value
    return None


def recompute_scores_in_csv(csv_path, output_csv=None, backup=True, backup_path=None, dry_run=False):
    """
    Fill period-specific metric and score columns without overwriting legacy score.

    This is an explicit one-time migration helper. It treats old generic metrics
    as TRAIN metrics and computes train_score/test_score/is_score/bo_score where
    possible. The legacy `score` column is preserved.
    """
    csv_path = Path(csv_path)
    output_csv = csv_path if output_csv is None else Path(output_csv)

    df = pd.read_csv(csv_path)
    updated_df = ensure_period_metric_columns(df)
    train_score_present = pd.to_numeric(updated_df["train_score"], errors="coerce").notna()
    bo_score_present = pd.to_numeric(updated_df["bo_score"], errors="coerce").notna()
    updated_df = order_result_columns(updated_df)

    resolved_backup_path = None
    if not dry_run:
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        if backup and output_csv.resolve() == csv_path.resolve():
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                backup_path = csv_path.with_name(f"{csv_path.stem}.backup_{timestamp}{csv_path.suffix}")
            resolved_backup_path = Path(backup_path)
            resolved_backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(csv_path, resolved_backup_path)

        updated_df.to_csv(output_csv, index=False)

    summary = {
        "input_csv": str(csv_path),
        "output_csv": str(output_csv),
        "backup_path": None if resolved_backup_path is None else str(resolved_backup_path),
        "rows_total": int(len(df)),
        "rows_with_train_score": int(train_score_present.sum()),
        "rows_with_bo_score": int(bo_score_present.sum()),
        "rows_without_bo_score": int((~bo_score_present).sum()),
        "dry_run": bool(dry_run),
    }

    print(
        f"Prepared period scores for {summary['rows_with_train_score']} row(s); "
        f"{summary['rows_with_bo_score']} row(s) have bo_score."
    )
    if resolved_backup_path is not None:
        print(f"Backup saved to: {resolved_backup_path}")
    if not dry_run:
        print(f"Updated CSV saved to: {output_csv}")

    return summary


def load_existing_results(user, log_path=None, universe=FIXED_UNIVERSE, region=FIXED_REGION):
    """Load all completed trials from CSV and convert them into BoTorch training data."""
    universe = normalise_universe(universe)

    if log_path is None:
        log_path = get_log_path(user=user, universe=universe, region=region)

    if not log_path.exists():
        return [], None, None

    df = pd.read_csv(log_path)
    if df.empty:
        return [], None, None

    df = ensure_period_metric_columns(df)

    missing_columns = {"params"} - set(df.columns)
    if missing_columns:
        raise ValueError(f"{log_path} is missing required column(s): {sorted(missing_columns)}")

    existing_results = df.to_dict("records")
    train_x = []
    train_y = []
    skipped_rows = 0

    for _, row in df.iterrows():
        try:
            params = canonicalise_params(row["params"])
            score = get_bo_training_score(row)
            if score is None:
                raise ValueError("missing usable bo_score, score, or train_score")
            train_x.append(encode_params_to_latent(params))
            train_y.append([score])
        except (ValueError, TypeError, SyntaxError) as error:
            skipped_rows += 1
            print(f"Skipping one unreadable saved trial in {log_path}: {error}")

    train_x = torch.tensor(train_x) if train_x else None
    train_y = torch.tensor(train_y) if train_y else None

    if skipped_rows:
        print(f"Loaded {len(train_x) if train_x is not None else 0} usable trial(s); skipped {skipped_rows}.")

    return existing_results, train_x, train_y


def append_result_to_csv(result, user, log_path=None, universe=FIXED_UNIVERSE, region=FIXED_REGION):
    """Append one completed result immediately, so progress survives kernel stops."""
    universe = normalise_universe(universe)

    if log_path is None:
        log_path = get_log_path(user=user, universe=universe, region=region)

    result = dict(result)
    result["run_timestamp"] = datetime.now().isoformat(sep=" ", timespec="seconds")
    result.setdefault("region", str(region).strip().upper())
    result.setdefault("universe", universe)
    result.setdefault("user", normalise_user_name(user))

    if log_path.exists() and log_path.stat().st_size > 0:
        existing_df = pd.read_csv(log_path)
        result_df = pd.DataFrame([result])
        combined_columns_df = order_result_columns(
            pd.concat([existing_df, result_df], ignore_index=True, sort=False)
        )

        if list(existing_df.columns) == list(combined_columns_df.columns):
            result_df = result_df.reindex(columns=combined_columns_df.columns)
            result_df.to_csv(log_path, mode="a", header=False, index=False)
        else:
            combined_columns_df.to_csv(log_path, index=False)
    else:
        result_df = order_result_columns(pd.DataFrame([result]))
        result_df.to_csv(log_path, index=False)

    return result


def ask_float(prompt, allow_stop=False):
    """Ask for a float. Empty input skips the candidate; `stop` ends a batch."""
    while True:
        value = input(prompt).strip()
        if value == "":
            return None
        if allow_stop and value.lower() == "stop":
            return STOP_BATCH
        try:
            return float(value)
        except ValueError:
            stop_text = ", type stop to end the batch" if allow_stop else ""
            print(f"Please enter a number, or press Enter to skip this candidate{stop_text}.")


def read_multiline_block(prompt, end_token="DONE", allow_stop=False):
    """Read pasted text until end_token. Return None to cancel, STOP_BATCH to stop a batch."""
    print(prompt)
    lines = []
    end_token = end_token.strip().lower()

    while True:
        line = input()
        stripped = line.strip()
        lowered = stripped.lower()

        if lowered == end_token:
            break
        if lowered == "cancel":
            return None
        if allow_stop and lowered == "stop":
            return STOP_BATCH

        lines.append(line)

    text = "\n".join(lines).strip()
    if not text:
        return None

    return text


def _parse_number(value):
    return float(value.replace(",", ""))


def parse_aggregate_data_block(text):
    """Parse a pasted BRAIN Aggregate Data block into numeric metric values."""
    parsed = {}
    missing = []

    for output_key, label in AGGREGATE_DATA_FIELDS.items():
        pattern = rf"\b{re.escape(label)}\b(?:\s*\([^)]*\))?\s*:?\s*({NUMBER_PATTERN})\s*[%‱]?"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is None:
            missing.append(label)
            continue
        parsed[output_key] = _parse_number(match.group(1))

    if missing:
        raise ValueError(f"Could not find required Aggregate Data metric(s): {', '.join(missing)}")

    return parsed


def ask_yes_no(prompt, allow_stop=False):
    """Ask for a yes/no confirmation. Empty input means no."""
    while True:
        value = input(prompt).strip().lower()
        if value == "":
            return False
        if allow_stop and value == "stop":
            return STOP_BATCH
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        stop_text = ", or type stop to end the batch" if allow_stop else ""
        print(f"Please enter y/n{stop_text}.")


def print_parsed_aggregate_metrics(metrics, period=None):
    label = "" if period is None else f" {period.upper()}"
    print(f"\nParsed{label} Aggregate Data metrics:")
    print(f"Sharpe:        {metrics['sharpe']}")
    print(f"Turnover (%):  {metrics['turnover_pct']}")
    print(f"Fitness:       {metrics['fitness']}")
    print(f"Returns (%):   {metrics['returns_pct']}")
    print(f"Drawdown (%):  {metrics['drawdown_pct']}")
    print(f"Margin (‱):    {metrics['margin_permyriad']}")


def ask_aggregate_data_metrics(period="train", allow_stop=False, include_raw=False):
    """Ask for a pasted Aggregate Data block and return parsed metrics."""
    period_label = period.upper()
    prompt = (
        f"Paste {period_label} Aggregate Data block. Type DONE on a new line when finished.\n"
        "Type cancel to skip this candidate"
        + (" or stop to end the remaining batch" if allow_stop else "")
        + "."
    )

    while True:
        block = read_multiline_block(prompt, allow_stop=allow_stop)
        if block is STOP_BATCH:
            return STOP_BATCH
        if block is None:
            return None

        try:
            metrics = parse_aggregate_data_block(block)
        except ValueError as error:
            print(f"\nCould not parse Aggregate Data block: {error}")
            print("Please paste the block again, or type cancel then DONE to skip this candidate.")
            continue

        print_parsed_aggregate_metrics(metrics, period=period)
        confirmed = ask_yes_no("Use these parsed metrics? y/n: ", allow_stop=allow_stop)
        if confirmed is STOP_BATCH:
            return STOP_BATCH
        if confirmed:
            if include_raw:
                return metrics, block.strip()
            return metrics

        print("Okay, paste the Aggregate Data block again.")


def ask_brain_rating(prompt, allow_stop=False):
    """Ask for a BRAIN rating. Empty input skips the candidate; `stop` ends a batch."""
    allowed_text = " / ".join(BRAIN_RATINGS)
    while True:
        value = input(prompt).strip()
        if value == "":
            return None
        if allow_stop and value.lower() == "stop":
            return STOP_BATCH

        rating = normalise_brain_rating(value)
        if rating is not None:
            return rating

        stop_text = ", type stop to end the batch" if allow_stop else ""
        print(f"Please enter one of: {allowed_text}. Press Enter to skip this candidate{stop_text}.")


def build_candidate(params, user, universe=FIXED_UNIVERSE, batch_candidate_id=None):
    """Create a candidate dict with all BRAIN-ready fields, before metrics exist."""
    universe = normalise_universe(universe)
    alpha, settings = make_alpha(params, universe=universe)
    metadata = build_alpha_metadata(params, universe=universe)

    return {
        **metadata,
        "alpha": alpha,
        "settings": settings,
        "region": FIXED_REGION,
        "universe": universe,
        "user": normalise_user_name(user),
        "batch_candidate_id": batch_candidate_id,
        "params": canonicalise_params(params),
    }


def score_period_metrics(metrics):
    return compute_period_score(
        sharpe=metrics["sharpe"],
        turnover_pct=metrics["turnover_pct"],
        fitness=metrics["fitness"],
        returns_pct=metrics["returns_pct"],
        drawdown_pct=metrics["drawdown_pct"],
        margin_permyriad=metrics["margin_permyriad"],
    )


def build_period_result_fields(metrics, period, aggregate_data=None):
    fields = add_period_prefix(metrics, period)
    if aggregate_data is not None:
        fields[period_aggregate_data_column(period)] = str(aggregate_data).strip()
    fields[PERIOD_SCORE_COLUMNS[period]] = score_period_metrics(metrics)
    return fields


def suggest_one_candidate(user, universe, train_x, train_y, seed_threshold, used_keys=None):
    """Suggest one decoded candidate and avoid duplicate final settings."""
    used_keys = set() if used_keys is None else used_keys
    use_botorch = train_x is not None and train_x.shape[0] >= seed_threshold
    last_candidate = None

    for attempt in range(MAX_CANDIDATE_RETRIES):
        if use_botorch and attempt == 0:
            raw_candidate = suggest_botorch_candidate(train_x, train_y)
        else:
            raw_candidate = suggest_random_candidate()

        candidate = build_candidate(
            params=decode_params(raw_candidate),
            user=user,
            universe=universe,
        )
        last_candidate = candidate

        if candidate_key(candidate["params"]) not in used_keys:
            return candidate

    return last_candidate


def print_candidate(candidate, index=None, batch=None):
    """Print a candidate in a BRAIN-ready block."""
    if index is None or batch is None:
        print("\nBO candidate")
    else:
        print(f"\nBO candidate {index}/{batch}")

    print("-" * 50)
    if candidate.get("batch_candidate_id") is not None:
        print("\nBatch candidate ID:")
        print(candidate["batch_candidate_id"])
    print("\nAlpha expression:")
    print(candidate["alpha"])
    print("\nSuggested settings:")
    print(candidate["settings"])
    print("")
    print("Suggested alpha name:")
    print(candidate["alpha_name"])
    print("\nCategory:")
    print(candidate["alpha_category"])
    print("\nDescription:")
    print(candidate["alpha_description"])


def collect_metrics_for_candidate(candidate, index=None, batch=None, allow_stop=False):
    """Collect Aggregate Data metrics and return a completed result dict."""
    if index is None or batch is None:
        print("\nAfter the simulation finishes, enter the TRAIN Aggregate Data metrics:")
        print("Paste the copied TRAIN Aggregate Data block below.")
    else:
        print(f"\nEnter metrics for candidate {index}/{batch}: {candidate['alpha_name']}")
        print("\nAlpha expression:")
        print(candidate["alpha"])
        stop_hint = " Type stop at any prompt to end the remaining batch." if allow_stop else ""
        print(f"Paste the copied TRAIN Aggregate Data block below.{stop_hint}")

    train_entry = ask_aggregate_data_metrics(period="train", allow_stop=allow_stop, include_raw=True)
    if train_entry is STOP_BATCH:
        return STOP_BATCH
    if train_entry is None:
        return None
    train_metrics, train_aggregate_data = train_entry

    result_fields = build_period_result_fields(
        train_metrics,
        "train",
        aggregate_data=train_aggregate_data,
    )

    add_test = ask_yes_no("Add TEST metrics now? y/n: ", allow_stop=allow_stop)
    if add_test is STOP_BATCH:
        return STOP_BATCH
    if add_test:
        test_entry = ask_aggregate_data_metrics(period="test", allow_stop=allow_stop, include_raw=True)
        if test_entry is STOP_BATCH:
            return STOP_BATCH
        if test_entry is None:
            return None
        test_metrics, test_aggregate_data = test_entry
        result_fields.update(
            build_period_result_fields(
                test_metrics,
                "test",
                aggregate_data=test_aggregate_data,
            )
        )

    add_is = ask_yes_no("Add IS metrics now? y/n: ", allow_stop=allow_stop)
    if add_is is STOP_BATCH:
        return STOP_BATCH
    if add_is:
        is_entry = ask_aggregate_data_metrics(period="is", allow_stop=allow_stop, include_raw=True)
        if is_entry is STOP_BATCH:
            return STOP_BATCH
        if is_entry is None:
            return None
        is_metrics, is_aggregate_data = is_entry
        result_fields.update(
            build_period_result_fields(
                is_metrics,
                "is",
                aggregate_data=is_aggregate_data,
            )
        )

    rating_prompt = "BRAIN rating [Spectacular / Excellent / Good / Average / Needs Improvement]: "
    brain_rating = ask_brain_rating(rating_prompt, allow_stop=allow_stop)
    if brain_rating is STOP_BATCH:
        return STOP_BATCH
    if brain_rating is None:
        return None

    bo_score = compute_bo_score(
        train_score=result_fields.get("train_score"),
        test_score=result_fields.get("test_score"),
    )
    result_fields["bo_score"] = bo_score

    return {
        **candidate,
        **result_fields,
        "brain_rating": brain_rating,
    }


def ask_user_for_metrics(params, user, universe=FIXED_UNIVERSE):
    """Backward-compatible single-candidate prompt helper."""
    candidate = build_candidate(params=params, user=user, universe=universe)
    print_candidate(candidate)
    return collect_metrics_for_candidate(candidate)


def suggest_random_candidate():
    return torch.rand(LATENT_DIM)


def suggest_botorch_candidate(train_x, train_y):
    """Fit a GP model over latent [0, 1]^d coordinates and suggest the next candidate."""
    train_y_std = standardize(train_y)

    model = SingleTaskGP(train_x, train_y_std)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    best_f = train_y_std.max()
    acqf = LogExpectedImprovement(model=model, best_f=best_f)

    candidate_norm, _ = optimize_acqf(
        acq_function=acqf,
        bounds=LATENT_BOUNDS,
        q=1,
        num_restarts=10,
        raw_samples=128,
    )

    return candidate_norm.detach().squeeze(0)


def run_one_trial(user, universe=FIXED_UNIVERSE, batch=1, seed_threshold=50):
    """Run one or more suggest → manual simulate → save cycles.

    Re-run this function anytime. Completed trials are appended to CSV immediately,
    and the CSV is reloaded automatically at the start of each call.

    Use different `user` and `universe` values to maintain separate BO campaigns, e.g.
    run_one_trial(user="Angze", universe="TOP3000")
    or run_one_trial(user="Devan", universe="TOP1000").
    """
    universe = normalise_universe(universe)
    batch = int(batch)
    if batch < 1:
        raise ValueError("batch must be at least 1.")

    user_slug = normalise_user_name(user)

    log_path = get_log_path(user=user_slug, universe=universe, region=FIXED_REGION)

    results, train_x, train_y = load_existing_results(user=user_slug, log_path=log_path, universe=universe)

    n_existing = 0 if train_x is None else train_x.shape[0]
    print(f"Loaded {n_existing} existing completed trials from {log_path}.")

    if train_x is None or train_x.shape[0] < seed_threshold:
        print(f"Using random candidates because fewer than {seed_threshold} trials are available.")
    else:
        print("Using BoTorch LogExpectedImprovement candidates.")

    candidates = []
    used_keys = set()
    for result in results:
        try:
            used_keys.add(candidate_key(result["params"]))
        except (KeyError, ValueError, TypeError, SyntaxError):
            pass

    for index in range(1, batch + 1):
        candidate = suggest_one_candidate(
            user=user_slug,
            universe=universe,
            train_x=train_x,
            train_y=train_y,
            seed_threshold=seed_threshold,
            used_keys=used_keys,
        )
        candidate["batch_candidate_id"] = index
        used_keys.add(candidate_key(candidate["params"]))
        candidates.append(candidate)

    for index, candidate in enumerate(candidates, start=1):
        print_candidate(
            candidate,
            index=index if batch > 1 else None,
            batch=batch if batch > 1 else None,
        )

    saved_results = []
    for index, candidate in enumerate(candidates, start=1):
        completed_result = collect_metrics_for_candidate(
            candidate,
            index=index if batch > 1 else None,
            batch=batch if batch > 1 else None,
            allow_stop=batch > 1,
        )

        if completed_result is STOP_BATCH:
            print("Batch stopped. Remaining candidates were not saved.")
            break

        if completed_result is None:
            if batch > 1:
                print(f"Candidate {index} skipped. Nothing was saved for this candidate.")
            else:
                print("Trial cancelled. Nothing was saved.")
            continue

        completed_result = append_result_to_csv(
            completed_result,
            user=user_slug,
            log_path=log_path,
            universe=universe,
        )
        saved_results.append(completed_result)
        print(f"Saved candidate {index} to {log_path}.")

    if batch == 1:
        return saved_results[0] if saved_results else None

    return saved_results


def run_batch_trials(user, universe=FIXED_UNIVERSE, batch=1, seed_threshold=50):
    """Run a batch of human-in-the-loop BO candidates."""
    return run_one_trial(
        user=user,
        universe=universe,
        batch=batch,
        seed_threshold=seed_threshold,
    )
