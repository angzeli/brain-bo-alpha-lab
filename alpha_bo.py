import ast
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
    "range_position",
    "time_series_rank",
    "short_long_trend",
    "volume_surprise",
    "price_volume_momentum",
    "price_volume_reversal",
    "intraday_position",
]
PRICE_FIELDS = ["close", "open", "high", "low", "vwap"]
TRANSFORMS = ["rank", "zscore", "scale"]
NEUTRALISATIONS = ["None", "Market", "Sector", "Industry", "Subindustry"]
BOOLEAN_SETTINGS = ["Off", "On"]

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
    "sharpe",
    "turnover_pct",
    "fitness",
    "returns_pct",
    "drawdown_pct",
    "margin_permyriad",
    "brain_rating",
    "score",
]

TEMPLATE_CATEGORY_MAP = {
    "price_momentum": "price momentum",
    "price_reversion": "price reversion",
    "low_volatility": "price reversion",
    "volume_ratio": "volume",
    "range_position": "price momentum",
    "time_series_rank": "price momentum",
    "short_long_trend": "price momentum",
    "volume_surprise": "volume",
    "price_volume_momentum": "price volume",
    "price_volume_reversal": "price volume",
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
LATENT_DIM = 10

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
# Region, universe, delay, unit handling, and test period are fixed to keep scores comparable across trials.
LATENT_BOUNDS = torch.tensor(
    [[0.0] * LATENT_DIM, [1.0] * LATENT_DIM],
    dtype=torch.double,
)


def canonicalise_params(params):
    """Normalise old/new saved params into the current 10-parameter schema.

    Current schema:
    [n, m, decay, truncation, template_type, price_field, transform, neutralisation,
     pasteurisation, nan_handling]

    Region, universe, delay, unit handling, and test period are intentionally fixed
    outside the optimised params so the BO objective is comparable across trials.
    """
    if isinstance(params, str):
        params = ast.literal_eval(params)

    # Old 3-parameter schema: [n, m, signal/template_type]
    if len(params) == 3:
        return [
            int(params[0]), int(params[1]), 1, 0.01,
            canonicalise_template_type(params[2]), "close", "rank", "None", "On", "Off",
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
            neutralisation, "On", "Off",
        ]

    # Previous 13-parameter schema:
    # [n, m, decay, truncation, signal/template_type, price_field, transform,
    #  neutralisation, universe, pasteurisation, nan_handling,
    #  test_years, test_months]
    if len(params) == 13:
        return [
            int(params[0]), int(params[1]), int(params[2]), float(params[3]),
            canonicalise_template_type(params[4]), str(params[5]), str(params[6]), str(params[7]),
            str(params[9]), str(params[10]),
        ]

    # Current 10-parameter schema without delay.
    if len(params) == 10:
        return [
            int(params[0]), int(params[1]), int(params[2]), float(params[3]),
            canonicalise_template_type(params[4]), str(params[5]), str(params[6]), str(params[7]),
            str(params[8]), str(params[9]),
        ]

    # Previous 11-parameter schema with optimised delay. Drop saved delay and use FIXED_DELAY.
    if len(params) == 11:
        return [
            int(params[0]), int(params[1]), int(params[3]), float(params[4]),
            canonicalise_template_type(params[5]), str(params[6]), str(params[7]), str(params[8]),
            str(params[9]), str(params[10]),
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
        neutralisation, pasteurisation, nan_handling,
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
        neutralisation, pasteurisation, nan_handling,
    ) = canonicalise_params(params)

    if template_type == "price_momentum":
        return f"ts_delta({price_field}, {n}) / ts_std_dev({price_field}, {m})"
    if template_type == "price_reversion":
        return f"-ts_delta({price_field}, {n}) / ts_std_dev({price_field}, {m})"
    if template_type == "low_volatility":
        return f"-ts_std_dev({price_field}, {n})"
    if template_type == "volume_ratio":
        return f"ts_mean(volume, {n}) / ts_mean(volume, {m})"
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
    if template_type == "intraday_position":
        return "(close - open) / (high - low)"

    raise ValueError(f"Unknown template_type: {template_type}")


def make_alpha(params, universe=FIXED_UNIVERSE):
    universe = normalise_universe(universe)
    (
        n, m, decay, truncation, template_type, price_field, transform,
        neutralisation, pasteurisation, nan_handling,
    ) = canonicalise_params(params)

    base_expression = build_base_expression(params)
    transformed_expression = apply_transform(base_expression, transform)
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
        neutralisation, pasteurisation, nan_handling,
    ) = canonicalise_params(params)

    alpha_category = TEMPLATE_CATEGORY_MAP[template_type]
    short_window, long_window = trend_windows(n, m)

    if template_type == "price_momentum":
        alpha_name = f"mom_{price_field}_{n}_volnorm_{m}"
        uses = f"Uses a {n}-day {price_field} price change normalised by {m}-day {price_field} volatility"
    elif template_type == "price_reversion":
        alpha_name = f"rev_{price_field}_{n}_volnorm_{m}"
        uses = f"Uses a negative {n}-day {price_field} price change normalised by {m}-day {price_field} volatility"
    elif template_type == "low_volatility":
        alpha_name = f"low_vol_{price_field}_{n}_{transform}"
        uses = f"Uses negative {n}-day {price_field} volatility"
    elif template_type == "volume_ratio":
        alpha_name = f"vol_ratio_{n}_{m}_{transform}"
        uses = f"Uses the ratio of {n}-day average volume to {m}-day average volume"
    elif template_type == "range_position":
        alpha_name = f"range_scale_{price_field}_{n}_{transform}"
        uses = f"Uses the time-series scaled position of {price_field} over a {n}-day lookback"
    elif template_type == "time_series_rank":
        alpha_name = f"ts_rank_{price_field}_{n}_{transform}"
        uses = f"Uses the {n}-day time-series rank of {price_field}"
    elif template_type == "short_long_trend":
        alpha_name = f"trend_{price_field}_{short_window}_{long_window}_{transform}"
        uses = f"Uses the ratio of {short_window}-day to {long_window}-day average {price_field} levels"
    elif template_type == "volume_surprise":
        alpha_name = f"vol_surprise_{n}_{m}_{transform}"
        uses = f"Uses current volume versus its {n}-day average, normalised by {m}-day volume volatility"
    elif template_type == "price_volume_momentum":
        alpha_name = f"pv_mom_{price_field}_{n}_{m}_{transform}"
        uses = f"Uses {n}-day {price_field} price change scaled by relative volume versus its {m}-day average"
    elif template_type == "price_volume_reversal":
        alpha_name = f"pv_rev_{price_field}_{n}_{m}_{transform}"
        uses = f"Uses negative {n}-day {price_field} price change scaled by relative volume versus its {m}-day average"
    elif template_type == "intraday_position":
        alpha_name = f"intraday_pos_{transform}"
        uses = "Uses where the close sits relative to the open inside the daily high-low range"
    else:
        raise ValueError(f"Unknown template_type: {template_type}")

    alpha_description = (
        f"{uses}, transformed with {transform}. "
        f"Tested with Delay{FIXED_DELAY}, {neutralisation} neutralisation, Decay {decay}, "
        f"Truncation {truncation}, Pasteurisation {pasteurisation}, "
        f"NaN handling {nan_handling}, Universe {universe}."
    )

    return {
        "alpha_name": alpha_name,
        "alpha_category": alpha_category,
        "alpha_description": alpha_description,
    }


def compute_score(sharpe, turnover_pct, fitness, returns_pct, drawdown_pct, margin_permyriad):
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


def recompute_scores_in_csv(csv_path, output_csv=None, backup=True, backup_path=None, dry_run=False):
    """
    Recompute saved score values for rows with the full Aggregate Data metric set.

    This is an explicit one-time migration helper. Normal BO resume/loading still
    uses the score already saved in the CSV and does not call this function.
    """
    csv_path = Path(csv_path)
    output_csv = csv_path if output_csv is None else Path(output_csv)

    required_metrics = [
        "sharpe",
        "turnover_pct",
        "fitness",
        "returns_pct",
        "drawdown_pct",
        "margin_permyriad",
    ]

    df = pd.read_csv(csv_path)
    missing_columns = [column for column in required_metrics if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{csv_path} is missing metric column(s) needed for score recompute: {missing_columns}"
        )

    numeric_metrics = {
        column: pd.to_numeric(df[column], errors="coerce")
        for column in required_metrics
    }
    eligible_mask = pd.DataFrame(numeric_metrics).notna().all(axis=1)

    updated_scores = []
    for idx in df.index[eligible_mask]:
        updated_scores.append(
            compute_score(
                sharpe=float(numeric_metrics["sharpe"].loc[idx]),
                turnover_pct=float(numeric_metrics["turnover_pct"].loc[idx]),
                fitness=float(numeric_metrics["fitness"].loc[idx]),
                returns_pct=float(numeric_metrics["returns_pct"].loc[idx]),
                drawdown_pct=float(numeric_metrics["drawdown_pct"].loc[idx]),
                margin_permyriad=float(numeric_metrics["margin_permyriad"].loc[idx]),
            )
        )

    updated_df = df.copy()
    if "score" not in updated_df.columns:
        updated_df["score"] = pd.NA
    updated_df.loc[eligible_mask, "score"] = updated_scores

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
        "rows_updated": int(eligible_mask.sum()),
        "rows_skipped": int((~eligible_mask).sum()),
        "dry_run": bool(dry_run),
    }

    print(
        f"Recomputed scores for {summary['rows_updated']} row(s); "
        f"skipped {summary['rows_skipped']} row(s)."
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

    missing_columns = {"params", "score"} - set(df.columns)
    if missing_columns:
        raise ValueError(f"{log_path} is missing required column(s): {sorted(missing_columns)}")

    existing_results = df.to_dict("records")
    train_x = []
    train_y = []
    skipped_rows = 0

    for _, row in df.iterrows():
        try:
            params = canonicalise_params(row["params"])
            score = float(row["score"])
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
        print("\nAfter the simulation finishes, enter the Aggregate Data metrics:")
        print("Press Enter on an empty prompt to cancel without saving this trial.")
    else:
        print(f"\nEnter metrics for candidate {index}/{batch}: {candidate['alpha_name']}")
        print("\nAlpha expression:")
        print(candidate["alpha"])
        stop_hint = " Type stop at any prompt to end the remaining batch." if allow_stop else ""
        print(f"Press Enter on an empty prompt to skip this candidate without saving it.{stop_hint}")

    sharpe = ask_float("Sharpe: ", allow_stop=allow_stop)
    if sharpe is STOP_BATCH:
        return STOP_BATCH
    if sharpe is None:
        return None

    turnover_pct = ask_float("Turnover (%): ", allow_stop=allow_stop)
    if turnover_pct is STOP_BATCH:
        return STOP_BATCH
    if turnover_pct is None:
        return None

    fitness = ask_float("Fitness: ", allow_stop=allow_stop)
    if fitness is STOP_BATCH:
        return STOP_BATCH
    if fitness is None:
        return None

    returns_pct = ask_float("Returns (%): ", allow_stop=allow_stop)
    if returns_pct is STOP_BATCH:
        return STOP_BATCH
    if returns_pct is None:
        return None

    drawdown_pct = ask_float("Drawdown (%): ", allow_stop=allow_stop)
    if drawdown_pct is STOP_BATCH:
        return STOP_BATCH
    if drawdown_pct is None:
        return None

    margin_permyriad = ask_float("Margin (‱): ", allow_stop=allow_stop)
    if margin_permyriad is STOP_BATCH:
        return STOP_BATCH
    if margin_permyriad is None:
        return None

    rating_prompt = "BRAIN rating [Spectacular / Excellent / Good / Average / Needs Improvement]: "
    brain_rating = ask_brain_rating(rating_prompt, allow_stop=allow_stop)
    if brain_rating is STOP_BATCH:
        return STOP_BATCH
    if brain_rating is None:
        return None

    score = compute_score(
        sharpe=sharpe,
        turnover_pct=turnover_pct,
        fitness=fitness,
        returns_pct=returns_pct,
        drawdown_pct=drawdown_pct,
        margin_permyriad=margin_permyriad,
    )

    return {
        **candidate,
        "sharpe": sharpe,
        "turnover_pct": turnover_pct,
        "fitness": fitness,
        "returns_pct": returns_pct,
        "drawdown_pct": drawdown_pct,
        "margin_permyriad": margin_permyriad,
        "brain_rating": brain_rating,
        "score": score,
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
