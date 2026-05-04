import ast
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize, standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

FIXED_REGION = "USA"
FIXED_UNIVERSE = "TOP3000"
FIXED_UNIT_HANDLING = "Verify"
FIXED_TEST_YEARS = 1
FIXED_TEST_MONTHS = 0
FIXED_DELAY = 1

UNIVERSES = ["TOP3000", "TOP1000", "TOP500", "TOP200"]
SIGNAL_TYPES = ["momentum", "reversal", "volume", "volatility"]
PRICE_FIELDS = ["close", "open", "high", "low", "vwap"]
TRANSFORMS = ["rank", "zscore", "scale"]
NEUTRALISATIONS = ["None", "Market", "Sector", "Industry", "Subindustry"]
BOOLEAN_SETTINGS = ["Off", "On"]

CSV_COLUMN_ORDER = [
    "run_timestamp",
    "user",
    "region",
    "universe",
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
    "passed",
    "score",
]

ALPHA_CATEGORY_BY_SIGNAL = {
    "momentum": "price momentum",
    "reversal": "price reversion",
    "volume": "volume",
    "volatility": "price reversion",
}


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

# Search dimensions:
# x[0] = n, primary lookback window
# x[1] = m, secondary / normalisation lookback window
# x[2] = decay setting
# x[3] = truncation level
# x[4] = signal_type index
# x[5] = price field index
# x[6] = transform index
# x[7] = neutralisation index
# x[8] = pasteurisation index: 0 Off, 1 On
# x[9] = NaN handling index: 0 Off, 1 On
# Region, universe, delay, unit handling, and test period are fixed to keep scores comparable across trials.
BOUNDS = torch.tensor([
    [3.0, 3.0, 1.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [120.0, 120.0, 60.0, 0.20, 3.0, 4.0, 2.0, 4.0, 1.0, 1.0],
])


def canonicalise_params(params):
    """Normalise old/new saved params into the current 10-parameter schema.

    Current schema:
    [n, m, decay, truncation, signal_type, price_field, transform, neutralisation,
     pasteurisation, nan_handling]

    Region, universe, delay, unit handling, and test period are intentionally fixed
    outside the optimised params so the BO objective is comparable across trials.
    """
    if isinstance(params, str):
        params = ast.literal_eval(params)

    # Old 3-parameter schema: [n, m, signal_type]
    if len(params) == 3:
        return [
            int(params[0]), int(params[1]), 1, 0.01,
            str(params[2]), "close", "rank", "None", "On", "Off",
        ]

    # Previous 9-parameter schema:
    # [n, m, decay, truncation, signal_type, price_field, transform,
    #  neutralisation, universe]
    if len(params) == 9:
        neutralisation = str(params[7])
        neutralisation = "None" if neutralisation.lower() == "none" else neutralisation.capitalize()

        return [
            int(params[0]), int(params[1]), int(params[2]), float(params[3]),
            str(params[4]), str(params[5]), str(params[6]),
            neutralisation, "On", "Off",
        ]

    # Previous 13-parameter schema:
    # [n, m, decay, truncation, signal_type, price_field, transform,
    #  neutralisation, universe, pasteurisation, nan_handling,
    #  test_years, test_months]
    if len(params) == 13:
        return [
            int(params[0]), int(params[1]), int(params[2]), float(params[3]),
            str(params[4]), str(params[5]), str(params[6]), str(params[7]),
            str(params[9]), str(params[10]),
        ]

    # Current 10-parameter schema without delay.
    if len(params) == 10:
        return [
            int(params[0]), int(params[1]), int(params[2]), float(params[3]),
            str(params[4]), str(params[5]), str(params[6]), str(params[7]),
            str(params[8]), str(params[9]),
        ]

    # Previous 11-parameter schema with optimised delay. Drop saved delay and use FIXED_DELAY.
    if len(params) == 11:
        return [
            int(params[0]), int(params[1]), int(params[3]), float(params[4]),
            str(params[5]), str(params[6]), str(params[7]), str(params[8]),
            str(params[9]), str(params[10]),
        ]

    raise ValueError(f"Unsupported params format with length {len(params)}: {params}")


def decode_params(x):
    """Convert a continuous BoTorch candidate into valid alpha parameters."""
    n = int(round(float(x[0])))
    m = int(round(float(x[1])))
    decay = int(round(float(x[2])))
    truncation = round(float(x[3]), 4)

    signal_idx = int(round(float(x[4])))
    field_idx = int(round(float(x[5])))
    transform_idx = int(round(float(x[6])))
    neutralisation_idx = int(round(float(x[7])))
    pasteurisation_idx = int(round(float(x[8])))
    nan_handling_idx = int(round(float(x[9])))

    n = min(max(n, 3), 120)
    m = min(max(m, 3), 120)
    decay = min(max(decay, 1), 60)
    truncation = min(max(truncation, 0.001), 0.20)

    signal_idx = min(max(signal_idx, 0), len(SIGNAL_TYPES) - 1)
    field_idx = min(max(field_idx, 0), len(PRICE_FIELDS) - 1)
    transform_idx = min(max(transform_idx, 0), len(TRANSFORMS) - 1)
    neutralisation_idx = min(max(neutralisation_idx, 0), len(NEUTRALISATIONS) - 1)
    pasteurisation_idx = min(max(pasteurisation_idx, 0), len(BOOLEAN_SETTINGS) - 1)
    nan_handling_idx = min(max(nan_handling_idx, 0), len(BOOLEAN_SETTINGS) - 1)

    return [
        n, m, decay, truncation,
        SIGNAL_TYPES[signal_idx],
        PRICE_FIELDS[field_idx],
        TRANSFORMS[transform_idx],
        NEUTRALISATIONS[neutralisation_idx],
        BOOLEAN_SETTINGS[pasteurisation_idx],
        BOOLEAN_SETTINGS[nan_handling_idx],
    ]


def encode_params(params):
    """Convert stored alpha parameters into numerical BO coordinates."""
    (
        n, m, decay, truncation, signal_type, price_field, transform,
        neutralisation, pasteurisation, nan_handling,
    ) = canonicalise_params(params)

    return [
        float(n),
        float(m),
        float(decay),
        float(truncation),
        float(SIGNAL_TYPES.index(signal_type)),
        float(PRICE_FIELDS.index(price_field)),
        float(TRANSFORMS.index(transform)),
        float(NEUTRALISATIONS.index(neutralisation)),
        float(BOOLEAN_SETTINGS.index(pasteurisation)),
        float(BOOLEAN_SETTINGS.index(nan_handling)),
    ]


def apply_transform(expression, transform):
    if transform == "rank":
        return f"rank({expression})"
    if transform == "zscore":
        return f"zscore({expression})"
    if transform == "scale":
        return f"scale({expression})"
    raise ValueError(f"Unknown transform: {transform}")


def make_alpha(params, universe=FIXED_UNIVERSE):
    universe = normalise_universe(universe)
    (
        n, m, decay, truncation, signal_type, price_field, transform,
        neutralisation, pasteurisation, nan_handling,
    ) = canonicalise_params(params)

    if signal_type == "momentum":
        base_expression = f"ts_delta({price_field}, {n}) / ts_std_dev({price_field}, {m})"
    elif signal_type == "reversal":
        base_expression = f"-ts_delta({price_field}, {n}) / ts_std_dev({price_field}, {m})"
    elif signal_type == "volume":
        base_expression = f"ts_mean(volume, {n}) / ts_mean(volume, {m})"
    elif signal_type == "volatility":
        base_expression = f"-ts_std_dev({price_field}, {n})"
    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")

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
        n, m, decay, truncation, signal_type, price_field, transform,
        neutralisation, pasteurisation, nan_handling,
    ) = canonicalise_params(params)

    alpha_category = ALPHA_CATEGORY_BY_SIGNAL[signal_type]

    if signal_type == "momentum":
        alpha_name = f"mom_{price_field}_{n}_volnorm_{m}"
        uses = f"Uses a {n}-day {price_field} price change normalised by {m}-day {price_field} volatility"
    elif signal_type == "reversal":
        alpha_name = f"rev_{price_field}_{n}_volnorm_{m}"
        uses = f"Uses a negative {n}-day {price_field} price change normalised by {m}-day {price_field} volatility"
    elif signal_type == "volume":
        alpha_name = f"vol_surge_{n}_{m}_{transform}"
        uses = f"Uses the ratio of {n}-day average volume to {m}-day average volume"
    elif signal_type == "volatility":
        alpha_name = f"low_vol_{price_field}_{n}_{transform}"
        uses = f"Uses negative {n}-day {price_field} volatility"
    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")

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


def compute_score(sharpe, turnover_pct, fitness, returns_pct, drawdown_pct, margin_permyriad, passed):
    score = fitness + 0.5 * sharpe - 0.002 * turnover_pct
    if not passed:
        score -= 2
    return score


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
            train_x.append(encode_params(params))
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


def ask_float(prompt):
    """Ask for a float. Empty input cleanly cancels the current trial instead of crashing."""
    while True:
        value = input(prompt).strip()
        if value == "":
            return None
        try:
            return float(value)
        except ValueError:
            print("Please enter a number, or press Enter to cancel this trial.")


def ask_bool(prompt):
    """Ask for a yes/no value. Empty input cleanly cancels the current trial."""
    while True:
        value = input(prompt).strip().lower()
        if value == "":
            return None
        if value in {"y", "yes", "true", "1"}:
            return True
        if value in {"n", "no", "false", "0"}:
            return False
        print("Please enter y/n, or press Enter to cancel this trial.")


def ask_user_for_metrics(params, user, universe=FIXED_UNIVERSE):
    universe = normalise_universe(universe)
    alpha, settings = make_alpha(params, universe=universe)
    metadata = build_alpha_metadata(params, universe=universe)

    print("\nBO candidate")
    print("-" * 50)
    print("\nAlpha expression:")
    print(alpha)
    print("\nSuggested settings:")
    print(settings)
    print("")
    print("Suggested alpha name:")
    print(metadata["alpha_name"])
    print("\nCategory:")
    print(metadata["alpha_category"])
    print("\nDescription:")
    print(metadata["alpha_description"])
    print("\nAfter the simulation finishes, enter the Aggregate Data metrics:")
    print("Press Enter on an empty prompt to cancel without saving this trial.")

    sharpe = ask_float("Sharpe: ")
    if sharpe is None:
        return None

    turnover_pct = ask_float("Turnover (%): ")
    if turnover_pct is None:
        return None

    fitness = ask_float("Fitness: ")
    if fitness is None:
        return None

    returns_pct = ask_float("Returns (%): ")
    if returns_pct is None:
        return None

    drawdown_pct = ask_float("Drawdown (%): ")
    if drawdown_pct is None:
        return None

    margin_permyriad = ask_float("Margin (‱): ")
    if margin_permyriad is None:
        return None

    passed = ask_bool("Passed? y/n: ")
    if passed is None:
        return None

    score = compute_score(
        sharpe=sharpe,
        turnover_pct=turnover_pct,
        fitness=fitness,
        returns_pct=returns_pct,
        drawdown_pct=drawdown_pct,
        margin_permyriad=margin_permyriad,
        passed=passed,
    )
    alpha, settings = make_alpha(params, universe=universe)

    return {
        **metadata,
        "alpha": alpha,
        "settings": settings,
        "region": FIXED_REGION,
        "universe": universe,
        "user": normalise_user_name(user),
        "params": canonicalise_params(params),
        "sharpe": sharpe,
        "turnover_pct": turnover_pct,
        "fitness": fitness,
        "returns_pct": returns_pct,
        "drawdown_pct": drawdown_pct,
        "margin_permyriad": margin_permyriad,
        "passed": passed,
        "score": score,
    }


def suggest_random_candidate():
    candidate = torch.empty(10)
    candidate[0] = torch.randint(3, 121, size=(1,)).item()
    candidate[1] = torch.randint(3, 121, size=(1,)).item()
    candidate[2] = torch.randint(1, 61, size=(1,)).item()
    candidate[3] = torch.empty(1).uniform_(0.001, 0.20).item()
    candidate[4] = torch.randint(0, len(SIGNAL_TYPES), size=(1,)).item()
    candidate[5] = torch.randint(0, len(PRICE_FIELDS), size=(1,)).item()
    candidate[6] = torch.randint(0, len(TRANSFORMS), size=(1,)).item()
    candidate[7] = torch.randint(0, len(NEUTRALISATIONS), size=(1,)).item()
    candidate[8] = torch.randint(0, len(BOOLEAN_SETTINGS), size=(1,)).item()
    candidate[9] = torch.randint(0, len(BOOLEAN_SETTINGS), size=(1,)).item()
    return candidate


def suggest_botorch_candidate(train_x, train_y):
    """Fit a GP model and suggest the next candidate using LogEI."""
    train_x_norm = normalize(train_x, BOUNDS)
    train_y_std = standardize(train_y)

    model = SingleTaskGP(train_x_norm, train_y_std)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    best_f = train_y_std.max()
    acqf = LogExpectedImprovement(model=model, best_f=best_f)

    candidate_norm, _ = optimize_acqf(
        acq_function=acqf,
        bounds=torch.stack([torch.zeros(10), torch.ones(10)]),
        q=1,
        num_restarts=10,
        raw_samples=128,
    )

    candidate = unnormalize(candidate_norm.detach().squeeze(0), BOUNDS)
    return candidate


def run_one_trial(user, universe=FIXED_UNIVERSE, seed_threshold=50):
    """Run one suggest → manual simulate → save cycle.

    Re-run this function anytime. Completed trials are appended to CSV immediately,
    and the CSV is reloaded automatically at the start of each call.

    Use different `user` and `universe` values to maintain separate BO campaigns, e.g.
    run_one_trial(user="Angze", universe="TOP3000")
    or run_one_trial(user="Devan", universe="TOP1000").
    """
    universe = normalise_universe(universe)

    user_slug = normalise_user_name(user)

    log_path = get_log_path(user=user_slug, universe=universe, region=FIXED_REGION)

    results, train_x, train_y = load_existing_results(user=user_slug, log_path=log_path, universe=universe)

    n_existing = 0 if train_x is None else train_x.shape[0]
    print(f"Loaded {n_existing} existing completed trials from {log_path}.")

    if train_x is None or train_x.shape[0] < seed_threshold:
        raw_candidate = suggest_random_candidate()
        print(f"Using random candidate because fewer than {seed_threshold} trials are available.")
    else:
        raw_candidate = suggest_botorch_candidate(train_x, train_y)
        print("Using BoTorch LogExpectedImprovement candidate.")

    params = decode_params(raw_candidate)
    new_result = ask_user_for_metrics(params, user=user_slug, universe=universe)

    if new_result is None:
        print("Trial cancelled. Nothing was saved.")
        return None

    new_result = append_result_to_csv(new_result, user=user_slug, log_path=log_path, universe=universe)
    print(f"Saved completed trial to {log_path}.")
    return new_result
