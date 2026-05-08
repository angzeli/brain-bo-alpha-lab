# BRAIN Operator Playbook

## Why this document exists

This is a project-specific guide to the BRAIN operators we use when designing alpha templates.
It is not a replacement for the official WorldQuant BRAIN documentation.

The goal is to explain which operators matter for this repo, what alpha hypotheses they help us test, and which operators should be used now versus postponed.

Operators are grouped by how they help us build interpretable alpha expressions.

---

## 1. Core idea: operators as alpha-building blocks

Most generated alpha expressions combine:

- a raw data field, such as `close`, `high`, `low`, `vwap`, or `volume`;
- a time-series transformation, such as `ts_delta` or `ts_mean`;
- a cross-sectional transformation, such as `rank` or `zscore`;
- optional smoothing, normalisation, or conditional logic.

Example:

```text
rank(ts_delta(high, 30) / ts_std_dev(high, 80))
```

Interpretation:

- `ts_delta(high, 30)` measures high-price momentum.
- `ts_std_dev(high, 80)` normalises by recent high-price volatility.
- `rank(...)` compares stocks cross-sectionally on the same day.

This is the pattern we want: simple enough to inspect, structured enough for Bayesian optimisation to tune.

---

## 2. Safe core operators

These operators are useful for the current template library.

### Arithmetic

Arithmetic operators combine or reshape signals:

- `add(x, y)` / `x + y`
- `subtract(x, y)` / `x - y`
- `multiply(x, y)` / `x * y`
- `divide(x, y)` / `x / y`
- `reverse(x)` / `-x`
- `abs(x)`
- `log(x)` — use carefully, only when the input is positive
- `sqrt(x)` — use carefully, only when the input is non-negative
- `signed_power(x, y)` — useful for sign-preserving transformations

Examples:

```text
rank(ts_delta(close, 20) / ts_std_dev(close, 60))
```

```text
rank(-ts_delta(close, 20))
```

### Time-series operators

Time-series operators compare a stock with its own history.

#### `ts_delta(x, d)`

Use for momentum and reversal.

```text
rank(ts_delta(close, 20))
```

#### `ts_mean(x, d)`

Use for smoothing, rolling averages, and volume ratios.

```text
rank(ts_mean(volume, 20) / ts_mean(volume, 60))
```

#### `ts_std_dev(x, d)`

Use for volatility and volatility normalisation.

```text
rank(ts_delta(high, 30) / ts_std_dev(high, 80))
```

#### `ts_zscore(x, d)`

Use for surprise or deviation from recent history.

```text
rank(ts_zscore(volume, 60))
```

#### `ts_rank(x, d)`

Use for "where is today relative to recent history?"

```text
rank(ts_rank(close, 60))
```

#### `ts_scale(x, d)`

Use as the replacement for unavailable `ts_min` / `ts_max`.

```text
rank(ts_scale(close, 60))
```

#### `ts_av_diff(x, d)`

Use for deviation from a recent average.

```text
rank(ts_av_diff(close, 30))
```

#### `ts_decay_linear(x, d)`

Use for smoothing and reducing noise.

```text
rank(ts_decay_linear(ts_delta(high, 30), 5))
```

#### `ts_corr(x, y, d)`

Use for price-volume interaction or relationship-style hypotheses.

```text
rank(ts_corr(close, volume, 60))
```

### Cross-sectional operators

Cross-sectional operators compare stocks with other stocks on the same day.

#### `rank(x)`

This is the most robust default transform.

```text
rank(ts_delta(close, 20))
```

#### `zscore(x)`

Useful when relative magnitude matters.

```text
zscore(ts_delta(high, 30) / ts_std_dev(high, 80))
```

#### `scale(x)`

Useful for exposure scaling.

```text
scale(ts_mean(volume, 20) / ts_mean(volume, 60))
```

#### `normalize(x)`

Useful for market-centering.

```text
normalize(ts_delta(close, 20))
```

#### `winsorize(x)`

Useful for reducing extreme outliers.

```text
rank(winsorize(ts_zscore(volume, 60), std=4))
```

---

## 3. Template families enabled by these operators

This section connects operators to the alpha families used in this repo.

### Price momentum

```text
rank(ts_delta(high, n) / ts_std_dev(high, m))
```

Hypothesis:

Stocks making stronger recent highs, adjusted by volatility, may continue to outperform.

### Price reversion

```text
rank(-ts_delta(close, n))
```

Hypothesis:

Stocks that moved up too much may revert.

### Low volatility

```text
rank(-ts_std_dev(close, n))
```

Hypothesis:

Lower-volatility stocks may have more stable performance.

### Volume ratio

```text
rank(ts_mean(volume, n) / ts_mean(volume, m))
```

Hypothesis:

Recent trading activity relative to longer-term activity may contain information.

### Volume surprise

```text
rank(ts_zscore(volume, n))
```

Hypothesis:

Unusual volume may indicate attention or information flow.

### Price-volume interaction

```text
rank(ts_delta(close, n) * (volume / ts_mean(volume, m)))
```

Hypothesis:

Price moves confirmed by abnormal volume may be more meaningful.

### Time-series position

```text
rank(ts_rank(close, n))
```

```text
rank(ts_scale(close, n))
```

Hypothesis:

A stock's position relative to its recent history may indicate breakout or reversal behaviour.

### Price-volume correlation

```text
rank(ts_corr(close, volume, n))
```

Hypothesis:

The recent relationship between price and volume may capture accumulation/distribution behaviour.

---

## 4. Operators to use carefully

These operators are useful, but they add complexity. Use them only when they support a clear hypothesis.

### `hump(x, hump=0.01)`

- Useful for reducing turnover.
- Can smooth alpha changes.
- May also weaken the signal.

Example:

```text
hump(rank(ts_delta(close, 20)), hump=0.01)
```

### `trade_when(x, y, z)`

- Useful for conditional trading.
- Can reduce turnover or restrict trading to active regimes.
- More complex than ordinary transforms, so test carefully.

Example idea:

```text
trade_when(volume > ts_mean(volume, 60), rank(ts_delta(close, 20)), -1)
```

Do not add `trade_when` blindly to every template.

### `ts_arg_max(x, d)` and `ts_arg_min(x, d)`

These return how many days ago the recent max or min occurred.

- Useful for breakout or recent-extreme logic.
- Interpretation is slightly unintuitive because `0` means today.

Examples:

```text
rank(-ts_arg_max(high, 60))
```

```text
rank(ts_arg_min(low, 60))
```

### Group operators

Examples:

- `group_rank(x, group)`
- `group_zscore(x, group)`
- `group_neutralize(x, group)`

These can compare stocks within sector, industry, or subindustry. Use them only after confirming valid group-field syntax in BRAIN.

Do not confuse the platform-level Neutralisation setting with explicit group operators inside the alpha expression.

Example idea:

```text
group_rank(ts_delta(close, 20), sector)
```

The group-field syntax must be tested before using this in batch generation.

---

## 5. Operators to postpone

These operators are valid, but they are not priorities for the current workflow skeleton:

- `ts_regression(...)`
- `bucket(...)`
- `group_backfill(...)`
- `group_mean(...)`
- vector operators like `vec_avg(...)` and `vec_sum(...)`

Reasons:

- more complex syntax;
- higher risk of invalid expressions;
- need specific fields or careful interpretation;
- not necessary for the current BO workflow.

---

## 6. Operators to avoid for now

Avoid:

```text
ts_min
ts_max
```

Reason:

They caused BRAIN parser errors in the current environment.

Use `ts_scale(x, d)` or `ts_rank(x, d)` instead.

Instead of:

```text
(close - ts_min(close, 60)) / (ts_max(close, 60) - ts_min(close, 60))
```

use:

```text
ts_scale(close, 60)
```

or:

```text
ts_rank(close, 60)
```

---

## 7. Current project-safe operator set

This is the current safest set for generated templates:

- `rank`
- `zscore`
- `scale`
- `normalize`
- `winsorize`
- `reverse`
- `ts_delta`
- `ts_mean`
- `ts_std_dev`
- `ts_zscore`
- `ts_rank`
- `ts_scale`
- `ts_av_diff`
- `ts_decay_linear`
- `ts_corr`
- `hump`

This list can expand as templates are tested successfully.

---

## 8. Design principles for adding new templates

1. Start from a hypothesis, not from an operator.
2. Prefer simple expressions first.
3. Add one source of complexity at a time.
4. Avoid unsupported or untested operators in batch generation.
5. Track whether a template improves TRAIN, TEST, and IS consistency.
6. Do not optimise for beautiful formulas; optimise for reproducible experimental learning.

Bad:

```text
Let's use every operator because it exists.
```

Good:

```text
Volume ratio showed some weak signal, so we test smoothed and inverse volume-ratio variants.
```

---

## 9. How this connects to BO

Bayesian optimisation does not understand financial meaning directly.

BO only sees:

```text
params -> bo_score
```

That means template design matters.

The operator playbook defines sensible template families. BO then tunes parameters such as:

- lookback windows;
- transform;
- neutralisation;
- decay;
- truncation;
- direction.

This keeps the search space structured instead of random.
