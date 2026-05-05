# 📝 BRAIN Alpha Parameters and Checks

## Why this document exists

This file helps understand the main WorldQuant BRAIN simulation outputs and how to record them in our local CSV workflow.

## 1. Key distinction in our workflow

It is useful to separate three ideas:

- `simulated`: the Alpha was run on BRAIN and recorded in our CSV.
- `passed`: the Alpha passed the relevant BRAIN checks.
- `submitted`: the user actually submitted the Alpha on BRAIN.

In this repo, a CSV row means the Alpha was simulated and logged. It does not automatically mean the Alpha passed BRAIN checks or was submitted.

The `passed` value in our Python prompt should be:

- `y` only if the relevant required BRAIN checks pass.
- `n` if one or more required checks fail.

Submission remains a separate manual action on BRAIN.

## 2. Aggregate Data metrics we record

Our Python prompt asks for BRAIN Aggregate Data metrics in this order:

```text
Sharpe:
Turnover (%):
Fitness:
Returns (%):
Drawdown (%):
Margin (‱):
Passed? y/n:
```

Enter numbers only. Do not type `%` or `‱` into numeric fields.

### Sharpe

Sharpe measures return while accounting for consistency. Higher Sharpe usually means the Alpha's daily PnL is more consistently positive relative to its volatility.

BRAIN also discusses Information Ratio, or IR:

```text
IR = mean daily PnL / standard deviation of daily PnL
```

Sharpe is the annualized version:

```text
Sharpe = sqrt(252) * IR ≈ 15.8 * IR
```

The `252` represents the approximate number of US trading days in a year. BRAIN definitions may differ from definitions used elsewhere, so use the values shown on BRAIN when recording results.

### Turnover (%)

Turnover measures how much the portfolio trades. Daily Turnover is the dollar trading volume divided by book size.

High turnover often means the Alpha changes positions too aggressively. BRAIN checks may require turnover to be above a minimum and below a maximum.

### Fitness

Fitness is a BRAIN quality metric based on Sharpe, Returns, and Turnover:

```text
Fitness = Sharpe * sqrt(abs(Returns) / max(Turnover, 0.125))
```

Good Alphas tend to have high Fitness. To improve Fitness, users generally try to:

- increase Sharpe,
- increase Returns,
- reduce Turnover.

These goals can conflict. For example, smoothing an Alpha may reduce turnover but also reduce returns.

### Returns (%)

Return is the gain or loss of a security or portfolio over a period.

In BRAIN:

```text
Return = annualized PnL / half of book size
```

Returns are shown in percentage terms.

### Drawdown (%)

Drawdown is the largest peak-to-trough reduction in PnL during a period, expressed as a percentage.

Lower drawdown is usually better because it means the Alpha has smaller losses from previous highs.

### Margin (‱)

Margin measures profit per dollar traded. In BRAIN terms, it is PnL divided by total dollars traded.

It is displayed in permyriad units, shown as `‱`. In our CSV we store this as `margin_permyriad`.

### Passed? y/n

This is our manual summary of BRAIN checks.

Enter `y` only if the Alpha passes the relevant required BRAIN checks. Enter `n` if any required check fails.

## 3. Fitness and BRAIN rating labels

The Stats tab in BRAIN Results includes a ratings panel. It labels an Alpha as:

- Spectacular
- Excellent
- Good
- Average
- Needs Improvement

These labels are based on Fitness. The exact thresholds can vary depending on context or table, so do not treat any single cutoff as universal.

As a practical guide:

- Fitness above roughly `2.5` or `3.25` may be very strong / Spectacular.
- Fitness above roughly `2` or `2.6` may be Excellent.
- Fitness above roughly `1.5` or `1.95` may be Good.
- Fitness above roughly `1` or `1.3` may be Average.
- Fitness below those levels usually Needs Improvement.

When BRAIN displays the rating directly, rely on the platform label.

## 4. PnL, drawdown, and robustness

The cumulative PnL chart shows Alpha performance over the full simulation.

Look for:

- an upward PnL trend,
- high Sharpe,
- low drawdown,
- no obvious instability or one-off jumps.

The chart can also show Sharpe over time. A strong Alpha should not depend only on one short lucky period.

BRAIN uses a constant book size of `$20 million`. Performance metrics such as Returns and Sharpe are computed on a base of `$10 million`, or half of book size.

Profit is not reinvested. Losses are replaced by cash injection into the portfolio. This keeps the simulation comparable over time.

## 5. IS, Test, Semi-OS, and OS

BRAIN separates time periods to reduce overfitting.

- `IS`: In-Sample.
- `Semi-OS`: Semi-Out-of-Sample.
- `OS`: Out-of-Sample.

The rolling 5-year IS simulation period begins seven years ago and ends two years ago, updating daily. Consultants may have access to a 10-year in-sample period instead of 5 years.

Users can divide IS into Train and Test using simulation settings:

- Train period: used for development.
- Test period: used for validation.

The latest two years of data are hidden as Semi-OS for scoring/testing. OS is also hidden from the user and helps evaluate robustness.

Practical takeaway: an Alpha that looks good only in Train may be overfit. Hidden Semi-OS / OS checks help test whether the idea generalizes.

## 6. Alpha checks and pass/fail decisions

BRAIN may show check messages like:

```text
Turnover of 9.42% is above cutoff of 1%.
Turnover of 9.42% is below cutoff of 70%.
Sharpe of 0.24 is below cutoff of 1.25.
Fitness of 0.17 is below cutoff of 1.
Weight concentration 36.81% is above cutoff of 10% on 9/26/2019.
Sub-universe Sharpe of -0.1 is below cutoff of 0.1.
```

This Alpha should be recorded as:

```text
Passed? n
```

Why? It passes some checks, such as turnover being below `70%`, but it fails others, such as Sharpe, Fitness, weight concentration, and sub-universe Sharpe.

An Alpha can pass some checks and fail others. The overall answer should be `passed = y` only if the relevant required BRAIN checks pass.

## 7. Common failure modes and what to try

### Turnover too high

What it means: the Alpha trades too aggressively.

Things to try:

- Increase Decay.
- Use smoother or larger lookback windows.
- Avoid overly noisy short-term signals.

### Sharpe too low

What it means: the signal is not consistent enough relative to its PnL volatility.

Things to try:

- Try a different template.
- Try a different price field.
- Try a different neutralisation.
- Try different lookback windows.

### Fitness too low

What it means: the combination of Sharpe, Returns, and Turnover is not strong enough.

Things to try:

- Improve Sharpe.
- Improve Returns.
- Reduce Turnover.

### Weight concentration too high

What it means: too much portfolio weight is concentrated in a small number of stocks.

Things to try:

- Lower Truncation.
- Use rank or scale transforms.
- Try a broader universe.
- Try stronger neutralisation.

### Sub-universe Sharpe too low

What it means: the Alpha may not generalize well to a smaller or restricted universe.

Things to try:

- Use more robust templates.
- Avoid signals that depend on a narrow group of stocks.
- Reduce dependence on illiquid names.

### Self-correlation too high

What it means: the Alpha may be too similar to existing submitted Alphas.

Things to try:

- Try a different signal family.
- Try a different data field.
- Try a different transformation.

## 8. Self-correlation and diversity

Self-correlation shows the most correlated Alphas previously submitted that qualified for OS testing.

This helps users maintain a diverse set of Alphas. Even a good-looking Alpha may be less useful if it is too similar to existing submissions.

In this repo, we still log the result because it helps the BO workflow learn. But for actual BRAIN submission decisions, similarity to existing Alphas matters.

## 9. Alpha statuses

BRAIN Alpha statuses include:

- `UNSUBMITTED`: after successful simulation, before submission.
- `ACTIVE`: after submission.
- `DECOMMISSIONED`: if the dataset is no longer available or there is prolonged OS underperformance.

For this project:

- A CSV row means `simulated`.
- `passed = y` means the Alpha passed the relevant BRAIN checks.
- `submitted` means the user manually submitted it on BRAIN.

These are related but not the same.