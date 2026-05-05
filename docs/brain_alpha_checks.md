# 📝 BRAIN Alpha Checks

When an alpha is simulated on WorldQuant BRAIN, the platform may show several checks that determine whether the alpha passes or fails.

In our CSV workflow, the `passed` column means:

> Did the alpha pass the relevant BRAIN checks and become eligible/useful for submission?

It does **not** simply mean “did the alpha have a positive return”.

---

## Common checks

Example BRAIN output:

    Turnover of 9.42% is above cutoff of 1%.
    Turnover of 9.42% is below cutoff of 70%.
    Sharpe of 0.24 is below cutoff of 1.25.
    Fitness of 0.17 is below cutoff of 1.
    Weight concentration 36.81% is above cutoff of 10% on 9/26/2019.
    Sub-universe Sharpe of -0.1 is below cutoff of 0.1.

This alpha should be recorded as:

    Passed? n

because at least one required check failed.

---

## How to interpret the checks

### Turnover

Turnover measures how much the portfolio changes through time.

High turnover often means the alpha trades too aggressively.

Possible fixes:

- increase Decay,
- use smoother lookback windows,
- avoid very short noisy signals,
- try stronger neutralisation or transform choices.

### Sharpe

Sharpe measures risk-adjusted performance.

If Sharpe is below cutoff, the alpha is not strong enough relative to its volatility.

Possible fixes:

- improve the signal template,
- test a different lookback window,
- try a different universe,
- combine price and volume information.

### Fitness

Fitness is a BRAIN-specific quality metric combining performance and robustness.

If Fitness is below cutoff, the alpha is usually too weak for submission.

Possible fixes:

- improve signal quality,
- reduce turnover,
- reduce concentration,
- try a different neutralisation setting.

### Weight concentration

Weight concentration means too much position weight is concentrated in a small number of stocks.

If weight concentration is too high, the alpha may be unstable or too dependent on a few names.

Possible fixes:

- lower Truncation,
- use rank/scale transforms,
- try a broader universe,
- try stronger neutralisation.

### Sub-universe Sharpe

Sub-universe Sharpe checks whether the alpha remains robust on a smaller or restricted universe.

If this is too low, the alpha may not generalise well.

Possible fixes:

- avoid overly narrow signals,
- test more robust templates,
- reduce dependence on illiquid names,
- try different neutralisation.

---

## Rule for entering `passed`

Enter:

    y

only if the alpha passes the relevant BRAIN checks.

Enter:

    n

if one or more required checks fail.

Bad simulations are still useful and should still be logged, because they help the BO workflow learn which regions of the search space are poor.