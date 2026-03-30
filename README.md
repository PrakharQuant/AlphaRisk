# 🛡️ AlphaRisk : Objective Risk Measures Beyond the Sharpe Ratio

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Live Demo →** [riskindices.streamlit.app](https://your-app.streamlit.app)

AlphaRisk computes two **axiomatic, decision-theoretic risk indices** : the **Aumann-Serrano (AS) Index** and the **Foster-Hart (FH) Index** — for any asset fetchable via Yahoo Finance. These measures are grounded in expected-utility theory and provide guarantees that the Sharpe ratio fundamentally cannot: they are monotone under first- and second-order stochastic dominance, and the FH index directly implies a *bankruptcy-avoidance wealth threshold*.

---

## Motivation

The Sharpe ratio is mean-variance and implicitly assumes Gaussian returns. Real asset returns, especially crypto, options, and emerging markets, exhibit **fat tails, skewness, and excess kurtosis** that Sharpe ignores entirely. The AS and FH indices instead operate directly on the **full empirical return distribution**, making them robust to higher moments by construction.

| Property | Sharpe Ratio | AS Index | FH Index |
|---|:---:|:---:|:---:|
| Accounts for fat tails | ✗ | ✓ | ✓ |
| Monotone under FOSD | ✗ | ✓ | ✓ |
| Monotone under SOSD | ✗ | ✓ | ✓ |
| Bankruptcy interpretation | ✗ | ✗ | ✓ |
| Requires distributional assumption | ✗ | ✗ | ✗ |

---

## Theory

### Aumann-Serrano Index
<img src="as-indice.png" width="500" alt="Aumann-Serrano Risk Index"/>

Introduced by Aumann & Serrano (2008), the AS index ρ(X) of a gamble (return distribution) X is defined as the unique solution to:

$$\mathbb{E}\left[e^{-X/\rho}\right] = 1$$

Equivalently, ρ = 1/α where α solves the MGF equation M_X(−α) = 1. This can be interpreted as: the riskiness equals the reciprocal of the **absolute risk-aversion coefficient** of the CARA agent who is exactly indifferent to taking the gamble.

**Key properties:**
- **Homogeneous of degree 1:** scaling a position by λ scales its AS index by λ
- **Duality with CARA utility:** ρ(X) is the wealth level at which an agent with u(w) = −e^(−w/ρ) is indifferent
- **Lower is safer** : BTC will show a far higher AS index than SPY


### Foster-Hart Index

<img src="fh-indice.png" width="500" alt="Foster-Hart Risk Measure"/>

Foster & Hart (2009, 2013) define the riskiness R(X) as the minimum initial wealth R such that **no-bankruptcy is achievable** under Kelly-optimal play. It solves:

$$\mathbb{E}\left[\log\left(1 + \frac{X}{R}\right)\right] = 0$$

This has a striking interpretation: if an agent's current wealth $w < R(X)$, there **exists no betting strategy** that guarantees avoiding bankruptcy in the long run. If $w \geq R(X)$, a safe strategy exists.

**Key properties:**
- **Bankruptcy threshold:** investing $1 in an asset with FH index R is safe only if total wealth ≥ R
- **Consistent with all risk-averse preferences:** any risk-averse expected utility maximiser will prefer a gamble with lower FH index
- **Naturally handles left-tail events** (unlike Sharpe, which is symmetric)

---

## App Screenshot

```
┌─────────────────────────────────────────────┐
│  🛡️ AlphaRisk: Objective Risk Measures       │
│                                              │
│  Aumann-Serrano Index    Foster-Hart Index   │
│       0.0312                 $31,847.22      │
│                                              │
│  [Price history chart: BTC-USD]              │
└─────────────────────────────────────────────┘
```

---
![Rolling Foster-Hart Risk Index — BTC-USD](risk_plot.png)
## Quickstart

```bash
git clone https://github.com/PrakharQuant/alpharisk.git
cd alpharisk
pip install -r requirements.txt
streamlit run app.py
```

**Requirements:**
```
streamlit
yfinance
numpy
pandas
scipy
```

---

## Usage

1. Enter any Yahoo Finance ticker : equities (`TSLA`, `NIFTY50.NS`), crypto (`BTC-USD`, `ETH-USD`), or ETFs (`SPY`, `GLD`)
2. Set the lookback window (30–730 days)
3. Hit **Analyze Risk Profile**

The app returns:
- **AS Index** : lower means intrinsically safer; compare across assets on the same scale
- **FH Minimum Wealth** : for every ₹1 (or $1) invested, you should hold at least this much total wealth to be in the "mathematically safe" regime

---

## Implementation Notes

Both indices are computed numerically via **Brent's root-finding method** (`scipy.optimize.brentq`), which guarantees convergence for continuous monotone objectives on a bracketed interval.

```python
def calculate_as_index(returns):
    """Solve E[exp(-alpha * X)] = 1 for alpha*, return 1/alpha*."""
    def objective(alpha):
        return np.mean(np.exp(np.clip(-alpha * returns, -100, 100))) - 1
    alpha_star = brentq(objective, 1e-10, 100)
    return 1 / alpha_star

def calculate_fh_index(returns):
    """Solve E[log(1 + X/R)] = 0 for R, where R > max_loss."""
    max_loss = -np.min(returns)
    def objective(R):
        return np.mean(np.log(1 + returns / R))
    return brentq(objective, max_loss + 1e-8, max_loss * 100_000)
```

Edge cases handled:
- Non-positive mean returns → AS Index reported as `∞` (no finite indifferent CARA agent exists)
- Zero maximum loss → FH Index = 0 (no bankruptcy risk)
- Numerical overflow in the exponential is guarded via `np.clip`

---

## Interpretation Guide

| AS Index Range | Interpretation |
|---|---|
| < 0.05 | Low intrinsic risk (government bonds, stablecoin-like) |
| 0.05 – 0.15 | Moderate risk (large-cap equities) |
| > 0.15 | High risk (crypto, small-caps, leveraged products) |

The **FH Index** is best read as a **leverage constraint**: an FH value of \$30,000 means you should only bet \$1 in this asset if your total portfolio is at least \$30,000. Equivalently, safe leverage ≤ 1/30,000 per dollar of wealth.

---

## References

1. Aumann, R. J., & Serrano, R. (2008). *An economic index of riskiness*. Journal of Political Economy, 116(5), 810–836.
2. Foster, D. P., & Hart, S. (2009). *An operational measure of riskiness*. Journal of Political Economy, 117(5), 785–814.
3. Foster, D. P., & Hart, S. (2013). *A wealth-requirement axiomatization of riskiness*. Theoretical Economics, 8(2), 591–620.
4. Hart, S. (2011). *Comparing risks by acceptance and rejection*. Journal of Political Economy, 119(4), 617–638.

---

## Related Projects

| Project | Description |
|---|---|
| [Monte Carlo VaR](https://montecarlovar.streamlit.app) | Parametric & simulation-based Value-at-Risk |
| [Agnostic Risk](https://agnostic-risk-prakharquant.streamlit.app) | Distribution-free risk via Chebyshev & Markov bounds |
| [Quant NN Backprop](https://github.com/PrakharQuant/quant-nn-backprop) | MLP applied to cross-sectional return prediction |

---

## Author

**Prakhar** · [@PrakharQuant](https://github.com/PrakharQuant)

*Building a rigorous quant finance portfolio at the intersection of decision theory, probability, and markets.*
