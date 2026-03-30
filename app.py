import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from risk_engine import calculate_as_index, calculate_fh_index

st.set_page_config(page_title="AlphaRisk Index", page_icon="🛡️")
st.title("🛡️ AlphaRisk: Objective Risk Measures")
st.markdown("""
This tool calculates the **Aumann-Serrano** and **Foster-Hart** risk indices.
Unlike the Sharpe Ratio, these measures account for fat-tail risks and bankruptcy potential.
""")

tab1, tab2 = st.tabs(["📈 Market Data", "🎲 Gamble Simulator"])

# --- TAB 1: existing code ---
with tab1:
    st.sidebar.header("Settings")
    ticker = st.sidebar.text_input("Enter Ticker (e.g., BTC-USD, TSLA, SPY)", "BTC-USD")
    days = st.sidebar.slider("Analysis Window (Days)", 30, 730, 365)

    if st.button("Analyze Risk Profile"):
        with st.spinner(f'Fetching data for {ticker}...'):
            data = yf.download(ticker, period=f"{days}d")

            if data.empty:
                st.error("No data found! Please check the ticker symbol.")
            else:
                if isinstance(data.columns, pd.MultiIndex):
                    prices = data['Close'][ticker]
                else:
                    prices = data['Close']

                returns = prices.pct_change().dropna().values

                as_val = calculate_as_index(returns)
                fh_val = calculate_fh_index(returns)

                st.divider()
                c1, c2 = st.columns(2)
                c1.metric("Aumann-Serrano Index", f"{as_val:.4f}")
                c2.metric("Foster-Hart (Min Wealth)", f"${fh_val:,.2f}")

                st.subheader(f"Price History: {ticker}")
                st.line_chart(prices)

                import matplotlib.pyplot as plt
                st.subheader(f"Rolling 30-Day Foster-Hart Index: {ticker}")
                rolling_fh = pd.Series(returns).rolling(window=30).apply(calculate_fh_index)
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(rolling_fh.values, color='crimson', linewidth=1.5, label='FH Index (30d rolling)')
                ax.set_ylabel("Min. Safe Wealth per $1 invested")
                ax.set_title(f"Riskiness Over Time — {ticker}")
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

                st.info(f"""
                **What does this mean?**
                * **AS Index:** Measures intrinsic riskiness. Lower is safer.
                * **FH Index:** You should hold at least **${fh_val:,.2f}** in total wealth
                for every $1 invested in {ticker} to avoid eventual ruin.
                """)

# --- TAB 2: Gamble Simulator ---
with tab2:
    st.subheader("Hypothetical Gamble Simulator")
    st.markdown("""
    Design a two-outcome gamble and compute its **exact** Aumann-Serrano and Foster-Hart indices.
    Unlike the Sharpe ratio, these measures capture the full risk of the gamble — not just variance.
    """)

    c1, c2, c3 = st.columns(3)
    upside = c1.number_input("Upside ($)", value=100, step=10)
    downside = c2.number_input("Downside ($)", value=-50, step=10)
    prob = c3.slider("Probability of Upside", 0.01, 0.99, 0.50)

    ev = upside * prob + downside * (1 - prob)
    st.metric("Expected Value", f"${ev:.2f}")

    if ev <= 0:
        st.warning("Expected value must be positive for AS/FH indices to exist.")
    elif downside >= 0:
        st.warning("Downside must be negative for a meaningful gamble.")
    else:
        n = 10000
        returns = np.array(
            [upside / 100] * int(prob * n) + [downside / 100] * int((1 - prob) * n)
        )

        as_val = calculate_as_index(returns)
        fh_val = calculate_fh_index(returns)

        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Aumann-Serrano Index", f"{as_val:.4f}")
        c2.metric("Foster-Hart (Min Wealth per $1)", f"${fh_val:,.2f}")

        st.divider()

        # AS Explanation
        st.subheader("🔵 Aumann-Serrano Index — What It Means")
        st.markdown(f"""
        The AS index solves:

        **E\[exp(−g / R)] = 1**

        where g is the gamble outcome and R is the index value.

        **Your gamble has an AS index of {as_val:.4f}.**

        This means an investor with CARA utility and absolute risk-aversion coefficient
        of **1 / {as_val:.2f} = {1/as_val:.5f}** is exactly indifferent between
        taking this gamble and receiving nothing.

        - If your risk aversion < {1/as_val:.5f} → you **accept** the gamble
        - If your risk aversion > {1/as_val:.5f} → you **reject** the gamble

        The AS index is **homogeneous of degree 1**: doubling the stakes doubles the index.
        It is also monotone under first and second-order stochastic dominance —
        a safer gamble always has a lower AS index.
        """)

        st.divider()

        # FH Explanation
        st.subheader("🟠 Foster-Hart Index — What It Means")
        st.markdown(f"""
        The FH index solves:

        **E\[ln(1 + g / R)] = 0**

        This is the minimum initial wealth R such that a Kelly-optimal investor
        can avoid bankruptcy in the long run.

        **Your gamble has an FH index of ${fh_val:,.2f}.**

        This means:
        - If your total wealth < **${fh_val:,.2f}** → *no strategy* can prevent eventual ruin
        - If your total wealth ≥ **${fh_val:,.2f}** → a safe betting strategy exists

        Equivalently, the maximum safe fraction of wealth to bet is
        **$1 / ${fh_val:,.2f} = {1/fh_val:.5f}** per dollar of wealth.

        The FH index is more conservative than the AS index because it accounts
        for the path-dependent risk of ruin under repeated play — not just a
        single-period risk-aversion threshold.
        """)

        st.divider()

        # Comparison
        st.subheader("⚖️ AS vs FH — Key Difference")
        st.markdown(f"""
        | | Aumann-Serrano | Foster-Hart |
        |---|---|---|
        | **Answers** | Who accepts this gamble? | How much wealth do you need? |
        | **Framework** | Single-period CARA utility | Repeated play, Kelly criterion |
        | **Your value** | {as_val:.4f} | ${fh_val:,.2f} |
        | **Conservative?** | Less | More |

        Both indices agree that this gamble is **{"relatively safe" if as_val < 0.1 else "moderately risky" if as_val < 0.3 else "high risk"}**
        based on the AS index, and requires **{"modest" if fh_val < 500 else "substantial" if fh_val < 5000 else "very large"}**
        capital to undertake safely according to Foster-Hart.
        """)
