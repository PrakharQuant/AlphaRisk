import sys
import os
sys.path.append(os.path.dirname(__file__))

import streamlit as st
import yfinance as yf
# ... rest of your imports

import streamlit as st
import yfinance as yf
import numpy as np
from risk_engine import calculate_as_index, calculate_fh_index

st.set_page_config(page_title="AlphaRisk", page_icon="🛡️")
st.title("🛡️ AlphaRisk: AS & FH Index Dashboard")

ticker = st.text_input("Enter Ticker (e.g., TSLA, BTC-USD)", "BTC-USD")
days = st.slider("Analysis Period (Days)", 30, 730, 365)

if st.button("Analyze Risk"):
    with st.spinner('Downloading data...'):
        # 1. Download data
        data = yf.download(ticker, period=f"{days}d")
        
        # 2. Check if data is empty
        if data.empty:
            st.error(f"No data found for {ticker}. Check the ticker symbol.")
        else:
            # 3. Safely extract returns (handles MultiIndex)
            # We use .iloc[:, 0] to grab the first column of Adj Close
            try:
                adj_close = data['Adj Close']
                returns = adj_close.pct_change().dropna().values.flatten()
                
                if len(returns) < 5:
                    st.warning("Not enough data points for a reliable calculation.")
                else:
                    # 4. Calculate
                    as_val = calculate_as_index(returns)
                    fh_val = calculate_fh_index(returns)
                    
                    # 5. Display
                    col1, col2 = st.columns(2)
                    col1.metric("Aumann-Serrano Index", f"{as_val:.4f}")
                    col2.metric("Foster-Hart (Min Wealth)", f"${fh_val:,.2f}")
                    
                    st.subheader(f"Price Action: {ticker}")
                    st.line_chart(data['Adj Close'])
                    
                    st.info(f"💡 **Insight:** The Foster-Hart index of ${fh_val:,.2f} suggests you should have this much 'reserve wealth' for every 1 unit of this asset to avoid long-term ruin.")
            except Exception as e:
                st.error(f"Error processing data: {e}")
