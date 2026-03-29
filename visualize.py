import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from risk_engine import calculate_as_index, calculate_fh_index

def create_risk_chart(ticker="BTC-USD"):
    # Fetch data
    data = yf.download(ticker, period="1y")
    returns = data['Adj Close'].pct_change().dropna()
    
    # Calculate rolling FH Index (30-day window)
    rolling_fh = returns.rolling(window=30).apply(calculate_fh_index)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_fh, color='red', label='Foster-Hart Risk Index')
    plt.title(f'Riskiness Over Time: {ticker}')
    plt.ylabel('Risk Index (Higher = More Dangerous)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save it for your README
    plt.savefig('risk_plot.png')
    print("Chart saved as risk_plot.png!")

if __name__ == "__main__":
    create_risk_chart()
