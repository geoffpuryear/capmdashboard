import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime, timedelta

st.set_page_config(page_title="CAPM Dashboard", layout="wide")
st.title("📈 CAPM Dashboard")

# Sidebar inputs
tickers = st.sidebar.text_input("Enter stock tickers (comma separated)", "AAPL,MSFT,GOOGL").upper().split(',')
benchmark_ticker = st.sidebar.text_input("Enter market benchmark ticker", "SPY")  # More reliable than ^GSPC
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())
risk_free_rate = st.sidebar.number_input("Risk-free rate (%)", value=4.0) / 100

st.write(f"Analyzing CAPM for: **{', '.join(tickers)}** relative to **{benchmark_ticker}**")

# Download and process return data
def get_returns(ticker):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data.empty:
            return pd.Series(dtype='float64')

        # Use 'Adj Close' if available, else fall back to 'Close' silently
        if 'Adj Close' in data.columns:
            price = data['Adj Close']
        elif 'Close' in data.columns:
            price = data['Close']
        else:
            return pd.Series(dtype='float64')

        return price.pct_change().dropna()
    except Exception as e:
        return pd.Series(dtype='float64')

# Get market returns
market_returns = get_returns(benchmark_ticker)

if market_returns.empty:
    st.error("❌ Market data could not be retrieved. Please check the benchmark ticker.")
else:
    # Analyze each stock
    for ticker in tickers:
        st.subheader(f"📊 {ticker} Analysis")
        stock_returns = get_returns(ticker)

        if stock_returns.empty:
            st.error(f"❌ No data available for {ticker}. Skipping...")
            continue

        # Match dates
        common_dates = market_returns.index.intersection(stock_returns.index)
        X = market_returns.loc[common_dates].values
        y = stock_returns.loc[common_dates].values

        if len(X) < 2:
            st.warning(f"⚠️ Not enough data to perform regression for {ticker}.")
            continue

        # Linear regression for beta and alpha
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()
        beta = model.params[1]
        alpha = model.params[0]
        r_squared = model.rsquared

        # CAPM expected return
        market_mean_return = float(market_returns.loc[common_dates].mean())
        expected_return = risk_free_rate + beta * (market_mean_return - risk_free_rate)
        mean_stock_return = float(stock_returns.loc[common_dates].mean())

        # Display metrics
        st.markdown(f"""
        - **Beta (β):** {beta:.3f}  
        - **Alpha (α):** {alpha:.3%}  
        - **R² of Regression:** {r_squared:.2f}  
        - **Expected Return (CAPM):** {expected_return:.2%}  
        - **Mean Historical Return:** {mean_stock_return:.2%}
        """)

        # Plot regression
        fig, ax = plt.subplots()
        ax.scatter(market_returns.loc[common_dates], stock_returns.loc[common_dates], alpha=0.5, label='Data')
        ax.plot(market_returns.loc[common_dates], model.predict(X_const), color='red', label='CAPM Fit')
        ax.set_xlabel("Market Returns")
        ax.set_ylabel(f"{ticker} Returns")
        ax.set_title(f"{ticker} vs. Market")
        ax.legend()
        st.pyplot(fig)
