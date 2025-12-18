import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

st.title("Bitcoin Price Prediction Web App (Random Forest)")

# ----------------------------------------
# Helper: RSI
# ----------------------------------------
def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Button to fetch & train
if st.button("Run Model & Predict"):
    with st.spinner("Fetching data from CoinGecko..."):
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {"vs_currency": "usd", "days": "365", "interval": "daily"}
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        if 'prices' not in data or 'total_volumes' not in data:
            st.error("API data missing!")
            st.stop()

        prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])

        df = pd.merge(prices, volumes, on='timestamp')
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('Date', inplace=True)
        df.drop(columns=['timestamp'], inplace=True)

        df['Open'] = df['price']
        df['High'] = df['price']
        df['Low'] = df['price']
        df['Close'] = df['price']
        df.rename(columns={'volume': 'Volume'}, inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    st.success("Data Loaded Successfully!")

    # Feature engineering
    df['MA_7'] = df['Close'].rolling(7).mean()
    df['MA_30'] = df['Close'].rolling(30).mean()
    df['EMA_7'] = df['Close'].ewm(span=7, adjust=False).mean()
    df['EMA_30'] = df['Close'].ewm(span=30, adjust=False).mean()
    df['RSI_14'] = compute_rsi(df['Close'], 14)
    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month

    df.dropna(inplace=True)

    feature_cols = ['Open','High','Low','Volume','MA_7','MA_30','EMA_7','EMA_30','RSI_14','Day_of_Week','Month']
    X = df[feature_cols]
    y = df['Close']

    # Train model
    tscv = TimeSeriesSplit(n_splits=5)
    model = RandomForestRegressor(n_estimators=200, random_state=42)

    for train_index, test_index in tscv.split(X):
        model.fit(X.iloc[train_index], y.iloc[train_index])

    joblib.dump(model, "rf_model.pkl")

    st.success("Model Trained Successfully!")

    # Predict future 30 days
    future_days = 30
    last_row = df.iloc[-1]
    future_index = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=future_days)
    future_df = pd.DataFrame(index=future_index)

    for col in ['Open','High','Low','Volume','MA_7','MA_30','EMA_7','EMA_30','RSI_14']:
        future_df[col] = last_row[col]

    future_df['Day_of_Week'] = future_df.index.dayofweek
    future_df['Month'] = future_df.index.month

    future_X = future_df[feature_cols]
    future_pred = model.predict(future_X)

    result = pd.DataFrame({"Predicted_Close": future_pred}, index=future_index)

    st.subheader("Future 30‑Day Predictions")
    st.dataframe(result)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(df.index, df['Close'], label="Historical Close")
    ax.plot(result.index, result['Predicted_Close'], label="Predicted", linestyle='--')
    ax.legend()
    st.pyplot(fig)

    st.success("Done! Your Streamlit prediction app is running.")
