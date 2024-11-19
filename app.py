import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import requests
from datetime import datetime, timedelta

# Function to fetch news articles for the stock
def fetch_news(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey=219ac0aeef144a1fbe07b33b48eae6ad"  # Replace with your News API key
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        return articles[:10]  # Limit to 10 articles
    else:
        return []

# Function to get stock information
def get_stock_info(ticker):
    stock = yf.Ticker(ticker)
    return {
        "Name": stock.info.get("longName", "N/A"),
        "Sector": stock.info.get("sector", "N/A"),
        "Industry": stock.info.get("industry", "N/A"),
        "Description": stock.info.get("longBusinessSummary", "N/A"),
        "Logo": stock.info.get("logo_url", None)
    }

# Function to get historical data
def get_historical_data(ticker, start, end):
    stock = yf.Ticker(ticker)
    return stock.history(start=start, end=end)

# Function to build and train the LSTM model
def build_and_train_lstm(data, prediction_days=60, epochs=20):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    X_train, y_train = [], []
    for x in range(prediction_days, len(scaled_data)):
        X_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    # Convert to NumPy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Ensure the input has three dimensions for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)

    return model, scaler

# Function to predict stock prices
def predict_stock_price(model, data, scaler, prediction_days=60, forecast_days=30):
    last_data = data['Close'].values[-prediction_days:]
    scaled_last_data = scaler.transform(last_data.reshape(-1, 1))

    # Prepare input for prediction
    X_test = []
    X_test.append(scaled_last_data)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_prices = []

    # Predict for forecast_days
    for _ in range(forecast_days):
        predicted_price = model.predict(X_test)
        predicted_prices.append(predicted_price[0][0])

        # Update the test set with the new predicted price
        new_input = np.append(X_test[0][1:], predicted_price)
        X_test = np.array([new_input]).reshape(1, prediction_days, 1)

    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
    return predicted_prices.flatten()

# Streamlit web app
def main():
    st.title("Stock Price Prediction App")
    st.write("Enter a stock ticker symbol (e.g., ADANIENT.NS) to get predictions and details.")

    ticker = st.text_input("Enter Stock Ticker Symbol", value="AAPl").upper()
    
    # User input for date range selection
    start_date = st.date_input("Select Start Date", value=pd.to_datetime('2020-01-01'))
    end_date = st.date_input("Select End Date", value=pd.to_datetime('today'))

    # User input for prediction period
    forecast_days = st.number_input("Number of Days to Forecast", min_value=1, max_value=365, value=30)

    if ticker:
        try:
            # Display stock information
            stock_info = get_stock_info(ticker)
            st.subheader("Stock Information")
            if stock_info["Logo"]:
                st.image(stock_info["Logo"])
            st.write(f"**Name:** {stock_info['Name']}")
            st.write(f"**Sector:** {stock_info['Sector']}")
            st.write(f"**Industry:** {stock_info['Industry']}")
            st.write(f"**Description:** {stock_info['Description']}")

            # Get historical data and display
            data = get_historical_data(ticker, start_date, end_date)
            st.subheader("Historical Stock Data")
            st.dataframe(data)

            # Plot historical Open and Close prices
            st.subheader("Historical Price Chart")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Open'], mode='lines', name='Open Price', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='orange')))
            fig.update_layout(title=f"{ticker} Historical Prices", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig)

            # Check if enough data is available for prediction
            if len(data) < 60:
                st.warning("Not enough data to perform predictions. Please select a longer date range.")
                return

            # Predict stock prices
            st.subheader("Stock Price Prediction")
            model, scaler = build_and_train_lstm(data)
            predicted_prices = predict_stock_price(model, data, scaler, forecast_days=forecast_days)

            # Prepare prediction DataFrame
            future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
            forecast_df = pd.DataFrame(data={'Date': future_dates, 'Predicted Close Price': predicted_prices})

            # Display prediction results
            st.write(f"The predicted closing prices for **{ticker}** for the next **{forecast_days} days**:")
            st.dataframe(forecast_df)

            # Improved predicted price chart
            st.subheader("Predicted Price Chart")
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Close Price', line=dict(color='orange')))
            fig_pred.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Predicted Close Price'], mode='lines+markers', name='Predicted Close Price', line=dict(color='red')))
            fig_pred.update_layout(title=f"{ticker} Predicted Prices", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig_pred)

            # Display recent news articles
            st.subheader("Recent News Articles")
            news_articles = fetch_news(ticker)
            if news_articles:
                for article in news_articles:
                    st.write(f"**Title:** {article['title']}")
                    st.write(f"**Source:** {article['source']['name']}")
                    st.write(f"**Published At:** {article['publishedAt']}")
                    st.write(f"[Read more]({article['url']})")
                    st.markdown("---")
            else:
                st.write("No news articles found.")

        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")

if __name__ == "__main__":
    main()
