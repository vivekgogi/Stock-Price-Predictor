import datetime
import plotly.graph_objs as go
import streamlit as st

from stock_price_predictor import load_stock_data, lstm_predict

STOCK_SYMBOLS = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ITC Ltd": "ITC.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Axis Bank": "AXISBANK.NS",
    "Maruti Suzuki": "MARUTI.NS",
}

def main():
    st.set_page_config(layout="wide")
    st.title('Stock Price Predictor')
    st.write('Forecasting the stock prices of renowned Indian corporations traded on either the National Stock Exchange (NSE) or the Bombay Stock Exchange (BSE).\n')

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_stock = st.selectbox("Select Stock", list(STOCK_SYMBOLS.keys()))

    with col2:
        start_date = st.date_input("Start Date", min_value=datetime.date(2001, 1, 1), max_value=datetime.date.today() - datetime.timedelta(days=1), value=datetime.date(2001, 1, 1))

    with col3:
        end_date = st.date_input("End Date", min_value=datetime.date(2000, 1, 1), max_value=datetime.date.today(), value=datetime.date.today())

    if start_date > end_date:
        st.error("Error: End date must be after start date.")

    stock_data = load_stock_data(STOCK_SYMBOLS[selected_stock], start_date, end_date)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["Close"], name="Close"))
    fig.update_layout(title=f"{selected_stock} Stock Price", width=1050)
    st.plotly_chart(fig)

    if st.button("Predict Price"):
        try:
            predicted_prices = lstm_predict(stock_data)

            actual_prices = stock_data["Close"].tolist()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stock_data.index, y=actual_prices, name="Actual"))
            fig.add_trace(go.Scatter(
                x=stock_data.index[-len(predicted_prices):],
                y=predicted_prices,
                name="Predicted",
            ))
            fig.update_layout(title=f"{selected_stock} Stock Price Prediction", width=1050)
            st.write("Predicting the stock price of ", selected_stock, "over the past 60 days.")
            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error occurred during prediction: {e}")

if __name__ == "__main__":
    main()
