import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from datetime import datetime
from tensorflow.keras.models import load_model
import streamlit as st
from tensorflow.keras.models import load_model

from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go


# Function to get company details using ticker symbol
def get_company_details(ticker):
    # Create a Ticker object
    company = yf.Ticker(ticker)

    # Get company info
    info = company.info

    return info

st.markdown("<h1 style='text-align: center;font-size: 4em; padding-bottom: 6px;'>STOCKERR</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;font-size: 1em; padding: 0; margin-bottom:30px;'>Stock analysis and screening tool for investors in India.</h1>", unsafe_allow_html=True)
user_input=st.text_input('Enter stock ticker','AAPL')
df = yf.download(user_input, start = '2012-01-01', end=datetime.now().date())
last_row = df.iloc[-1]
company_details = get_company_details(user_input)
st.subheader(company_details['shortName'])
st.write(company_details['longBusinessSummary'])
ticker = user_input
st.subheader('Stock Visualisation')

# Assuming your DataFrame is named df
last_row = df.iloc[-1]
last_row = last_row.round(2)
max_price = round(df['High'].max(), 2)
min_price = round(df['Low'].min(), 2)

# Sample data
data = {
    "Metric": ["Open", "DayHigh","Max price","currentRatio", "Close","Volume", "DayLow", "Min Price", "revenueGrowth", "Adj Close"],
    "Value": [last_row['Open'], last_row['High'], max_price,company_details['currentRatio'], last_row['Close'],last_row['Volume'],last_row['Low'], min_price,company_details['revenueGrowth'],last_row['Adj Close']]
}
df_details = pd.DataFrame(data)

# Split the DataFrame into two parts
num_rows = len(df_details)
mid_point = num_rows // 2
df_details1 = df_details.iloc[:mid_point]
df_details2 = df_details.iloc[mid_point:]

# Display the two DataFrames side by side using columns layout
col1, col2 = st.columns(2)

with col1:
    st.table(df_details1.set_index('Metric')) # Set 'Metric' column as index

with col2:
    st.table(df_details2.set_index('Metric')) # Set 'Metric' column as index

text=st.text_input('Please provide any additional details you would like to retrieve:','eg: bookValue')
if text in company_details.keys():
    st.write(f"{text} : {company_details[text]}")
st.write(f"To discover more about the {user_input}, please visit the {company_details['website']}.")

st.subheader('Closing Price vs Time chart')

START="25-01-01"
TODAY=datetime.today().strftime("%Y-%m-%d")

data=df
data.reset_index(inplace=True)

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open',line=dict(color='green')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close',line=dict(color='red')))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()




# Assuming you already have 'df' (DataFrame) containing the stock data

# Calculate the 100-day moving average
ma100 = df['Close'].rolling(100).mean()

# Create a Plotly figure
fig = go.Figure()

# Add trace for stock price
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Stock Price'))

# Add trace for 100-day moving average
fig.add_trace(go.Scatter(x=df['Date'], y=ma100, name='100-day Moving Average', line=dict(color='red', width=3)))

# Update layout with title and x-axis rangeslider
fig.update_layout(title='Stock Price with Moving Average',
                  xaxis_rangeslider_visible=True)

# Display the Plotly chart
st.plotly_chart(fig, use_container_width=True)


# Calculate the moving averages
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()

# Create a Plotly figure
fig = go.Figure()

# Add trace for 100-day moving average
fig.add_trace(go.Scatter(x=df['Date'], y=ma100, name='100-day MA', line=dict(color='red', width=3)))

# Add trace for 200-day moving average
fig.add_trace(go.Scatter(x=df['Date'], y=ma200, name='200-day MA', line=dict(color='green', width=3)))

# Add trace for closing price
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Closing Price', line=dict(color='blue')))

# Update layout with title and x-axis rangeslider
fig.update_layout(title='Closing Price vs. Time Chart with 100MA & 200MA',
                  xaxis_rangeslider_visible=True)

# Display the Plotly chart
st.plotly_chart(fig, use_container_width=True)




START = "2012-01-01"
TODAY = datetime.today().strftime("%Y-%m-%d")

st.markdown("<h1 style='font-size: 2em; padding: 0; margin-bottom:30px;'>Price Prediction</h1>", unsafe_allow_html=True)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365



def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(user_input)
data_load_state.text('Loading data... done!')


# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecasted data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = m.plot(forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)