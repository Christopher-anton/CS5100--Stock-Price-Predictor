# print("********SYSTEM BOOTING UP********")
# from dataset_setup import train_data
# from train_models import Models
# print("********SYSTEM BOOT UP COMPLETE********")

# print("********INITIATING DATASET PREPARATION PROTOCOLS********")
# obj = train_data('dataset/AMZN.csv', 14, 1)
# trainX, trainY, valX, valY, testX, testY, scaler, dates = obj.csv_read()
# print("********DATASET PREPARATION PROTOCOLS COMPLETE********")

# c_model = Models(trainX, trainY, valX, valY, testX, testY, scaler, dates)

# answer = input("Select Model:\n LSTM\n GRU\n CNN1d\n\n")
# print()

# if answer == "LSTM":
#     print("********INITIATING LSTM MODEL PROTOCOLS********")
#     model = c_model.LSTM()
#     print("********SYSTEM TERMINATING********")
# elif answer == "GRU":
#     print("********INITIATING GRU MODEL PROTOCOLS********")
#     model = c_model.GRU()
#     print("********SYSTEM TERMINATING********")
# elif answer == "CNN1d":
#     print("********INITIATING CNN1d MODEL PROTOCOLS********")
#     model = c_model.CNN1d()
#     print("********SYSTEM TERMINATING********")
# else:
#     print("********INVALID OPTION********")
#     print("********SYSTEM TERMINATING********")
    
# Raw Package
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr

# Market Data 
import yfinance as yf

#Graphing/Visualization
import datetime as dt 
import plotly.graph_objs as go 

# Override Yahoo Finance 
yf.pdr_override()

# Create input field for our desired stock 
stock=input("Enter a stock ticker symbol: ")

# Retrieve stock data frame (df) from yfinance API at an interval of 1m 
df = yf.download(tickers=stock,period='5d',interval='1m')

print(df)

# Declare plotly figure (go)
fig=go.Figure()

fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'], name = 'market data'))

fig.update_layout(
    title= str(stock)+' Live Share Price:',
    yaxis_title='Stock Price (USD per Shares)')               

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            dict(count=45, label="45m", step="minute", stepmode="backward"),
            dict(count=1, label="HTD", step="hour", stepmode="todate"),
            dict(count=3, label="3h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
    )
)

fig.show()