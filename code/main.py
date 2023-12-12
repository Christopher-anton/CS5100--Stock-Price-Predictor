print("********SYSTEM BOOTING UP********")
from dataset_setup import train_data
from train_models import Models
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
print("********SYSTEM BOOT UP COMPLETE********\n")

import yfinance as yf 
import pandas as pd

yf.pdr_override()

# # Create input field for our desired stock 
# stock=input("Enter a stock ticker symbol: ")

# # Retrieve stock data frame (df) from yfinance API at an interval of 1m 
# df = yf.download(tickers=stock,period='5d',interval='1m')
# print(df.head())
# df = pd.DataFrame(df)
# # df.index = pd.to_datetime(df.index)
# print(df.index)

# high = df['High']
# date = df['Datetime']
# high = pd.DataFrame(high)
# date = pd.DataFrame(date)
# # print(high.head())
# combined_df = pd.concat([high, date], axis=0)


# # Save the combined DataFrame to a CSV file
# # combined_df.to_csv('combined_data.csv', index=False) 
# file_path = stock +'_' +'.csv'
# print (file_path)
# combined_df.to_csv(file_path, sep='\t', index=False, header=True)
# print(combined_df.head())

print("********INITIATING DATASET PREPARATION PROTOCOLS********\n")
path_price = 'code\dataset\TSLA.csv'
path_senti = 'code\dataset\TSLA_Senti.csv'
obj = train_data(path_price, path_senti, 14, 1)
trainX, trainY, valX, valY, testX, testY, scaler, dates = obj.csv_read()
# df = pd.read_csv(path_price)
# dates = len(df)
# dates = len(testX)
print("********DATASET PREPARATION PROTOCOLS COMPLETE********\n")
plot_type = False
lstm_model = Models(trainX, trainY, valX, valY, testX, testY, scaler, dates, plot_type)
gru_model = Models(trainX, trainY, valX, valY, testX, testY, scaler, dates, plot_type)
cnn_model = Models(trainX, trainY, valX, valY, testX, testY, scaler, dates, plot_type)

print("********INITIATING LSTM MODEL PROTOCOLS********\n")
true_val1, pred_val1 = lstm_model.LSTM()


print("********INITIATING GRU MODEL PROTOCOLS********\n")
true_val2, pred_val2 = gru_model.GRU()



print("********INITIATING CNN1D MODEL PROTOCOLS********\n")
true_val3, pred_val3 = cnn_model.CNN1D()



# print("********SYSTEM TERMINATING********\n")
print("XXXXXXXXXXXXXX LSTM MAPE XXXXXXXXXXX")
print(lstm_model.MAPE(true_val1, pred_val1))
print("XXXXXXXXXXXXXX GRU MAPE XXXXXXXXXXX")
print(gru_model.MAPE(true_val2, pred_val2))
print("XXXXXXXXXXXXXX CNN MAPE XXXXXXXXXXX")
print(cnn_model.MAPE(true_val3, pred_val3))

# if plot_type == False:
np.save("LSTM_Truex.npy", true_val1)
np.save("GRU_Truex.npy", true_val2)
np.save("CNN_Truex.npy", true_val3)
np.save("LSTM_Predx.npy", pred_val1)
np.save("GRU_Predx.npy", pred_val2)
np.save("CNN_Predx.npy", pred_val3)
np.save("datesx.npy", dates)
# data1 = np.load('LSTM_Predx.npy')
# data2 = np.load('A:\Academics_NEU\sem_6\CS5100--Stock-Price-Predictor\LSTM_Truex.npy')
# dates  = np.load('dates.npy')
# plt.plot(dates, data2)
# plt.plot(dates, data1)
# plt.xlabel("Day")
# plt.ylabel("Stock Price in ($)")
# # plt.title(f"{name} MODEL OUTPUT\nMEAN ABSOLUTE PERCENTAGE ERROR: {self.MAPE(y, predictions)}%")
# plt.legend(["PREDICTION", "ACTUAL"])
# plt.show()
# # answer = input("Select Model:\n LSTM\n GRU\n CNN1d\n\n")
# # print()

# # if answer == "LSTM":
# #     print("********INITIATING LSTM MODEL PROTOCOLS********")
# #     model = c_model.LSTM()
# #     print("********SYSTEM TERMINATING********")
# # elif answer == "GRU":
# #     print("********INITIATING GRU MODEL PROTOCOLS********")
# #     model = c_model.GRU()
# #     print("********SYSTEM TERMINATING********")
# # elif answer == "CNN1d":
# #     print("********INITIATING CNN1d MODEL PROTOCOLS********")
# #     model = c_model.CNN1d()
# #     print("********SYSTEM TERMINATING********")

print("********SYSTEM TERMINATING********")