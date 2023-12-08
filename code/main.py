print("********SYSTEM BOOTING UP********")
from dataset_setup import train_data
from train_models import Models
import numpy as np
print("********SYSTEM BOOT UP COMPLETE********\n")

print("********INITIATING DATASET PREPARATION PROTOCOLS********\n")
path_price = 'dataset/AMZN.csv'
path_senti = 'dataset/AMZN_Senti.csv'
obj = train_data(path_price, path_senti, 14, 1)
trainX, trainY, valX, valY, testX, testY, scaler, dates = obj.csv_read()
print("********DATASET PREPARATION PROTOCOLS COMPLETE********\n")
plot_type = True
lstm_model = Models(trainX, trainY, valX, valY, testX, testY, scaler, dates, plot_type)
gru_model = Models(trainX, trainY, valX, valY, testX, testY, scaler, dates, plot_type)
cnn_model = Models(trainX, trainY, valX, valY, testX, testY, scaler, dates, plot_type)

print("********INITIATING LSTM MODEL PROTOCOLS********\n")
true_val1, pred_val1 = lstm_model.LSTM()

print("********INITIATING GRU MODEL PROTOCOLS********\n")
true_val2, pred_val2 = gru_model.GRU()

print("********INITIATING CNN1D MODEL PROTOCOLS********\n")
true_val3, pred_val3 = cnn_model.CNN1D()
print("********SYSTEM TERMINATING********\n")

if plot_type == False:
    np.save("LSTM_True.npy", true_val1)
    np.save("GRU_True.npy", true_val2)
    np.save("CNN_True.npy", true_val3)
    np.save("LSTM_Pred.npy", pred_val1)
    np.save("GRU_Pred.npy", pred_val2)
    np.save("CNN_Pred.npy", pred_val3)

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