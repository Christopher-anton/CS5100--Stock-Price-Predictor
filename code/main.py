print("********SYSTEM BOOTING UP********")
from dataset_setup import train_data
from train_models import Models
print("********SYSTEM BOOT UP COMPLETE********\n")

print("********INITIATING DATASET PREPARATION PROTOCOLS********\n")
path = 'dataset/AMZN.csv'
obj = train_data(path, 14, 1)
trainX, trainY, valX, valY, testX, testY, scaler, dates = obj.csv_read()
print("********DATASET PREPARATION PROTOCOLS COMPLETE********\n")

c_model = Models(trainX, trainY, valX, valY, testX, testY, scaler, dates)

print("********INITIATING LSTM MODEL PROTOCOLS********\n")
c_model.LSTM()

print("********INITIATING GRU MODEL PROTOCOLS********\n")
c_model.GRU()

print("********INITIATING CNN1D MODEL PROTOCOLS********\n")
c_model.CNN1D()
print("********SYSTEM TERMINATING********")

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