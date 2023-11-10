print("********SYSTEM BOOTING UP********")
from dataset_setup import train_data
from train_models import Models
print("********SYSTEM BOOT UP COMPLETE********")

print("********INITIATING DATASET PREPARATION PROTOCOLS********")
obj = train_data('dataset/TSLA.csv', 14, 1)
trainX, trainY, valX, valY, testX, testY, scaler = obj.csv_read()
print("********DATASET PREPARATION PROTOCOLS COMPLETE********")

c_model = Models(trainX, trainY, valX, valY, testX, testY, scaler)

answer = input("Select Model:\n LSTM\n GRU\n CNN1d\n\n")
print()

if answer == "LSTM":
    print("********INITIATING LSTM MODEL PROTOCOLS********")
    model = c_model.LSTM()
    print("********SYSTEM TERMINATING********")
elif answer == "GRU":
    print("********INITIATING GRU MODEL PROTOCOLS********")
    model = c_model.GRU()
    print("********SYSTEM TERMINATING********")
elif answer == "CNN1d":
    print("********INITIATING CNN1d MODEL PROTOCOLS********")
    model = c_model.CNN1d()
    print("********SYSTEM TERMINATING********")
else:
    print("********INVALID OPTION********")
    print("********SYSTEM TERMINATING********")