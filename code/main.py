from dataset_setup import train_data
from LSTM import Models
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

input_dim = 5
hidden_dim = 32
num_layers = 2
output_dim = 2

past_days = 14
future = 1

# dateset class
obj = train_data('dataset/TSLA.csv',past_days,future)
trainX, trainY, valX, valY, testX, testY = obj.csv_read()

print(f"trainX: {trainX.shape}\ntrainY: {trainX.shape}\nvalX: {valX.shape}\nvalY: {valY.shape}\ntestX: {testX.shape}\ntestY: {testY.shape}")

# plt.plot([i for i in range(len(trainY))], trainY)
# plt.show()

c_model = Models(trainX, trainY, valX, valY, testX, testY)
model = c_model.LSTM()