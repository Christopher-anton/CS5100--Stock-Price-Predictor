import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import pandas as pd

class Models():
    def __init__(self, trainX, trainY, valX, valY, testX, testY) -> None:
        self.trainX = trainX
        self.trainY = trainY
        self.valX = valX
        self.valY = valY
        self.testX = testX
        self.testY = testY

    def plot_predictions1(self, model, X, y, start=0, end=50):
        predictions = model.predict(X).flatten()
        # print(predictions)
        # df = pd.DataFrame(data={'Predictions':predictions, 'Actuals':y})
        plt.plot([i for i in range(len(predictions))], predictions)
        plt.plot([i for i in range(len(y))], y)
        plt.title("LSTM MODEL OUTPUT")
        plt.legend(["PREDICTION", "ACTUAL"])
        plt.show()
        print("PLOTTED")
        # return df, mse(y, predictions)
    
    def LSTM(self):
        model1 = Sequential()
        model1.add(InputLayer((14, 5)))
        model1.add(LSTM(64))
        model1.add(Dense(8, 'relu'))
        model1.add(Dense(1, 'linear'))

        cp1 = ModelCheckpoint('model4/', save_best_only=True)
        model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

        model1.fit(self.trainX, self.trainY, validation_data=(self.valX, self.valY), epochs=10, callbacks=[cp1])

        self.plot_predictions1(model1, self.testX, self.testY)

        return model1