from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

class Models():
    def __init__(self, trainX, trainY, valX, valY, testX, testY, scaler, dates) -> None:
        self.trainX = trainX
        self.trainY = trainY
        self.valX = valX
        self.valY = valY
        self.testX = testX
        self.testY = testY
        self.scaler = scaler
        self.dates = dates

    def plot_predictions1(self, model, X, y):
        print("********PREDICTING THE FUTURE********")
        predictions = model.predict(X).flatten()
        r_predictions = []
        for i in range(50):
            r_predictions += [[predictions[i], predictions[i], predictions[i], predictions[i], predictions[i]]]
        final_predictions = self.scaler.inverse_transform(r_predictions)[:, 0]

        r_y = np.repeat(y, 5, axis = -1)
        final_y = self.scaler.inverse_transform(r_y)[:, 0]

        plt.plot(self.dates, final_predictions)
        plt.plot(self.dates, final_y)
        plt.xlabel("Day")
        plt.ylabel("Stock Price in ($)")
        plt.title(f"LSTM MODEL OUTPUT\nMEAN SQUARED ERROR: {mse(y, predictions)}")
        plt.legend(["PREDICTION", "ACTUAL"])
        plt.show()
    
    def LSTM(self):
        model1 = Sequential()
        model1.add(InputLayer((14, 5)))
        model1.add(LSTM(64))
        model1.add(Dense(8, 'relu'))
        model1.add(Dense(1, 'linear'))

        cp1 = ModelCheckpoint('lstm_model/', save_best_only=True)
        model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

        model1.fit(self.trainX, self.trainY, validation_data=(self.valX, self.valY), epochs=10, callbacks=[cp1])
        print("********LSTM MODEL HAS SUCCESSFULLY BEEN TRAINED********")

        self.plot_predictions1(model1, self.testX, self.testY)

        return model1
    
    # def LSTM(self):
    #     model1 = Sequential()
    #     model1.add(InputLayer((14, 5)))
    #     model1.add(LSTM(64, return_sequences=True))
    #     model1.add(LSTM(32))
    #     model1.add(Dense(8, 'relu'))
    #     model1.add(Dense(1, 'linear'))

    #     cp1 = ModelCheckpoint('lstm_model/', save_best_only=True)
    #     model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

    #     model1.fit(self.trainX, self.trainY, validation_data=(self.valX, self.valY), epochs=10, callbacks=[cp1])
    #     print("********LSTM MODEL HAS SUCCESSFULLY BEEN TRAINED********")

    #     self.plot_predictions1(model1, self.testX, self.testY)

    #     return model1

    # def LSTM(self):
    #     model1 = Sequential()
    #     model1.add(InputLayer((14, 5)))
    #     model1.add(LSTM(32, return_sequences=True))
    #     model1.add(LSTM(64))
    #     model1.add(Dense(8, 'relu'))
    #     model1.add(Dense(1, 'linear'))

    #     cp1 = ModelCheckpoint('lstm_model/', save_best_only=True)
    #     model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

    #     model1.fit(self.trainX, self.trainY, validation_data=(self.valX, self.valY), epochs=10, callbacks=[cp1])
    #     print("********LSTM MODEL HAS SUCCESSFULLY BEEN TRAINED********")

    #     self.plot_predictions1(model1, self.testX, self.testY)

    #     return model1
    
    def GRU(self):
        model2 = Sequential()
        model2.add(InputLayer((14, 5)))
        model2.add(GRU(64))
        model2.add(Dense(8, 'relu'))
        model2.add(Dense(1, 'linear'))

        cp2 = ModelCheckpoint('gru_model/', save_best_only=True)
        model2.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

        model2.fit(self.trainX, self.trainY, validation_data=(self.valX, self.valY), epochs=10, callbacks=[cp2])
        print("********GRU MODEL HAS SUCCESSFULLY BEEN TRAINED********")

        self.plot_predictions1(model2, self.testX, self.testY)

        return model2
    
    # def GRU(self):
    #     model2 = Sequential()
    #     model2.add(InputLayer((14, 5)))
    #     model2.add(GRU(64, return_sequences=True))
    #     model2.add(GRU(32))
    #     model2.add(Dense(8, 'relu'))
    #     model2.add(Dense(1, 'linear'))

    #     cp2 = ModelCheckpoint('gru_model/', save_best_only=True)
    #     model2.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

    #     model2.fit(self.trainX, self.trainY, validation_data=(self.valX, self.valY), epochs=10, callbacks=[cp2])
    #     print("********GRU MODEL HAS SUCCESSFULLY BEEN TRAINED********")

    #     self.plot_predictions1(model2, self.testX, self.testY)

    #     return model2

    # def GRU(self):
    #     model2 = Sequential()
    #     model2.add(InputLayer((14, 5)))
    #     model2.add(GRU(32, return_sequences=True))
    #     model2.add(GRU(64))
    #     model2.add(Dense(8, 'relu'))
    #     model2.add(Dense(1, 'linear'))

    #     cp2 = ModelCheckpoint('gru_model/', save_best_only=True)
    #     model2.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

    #     model2.fit(self.trainX, self.trainY, validation_data=(self.valX, self.valY), epochs=10, callbacks=[cp2])
    #     print("********GRU MODEL HAS SUCCESSFULLY BEEN TRAINED********")

    #     self.plot_predictions1(model2, self.testX, self.testY)

    #     return model2

    def CNN1d(self):
        model3 = Sequential()
        model3.add(InputLayer((14, 5)))
        model3.add(Conv1D(64, kernel_size=2))
        model3.add(Flatten())
        model3.add(Dense(8, 'relu'))
        model3.add(Dense(1, 'linear'))

        cp3 = ModelCheckpoint('cnn1d_model/', save_best_only=True)
        model3.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

        model3.fit(self.trainX, self.trainY, validation_data=(self.valX, self.valY), epochs=10, callbacks=[cp3])
        print("********GRU MODEL HAS SUCCESSFULLY BEEN TRAINED********")

        self.plot_predictions1(model3, self.testX, self.testY)