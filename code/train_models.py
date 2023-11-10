from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class Models():
    def __init__(self, trainX, trainY, valX, valY, testX, testY) -> None:
        self.trainX = trainX
        self.trainY = trainY
        self.valX = valX
        self.valY = valY
        self.testX = testX
        self.testY = testY

    def plot_predictions1(self, model, X, y, start=0, end=50):
        scaler = StandardScaler()

        print("********PREDICTING THE FUTURE********")
        predictions = model.predict(X).flatten()
        # r_predictions = np.repeat(predictions, 5, axis = -1)
        # final_predictions = scaler.inverse_transform(r_predictions)[:, 0]

        # r_y = np.repeat(y, 5, axis=1)
        # final_y = scaler.inverse_transform(r_y)[:, 0]
        
        plt.plot([i for i in range(len(predictions))], predictions)
        plt.plot([i for i in range(len(y))], y)
        # plt.plot([i for i in range(len(final_predictions))], final_predictions)
        # plt.plot([i for i in range(len(r_y))], r_y)
        plt.title(f"LSTM MODEL OUTPUT\MEAN SQUARED ERROR: {mse(y, predictions)}")
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
    
    def GRU(self):
        print("GRU IS A WORK IN PROGRESS")

    def CNN1d(self):
        print("CNN1d IS A WORK IN PROGRESS")