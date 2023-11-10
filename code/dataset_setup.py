import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class train_data:
    def __init__(self, csv, past, future) -> None:
        self.csv = csv
        self.past = past
        self.future = future
    
    def csv_read(self):
        df = pd.read_csv(self.csv)
        dates = pd.to_datetime(df['Date'])
        
        cols = list(df)[1:6]
        for_training = df[cols].astype(float)
        
        scaler = StandardScaler()
        scaler  = scaler.fit(for_training)
        scaled_for_training = scaler.transform(for_training)
        
        trainX = []
        trainY = []
        valX = []
        valY = []
        testX = []
        testY = []
        
        for i in tqdm(range(self.past, len(scaled_for_training) - self.future + 1)):
            if i < len(scaled_for_training) - 200:
                trainX.append(scaled_for_training[i - self.past:i, 0:for_training.shape[1]])
                trainY.append(scaled_for_training[i + self.future - 1:i + self.future, 0])
            elif i < len(scaled_for_training) - 50:
                valX.append(scaled_for_training[i - self.past:i, 0:for_training.shape[1]])
                valY.append(scaled_for_training[i + self.future - 1:i + self.future, 0])
            else:
                testX.append(scaled_for_training[i - self.past:i, 0:for_training.shape[1]])
                testY.append(scaled_for_training[i + self.future - 1:i + self.future, 0])
                
        trainX, trainY, valX, valY, testX, testY = np.array(trainX), np.array(trainY), np.array(valX), np.array(valY), np.array(testX), np.array(testY)
        return trainX, trainY, valX, valY, testX, testY, scaler, dates[-50:]