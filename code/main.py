from dataset_setup import train_data

obj = train_data('A:\Academics_NEU\sem_6\TSLA.csv',13,1)
trainX , trainY =obj.csv_read()
print(trainX)