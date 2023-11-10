from dataset_setup import train_data
from LSTM import LSTM
import torch    
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
trainX , trainY =obj.csv_read()

# plt.plot([i for i in range(len(trainY))], trainY)
# plt.show()

# LSTM model
model = LSTM( input_dim=input_dim, hidden_dim=hidden_dim, output_dim = output_dim, num_layers=num_layers)

loss_fn = torch.nn.MSELoss()

optimiser = torch.optim.Adam(model.parameters(), lr = 0.01)
print(model)
print(len(list(model.parameters())))
for i in tqdm(range(len(list(model.parameters())))):
    print(list(model.parameters())[i].size())

num_epochs = 100
hist = np.zeros(num_epochs)
seq_dim = past_days

for t in tqdm(range(num_epochs)):
    y_train_pred = model(trainX)
    loss = loss_fn(y_train_pred, trainY)
    if t%10==0 and t!=0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t]= loss.item()

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()