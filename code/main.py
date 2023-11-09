from dataset_setup import train_data
from LSTM import LSTM
import torch    

input_dim = 5
hidden_dim = 32
num_layers = 2
output_dim = 2

# dateset class
obj = train_data('A:\Academics_NEU\sem_6\TSLA.csv',13,1)
trainX , trainY =obj.csv_read()
# print(trainX)
# ***********************************
# LSTM model
model = LSTM( input_dim=input_dim, hidden_dim=hidden_dim, output_dim = output_dim, num_layers=num_layers)

loss_fn = torch.nn.MSELoss()

optimiser = torch.optim.Adam(model.parameters(), lr = 0.01)
print(model)
print(len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())