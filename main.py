import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from Dataset import MyDataset
import matplotlib.pyplot as plt
from GNN import GATRegressor, train
from sklearn.metrics import r2_score
import numpy as np


# Load Dataset
dataset = MyDataset(root='data/')
#dataset = dataset.shuffle()
data = dataset[0]
print(data.node_index)

num_data = len(dataset)
print(num_data)
num_train = int(0.8 * num_data)
num_val = int(0.1 * num_data)  # Set the proportion of validation sets
num_test = num_data - num_train - num_val
train_dataset = dataset[: num_train]
val_dataset = dataset[num_train:num_train + num_val]  # Divide the validation set from the remaining parts
test_dataset = dataset[num_train + num_val:]

print('train_dataset', len(train_dataset))
print('val_dataset', len(val_dataset))
print('test_dataset', len(test_dataset))

# Building a Data Loader
batch_size = 512
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size = batch_size)
test_loader = DataLoader(test_dataset, batch_size = len(test_dataset))

#print(test_dataset.graph_label.view(-1, 1).float())
# initial model
in_features = dataset.num_node_features
hidden_features = 31
out_features = 31
num_layers = 4
num_heads = 2
model = GATRegressor(in_features, hidden_features, out_features, num_layers, num_heads)
print(model)

# Define Loss function and optimizer
#criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 200

# Test-and-set equipment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# training model
best_train_loss, best_weights = train(model, train_loader, train_loader, optimizer , num_epochs)

# Create two lists to collect all predicted and actual values
all_outputs = []
all_targets = []

# Use the weight parameters corresponding to the minimum training loss
model.load_state_dict(best_weights)

# Evaluate model performance on the test set
model.eval()
with torch.no_grad():  # Gradient calculation is not required during the evaluation process
    for data in test_loader:
        output = model(data)
        for i in range(len(output)):
            print(output[i], data.y[i])
            all_outputs.append(output[i].item())
            all_targets.append(data.y[i].item())

# Convert the list to a numpy array
all_outputs = np.array(all_outputs)
all_targets = np.array(all_targets)

# Calculate the R^2 score
r2 = r2_score(all_targets, all_outputs)

plt.scatter(all_targets, all_outputs)

# Add text in the upper right corner and insert the R ^ 2 score into the string
plt.text(0.1, 0.9, '$R^2$ = {:.3f}'.format(r2), transform=plt.gca().transAxes,  fontsize=15)

line_x = np.linspace(min(all_targets), max(all_targets), 100) 
line_y = line_x 
plt.plot(line_x, line_y, 'r--')

plt.title("Benchmark")
plt.xlabel("Expectation Values of the Noisy Device")
plt.ylabel("Predicted Expectation Values")
plt.show()
