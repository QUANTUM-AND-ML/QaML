import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

leaky_relu = nn.LeakyReLU(negative_slope = 0.02)

# Defining the GAT model
class GATRegressor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads):
        super(GATRegressor, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels , heads=heads, concat=False)
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels , hidden_channels, heads=heads, concat=False))
        self.conv_last = GATConv(hidden_channels , out_channels, heads=heads, concat=False)
        self.fc1 = torch.nn.Linear(3, 12)
        self.fc2 = torch.nn.Linear(12, 12)
        self.lin_layer1 = torch.nn.Linear(out_channels*4 + 12, 256)
        self.lin_layer2 = torch.nn.Linear(256, 128)
        self.lin_layer3 = torch.nn.Linear(128, 64)
        self.lin_layer4 = torch.nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        last_node_indices = data.node_index.long()
        x = self.conv1(x, edge_index)
        x = leaky_relu(x)

        # Select global features based on the selected node index
        global_features = x[last_node_indices].float()
        #print(global_features)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = leaky_relu(x)
            global_features = torch.cat((global_features, x[last_node_indices]), dim=1).float()

        x = self.conv_last(x, edge_index)
        #print(len(x))
        x = leaky_relu(x)

        # Node pooling using expectation testing
        global_features = torch.cat((global_features, x[last_node_indices]), dim=1).float()

        split_tensors = torch.split(data.graph_attr, 3)
        data.graph_attr = torch.stack(split_tensors)
        y = self.fc1(data.graph_attr.float())
        y = leaky_relu(y)
        y = self.fc2(y)
        y = leaky_relu(y)
        x = torch.cat([y, global_features], dim=1).float()
        x = self.lin_layer1(x)
        x = leaky_relu(x)
        x = self.lin_layer2(x)
        x = leaky_relu(x)
        x = self.lin_layer3(x)
        x = leaky_relu(x)
        x = self.lin_layer4(x)
        return x

# Define the loss function
criterion = torch.nn.MSELoss()

# Define the training function
def train(model, train_loader, val_loader, optimizer, num_epochs):
    best_val_loss = float('inf')
    best_model_weights = None
    for epoch in range(num_epochs):
        #total_loss = 0  # Initialize the total loss to 0
        # Training model
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for data in train_loader:
            optimizer.zero_grad()
            output = model(data)
            target = data.y  # Use the labels of the entire image as the target
            # Adjust the shape of the target before calculating the loss
            target = target.unsqueeze(1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = output
            train_total += data.y.size(0)
            threshold = 0.05
            train_correct += (torch.abs(predicted - data.y) < threshold).sum().item()

            #total_loss += loss.item() * data.num_graphs  # Accumulated total loss, considering the number of graphs
        train_acc = 100 * train_correct / train_total
        train_loss /= len(train_loader)
        
        # Validate Dataset
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data in val_loader:
                output = model(data)
                target = data.y  # Use the labels of the entire image as the target
                # Adjust the shape of the target before calculating the loss
                target = target.unsqueeze(1)
                loss = criterion(output, target)
                val_loss += loss.item()
                predicted1 = output
                val_total += data.y.size(0)
                threshold = 0.05
                train_correct += (torch.abs(predicted1 - data.y) < threshold).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()
        # Print training and verification information
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print('---------------------------------------')

    return best_val_loss, best_model_weights
