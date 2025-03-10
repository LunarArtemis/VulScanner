import torch
from torch.nn import Linear, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool as gap, global_max_pool as gmp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Hyperparameters
embedding_size = 128
dropout_rate = 0.3
learning_rate = 0.005
patience = 10
num_epochs = 100

class GCN(torch.nn.Module):
    def __init__(self):
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.initial_conv = GCNConv(dataset.num_features, embedding_size) #to  translate our node features into the size of the embedding
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        # pooling layer
        #self.pool = TopKPooling(embedding_size, ratio=0.8)
        #dropout layer
        #self.dropout = Dropout(p=0.2)

        # Output layer
        self.lin1 = Linear(embedding_size*2, 128) # linear output layer ensures that we get a continuous unbounded output value. It input is the flattened vector (embedding size *2) from the pooling layer (mean and max)
        self.lin2 = Linear(128, 128)
        self.lin3 = Linear(128, 1)

        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, x, edge_index, batch_index):
        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = F.relu(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = F.relu(hidden)

        hidden = self.conv2(hidden, edge_index)
        hidden = F.relu(hidden)
        #hidden = self.dropout(hidden)
        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index), 
                            gap(hidden, batch_index)], dim=1)
        # Apply a final (linear) classifier.
        out = self.lin1(hidden)
        out = self.act1(out)
        out = self.lin2(out)
        out = self.act2(out)
        #out = F.dropout(out, p=0.5, training=self.training)
        out = self.lin3(out)
        out = torch.sigmoid(out)

        # return out, hidden
        return out

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize model
#model = GCN(num_features=dataset.num_features).to(device)
model = GCN().to(device)
# Update optimizer with weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
loss_fn = torch.nn.BCELoss()

# Training function
def train():
    model.train()

    loss_all = 0
    for data in train_dataset_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x.float(), data.edge_index, data.batch)
        label = data.y.to(device)
        #loss = torch.sqrt(loss_fn(output, label))  
        loss = loss_fn(output.squeeze(), label.float())  
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)

# Evaluation function
def evaluate(loader):
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for data in loader:

            data = data.to(device)
            # pred = model(data.x.float(), data.edge_index, data.batch).detach().cpu().numpy()
            pred = model(data.x.float(), data.edge_index, data.batch)
            label_true = data.y.to(device)
            label = data.y.detach().cpu().numpy()
            # predictions.append(pred)
            # labels.append(label)
            predictions.append(np.rint(pred.cpu().detach().numpy()))
            labels.append(label)
            loss = loss_fn(pred.squeeze(), label_true.float())
    # predictions = np.hstack(predictions)
    # labels = np.hstack(labels)
    predictions = np.concatenate(predictions).ravel()
    labels = np.concatenate(labels).ravel()

    # print(predictions)
    # print(labels)
    return accuracy_score(labels, predictions), loss