import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdmolops
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Function to convert SMILES to graph representation
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    #mol = Chem.AddHs(mol)
    adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
    
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom.GetAtomicNum())  # Using atomic number as a simple feature
    atom_features = np.array(atom_features)
    
    edge_features = []
    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        bond_features = [0, 0, 0]
        if bond_type == Chem.rdchem.BondType.SINGLE:
            bond_features[0] = 1
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            bond_features[1] = 1
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            bond_features[2] = 1
        edge_features.append(bond_features)
    
    return mol, adj_matrix, atom_features, edge_features

# Function to pad graph features to the same size
def pad_graph(mol, adj_matrix, atom_features, edge_features, max_nodes):
    num_nodes = len(atom_features)
    
    # Pad adjacency matrix
    padded_adj = np.zeros((max_nodes, max_nodes))
    padded_adj[:num_nodes, :num_nodes] = adj_matrix
    
    # Pad atom features
    padded_atom_features = np.zeros((max_nodes, 1))  # Assuming node_dim is 1
    padded_atom_features[:num_nodes, 0] = atom_features
    
    # Pad edge features
    padded_edge_features = np.zeros((max_nodes, max_nodes, len(edge_features[0])))
    bond_dict = {frozenset([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]): edge_features[i] for i, bond in enumerate(mol.GetBonds())}
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] == 1:
                bond = frozenset([i, j])
                if bond in bond_dict:
                    padded_edge_features[i, j] = bond_dict[bond]
    
    return padded_adj, padded_atom_features, padded_edge_features, num_nodes

class MoleculeDataset(Dataset):
    def __init__(self, smiles_list, targets, max_nodes):
        self.smiles_list = smiles_list
        self.targets = targets
        self.max_nodes = max_nodes

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        mol, adj_matrix, atom_features, edge_features = smiles_to_graph(smiles)
        adj_matrix, atom_features, edge_features, num_nodes = pad_graph(mol, adj_matrix, atom_features, edge_features, self.max_nodes)
        target = self.targets[idx]
        return adj_matrix, atom_features, edge_features, target, num_nodes

def collate_fn(batch):
    adj_matrices, atom_features, edge_features, targets, num_nodes = zip(*batch)
    adj_matrices = torch.tensor(adj_matrices, dtype=torch.float32)
    atom_features = torch.tensor(atom_features, dtype=torch.float32).view(len(batch), -1, 1)
    edge_features = torch.tensor(edge_features, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    num_nodes = torch.tensor(num_nodes, dtype=torch.long)
    return adj_matrices, atom_features, edge_features, targets, num_nodes


class DMPNNLayer(nn.Module):
    def __init__(self, hidden_dim, edge_dim, dropout=0.5):
        super(DMPNNLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.W = nn.Linear(hidden_dim + edge_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_attr, adj, num_nodes):
        batch_size, max_nodes, _ = h.size()
        messages = torch.zeros_like(h)

        for i in range(max_nodes):
            for j in range(max_nodes):
                if adj[:, i, j].sum() > 0:
                    edge_features = edge_attr[:, i, j, :]
                    combined = torch.cat([h[:, i, :], edge_features], dim=-1)
                    message = self.W(combined)
                    messages[:, j, :] += message

        mask = torch.arange(max_nodes).unsqueeze(0) < num_nodes.unsqueeze(1)
        mask = mask.unsqueeze(2).to(h.device)
        messages = messages * mask

        h_new = self.U(h + messages)
        #h_new = self.dropout(h_new)  # Apply dropout
        return h_new

class DMPNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers, dropout=0.5):
        super(DMPNN, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList([DMPNNLayer(hidden_dim, edge_dim, dropout) for _ in range(num_layers)])
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, node_features, edge_features, adj, num_nodes):
        h = self.node_embedding(node_features)
        for layer in self.layers:
            h = layer(h, edge_features, adj, num_nodes)
        graph_rep = torch.stack([torch.sum(h[batch, :num_nodes[batch], :], dim=0) for batch in range(h.size(0))])
        output = self.readout(graph_rep)
        return output

# Example SMILES list and targets

import pandas as pd
data = pd.read_csv('./db.csv')[:50]
smiles_list = data['smi']
targets     = data['s_energy(ev)']

#smiles_list = ["CCO", "CCN", "CCC", "CCCl"]  # Replace with your dataset
#targets = [40, 38, 36, 59]  # Example targets (replace with actual data)
max_nodes = 50  # Set to the desired maximum number of nodes for padding

dataset = MoleculeDataset(smiles_list, targets, max_nodes)
data_loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)

# Initialize DMPNN
node_dim = 1  # Atom feature dimension (atomic number)
edge_dim = 3  # Edge feature dimension (one-hot encoded bond type)
hidden_dim = 3  # Hidden dimension for messages
num_layers = 3

dmpnn = DMPNN(node_dim, edge_dim, hidden_dim, num_layers)

# Training loop
optimizer = torch.optim.Adam(dmpnn.parameters(), lr=0.001)
criterion = nn.MSELoss()  # Example loss function
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.001)

num_epochs = 100  # Number of epochs for training

for epoch in range(num_epochs):
    for adj_matrices, atom_features, edge_features, targets, num_nodes in data_loader:
        optimizer.zero_grad()
        outputs = dmpnn(atom_features, edge_features, adj_matrices, num_nodes)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Prediction on new data
def predict(smiles_list, model, max_nodes):
    dataset = MoleculeDataset(smiles_list, [0]*len(smiles_list), max_nodes)  # Dummy targets
    data_loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
    
    model.eval()
    predictions = []
    with torch.no_grad():
        for adj_matrices, atom_features, edge_features, _, num_nodes in data_loader:
            outputs = model(atom_features, edge_features, adj_matrices, num_nodes)
            predictions.append(outputs.item())
    return predictions

# Example prediction
data_t = pd.read_csv('./db_t.csv')[:10]
new_smiles_list = data_t['smi']
target_t     = data_t['s_energy(ev)']


predictions = predict(new_smiles_list, dmpnn, max_nodes)
print((torch.tensor(target_t)))
print(torch.tensor(predictions, dtype = torch.float64))
print(torch.mean((torch.tensor(target_t) - torch.tensor(predictions, dtype = torch.float64))))
print("Predictions:", predictions)
