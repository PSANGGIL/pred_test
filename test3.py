import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import rdmolops
import numpy as np

class DMPNNLayer(nn.Module):
    def __init__(self, hidden_dim, edge_dim):
        super(DMPNNLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.W = nn.Linear(hidden_dim + edge_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h, edge_attr, adj):
        # h: node hidden states [num_nodes, hidden_dim]
        # edge_attr: edge attributes [num_edges, edge_dim]
        # adj: adjacency matrix [num_nodes, num_nodes]

        messages = torch.zeros_like(h)

        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i, j] == 1:
                    edge_features = edge_attr[i, j]
                    combined = torch.cat([h[i], edge_features], dim=-1)
                    message = self.W(combined)
                    messages[j] += message

        h_new = self.U(h + messages)
        return h_new

class DMPNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers):
        super(DMPNN, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList([DMPNNLayer(hidden_dim, edge_dim) for _ in range(num_layers)])
        self.readout = nn.Linear(hidden_dim, 1)  # Example readout layer for regression

    def forward(self, node_features, edge_features, adj):
        h = self.node_embedding(node_features)
        for layer in self.layers:
            h = layer(h, edge_features, adj)
        graph_rep = torch.sum(h, dim=0)  # Simple sum pooling for readout
        output = self.readout(graph_rep)
        return output

# Function to convert SMILES to graph representation
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
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
    
    return padded_adj, padded_atom_features, padded_edge_features

# Example usage
smiles = "CCO"  # Ethanol
mol, adj_matrix, atom_features, edge_features = smiles_to_graph(smiles)

# Determine max number of nodes for padding
max_nodes = 10  # Set to the desired maximum number of nodes for padding

# Pad graph features
adj_matrix, atom_features, edge_features = pad_graph(mol, adj_matrix, atom_features, edge_features, max_nodes)

# Initialize DMPNN
node_dim = 1  # Atom feature dimension (atomic number)
edge_dim = 3  # Edge feature dimension (one-hot encoded bond type)
hidden_dim = 16  # Hidden dimension for messages
num_layers = 3

dmpnn = DMPNN(node_dim, edge_dim, hidden_dim, num_layers)

# Forward pass through the DMPNN
atom_features = torch.tensor(atom_features, dtype=torch.float32).view(-1, 1)
edge_features = torch.tensor(edge_features, dtype=torch.float32)
adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)

output = dmpnn(atom_features, edge_features, adj_matrix)
print("Output:", output.item())

