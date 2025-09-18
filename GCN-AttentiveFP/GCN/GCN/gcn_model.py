import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNModel(nn.Module):
    def __init__(self, radius, T, input_feature_dim, input_bond_dim,
                 fingerprint_dim, output_units_num, p_dropout):
        super(GCNModel, self).__init__()
        
        self.radius = radius
        self.T = T
        
        # Increase the dimensionality of hidden layers
        hidden_dim = fingerprint_dim * 4  # Significantly larger hidden dimension
        
        self.atom_fc = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, fingerprint_dim)
        )
        
        self.neighbor_fc = nn.Sequential(
            nn.Linear(input_feature_dim + input_bond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, fingerprint_dim)
        )
        
        # Increase the number and size of GCN layers
        self.gcn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fingerprint_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, fingerprint_dim)
            ) for _ in range(radius * 2)  # Double the number of GCN layers
        ])
        
        self.dropout = nn.Dropout(p=p_dropout)
        
        # More complex output layer
        self.output = nn.Sequential(
            nn.Linear(fingerprint_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_units_num)
        )

    def forward(self, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask):
        batch_size, mol_length, num_atom_feat = atom_list.size()
        atom_mask = atom_mask.unsqueeze(2)
        
        atom_feature = self.atom_fc(atom_list)
        neighbor_feature = self.get_neighbor_feature(atom_list, bond_list, atom_degree_list, bond_degree_list, batch_size, mol_length)
        
        for layer in self.gcn_layers:
            neighbor_feature = layer(neighbor_feature)
            atom_feature = atom_feature + neighbor_feature
            atom_feature = self.dropout(atom_feature)
        
        mol_feature = torch.sum(F.relu(atom_feature) * atom_mask, dim=1)
        mol_prediction = self.output(self.dropout(mol_feature))
        
        return atom_feature, mol_prediction


    def get_neighbor_feature(self, atom_list, bond_list, atom_degree_list, bond_degree_list, batch_size, mol_length):
        neighbor_feature = []
        for i in range(batch_size):
            atom_degrees = atom_degree_list[i]
            bond_degrees = bond_degree_list[i]
        
            # Ensure indices are within bounds
            atom_degrees = torch.clamp(atom_degrees, 0, mol_length - 1)
            bond_degrees = torch.clamp(bond_degrees, 0, bond_list.size(1) - 1)
        
        # No need to pad, use the original tensors
            atom_features = atom_list[i]
            bond_features = bond_list[i]
        
        # Gather neighbor features
            neighbor_atoms = atom_features[atom_degrees]
            neighbor_bonds = bond_features[bond_degrees]
            
        # Combine atom and bond features
            mol_neighbor_feature = torch.cat([neighbor_atoms, neighbor_bonds], dim=-1)
        
        # Apply neighbor_fc to each atom's neighborhood and sum
            mol_neighbor_feature = self.neighbor_fc(mol_neighbor_feature)
            mol_neighbor_feature = mol_neighbor_feature.sum(dim=1)
        
            neighbor_feature.append(mol_neighbor_feature)
    
        neighbor_feature = torch.stack(neighbor_feature, dim=0)
        return F.relu(neighbor_feature)
    
    
    
    
    