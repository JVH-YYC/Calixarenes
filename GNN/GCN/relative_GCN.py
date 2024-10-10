import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativeGCNModel(nn.Module):
    def __init__(self, radius, T, input_feature_dim, input_bond_dim,
                 fingerprint_dim, output_units_num, p_dropout):
        super(RelativeGCNModel, self).__init__()
        
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
        
        # More complex output layer for pairwise prediction
        self.output = nn.Sequential(
            nn.Linear(fingerprint_dim * 2, hidden_dim),  # Input is concatenated features of two molecules
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_units_num)
        )

    def forward(self, atom_list1, bond_list1, atom_degree_list1, bond_degree_list1, atom_mask1,
                atom_list2, bond_list2, atom_degree_list2, bond_degree_list2, atom_mask2):
        
        _size, mol_length, num_atom_feat = atom_list1.size()

        # Process first molecule
        atom_feature1, mol_feature1 = self.process_molecule(atom_list1, bond_list1, atom_degree_list1, bond_degree_list1, atom_mask1)
        
        # Process second molecule
        atom_feature2, mol_feature2 = self.process_molecule(atom_list2, bond_list2, atom_degree_list2, bond_degree_list2, atom_mask2)
        
        # Concatenate features of both molecules and predict pairwise difference
        pairwise_feature = torch.cat([mol_feature1, mol_feature2], dim=1)
        pairwise_prediction = self.output(self.dropout(pairwise_feature))
        
        return atom_feature1, atom_feature2, pairwise_prediction

    def process_molecule(self, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask):
        batch_size, mol_length, num_atom_feat = atom_list.size()
        atom_mask = atom_mask.unsqueeze(2)
        
        #print("1. atom_list shape:", atom_list.shape)
        atom_feature = self.atom_fc(atom_list)
        #print("2. atom_feature shape after atom_fc:", atom_feature.shape)
        
        neighbor_feature = self.get_neighbor_feature(atom_list, bond_list, atom_degree_list, bond_degree_list, batch_size, mol_length)
        #print("3. neighbor_feature shape:", neighbor_feature.shape)
        
        for i, layer in enumerate(self.gcn_layers):
            neighbor_feature = layer(neighbor_feature)
           # print(f"4. neighbor_feature shape after gcn layer {i}:", neighbor_feature.shape)
            atom_feature = atom_feature + neighbor_feature
            atom_feature = self.dropout(atom_feature)
        #    print(f"5. atom_feature shape after layer {i}:", atom_feature.shape)
        
        mol_feature = torch.sum(F.relu(atom_feature) * atom_mask, dim=1)
       # print("6. mol_feature shape:", mol_feature.shape)
        return atom_feature, mol_feature

    def get_neighbor_feature(self, atom_list, bond_list, atom_degree_list, bond_degree_list, batch_size, mol_length):
        neighbor_feature = []
        for i in range(batch_size):
            atom_degrees = atom_degree_list[i]
            bond_degrees = bond_degree_list[i]
        
            # Ensure indices are within bounds
            atom_degrees = torch.clamp(atom_degrees, 0, mol_length - 1)
            bond_degrees = torch.clamp(bond_degrees, 0, bond_list.size(1) - 1)
        
            atom_features = atom_list[i]
            bond_features = bond_list[i]
        
            # Gather neighbor features
            neighbor_atoms = atom_features[atom_degrees]
            neighbor_bonds = bond_features[bond_degrees]
            
            # Combine atom and bond features
            mol_neighbor_feature = torch.cat([neighbor_atoms, neighbor_bonds], dim=-1)
        
            # Apply neighbor_fc to each atom's neighborhood
            mol_neighbor_feature = self.neighbor_fc(mol_neighbor_feature)
            
            # Instead of summing, we'll keep the full neighborhood information
            neighbor_feature.append(mol_neighbor_feature)
            
        neighbor_feature = torch.stack(neighbor_feature, dim=0)
        neighbor_feature = neighbor_feature.mean(dim=2)   #?) It is to reduce the dimension! Is it ok to keep it?? is there any feature missing? 

        return F.relu(neighbor_feature)