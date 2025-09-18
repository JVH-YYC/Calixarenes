import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Fingerprint(nn.Module):
    def __init__(self, radius, T, input_feature_dim, input_bond_dim,
                 fingerprint_dim, output_units_num, p_dropout):
        super(Fingerprint, self).__init__()
        
        self.radius = radius
        self.T = T
        
        # Atom embedding
        self.atom_fc = nn.Linear(input_feature_dim, fingerprint_dim)
        self.neighbor_fc = nn.Linear(input_feature_dim + input_bond_dim, fingerprint_dim)
        self.gru_cells = nn.ModuleList([nn.GRUCell(fingerprint_dim, fingerprint_dim) for _ in range(radius)])
        self.align_layers = nn.ModuleList([nn.Linear(2 * fingerprint_dim, 1) for _ in range(radius)])
        self.attend_layers = nn.ModuleList([nn.Linear(fingerprint_dim, fingerprint_dim) for _ in range(radius)])
        
        # Molecule embedding
        self.mol_gru_cell = nn.GRUCell(fingerprint_dim, fingerprint_dim)
        self.mol_align = nn.Linear(2 * fingerprint_dim, 1)
        self.mol_attend = nn.Linear(fingerprint_dim, fingerprint_dim)
        
        self.dropout = nn.Dropout(p=p_dropout)
        self.output = nn.Linear(fingerprint_dim, output_units_num)

    def forward(self, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask):
        batch_size, mol_length, num_atom_feat = atom_list.size()
        atom_mask = atom_mask.unsqueeze(2)
        
        atom_feature = F.leaky_relu(self.atom_fc(atom_list))
        neighbor_feature = self.get_neighbor_feature(atom_list, bond_list, atom_degree_list, bond_degree_list, batch_size)
        
        attend_mask, softmax_mask = self.create_masks(atom_degree_list, mol_length)
        
        atom_feature = self.apply_atom_attention(atom_feature, neighbor_feature, attend_mask, softmax_mask)
        mol_feature = torch.sum(F.relu(atom_feature) * atom_mask, dim=-2)
        
        mol_feature = self.apply_molecule_attention(atom_feature, mol_feature, atom_mask)
        
        mol_prediction = self.output(self.dropout(mol_feature))
        
        return atom_feature, mol_prediction

    def get_neighbor_feature(self, atom_list, bond_list, atom_degree_list, bond_degree_list, batch_size):
        bond_neighbor = torch.stack([bond_list[i][bond_degree_list[i]] for i in range(batch_size)], dim=0)
        atom_neighbor = torch.stack([atom_list[i][atom_degree_list[i]] for i in range(batch_size)], dim=0)
        neighbor_feature = torch.cat([atom_neighbor, bond_neighbor], dim=-1)
        return F.leaky_relu(self.neighbor_fc(neighbor_feature))

    def create_masks(self, atom_degree_list, mol_length):
        attend_mask = (atom_degree_list != mol_length - 1).float().unsqueeze(-1)
        softmax_mask = torch.where(atom_degree_list == mol_length - 1, torch.tensor(-9e8).cuda(), torch.tensor(0.).cuda()).unsqueeze(-1)
        return attend_mask, softmax_mask

    def apply_atom_attention(self, atom_feature, neighbor_feature, attend_mask, softmax_mask):
        #print("Applying atom-level attention")
        batch_size, mol_length = atom_feature.shape[:2]
        
        for d in range(self.radius):
            atom_feature_expand = atom_feature.unsqueeze(-2).expand(-1, -1, neighbor_feature.size(2), -1)
            feature_align = torch.cat([atom_feature_expand, neighbor_feature], dim=-1)
            
            align_score = F.leaky_relu(self.align_layers[d](self.dropout(feature_align))) + softmax_mask
            attention_weight = F.softmax(align_score, -2) * attend_mask
            
            context = torch.sum(attention_weight * self.attend_layers[d](self.dropout(neighbor_feature)), -2)
            context = F.elu(context)
            
            atom_feature = self.gru_cells[d](
                context.view(batch_size * mol_length, -1),
                atom_feature.view(batch_size * mol_length, -1)
            ).view(batch_size, mol_length, -1)
            
            neighbor_feature = F.relu(atom_feature).unsqueeze(-2).expand(-1, -1, neighbor_feature.size(2), -1)
        
        return atom_feature

    def apply_molecule_attention(self, atom_feature, mol_feature, atom_mask):
        #print("Applying mol-level attention")
        batch_size, mol_length = atom_feature.shape[:2]
        mol_softmax_mask = torch.where(atom_mask.squeeze() == 0, torch.tensor(-9e8).cuda(), torch.tensor(0.).cuda())
        
        for _ in range(self.T):
            mol_prediction_expand = mol_feature.unsqueeze(-2).expand(-1, mol_length, -1)
            mol_align = torch.cat([mol_prediction_expand, atom_feature], dim=-1)
            mol_align_score = F.leaky_relu(self.mol_align(mol_align)) + mol_softmax_mask.unsqueeze(-1)
            mol_attention_weight = F.softmax(mol_align_score, -2) * atom_mask
            
            mol_context = torch.sum(mol_attention_weight * self.mol_attend(self.dropout(atom_feature)), -2)
            mol_context = F.elu(mol_context)
            mol_feature = self.mol_gru_cell(mol_context, mol_feature)
        
        return mol_feature