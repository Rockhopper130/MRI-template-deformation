import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim=None, activation=nn.ReLU, dropout=0.0):
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(a, b))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        final_out = out_dim if out_dim is not None else dims[-1]
        layers.append(nn.Linear(dims[-1], final_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class CRSM(nn.Module):
    def __init__(self, radial_channel_indices, conical_depth_indices=None, aggregation='mean', mlp_hidden=(128,), mlp_out_dim=None, mlp_activation=nn.ReLU, mlp_dropout=0.0):
        super().__init__()
        self.register_buffer('radial_idx', torch.tensor(radial_channel_indices, dtype=torch.long))
        self.register_buffer('conical_idx', torch.tensor(conical_depth_indices, dtype=torch.long))
        
        assert aggregation in ('mean', 'max')
        self.aggregation = aggregation
        
        in_dim = len(self.radial_idx) + len(self.conical_idx)
        out_dim = mlp_out_dim if mlp_out_dim is not None else in_dim
        self.mlp = MLP(in_dim, mlp_hidden, out_dim, activation=mlp_activation, dropout=mlp_dropout)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        device = x.device
        
        # Move indices to the same device as input tensor
        radial_idx = self.radial_idx.to(device)
        conical_idx = self.conical_idx.to(device)
        
        radial = x[:, radial_idx]
        ci = conical_idx
        conical_feats = x[:, ci]
        if edge_index is None:
            agg = conical_feats
        else:
            ei = edge_index
            if ei.dim() != 2 or ei.size(0) != 2:
                raise ValueError('edge_index must be shape [2, E]')
            i = ei[0]
            j = ei[1]
            sym_i = torch.cat([i, j], dim=0)
            sym_j = torch.cat([j, i], dim=0)
            indices = torch.stack([sym_i, sym_j], dim=0)
            values = torch.ones(indices.size(1), device=device)
            N = x.size(0)
            adj = torch.sparse_coo_tensor(indices, values, (N, N), device=device).coalesce()
            neighbor_sum = torch.sparse.mm(adj, conical_feats)
            deg = torch.bincount(indices[0], minlength=N).to(device).unsqueeze(1).clamp(min=1)
            if self.aggregation == 'mean':
                agg = neighbor_sum / deg
            else:
                row_indices = indices[0]
                col_indices = indices[1]
                src = conical_feats[col_indices]
                max_per_row = torch.full((N, conical_feats.size(1)), float('-inf'), device=device)
                for r, s in zip(row_indices.tolist(), src.tolist()):
                    s_tensor = torch.tensor(s, device=device)
                    max_per_row[r] = torch.maximum(max_per_row[r], s_tensor)
                isolated = (max_per_row == float('-inf'))
                max_per_row[isolated] = 0.0
                agg = max_per_row
        combined = torch.cat([radial, agg], dim=1)
        out = self.mlp(combined)
        data_out = type(data)()
        data_out.x = out
        data_out.edge_index = edge_index
        return data_out