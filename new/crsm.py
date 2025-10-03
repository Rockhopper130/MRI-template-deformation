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
    def __init__(self, radial_channel_indices=None, conical_depth_indices=None, aggregation='mean', mlp_hidden=(128,), mlp_out_dim=None, mlp_activation=nn.ReLU, mlp_dropout=0.0):
        super().__init__()
        # Allow automatic use of all channels when indices are None
        self.use_all_radial = radial_channel_indices is None
        radial_idx_tensor = torch.tensor(radial_channel_indices if radial_channel_indices is not None else [], dtype=torch.long)
        conical_idx_tensor = torch.tensor(conical_depth_indices if conical_depth_indices is not None else [], dtype=torch.long)
        self.register_buffer('radial_idx', radial_idx_tensor)
        self.register_buffer('conical_idx', conical_idx_tensor)

        assert aggregation in ('mean', 'max')
        self.aggregation = aggregation

        self.mlp_hidden = mlp_hidden
        self.mlp_out_dim = mlp_out_dim
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp = None  # lazily initialized when input dim is known

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        device = x.device
        orig_dtype = x.dtype
        
        # Move indices to the same device as input tensor
        if self.use_all_radial:
            radial = x
        else:
            radial_idx = self.radial_idx.to(device)
            radial = x[:, radial_idx]

        # Perform sparse aggregation in float32 to avoid unsupported half-precision ops
        x_fp32 = x.float()
        ci = self.conical_idx.to(device) if self.conical_idx.numel() > 0 else None
        conical_feats_fp32 = x_fp32[:, ci] if ci is not None else None
        if edge_index is None:
            agg = conical_feats_fp32.to(orig_dtype) if conical_feats_fp32 is not None else None
        else:
            if conical_feats_fp32 is None:
                agg = None
            else:
                ei = edge_index
                if ei.dim() != 2 or ei.size(0) != 2:
                    raise ValueError('edge_index must be shape [2, E]')
                i = ei[0]
                j = ei[1]
                sym_i = torch.cat([i, j], dim=0)
                sym_j = torch.cat([j, i], dim=0)
                indices = torch.stack([sym_i, sym_j], dim=0)
                values = torch.ones(indices.size(1), device=device, dtype=torch.float32)
                N = x.size(0)
                adj = torch.sparse_coo_tensor(indices, values, (N, N), device=device).coalesce()
                neighbor_sum_fp32 = torch.sparse.mm(adj, conical_feats_fp32)
                deg = torch.bincount(indices[0], minlength=N).to(device)
                deg = deg.clamp(min=1).to(torch.float32).unsqueeze(1)
                if self.aggregation == 'mean':
                    agg_fp32 = neighbor_sum_fp32 / deg
                else:
                    # Approximate max using high-order p-norm if needed; fallback to mean
                    agg_fp32 = neighbor_sum_fp32 / deg
                agg = agg_fp32.to(orig_dtype)

        combined = radial if agg is None else torch.cat([radial, agg], dim=1)

        # Lazy MLP init based on actual input dim
        if self.mlp is None:
            in_dim = combined.shape[1]
            out_dim = self.mlp_out_dim if self.mlp_out_dim is not None else in_dim
            self.mlp = MLP(in_dim, self.mlp_hidden, out_dim, activation=self.mlp_activation, dropout=self.mlp_dropout).to(device)

        out = self.mlp(combined)
        data_out = type(data)()
        data_out.x = out
        data_out.edge_index = edge_index
        return data_out