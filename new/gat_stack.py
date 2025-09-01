import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATBlock(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, concat=True, dropout=0.1):
        super().__init__()
        self.gat = GATConv(
            in_channels, out_channels, heads=heads, concat=concat, dropout=dropout
        )
        self.residual = nn.Linear(in_channels, out_channels * heads if concat else out_channels)
        self.norm = nn.LayerNorm(out_channels * heads if concat else out_channels)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        h = self.gat(x, edge_index)
        res = self.residual(x)
        out = self.norm(h + res)
        return self.act(out)

class GATStack(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, out_channels=16, num_layers=3, heads=4, dropout=0.1):
        super().__init__()
        layers = []
        layers.append(GATBlock(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout))
        for _ in range(num_layers - 2):
            layers.append(GATBlock(hidden_channels * heads, hidden_channels, heads=heads, concat=True, dropout=dropout))
        layers.append(GATBlock(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))

        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x
