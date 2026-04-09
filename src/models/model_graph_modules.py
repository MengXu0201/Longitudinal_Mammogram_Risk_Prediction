import torch
import torch.nn as nn


class NodeFeatureProjector(nn.Module):
    """
    Project CNN node features plus normalized coordinates back to the hidden size.
    """

    def __init__(self, feature_dim, coord_dim=2, hidden_dim=512):
        super().__init__()
        self.proj = nn.Linear(feature_dim + coord_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, node_features, node_coords, node_mask):
        x = torch.cat([node_features, node_coords], dim=-1)
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x * node_mask.unsqueeze(-1).to(x.dtype)


class GraphConvBlock(nn.Module):
    """
    Dense masked GraphSAGE-style message passing block.
    """

    def __init__(self, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.neigh_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features, adjacency, node_mask):
        mask = node_mask.unsqueeze(-1).to(node_features.dtype)
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neighborhood = torch.bmm(adjacency, node_features) / degree

        updated = self.self_proj(node_features) + self.neigh_proj(neighborhood)
        updated = self.norm(updated)
        updated = self.act(updated)
        updated = self.dropout(updated)

        # Residual connection keeps optimization stable while preserving masking.
        updated = (updated + node_features) * mask
        return updated


class GraphReadout(nn.Module):
    """
    Masked mean graph pooling to produce one image-level vector per graph.
    """

    def forward(self, node_features, node_mask):
        weights = node_mask.unsqueeze(-1).to(node_features.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        pooled = (node_features * weights).sum(dim=1) / denom
        return pooled
