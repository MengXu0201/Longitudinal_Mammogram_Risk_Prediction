import torch
import torch.nn as nn

from src.models.model_graph_modules import (
    GraphConvBlock,
    GraphReadout,
    NodeFeatureProjector,
)
from src.models.model_risk_prediction import CumulativeProbabilityLayer


class GraphBaselineMaskedGridRiskHead(nn.Module):
    """
    Match the CNN baseline output contract while operating on graph-pooled vectors.
    """

    def __init__(self, hidden_dim=512, num_years=5):
        super().__init__()
        self.cumulative_prob_layer_fused = CumulativeProbabilityLayer(
            num_features=2 * hidden_dim, max_followup=num_years + 1
        )
        self.cumulative_prob_layer_cur = CumulativeProbabilityLayer(
            num_features=hidden_dim, max_followup=num_years + 1
        )
        self.cumulative_prob_layer_pri = CumulativeProbabilityLayer(
            num_features=hidden_dim, max_followup=num_years + 1
        )

    def forward(self, current_graph_vector, prior_graph_vector, time_gap=None):
        fused = torch.cat([current_graph_vector, prior_graph_vector], dim=1)
        return {
            "pred_fused": torch.sigmoid(self.cumulative_prob_layer_fused(fused)),
            "pred_cur": torch.sigmoid(self.cumulative_prob_layer_cur(current_graph_vector)),
            "pred_pri": torch.sigmoid(self.cumulative_prob_layer_pri(prior_graph_vector)),
        }


class GraphBaselineMaskedGridRiskModel(nn.Module):
    """
    No-alignment graph baseline built from breast-masked feature-grid nodes.
    """

    def __init__(
        self,
        hidden_dim=512,
        num_years=5,
        num_graph_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.node_projector = NodeFeatureProjector(
            feature_dim=512,
            coord_dim=2,
            hidden_dim=hidden_dim,
        )
        self.graph_layers = nn.ModuleList(
            [GraphConvBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_graph_layers)]
        )
        self.readout = GraphReadout()
        self.risk_head = GraphBaselineMaskedGridRiskHead(
            hidden_dim=hidden_dim,
            num_years=num_years,
        )

    def _encode_graph(self, graph_batch):
        node_features = self.node_projector(
            graph_batch["node_features"],
            graph_batch["node_coords"],
            graph_batch["node_mask"],
        )

        for graph_layer in self.graph_layers:
            node_features = graph_layer(
                node_features,
                graph_batch["adjacency"],
                graph_batch["node_mask"],
            )

        graph_vector = self.readout(node_features, graph_batch["node_mask"])
        return graph_vector, graph_batch

    def forward(self, current_graph, previous_graph, time_gap):
        g_cur, _ = self._encode_graph(current_graph)
        g_pri, _ = self._encode_graph(previous_graph)

        risk_out = self.risk_head(g_cur, g_pri, time_gap)

        return {
            "risk_prediction": risk_out,
        }
