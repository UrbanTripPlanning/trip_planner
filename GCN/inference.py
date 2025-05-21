import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph
from typing import List, Optional

from GCN.autoencoder import EdgeAutoEncoderMultiTask


class EdgeWeightPredictor:
    """
    Uses a pretrained multi-task GCN autoencoder to predict
    travel-time for each edge in a NetworkX graph and assign
    it as the 'weight' attribute.
    """

    def __init__(
            self,
            model_path: str,
            hidden_dims: List[int] = [64, 32],
            bottleneck_dim: int = 1,
            dropout: float = 0.2,
            device: Optional[str] = None
    ):
        # Select device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Instantiate model and load weights
        self.model = EdgeAutoEncoderMultiTask(
            in_ch=3,
            hidden_ch=hidden_dims,
            bottleneck_ch=bottleneck_dim,
            dropout=dropout
        ).to(self.device)

        # Load state dict (assumes you saved model.state_dict())
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        # Prepare line-graph transform
        self.lg_transform = LineGraph()

    def _nx_to_pyg(self, graph: nx.Graph) -> Data:
        """
        Convert a NetworkX graph with 'length', 'speed', 'time' edge
        attributes into a PyG Data object for line-graph construction.
        """
        nodes = list(graph.nodes())
        idx = {node: i for i, node in enumerate(nodes)}

        src, dst, attrs = [], [], []
        for u, v, data in graph.edges(data=True):
            src.append(idx[u])
            dst.append(idx[v])
            attrs.append([
                float(data.get("length", 0.0)),
                float(data.get("speed", 0.0)),
                float(data.get("time", 0.0)),
            ])

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(attrs, dtype=torch.float)

        # Dummy node features (not used by the encoder)
        x = torch.zeros((len(nodes), 1), dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def infer_edge_weights(
            self,
            graph: nx.Graph,
            min_weight: float = 1e-3
    ) -> List[float]:
        """
        Predicts a positive travel-time for each edge in the graph.
        Returns a list of weights in the same order as graph.edges().
        """
        # Convert to PyG and build line-graph
        orig = self._nx_to_pyg(graph).to(self.device)
        L = self.lg_transform(orig)
        L.x = orig.edge_attr  # use edge_attr as node features on line-graph

        with torch.no_grad():
            # Forward pass: we only need the time-prediction head
            _, t_pred, _ = self.model(L)  # t_pred shape: [E]
            w = t_pred.squeeze()  # shape [E]
            # Ensure strictly positive weights
            w = torch.clamp(w, min=min_weight)

        return w.cpu().tolist()

    def assign_weights_to_graph(
            self,
            graph: nx.Graph,
            weights: List[float]
    ) -> None:
        """
        Assigns predicted weights back to the NetworkX graph edges
        by setting the 'weight' attribute.
        """
        weighted_edges = [
            (u, v, float(w))
            for (u, v), w in zip(graph.edges(), weights)
        ]
        graph.add_weighted_edges_from(weighted_edges, weight="weight")
