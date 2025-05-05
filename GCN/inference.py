import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph
from typing import List, Optional

from GCN.autoencoder import EdgeAutoEncoder


class EdgeWeightPredictor:
    """
    Given a pretrained GCN autoencoder, build the line-graph of a road network
    and predict a scalar weight for each original edge, then assign it back.
    """

    def __init__(
        self,
        model_path: str,
        hidden_dims: List[int] = [64, 32],
        bottleneck_dim: int = 1,
        device: Optional[str] = None
    ):
        """
        :param model_path: path to the saved autoencoder (.pt)
        :param hidden_dims: encoder hidden sizes [enc1, enc2]
        :param bottleneck_dim: size of the bottleneck embedding (should be 1)
        :param device: 'cpu' or 'cuda'; auto‐selects if None
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 1) Instantiate autoencoder and load weights
        self.model = EdgeAutoEncoder(
            in_ch=3,
            hidden_ch=hidden_dims,
            bottleneck_ch=bottleneck_dim
        ).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        # 2) Prepare line‐graph transform
        self.lg_transform = LineGraph()

    def _nx_to_pyg(self, graph: nx.Graph) -> Data:
        """
        Convert a NetworkX graph with 'length','speed','time' edge attrs
        into a PyG Data object for line‐graph construction.
        """
        nodes = list(graph.nodes())
        idx = {n: i for i, n in enumerate(nodes)}

        src, dst, attrs = [], [], []
        for u, v, d in graph.edges(data=True):
            src.append(idx[u])
            dst.append(idx[v])
            attrs.append([
                float(d.get('length', 0.0)),
                float(d.get('speed', 0.0)),
                float(d.get('time', 0.0))
            ])

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.tensor(attrs,    dtype=torch.float)
        x = torch.zeros((len(nodes), 1),    dtype=torch.float)  # dummy node features

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def infer_edge_weights(
        self,
        graph: nx.Graph,
        scale_min: float = 0.1,
        scale_max: float = 0.9
    ) -> List[float]:
        """
        Predict a positive scalar weight for each edge of `graph`.
        Returns weights in the same iteration order as graph.edges().
        """
        # Convert to PyG and build line‐graph
        orig = self._nx_to_pyg(graph).to(self.device)
        L = self.lg_transform(orig)
        # Ensure line‐graph node features = original edge attributes
        L.x = orig.edge_attr

        with torch.no_grad():
            _, z = self.model(L)       # z shape: [E,1]
            z = z.squeeze(1)           # [E]
            w = torch.sigmoid(z)       # in (0,1)
            w = scale_min + (scale_max - scale_min) * w

        return w.cpu().tolist()

    def assign_weights_to_graph(
        self,
        graph: nx.Graph,
        weights: List[float]
    ) -> None:
        """
        Efficiently assign predicted weights back to graph edges using
        networkx.add_weighted_edges_from.
        """
        # Build (u, v, weight) tuples in the same order as graph.edges()
        triples = [
            (u, v, float(w))
            for (u, v), w in zip(graph.edges(), weights)
        ]
        # This will overwrite or set the 'weight' attribute on each edge
        graph.add_weighted_edges_from(triples, weight='weight')
