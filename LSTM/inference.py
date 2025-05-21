import torch
import networkx as nx
from typing import List, Optional

from LSTM.lstm_autoencoder import LSTMAutoEncoder

class LSTMEdgeWeightPredictor:
    """
    Predicts scalar weights for each edge in a road network using a pretrained
    LSTM Autoencoder. Edge attributes are processed into sequences, and the 
    bottleneck vector is used to derive a meaningful weight per edge.
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None
    ):
        # Determine device (GPU if available, else CPU)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model checkpoint from disk
        checkpoint = torch.load(model_path, map_location=device)

        # Extract model configuration from checkpoint if available
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Fallback values in case config was not saved
            config = {
                'input_size': 3,         # edge_attr dim (e.g., length, speed, time)
                'hidden_size': 64,
                'bottleneck_dim': 32,    # size of the latent vector `z`
                'output_size': 3,
            }
            print("[Warning] Config not found in checkpoint, using fallback values.")

        # Reconstruct model using saved (or fallback) config
        self.model = LSTMAutoEncoder(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            bottleneck_dim=config['bottleneck_dim'],
            output_size=config['output_size'],
        ).to(self.device)

        # Load trained model weights
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()  # Set to evaluation mode

    def _graph_to_sequence_tensor(self, graph: nx.Graph) -> torch.Tensor:
        """
        Convert graph edge attributes into a tensor of shape:
        [num_edges, seq_len=1, input_size=3]
        This assumes each edge has 3 features: length, speed, and time.
        """
        attrs = []
        for _, _, d in graph.edges(data=True):
            # Default to 0.0 if attribute is missing
            attrs.append([
                float(d.get('length', 0.0)),
                float(d.get('speed', 0.0)),
                float(d.get('time', 0.0))
            ])
        
        # Output shape: [E, 1, 3]
        return torch.tensor(attrs, dtype=torch.float32).unsqueeze(1)

    def infer_edge_weights(
        self,
        graph: nx.Graph,
        scale_min: float = 0.1,
        scale_max: float = 0.9
    ) -> List[float]:
        """
        Predict a scalar weight for each edge based on its encoded bottleneck vector.
        Output is a list of floats scaled between `scale_min` and `scale_max`.
        """
        # Prepare input sequence tensor: [E, 1, 3]
        x_seq = self._graph_to_sequence_tensor(graph).to(self.device)

        with torch.no_grad():
            # Pass through model → return reconstructed x and bottleneck vector z
            _, z = self.model(x_seq)  # z shape: [E, bottleneck_dim]

            # Compute L2 norm of each bottleneck vector → [E]
            z_score = z.norm(p=2, dim=1)

            # Normalize using sigmoid, then rescale to desired range
            w = torch.sigmoid(z_score)                    # → [0,1]
            w = scale_min + (scale_max - scale_min) * w   # → [scale_min, scale_max]

        return w.cpu().tolist()  # Convert to Python list

    def assign_weights_to_graph(
        self,
        graph: nx.Graph,
        weights: List[float]
    ) -> None:
        """
        Attach predicted weights back to the input graph by setting the
        `weight` attribute for each edge.
        """
        for (u, v), weight in zip(graph.edges(), weights):
            graph[u][v]['weight'] = float(weight)
