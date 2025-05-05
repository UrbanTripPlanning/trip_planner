import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class EdgeAutoEncoder(torch.nn.Module):
    """
    GCN autoencoder on the line-graph of edges, with:
      Encoder:  3 → 64 → 32 → 1
      Decoder:  1 → 32 → 64 → 3
    """

    def __init__(
        self,
        in_ch: int = 3,
        hidden_ch: list = [64, 32],
        bottleneck_ch: int = 1
    ):
        super().__init__()
        # Encoder layers
        self.enc1 = GCNConv(in_ch,          hidden_ch[0])  # 3 → 64
        self.enc2 = GCNConv(hidden_ch[0],   hidden_ch[1])  # 64 → 32
        self.enc3 = GCNConv(hidden_ch[1],   bottleneck_ch) # 32 → 1

        # Decoder layers (mirror of encoder)
        self.dec1 = GCNConv(bottleneck_ch,  hidden_ch[1])  # 1 → 32
        self.dec2 = GCNConv(hidden_ch[1],   hidden_ch[0])  # 32 → 64
        self.dec3 = GCNConv(hidden_ch[0],   in_ch)         # 64 → 3

    def forward(self, data: Data):
        """
        :param data: PyG Data object for the line-graph, where
                     data.x is [E, 3] and data.edge_index encodes adjacency.
        :returns:
            recon: reconstructed edge attributes, shape [E, 3]
            z:      bottleneck embeddings, shape [E, 1]
        """
        x, edge_index = data.x, data.edge_index

        # ----- Encoder -----
        x = F.relu(self.enc1(x, edge_index))
        x = F.relu(self.enc2(x, edge_index))
        z = F.relu(self.enc3(x, edge_index))  # bottleneck feature [E, 1]

        # ----- Decoder -----
        x = F.relu(self.dec1(z, edge_index))
        x = F.relu(self.dec2(x, edge_index))
        recon = self.dec3(x, edge_index)      # reconstructed [E, 3]

        return recon, z
