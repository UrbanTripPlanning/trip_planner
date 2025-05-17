import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, BatchNorm


class EdgeAutoEncoderMultiTask(torch.nn.Module):
    """
    Multi-task GCN autoencoder on line-graph of edges:
      - Encoder: 3 → 64 → 32 → 1 (no ReLU at bottleneck), with residual projections
      - Decoder (feature reconstruction): MLP 1 → 32 → 64 → 3
      - Time-regression head: MLP 1 → 16 → 1
    """

    def __init__(
            self,
            in_ch: int = 3,
            hidden_ch: list = [64, 32],
            bottleneck_ch: int = 1,
            dropout: float = 0.2
    ):
        super().__init__()
        # Encoder conv layers
        self.enc1 = GCNConv(in_ch, hidden_ch[0])
        self.bn1 = BatchNorm(hidden_ch[0])
        self.res1 = torch.nn.Linear(in_ch, hidden_ch[0])  # projection 3→64

        self.enc2 = GCNConv(hidden_ch[0], hidden_ch[1])
        self.bn2 = BatchNorm(hidden_ch[1])
        self.res2 = torch.nn.Linear(hidden_ch[0], hidden_ch[1])  # projection 64→32

        self.enc3 = GCNConv(hidden_ch[1], bottleneck_ch)  # no activation

        # Decoder MLP layers for feature reconstruction
        self.dec1 = torch.nn.Linear(bottleneck_ch, hidden_ch[1])
        self.dec2 = torch.nn.Linear(hidden_ch[1], hidden_ch[0])
        self.dec3 = torch.nn.Linear(hidden_ch[0], in_ch)

        # Time-prediction head
        self.time_head = torch.nn.Sequential(
            torch.nn.Linear(bottleneck_ch, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(16, 1)
        )

        self.dropout = dropout

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        # --- Encoder with projected residual + BN + ReLU + Dropout ---
        # Block 1
        res = self.res1(x)  # [E, 64]
        x = self.enc1(x, edge_index)  # [E, 64]
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + res  # [E, 64]

        # Block 2
        res = self.res2(x)  # [E, 32]
        x = self.enc2(x, edge_index)  # [E, 32]
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + res  # [E, 32]

        # Bottleneck (no ReLU)
        z = self.enc3(x, edge_index)  # [E,1]

        # --- Decoder: reconstruct features ---
        d = F.relu(self.dec1(z))  # [E,32]
        d = F.dropout(d, p=self.dropout, training=self.training)
        d = F.relu(self.dec2(d))  # [E,64]
        d = F.dropout(d, p=self.dropout, training=self.training)
        recon = self.dec3(d)  # [E,3]

        # --- Time head: predict travel time ---
        t_pred = self.time_head(z).squeeze(1)  # [E]

        return recon, t_pred, z.squeeze(1)
