import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, bottleneck_dim: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, bottleneck_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x


class MLPDecoder(nn.Module):
    def __init__(self, bottleneck_dim: int, hidden_channels: int):
        super().__init__()
        self.fc1 = nn.Linear(2 * bottleneck_dim, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc3 = nn.Linear(hidden_channels // 2, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, edge_index, node_embeddings):
        src = node_embeddings[edge_index[0]]
        dst = node_embeddings[edge_index[1]]
        edge_emb = torch.cat([src, dst], dim=1)
        x = F.relu(self.fc1(edge_emb))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        out = self.fc3(x)
        return F.softplus(out.squeeze(-1))  # GUARANTEE output > 0


class EncoderDecoderModel(nn.Module):
    def __init__(self, edge_in_channels: int, gcn_hidden_channels: int, bottleneck_dim: int,
                 decoder_hidden_channels: int):
        super().__init__()
        self.encoder = GCNEncoder(edge_in_channels, gcn_hidden_channels, bottleneck_dim)
        self.decoder = MLPDecoder(bottleneck_dim, decoder_hidden_channels)

    def forward(self, data):
        node_embeddings = self.encoder(data.edge_features, data.edge_index)
        travel_time_pred = self.decoder(data.edge_index, node_embeddings)
        return travel_time_pred
