import torch
from torch import nn
from torch_geometric_temporal.nn.recurrent import GConvGRU


class SpatioTemporalAutoencoder(nn.Module):
    """
    Spatio-temporal autoencoder:
      - Encoder: GConvGRU over a sequence of graph snapshots → final node hidden states.
      - Decoder: GRU + MLP that reconstructs edge travel times for each snapshot,
                 using both temporal AND static edge features.
    """

    def __init__(
            self,
            node_in_feats: int,
            gconv_hidden: int,
            edge_time_feats: int,
            edge_static_feats: int,
            decoder_hidden: int,
            mlp_hidden: int
    ):
        super().__init__()
        # 1) Encoder: node‐level GConvGRU
        self.encoder = GConvGRU(
            in_channels=node_in_feats,
            out_channels=gconv_hidden,
            K=3,
            normalization='sym'
        )
        # 2) Decoder GRU: input = spatial_emb (2*gconv_hidden) + time_feats + static_feats
        total_edge_in = 2 * gconv_hidden + edge_time_feats + edge_static_feats
        self.decoder_gru = nn.GRU(
            input_size=total_edge_in,
            hidden_size=decoder_hidden,
            batch_first=False
        )
        # 3) MLP: maps GRU hidden → travel time
        self.mlp = nn.Sequential(
            nn.Linear(decoder_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1)
        )

    def encode(self, x_seq, edge_index):
        """
        Encode a list of T node‐feature tensors into final node hidden state.
        Args:
            x_seq      : List[Tensor[N, F_node]]
            edge_index : Tensor[2, E] (static)
        Returns:
            h: Tensor[N, gconv_hidden]
        """
        h = None
        for x in x_seq:
            # positional args: (X, edge_index, edge_weight, H)
            h = self.encoder(x, edge_index, None, h)
        return h

    def decode(self, h, edge_index, time_seq, static_feats):
        """
        Reconstruct T snapshots from bottleneck.
        Args:
            h           : Tensor[N, gconv_hidden]
            edge_index  : Tensor[2, E]
            time_seq    : Tensor[T, E, Ft]
            static_feats: Tensor[E, Fs]
        Returns:
            out: Tensor[T, E]
        """
        # spatial edge embedding
        src, dst = edge_index
        h_edge = torch.cat([h[src], h[dst]], dim=1)  # [E, 2*gconv_hidden]

        T = time_seq.size(0)
        # expand for each time step
        spatial_seq = h_edge.unsqueeze(0).expand(T, -1, -1)  # [T, E, 2H]
        static_seq = static_feats.unsqueeze(0).expand(T, -1, -1)  # [T, E, Fs]

        # decoder input
        gru_in = torch.cat([spatial_seq, time_seq, static_seq], dim=2)  # [T, E, 2H+Ft+Fs]
        gru_out, _ = self.decoder_gru(gru_in, None)  # [T, E, D]
        out = self.mlp(gru_out).squeeze(-1)  # [T, E]
        return out

    def forward(self, sequence):
        """
        Legacy forward: accepts List[Data].
        """
        x_seq = [d.x for d in sequence]
        edge_index = sequence[0].edge_index
        time_seq = torch.stack([d.time_feats for d in sequence], dim=0)
        static_feats = sequence[0].static_feats
        h = self.encode(x_seq, edge_index)
        return self.decode(h, edge_index, time_seq, static_feats)
