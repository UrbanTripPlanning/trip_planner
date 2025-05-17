import torch
import torch.nn as nn

class LSTMAutoEncoder(nn.Module):
    """
    A simple LSTM-based autoencoder for sequence data.
    - Encoder compresses input sequences into a bottleneck vector.
    - Decoder reconstructs the input sequence from this vector.
    """

    def __init__(self, input_size, hidden_size, bottleneck_dim, output_size):
        super().__init__()

        # LSTM encoder: encodes sequence into hidden state
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)

        # Bottleneck layer: compress hidden state to lower-dimensional representation
        self.bottleneck = nn.Linear(hidden_size, bottleneck_dim)

        # Expand bottleneck vector to initialize decoder hidden state
        self.decoder_input = nn.Linear(bottleneck_dim, hidden_size)

        # LSTM decoder: reconstructs the sequence from the hidden state
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

        # Output projection layer: maps hidden state to output size
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x: Tensor of shape [batch, seq_len, input_size]
        Returns:
            - Reconstructed sequence: [batch, seq_len, output_size]
            - Bottleneck vector: [batch, bottleneck_dim]
        """
        # Pass through encoder LSTM
        _, (h_n, _) = self.encoder(x)         # h_n: [1, batch, hidden_size]
        h = h_n.squeeze(0)                    # Remove LSTM layer dimension → [batch, hidden_size]

        # Bottleneck: compressed representation of input sequence
        z = self.bottleneck(h)                # [batch, bottleneck_dim]

        # Decode from bottleneck by first expanding to hidden_size
        dec_input = self.decoder_input(z)     # [batch, hidden_size]
        dec_input = dec_input.unsqueeze(0)    # Add sequence dim → [1, batch, hidden_size]
        dec_input = dec_input.repeat(x.size(1), 1, 1).permute(1, 0, 2)  # [batch, seq_len, hidden_size]

        # Reconstruct sequence using decoder LSTM
        dec_out, _ = self.decoder(dec_input)  # [batch, seq_len, hidden_size]

        # Map hidden states to output features
        out = self.output_layer(dec_out)      # [batch, seq_len, output_size]

        return out, z  # Return reconstructed sequence and bottleneck vector
