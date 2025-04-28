import torch
import numpy as np
import os
import pickle

from GCN.autoencoder import EncoderDecoderModel


def prepare_feature_vector(length: float, hour: int, lane: int, rain: float, avgSpeed: float) -> np.ndarray:
    """
    Prepare a feature vector [length, sin(hour), cos(hour), lane, rain, avgSpeed] for inference.
    """
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    feature_vector = np.array([[length, hour_sin, hour_cos, lane, rain, avgSpeed]])
    return feature_vector


def predict_travel_time(
        model: torch.nn.Module,
        feature_scaler,
        target_scaler,
        feature_vector: np.ndarray,
        device: torch.device
) -> float:
    """
    Predict car travel time for one edge given the feature vector.
    """
    # Normalize input
    feature_vector_norm = feature_scaler.transform(feature_vector)

    # Prepare tensor
    input_tensor = torch.tensor(feature_vector_norm, dtype=torch.float).to(device)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)  # Dummy self-loop

    # Forward pass
    model.eval()
    with torch.no_grad():
        node_embedding = model.encoder(input_tensor, edge_index)
        travel_time_pred_norm = model.decoder(edge_index, node_embedding)

    # Denormalize output
    travel_time_pred = target_scaler.inverse_transform(
        travel_time_pred_norm.cpu().unsqueeze(-1)
    ).flatten()[0]

    return float(travel_time_pred)


def load_model_and_scalers(model_dir: str, device: torch.device):
    """Load the trained GNN model and scalers from the given directory."""
    model = EncoderDecoderModel(
        edge_in_channels=6,
        gcn_hidden_channels=128,
        bottleneck_dim=64,
        decoder_hidden_channels=128
    ).to(device)

    model_path = os.path.join(model_dir, "best_encoder_decoder.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with open(os.path.join(model_dir, "feature_scaler.pkl"), 'rb') as f:
        feature_scaler = pickle.load(f)
    with open(os.path.join(model_dir, 'target_scaler.pkl'), 'rb') as f:
        target_scaler = pickle.load(f)

    return model, feature_scaler, target_scaler
