import os
import pickle
import torch

from typing import Tuple
from STGCN.autoencoder import SpatioTemporalAutoencoder


def load_model_and_scalers(
        model_dir: str,
        device: torch.device
) -> Tuple[torch.nn.Module, object, object]:
    """
    Load a trained SpatioTemporalAutoencoder and its fitted feature/target scalers.

    Args:
        model_dir: Path to directory containing:
            - autoencoder.pth
            - feature_scaler.pkl
            - target_scaler.pkl
        device:    torch device ('cpu' or 'cuda')

    Returns:
        model           : SpatioTemporalAutoencoder in eval() mode
        feature_scaler  : scaler for edge time/static features
        target_scaler   : scaler for travel-time targets
    """
    # 1) Recreate the same autoencoder architecture used in training
    model = SpatioTemporalAutoencoder(
        node_in_feats=1,       # number of node features used by encoder
        gconv_hidden=64,       # hidden size in GConvGRU
        edge_time_feats=3,     # sin(hour), cos(hour), rain
        edge_static_feats=3,   # length, lane, avgSpeed
        decoder_hidden=64,     # GRU hidden size in decoder
        mlp_hidden=32          # MLP hidden size in decoder
    ).to(device)

    # 2) Load checkpoint
    checkpoint_path = os.path.join(model_dir, "autoencoder.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 3) Strip any unwanted prefixes from state_dict keys
    cleaned_state = {}
    for key, value in checkpoint.items():
        new_key = key
        # remove DataParallel or other prefixes
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod."):]
        cleaned_state[new_key] = value

    # 4) Load weights into model
    model.load_state_dict(cleaned_state)
    model.eval()

    # 5) Load the pre-fitted scalers
    feat_scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
    targ_scaler_path = os.path.join(model_dir, "target_scaler.pkl")
    with open(feat_scaler_path, "rb") as f:
        feature_scaler = pickle.load(f)
    with open(targ_scaler_path, "rb") as f:
        target_scaler = pickle.load(f)

    return model, feature_scaler, target_scaler
