import os
import torch
from torch_geometric.data import Dataset, Data
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle


class RoadNetworkSnapshotDataset(Dataset):
    """
    PyTorch Geometric Dataset with normalized edge features including avgSpeed, and cyclical encoding of 'hour'.
    """

    def __init__(self, root: str, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.snapshot_files = sorted([
            os.path.join(root, fname)
            for fname in os.listdir(root)
            if fname.endswith('.pt')
        ])
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        # Pre-fit scalers
        self._fit_scalers()

        # Save scalers
        os.makedirs('models', exist_ok=True)
        with open('models/feature_scaler.pkl', 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        with open('models/target_scaler.pkl', 'wb') as f:
            pickle.dump(self.target_scaler, f)

    def _fit_scalers(self):
        features = []
        targets = []
        for path in self.snapshot_files:
            data = torch.load(path, weights_only=False)

            feature_keys = ["length", "hour", "lane", "rain", "avgSpeed"]
            target_key = "car_travel_time"

            feature_indices = [data.edge_attr_keys.index(k) for k in feature_keys]
            target_index = data.edge_attr_keys.index(target_key)

            raw_features = data.edge_attr[:, feature_indices].numpy()
            raw_targets = data.edge_attr[:, target_index].numpy()

            # Transform hour into cyclical features
            hour = raw_features[:, 1]
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)

            # Final features with avgSpeed
            augmented_features = np.column_stack([
                raw_features[:, 0],  # length
                hour_sin,            # sin(hour)
                hour_cos,            # cos(hour)
                raw_features[:, 2],  # lane
                raw_features[:, 3],  # rain
                raw_features[:, 4],  # avgSpeed
            ])

            features.append(augmented_features)
            targets.append(raw_targets)

        features = np.vstack(features)
        targets = np.hstack(targets)

        self.feature_scaler.fit(features)
        self.target_scaler.fit(targets.reshape(-1, 1))

    def len(self) -> int:
        return len(self.snapshot_files)

    def get(self, idx: int) -> Data:
        path = self.snapshot_files[idx]
        data = torch.load(path, weights_only=False)

        feature_keys = ["length", "hour", "lane", "rain", "avgSpeed"]
        target_key = "car_travel_time"

        feature_indices = [data.edge_attr_keys.index(k) for k in feature_keys]
        target_index = data.edge_attr_keys.index(target_key)

        raw_features = data.edge_attr[:, feature_indices].numpy()
        raw_target = data.edge_attr[:, target_index].numpy()

        # Transform hour into cyclical features
        hour = raw_features[:, 1]
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        augmented_features = np.column_stack([
            raw_features[:, 0],  # length
            hour_sin,
            hour_cos,
            raw_features[:, 2],  # lane
            raw_features[:, 3],  # rain
            raw_features[:, 4],  # avgSpeed
        ])

        # Normalize
        norm_features = self.feature_scaler.transform(augmented_features)
        norm_target = self.target_scaler.transform(raw_target.reshape(-1, 1)).flatten()

        data.edge_features = torch.tensor(norm_features, dtype=torch.float)
        data.y = torch.tensor(norm_target, dtype=torch.float)

        return data
