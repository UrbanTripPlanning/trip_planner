import re
from datetime import datetime
from torch.utils.data import Dataset
import os
import torch

# ==============================
# SequenceDataset
# ==============================

# A dataset that extracts sliding windows over edge-wise time series from a sequence of graph snapshots.
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, full_dataset, indices, window_size=12):
        self.edge_series = []  # Will hold sequences of shape (window_size, edge_dim)
        self.window_size = window_size

        # Assumes: number of edges and edge order is the same across all snapshots
        for edge_idx in range(full_dataset[0].edge_attr.shape[0]):
            edge_sequence = []

            # For this edge, extract its feature over time from selected snapshot indices
            for i in range(len(indices)):
                g = full_dataset[indices[i]]
                edge_sequence.append(g.edge_attr[edge_idx].unsqueeze(0))  # Shape: (1, edge_dim)

            # Concatenate over time → shape: (num_timesteps, edge_dim)
            edge_sequence = torch.cat(edge_sequence, dim=0)

            # Generate sliding windows of size `window_size`
            for t in range(len(edge_sequence) - window_size + 1):
                window = edge_sequence[t:t + window_size]  # (window_size, edge_dim)
                self.edge_series.append(window)

    def __len__(self):
        # Total number of sliding windows across all edges
        return len(self.edge_series)

    def __getitem__(self, idx):
        # Return a single (window_size, edge_dim) sample
        return self.edge_series[idx]


# ==============================
# InMemoryGraphDataset
# ==============================

# Loads graph snapshots into RAM and provides utility to split by time.
class InMemoryGraphDataset(Dataset):
    def __init__(self, snapshot_dir: str):
        # Collect all .pt file paths in the given directory
        self.paths = sorted([
            os.path.join(snapshot_dir, fn)
            for fn in os.listdir(snapshot_dir)
            if fn.endswith('.pt')
        ])

        if not self.paths:
            raise RuntimeError(f"No snapshots in {snapshot_dir}")

        print(f"[dataset] loading {len(self.paths)} snapshots into RAM…")

        # Load all graph objects (each one expected to be a torch_geometric.data.Data)
        self.data_list = [torch.load(p, weights_only=False) for p in self.paths]

        # Print the shape of edge attributes for verification
        print(self.data_list[0].edge_attr.shape)
        print("[dataset] loaded all snapshots.")

    def __len__(self):
        # Number of snapshots loaded
        return len(self.data_list)

    def __getitem__(self, idx):
        # Return the snapshot at a specific index
        return self.data_list[idx]

    def time_based_split(self):
        """
        Splits the dataset into train (10 months), val (1 month), and test (1 month)
        based on the encoded timestamp in the filename.
        Filename pattern: snapshot_<x>_<YYYYMMDD>Txx.pt
        """
        dates = []
        pat = re.compile(r"snapshot_[^_]+_(\d{8})T\d{2}\.pt$")

        # Extract date from filenames
        for p in self.paths:
            m = pat.search(p)
            dates.append(datetime.strptime(m.group(1), "%Y%m%d"))

        # Get unique year-month pairs
        months = sorted({(d.year, d.month) for d in dates})
        assert len(months) >= 12, "Need a full 12 months of data!"

        # Split into train (first 10), val (11th), test (12th)
        train_m, val_m, test_m = months[:10], months[10:11], months[11:12]

        train_idx = [i for i, d in enumerate(dates) if (d.year, d.month) in train_m]
        val_idx   = [i for i, d in enumerate(dates) if (d.year, d.month) in val_m]
        test_idx  = [i for i, d in enumerate(dates) if (d.year, d.month) in test_m]

        return train_idx, val_idx, test_idx
