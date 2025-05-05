import torch
import re
import os
from datetime import datetime
from torch.utils.data import Dataset
from torch_geometric.transforms import LineGraph


class InMemoryGraphDataset(Dataset):
    """
    Load all the .pt snapshots into RAM,
    transform each graph to line-graph and set
    x = edge_attr, edge_attr = edge_attr.
    """

    def __init__(self, snapshot_dir: str):
        self.paths = sorted([
            os.path.join(snapshot_dir, fn)
            for fn in os.listdir(snapshot_dir)
            if fn.endswith(".pt")
        ])
        if not self.paths:
            raise RuntimeError(f"No snapshots in {snapshot_dir}")
        print(f"[dataset] Loading {len(self.paths)} snapshots into RAM…")

        self.data_list = []
        lg = LineGraph()  # line-graph transform

        for p in self.paths:
            orig = torch.load(p, weights_only=False)
            L = lg(orig)  # build the line-graph
            # explicitly assigns edge_attr on L
            L.x = orig.edge_attr  # feature per each "node-edge"
            L.edge_attr = orig.edge_attr
            self.data_list.append(L)

        print("[dataset] Loaded all snapshots.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def time_based_split(dataset):
    """
    It divides the snapshots into 10 months train, 1 month val, 1 month test.
    """
    dates = []
    pat = re.compile(r"snapshot_[^_]+_(\d{8})T\d{2}\.pt$")
    for p in dataset.paths:
        m = pat.search(p)
        dates.append(datetime.strptime(m.group(1), "%Y%m%d"))
    months = sorted({(d.year, d.month) for d in dates})
    assert len(months) >= 12, "Need at least one year of data!"
    tr_m = months[:10];
    va_m = months[10:11];
    te_m = months[11:12]
    tr_idx = [i for i, d in enumerate(dates) if (d.year, d.month) in tr_m]
    va_idx = [i for i, d in enumerate(dates) if (d.year, d.month) in va_m]
    te_idx = [i for i, d in enumerate(dates) if (d.year, d.month) in te_m]
    return tr_idx, va_idx, te_idx
