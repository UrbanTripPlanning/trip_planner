from typing import Tuple
import torch
import torch.nn.functional as F
from GCN.inferer import prepare_feature_vector, predict_travel_time, load_model_and_scalers as load_gcn
from STGCN.inferer import load_model_and_scalers as load_stgcn
from STGCN.train import load_snapshots, preprocess


class EdgeWeightPredictor:
    def __init__(self, GNN: str, device: torch.device):
        self.GNN = GNN
        self.device = device
        self.model_dir = f"./{self.GNN}/models"
        self.snapshot_dir = f"./data/snapshots"

        if GNN == "GCN":
            # load your old GCN model + scalers
            self.model, self.feature_scaler, self.target_scaler = load_gcn(self.model_dir, device)

        elif GNN == "STGCN":
            # 1) load & preprocess
            snaps = load_snapshots(self.snapshot_dir)
            seq, _, _, _ = preprocess(snaps, val_ratio=0.0)
            # 2) load STGCN model + scalers
            self.model, self.feature_scaler, self.target_scaler = load_stgcn(self.model_dir, device)
            # 3) compute one-time node hiddens + edge_index
            with torch.no_grad():
                x_seq = [d.x.to(device) for d in seq]
                ei = seq[0].edge_index.to(device)
                self.h = self.model.encode(x_seq, ei)  # [N,H]
                src, dst = ei
                e_uv = (self.h[src] * self.h[dst]).sum(dim=1)  # [E]
                self.raw_w = F.softplus(e_uv).cpu().numpy()  # [E]
                self.map_uv = {(int(src[i]), int(dst[i])): float(self.raw_w[i])
                               for i in range(self.raw_w.shape[0])}

    def predict(self, length: float, hour: int, lane: int, rain: float, avgSpeed: float,
                u: Tuple[float, float], v: Tuple[float, float]) -> float:
        """
        Return a positive weight for edge (u,v).
        If GCN: use prepare_feature_vector + predict_travel_time.
        If STGCN: lookup from self.map_uv, fallback=length/(safe_speed).
        """
        # formula fallback
        MIN_SPEED = 0.1
        default_speed = avgSpeed
        penalty = 0.0  # you can inject your penalty logic here if you like
        safe_speed = max(default_speed * (1 + penalty), MIN_SPEED)
        car_time = length / (safe_speed / 3.6)

        if self.GNN == "GCN":
            fv = prepare_feature_vector(length, hour, lane, rain, avgSpeed)
            return predict_travel_time(
                self.model, self.feature_scaler, self.target_scaler, fv, self.device
            )
        else:
            return self.map_uv.get((u, v), car_time)
