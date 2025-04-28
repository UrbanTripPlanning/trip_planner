import os
import glob
import pickle
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from STGCN.autoencoder import SpatioTemporalAutoencoder
from dotenv import load_dotenv

# 1) Pin threads
load_dotenv()
torch.set_num_threads(os.cpu_count())


def load_snapshots(snapshot_dir):
    """Load all snapshot_*.pt (weights_only=False)"""
    paths = sorted(glob.glob(os.path.join(snapshot_dir, "snapshot_*.pt")))
    if not paths:
        raise RuntimeError(f"No snapshots in {snapshot_dir}")
    return [torch.load(p, weights_only=False) for p in paths]


def preprocess(seq, val_ratio=0.2):
    """Compute time_feats/static_feats, fit scalers, split train/val."""
    keys = seq[0].edge_attr_keys
    idx = {k: keys.index(k) for k in ["car_travel_time", "hour", "rain", "length", "lane", "avgSpeed"]}
    T = len(seq)
    all_y, all_sf = [], []
    for d in seq:
        ea = d.edge_attr.numpy()
        all_y.append(ea[:, idx["car_travel_time"]].reshape(-1, 1))
        all_sf.append(np.hstack([
            ea[:, idx["length"]].reshape(-1, 1),
            ea[:, idx["lane"]].reshape(-1, 1),
            ea[:, idx["avgSpeed"]].reshape(-1, 1),
        ]))
    all_y = np.vstack(all_y)
    all_sf = np.vstack(all_sf)
    scaler_y = StandardScaler().fit(all_y)
    scaler_sf = StandardScaler().fit(all_sf)

    for d in seq:
        ea = d.edge_attr.numpy()
        # time_feats
        h = ea[:, idx["hour"]]
        r = ea[:, idx["rain"]].reshape(-1, 1)
        rad = 2 * np.pi * (h / 24.0)
        tf = np.hstack([np.sin(rad).reshape(-1, 1),
                        np.cos(rad).reshape(-1, 1),
                        r])
        d.time_feats = torch.tensor(tf, dtype=torch.float)
        # static_feats
        sf_raw = np.hstack([
            ea[:, idx["length"]].reshape(-1, 1),
            ea[:, idx["lane"]].reshape(-1, 1),
            ea[:, idx["avgSpeed"]].reshape(-1, 1),
        ])
        d.static_feats = torch.tensor(
            scaler_sf.transform(sf_raw), dtype=torch.float
        )
        # target
        y_raw = ea[:, idx["car_travel_time"]].reshape(-1, 1)
        d.y = torch.tensor(
            scaler_y.transform(y_raw).flatten(), dtype=torch.float
        )

    n_val = max(1, int(T * val_ratio))
    return seq[:-n_val], seq[-n_val:], scaler_y, scaler_sf


def main():
    p = argparse.ArgumentParser()
    default_snap = os.getenv('DATA_PATH', './data') + '/snapshots'
    p.add_argument("-s", "--snapshots", default=default_snap)
    p.add_argument("-o", "--out_dir", default="./models")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--gh", type=int, default=64, help="gconv hidden")
    p.add_argument("--dh", type=int, default=64, help="decoder hidden")
    p.add_argument("--mh", type=int, default=32, help="MLP hidden")
    p.add_argument("-e", "--epochs", type=int, default=100)
    args = p.parse_args()

    if not os.path.isdir(args.snapshots):
        raise FileNotFoundError(args.snapshots)

    # Load + preprocess
    seq = load_snapshots(args.snapshots)
    train, val, scaler_y, scaler_sf = preprocess(seq, val_ratio=0.2)

    # Hoist tensors
    x_seq = [d.x for d in train]
    edge_index = train[0].edge_index
    time_seq = torch.stack([d.time_feats for d in train], dim=0)  # [T,E,Ft]
    static_feats = train[0].static_feats  # [E,Fs]
    target_seq = torch.stack([d.y for d in train], dim=0)  # [T,E]

    # Same for val
    x_seq_v = [d.x for d in val]
    time_seq_v = torch.stack([d.time_feats for d in val], dim=0)
    target_seq_v = torch.stack([d.y for d in val], dim=0)

    # Build model
    model = SpatioTemporalAutoencoder(
        node_in_feats=x_seq[0].size(1),
        gconv_hidden=args.gh,
        edge_time_feats=time_seq.size(2),
        edge_static_feats=static_feats.size(1),
        decoder_hidden=args.dh,
        mlp_hidden=args.mh
    ).to("cpu")
    # JIT‐compile for CPU speed (PyTorch ≥2.0)
    model = torch.compile(model)

    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                       factor=0.5, patience=5)
    crit = nn.MSELoss()

    # Single‐pass per epoch
    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        # forward on full graph
        h = model.encode(x_seq, edge_index)
        pred = model.decode(h, edge_index, time_seq, static_feats)
        loss = crit(pred, target_seq)
        loss.backward()
        opt.step()

        # val
        model.eval()
        with torch.no_grad():
            h_v = model.encode(x_seq_v, edge_index)
            pred_v = model.decode(h_v, edge_index, time_seq_v, static_feats)
            vloss = crit(pred_v, target_seq_v)

        sched.step(vloss)
        lr = opt.param_groups[0]["lr"]
        print(f"Epoch {ep:03d} | Train: {loss.item():.4f} | Val: {vloss.item():.4f} | LR={lr:.2e}")

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "autoencoder.pth"))
    with open(os.path.join(args.out_dir, "target_scaler.pkl"), "wb") as f:
        pickle.dump(scaler_y, f)
    with open(os.path.join(args.out_dir, "feature_scaler.pkl"), "wb") as f:
        pickle.dump(scaler_sf, f)

    print("Done.")


if __name__ == "__main__":
    main()
