import os
import math
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from GCN.autoencoder import EdgeAutoEncoder
from GCN.dataset import InMemoryGraphDataset, time_based_split


def train(
        snapshot_dir, model_path, ckpt_path, rmse_curve_path, mae_curve_path,
        batch_size: int = 64,
        lr: float = 1e-3,
        epochs: int = 500,
        patience: int = 20,
        hidden_dims: list = [64, 32],
        bottleneck_dim: int = 1
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}\n")

    # 1) Loading dataset and split
    full_ds = InMemoryGraphDataset(snapshot_dir)
    tr_idx, va_idx, te_idx = time_based_split(full_ds)
    tr_ds, va_ds, te_ds = Subset(full_ds, tr_idx), Subset(full_ds, va_idx), Subset(full_ds, te_idx)

    # 2) DataLoader of PyG (native batching)
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    te_ld = DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # 3) Model, optimizer and criteria
    model = EdgeAutoEncoder(3, hidden_dims, bottleneck_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()
    mae = torch.nn.L1Loss()

    # 4) Resume the checkpoint if it exists
    start_ep, best_val = 1, float('inf')
    if os.path.isfile(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck['model'])
        optim.load_state_dict(ck['optim'])
        start_ep, best_val = ck['epoch'] + 1, ck['best_val']
        print(f"Resuming from epoch {start_ep}, best_val(MSE)={best_val:.4f}\n")

    # 5) AMP scaler
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()

    # list for curves
    train_rmse, val_rmse = [], []
    train_mae, val_mae = [], []
    no_improve = 0

    for ep in range(start_ep, epochs + 1):
        # --- Training ---
        model.train()
        t_mse, t_mae = 0.0, 0.0
        for batch in tr_ld:
            batch = batch.to(device)
            optim.zero_grad()
            with autocast():
                recon, _ = model(batch)
                loss_mse = mse(recon, batch.edge_attr)
                loss_mae = mae(recon, batch.edge_attr)
            scaler.scale(loss_mse).backward()
            scaler.step(optim)
            scaler.update()
            t_mse += loss_mse.item()
            t_mae += loss_mae.item()
        avg_tr_mse = t_mse / len(tr_ld)
        avg_tr_mae = t_mae / len(tr_ld)
        avg_tr_rmse = math.sqrt(avg_tr_mse)

        # --- Validation ---
        model.eval()
        v_mse, v_mae = 0.0, 0.0
        with torch.no_grad():
            for batch in va_ld:
                batch = batch.to(device)
                recon, _ = model(batch)
                v_mse += mse(recon, batch.edge_attr).item()
                v_mae += mae(recon, batch.edge_attr).item()
        avg_va_mse = v_mse / len(va_ld)
        avg_va_mae = v_mae / len(va_ld)
        avg_va_rmse = math.sqrt(avg_va_mse)

        # print metrics
        print(f"Epoch {ep}/{epochs}  "
              f"Train[RMSE]={avg_tr_rmse:.2f}, MAE={avg_tr_mae:.2f}  |  "
              f"Val[RMSE]={avg_va_rmse:.2f}, MAE={avg_va_mae:.2f}")

        # update curves
        train_rmse.append(avg_tr_rmse)
        val_rmse.append(avg_va_rmse)
        train_mae.append(avg_tr_mae)
        val_mae.append(avg_va_mae)

        # --- Early stopping & checkpoint ---
        if avg_va_mse < best_val - 1e-4:
            best_val = avg_va_mse
            no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model (MSE) at epoch {ep}: {model_path}")
        else:
            no_improve += 1

        torch.save({
            'epoch': ep,
            'model': model.state_dict(),
            'optim': optim.state_dict(),
            'best_val': best_val,
        }, ckpt_path)

        if no_improve >= patience:
            print(f"Early stopping at epoch {ep} (no improvement for {patience} epochs)")
            break

    # --- Final Test ---
    print("\n▶ Testing on held-out month")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    te_mse, te_mae = 0.0, 0.0
    with torch.no_grad():
        for batch in te_ld:
            batch = batch.to(device)
            recon, _ = model(batch)
            te_mse += mse(recon, batch.edge_attr).item()
            te_mae += mae(recon, batch.edge_attr).item()
    avg_te_mse = te_mse / len(te_ld)
    avg_te_rmse = math.sqrt(avg_te_mse)
    avg_te_mae = te_mae / len(te_ld)
    print(f"→ Test metrics  RMSE={avg_te_rmse:.2f}, MAE={avg_te_mae:.2f}")

    # --- Plot RMSE learning curve ---
    plt.figure(figsize=(8, 5))
    plt.plot(train_rmse, label='Train RMSE')
    plt.plot(val_rmse, label='Val RMSE')
    plt.xlabel('Epoch');
    plt.ylabel('RMSE')
    plt.legend();
    plt.grid(True);
    plt.tight_layout()
    plt.savefig(rmse_curve_path)
    print(f"RMSE curve saved: {rmse_curve_path}")

    # --- Plot MAE learning curve ---
    plt.figure(figsize=(8, 5))
    plt.plot(train_mae, label='Train MAE')
    plt.plot(val_mae, label='Val MAE')
    plt.xlabel('Epoch');
    plt.ylabel('MAE')
    plt.legend();
    plt.grid(True);
    plt.tight_layout()
    plt.savefig(mae_curve_path)
    print(f"MAE curve saved: {mae_curve_path}")

    return model

if __name__ == "__main__":

    SNAPSHOT_DIR = '../data/snapshots'
    MODEL_OUT = './models/edge_autoencoder.pt'
    CKPT_OUT = './models/edge_autoencoder.pt'
    RMSE_CURVE_OUT = './models/rmse_curve.png'
    MAE_CURVE_OUT = './models/mae_curve.png'

    model = train(
        SNAPSHOT_DIR,
        MODEL_OUT,
        CKPT_OUT,
        rmse_curve_path=RMSE_CURVE_OUT,
        mae_curve_path=MAE_CURVE_OUT,
        batch_size=64, lr=1e-3, epochs=500, patience=20,
        hidden_dims=[64, 32], bottleneck_dim=1
    )
