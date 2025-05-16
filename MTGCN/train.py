import os
import math
import torch
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from MTGCN.dataset import InMemoryGraphDataset, time_based_split
from MTGCN.autoencoder import EdgeAutoEncoderMultiTask


def train_multitask(
        snapshot_dir: str,
        model_out: str,
        ckpt_out: str,
        epochs: int = 200,
        batch_size: int = 64,
        lr: float = 1e-3,
        lambda_time: float = 1.0,
        patience: int = 20,
        rmse_curve_path: str = None,
        mae_curve_path: str = None,
        scatter_path: str = None
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"▶ Training on {device}\n")

    # Dataset & split
    full_ds = InMemoryGraphDataset(snapshot_dir)
    tr_idx, va_idx, te_idx = time_based_split(full_ds)
    tr_ds, va_ds, te_ds = (
        Subset(full_ds, tr_idx),
        Subset(full_ds, va_idx),
        Subset(full_ds, te_idx),
    )
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    te_ld = DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model, optimizer, losses
    model = EdgeAutoEncoderMultiTask().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    mse_time = torch.nn.MSELoss()
    mae_time = torch.nn.L1Loss()
    scaler = GradScaler()

    # Try resuming from checkpoint
    start_ep = 1
    best_va = float('inf')
    no_imp = 0
    if os.path.isfile(ckpt_out):
        print(f"↻ Found checkpoint '{ckpt_out}', resuming training...")
        ckpt = torch.load(ckpt_out, map_location=device)
        model.load_state_dict(ckpt['model'])
        optim.load_state_dict(ckpt['optim'])
        start_ep = ckpt.get('epoch', 1) + 1
        best_va = ckpt.get('best_va', best_va)
        no_imp = ckpt.get('no_imp', no_imp)
        print(f"    Resumed at epoch {start_ep - 1}, best_val_MSE={best_va:.6f}, no_imp={no_imp}\n")

    # Containers for curves
    train_time_mse, val_time_mse = [], []
    train_time_rmse, val_time_rmse = [], []
    train_time_mae, val_time_mae = [], []

    try:
        for ep in range(start_ep, epochs + 1):
            print(f"\n--- Epoch {ep}/{epochs} ---")
            # Training
            model.train()
            t_mse, t_mae = 0.0, 0.0
            for batch in tr_ld:
                batch = batch.to(device)
                optim.zero_grad()
                with autocast():
                    _, t_pred, _ = model(batch)
                    loss_time_mse = mse_time(t_pred, batch.edge_attr[:, 2])
                    loss_time_mae = mae_time(t_pred, batch.edge_attr[:, 2])
                    loss = lambda_time * loss_time_mse
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
                t_mse += loss_time_mse.item()
                t_mae += loss_time_mae.item()
            avg_tr_mse = t_mse / len(tr_ld)
            avg_tr_rmse = math.sqrt(avg_tr_mse)
            avg_tr_mae = t_mae / len(tr_ld)
            train_time_mse.append(avg_tr_mse)
            train_time_rmse.append(avg_tr_rmse)
            train_time_mae.append(avg_tr_mae)

            # Validation
            model.eval()
            v_mse, v_mae = 0.0, 0.0
            with torch.no_grad():
                for batch in va_ld:
                    batch = batch.to(device)
                    _, t_pred, _ = model(batch)
                    v_mse += mse_time(t_pred, batch.edge_attr[:, 2]).item()
                    v_mae += mae_time(t_pred, batch.edge_attr[:, 2]).item()
            avg_va_mse = v_mse / len(va_ld)
            avg_va_rmse = math.sqrt(avg_va_mse)
            avg_va_mae = v_mae / len(va_ld)
            val_time_mse.append(avg_va_mse)
            val_time_rmse.append(avg_va_rmse)
            val_time_mae.append(avg_va_mae)

            print(f" Train → MSE: {avg_tr_mse:.6f}, RMSE: {avg_tr_rmse:.6f}, MAE: {avg_tr_mae:.6f}")
            print(f" Val   → MSE: {avg_va_mse:.6f}, RMSE: {avg_va_rmse:.6f}, MAE: {avg_va_mae:.6f}")

            # Save checkpoint
            torch.save({
                'epoch': ep,
                'model': model.state_dict(),
                'optim': optim.state_dict(),
                'best_va': best_va,
                'no_imp': no_imp
            }, ckpt_out)
            print(f" Saved checkpoint to '{ckpt_out}'")

            # Update best
            if avg_va_mse < best_va - 1e-4:
                best_va = avg_va_mse
                no_imp = 0
                torch.save(model.state_dict(), model_out)
                print(f" ✔ New best model saved to '{model_out}'")
            else:
                no_imp += 1
                print(f" No improvement ({no_imp}/{patience})")

            # Early stopping
            if no_imp >= patience:
                print(f"⏱ Early stopping at epoch {ep}")
                break

    except KeyboardInterrupt:
        print("\n⚠️ Manual interruption detected! Saving current state...")
        torch.save({
            'epoch': ep,
            'model': model.state_dict(),
            'optim': optim.state_dict(),
            'best_va': best_va,
            'no_imp': no_imp
        }, ckpt_out)
        print(f" Checkpoint after interruption saved to '{ckpt_out}'")
        print(" Exiting training loop early.\n")
        return model

    # Final test
    print("\n▶ Final testing on held-out month")
    model.load_state_dict(torch.load(model_out, map_location=device))
    model.eval()
    te_mse, te_mae = 0.0, 0.0
    true_times, pred_times = [], []
    with torch.no_grad():
        for batch in te_ld:
            batch = batch.to(device)
            _, t_pred, _ = model(batch)
            te_mse += mse_time(t_pred, batch.edge_attr[:, 2]).item()
            te_mae += mae_time(t_pred, batch.edge_attr[:, 2]).item()
            true_times += batch.edge_attr[:, 2].cpu().tolist()
            pred_times += t_pred.cpu().tolist()
    avg_te_mse = te_mse / len(te_ld)
    avg_te_rmse = math.sqrt(avg_te_mse)
    avg_te_mae = te_mae / len(te_ld)
    print(f" → Test    MSE: {avg_te_mse:.6f}, RMSE: {avg_te_rmse:.6f}, MAE: {avg_te_mae:.6f}")

    # Plotting
    if rmse_curve_path:
        plt.figure(figsize=(8, 5))
        plt.plot(train_time_rmse, label='Train RMSE')
        plt.plot(val_time_rmse, label='Val RMSE')
        plt.xlabel('Epoch');
        plt.ylabel('RMSE')
        plt.legend();
        plt.grid(True);
        plt.tight_layout()
        plt.savefig(rmse_curve_path)
        print(f" RMSE curve saved to '{rmse_curve_path}'")

    if mae_curve_path:
        plt.figure(figsize=(8, 5))
        plt.plot(train_time_mae, label='Train MAE')
        plt.plot(val_time_mae, label='Val MAE')
        plt.xlabel('Epoch');
        plt.ylabel('MAE')
        plt.legend();
        plt.grid(True);
        plt.tight_layout()
        plt.savefig(mae_curve_path)
        print(f" MAE curve saved to '{mae_curve_path}'")

    if scatter_path:
        plt.figure(figsize=(6, 6))
        plt.scatter(true_times, pred_times, alpha=0.3)
        lims = [min(true_times + pred_times), max(true_times + pred_times)]
        plt.plot(lims, lims, 'r--')
        plt.xlabel('True Time');
        plt.ylabel('Predicted Time')
        plt.title('Pred vs True Travel Time')
        plt.grid(True);
        plt.tight_layout()
        plt.savefig(scatter_path)
        print(f" Scatter plot saved to '{scatter_path}'")

    return model


if __name__ == "__main__":

    SNAPSHOT_DIR = '../data/snapshots'
    MODEL_OUT = './models/edge_autoencoder.pt'
    CKPT_OUT = './models/edge_autoencoder.ckpt'
    MAE_CURVE_OUT = './models/mae_curve.png'
    RMSE_CURVE_OUT = './models/rmse_curve.png'
    SCATTER_OUT = './models/scatter.png'

    train_multitask(
        snapshot_dir=SNAPSHOT_DIR,
        model_out=MODEL_OUT,
        ckpt_out=CKPT_OUT,
        epochs=200,
        batch_size=64,
        lr=1e-3,
        lambda_time=1.0,
        patience=20,
        rmse_curve_path=RMSE_CURVE_OUT,
        mae_curve_path=MAE_CURVE_OUT,
        scatter_path=SCATTER_OUT
    )
