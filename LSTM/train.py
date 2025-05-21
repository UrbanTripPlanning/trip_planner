import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from lstm_autoencoder import LSTMAutoEncoder
from dataset import SequenceDataset, InMemoryGraphDataset

def train_lstm(
    snapshot_dir, model_path, ckpt_path, curve_out,
    batch_size=32, lr=1e-3, epochs=50,
    hidden_dim=64, bottleneck_dim=32, window_size=12
):
    # Choose device: GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"▶ Training LSTM Autoencoder on {device}\n")

    # Load full dataset into memory
    full_ds = InMemoryGraphDataset(snapshot_dir)
    tr_idx, va_idx, _ = InMemoryGraphDataset.time_based_split(full_ds)

    # Build sequence datasets for training and validation
    tr_ds = SequenceDataset(full_ds, tr_idx, window_size)
    va_ds = SequenceDataset(full_ds, va_idx, window_size)

    # Wrap datasets into dataloaders
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    # Get input dimension from first sequence window (shape: [window, edge_dim])
    input_dim = tr_ds[0].shape[-1]
    print(f"🧠 Input feature dimension: {input_dim}")

    # Model config dict to be saved with checkpoint
    config = {
        'input_size': input_dim,
        'hidden_size': hidden_dim,
        'bottleneck_dim': bottleneck_dim,
        'output_size': input_dim,
    }

    # Initialize model and move to device
    model = LSTMAutoEncoder(
        input_size=input_dim,
        hidden_size=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        output_size=input_dim
    ).to(device)

    # Optimizer and loss function
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    # Load checkpoint if available
    start_ep, best_val = 1, float('inf')
    if os.path.isfile(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck['model'])
        optim.load_state_dict(ck['optim'])
        start_ep = ck['epoch'] + 1
        best_val = ck['best_val']
        print(f"↻ Resuming from epoch {start_ep}, best_val={best_val:.4f}\n")

    # Track losses for plotting
    train_losses, val_losses = [], []

    try:
        for ep in range(start_ep, epochs + 1):
            # ======= Training =======
            model.train()
            t_loss = 0
            for batch in tqdm(tr_ld, desc=f"Epoch {ep}/{epochs} [train]"):
                batch = batch.to(device)

                # Reshape to: [batch, window, edge_dim] → [batch * edges, window, edge_dim]
                batch = batch.view(batch.size(0), batch.size(1), -1, input_dim)
                batch = batch.permute(0, 2, 1, 3).reshape(-1, batch.size(1), input_dim)

                optim.zero_grad()
                recon, _ = model(batch)
                loss = mse(recon, batch)
                loss.backward()
                optim.step()
                t_loss += loss.item()

            avg_tr = t_loss / len(tr_ld)

            # ======= Validation =======
            model.eval()
            v_loss = 0
            for batch in tqdm(va_ld, desc=f"Epoch {ep}/{epochs} [val]  "):
                batch = batch.to(device)
                with torch.no_grad():
                    recon, _ = model(batch)
                    v_loss += mse(recon, batch).item()
            avg_va = v_loss / len(va_ld)

            print(f"\nEpoch {ep}/{epochs} → Train: {avg_tr:.4f} | Val: {avg_va:.4f}\n")
            train_losses.append(avg_tr)
            val_losses.append(avg_va)

            # Save best model
            if avg_va < best_val:
                best_val = avg_va
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save({
                    'model': model.state_dict(),
                    'config': config
                }, model_path)
                print(f"✔ Best model saved: {model_path}")

            # Save checkpoint for resuming
            torch.save({
                'epoch': ep,
                'model': model.state_dict(),
                'optim': optim.state_dict(),
                'best_val': best_val,
                'config': config
            }, ckpt_path)

    except KeyboardInterrupt:
        print("\n⏸ Interrupted, checkpoint saved.")

    # ======= Plot learning curves =======
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(curve_out)
    print(f"\n📈 Learning curve saved to: {curve_out}")

    return model


if __name__ == "__main__":
    # ======= Config and Paths =======
    SNAPSHOT_DIR = './data/snapshots'
    MODEL_OUT = './STGCN/models/lstm_autoencoder.pt'
    CKPT_OUT = './STGCN/models/lstm_autoencoder.pt'
    CURVE_OUT = './STGCN/models/lstm_curve.png'

    # ======= Start Training =======
    model = train_lstm(
        snapshot_dir=SNAPSHOT_DIR,
        model_path=MODEL_OUT,
        ckpt_path=CKPT_OUT,
        curve_out=CURVE_OUT,
        batch_size=128,
        lr=1e-3,
        epochs=500,
        hidden_dim=64,
        bottleneck_dim=32,
        window_size=12
    )
