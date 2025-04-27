import torch
from torch_geometric.loader import DataLoader
import os
from dotenv import load_dotenv
from torch.optim.lr_scheduler import ReduceLROnPlateau

load_dotenv()

from GCN.autoencoder import EncoderDecoderModel
from GCN.dataset import RoadNetworkSnapshotDataset


# --- Functions ---

def train(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_edges
    return total_loss / len(train_loader.dataset)


def evaluate(model, criterion, loader, device):
    model.eval()
    total_loss = 0
    total_mae = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y)
            mae = torch.mean(torch.abs(output - data.y))
            total_loss += loss.item() * data.num_edges
            total_mae += mae.item() * data.num_edges
    return total_loss / len(loader.dataset), total_mae / len(loader.dataset)


def train_model(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs=100, patience=10):
    criterion = torch.nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, optimizer, criterion, train_loader, device)
        val_loss, val_mae = evaluate(model, criterion, val_loader, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:03d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, Val MAE = {val_mae:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses


# --- Main ---

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    snapshot_dir = os.path.join(os.getenv('DATA_PATH'), 'snapshots')
    full_dataset = RoadNetworkSnapshotDataset(root=snapshot_dir)

    num_snapshots = len(full_dataset)
    num_train = int(0.6 * num_snapshots)
    num_val = int(0.2 * num_snapshots)
    num_test = num_snapshots - num_train - num_val

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [num_train, num_val, num_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = EncoderDecoderModel(
        edge_in_channels=6,  # ["length", "sin(hour)", "cos(hour)", "lane", "rain", #avgSpeed]
        gcn_hidden_channels=128,
        bottleneck_dim=64,
        decoder_hidden_channels=128
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    model, train_losses, val_losses = train_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=100,
        patience=10
    )

    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'best_encoder_decoder.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Best model saved at {model_path}")

    test_mse, test_mae = evaluate(model, torch.nn.MSELoss(), test_loader, device)
    print(f"Test MSE: {test_mse:.6f} | Test MAE: {test_mae:.6f}")
