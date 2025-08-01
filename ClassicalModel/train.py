
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import r2_score

from model import NeuralNetworkModel, ArcheologicalDataset

EPOCH_NUM = 200
BATCH_SIZE = 5
DEVICE = "cuda"

def train():
    dataset = ArcheologicalDataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    criterion = nn.HuberLoss()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = NeuralNetworkModel(dataset.features_length).to(device="cuda")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("---------------- Starting Training --------------------")
    for epoch in range(EPOCH_NUM):
        model.train()
        all_preds = []
        all_targets = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch.reshape((BATCH_SIZE, 1)))
            loss.backward()
            optimizer.step()
            all_preds.append(y_pred.clone().detach().cpu())
            all_targets.append(y_batch.clone().detach().cpu())

        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()

        r2 = r2_score(targets, preds)

        print(f"Epoch {epoch+1}/{EPOCH_NUM} — R² Score: {r2:.4f}")

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(DEVICE)
            y_pred = model(x_batch).cpu()
            all_preds.append(y_pred)
            all_targets.append(y_batch)

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    r2 = r2_score(targets, preds)
    print(f"Evaluation — R² Score: {r2:.4f}")


if __name__ == "__main__":
    train()
