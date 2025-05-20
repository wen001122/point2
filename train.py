import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import PointTransformerV3
from point import Point

class EarDataset(Dataset):
    def __init__(self, data_root, split="train"):
        split_dir = os.path.join(data_root, split)
        self.file_list = sorted([
            os.path.join(split_dir, fname)
            for fname in os.listdir(split_dir)
            if fname.endswith(".npy")
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        arr = np.load(self.file_list[idx])  # shape: (24000, 7)
        coord = arr[:, :3]                  # (N, 3)
        normal = arr[:, 3:6]                # (N, 3)
        label = arr[:, 6].astype(np.int64)  # (N,)
        feat = np.concatenate([coord, normal], axis=1)  # (N, 6)
        batch = np.zeros((arr.shape[0],), dtype=np.int64)

        return {
            "coord": torch.tensor(coord, dtype=torch.float32),
            "feat": torch.tensor(feat, dtype=torch.float32),
            "batch": torch.tensor(batch, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        for k in data:
            data[k] = data[k].to(device)

        point = Point(data)
        point.serialization()
        point.sparsify()
        out = model(point)  # out.feat: (N, C)

        pred = out.feat
        loss = criterion(pred, data['label'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = PointTransformerV3(in_channels=6, cls_mode=False, enable_flash=False)
    model = model.to(device)

    # Dataset and Dataloader
    train_set = EarDataset(data_root="data", split="train")
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    for epoch in range(1, 101):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

if __name__ == '__main__':
    main()
