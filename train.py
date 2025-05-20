# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ear_dataset import EarDataset
from collate_fn import collate_fn
from point import Point
from ptv3_model import PointTransformerV3  # 你上传的模型结构

def train():
    # ============ 配置参数 ============
    data_root = "data"
    batch_size = 1
    max_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============ 数据集 ============
    train_set = EarDataset(data_root=data_root, split="train")
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4
    )

    # ============ 模型 ============
    model = PointTransformerV3(input_dim=6, num_classes=2)  # 你模型中 forward 最后需输出 (N, num_classes)
    model.to(device)

    # ============ 优化器和损失 ============
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # ============ 训练循环 ============
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0

        for batch in train_loader:
            point = Point(batch).sparsify()
            inputs = point.sparse_conv_feat.to(device)          # 稀疏输入
            labels = batch["label"].to(device)                  # 标签 (N,)

            outputs = model(inputs)                             # 输出 (N, num_classes)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    train()
