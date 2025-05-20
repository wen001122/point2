# collate_fn.py

import torch

def collate_fn(batch_list):
    # 合并多个样本为一个 batch
    coord = torch.cat([b["coord"] for b in batch_list], dim=0)
    feat = torch.cat([b["feat"] for b in batch_list], dim=0)
    label = torch.cat([b["label"] for b in batch_list], dim=0)

    # 每个样本对应一个 batch_id：0, 1, 2...
    batch = []
    for i, b in enumerate(batch_list):
        num_points = b["coord"].shape[0]
        batch.append(torch.full((num_points,), i, dtype=torch.long))
    batch = torch.cat(batch, dim=0)

    return {
        "coord": coord,
        "feat": feat,
        "label": label,
        "batch": batch,
        "grid_size": batch_list[0]["grid_size"]  # 所有样本 grid_size 相同
    }
