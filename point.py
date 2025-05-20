# model/point.py

import torch
import spconv.pytorch as spconv

class Point:
    def __init__(self, data_dict):
        self.coord = data_dict["coord"]  # (N, 3)
        self.feat = data_dict["feat"]    # (N, C)
        self.label = data_dict.get("label", None)  # (N,)
        self.batch = data_dict["batch"].view(-1)   # (N,)
        self.grid_size = data_dict["grid_size"]    # float

        self.grid_coord = (self.coord / self.grid_size).floor().int()  # voxel coord
        self.offset = self._batch2offset(self.batch)

    def _batch2offset(self, batch):
        batch = batch.view(-1).to("cpu")
        count = torch.bincount(batch)
        return torch.cumsum(count, dim=0).long()

    def sparsify(self):
        batch_col = self.batch.view(-1, 1).int()  # (N, 1)
        indices = torch.cat([batch_col, self.grid_coord], dim=1)  # (N, 4)
        spatial_shape = self.grid_coord.max(dim=0).values + 1

        self.sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=indices,
            spatial_shape=spatial_shape.tolist(),
            batch_size=int(self.batch.max().item()) + 1
        )
        return self
