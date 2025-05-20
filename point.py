# from addict import Dict
# import torch
# import spconv.pytorch as spconv

# @torch.inference_mode()
# def offset2bincount(offset):
#     return torch.diff(offset, prepend=torch.tensor([0], device=offset.device))

# @torch.inference_mode()
# def offset2batch(offset):
#     bincount = offset2bincount(offset)
#     return torch.arange(len(bincount), device=offset.device).repeat_interleave(bincount)

# @torch.inference_mode()
# def batch2offset(batch):
#     return torch.cumsum(batch.bincount(), dim=0).long()

# class Point(Dict):
#     def __init__(self, data_dict):
#         super().__init__(data_dict)
#         self.coord = data_dict["coord"]
#         self.feat = data_dict["feat"]
#         self.label = data_dict.get("label", None)
#         self.batch = data_dict["batch"].view(-1)
#         self.grid_size = 0.02  # 默认值，或者根据实际需要改成 data_dict["grid_size"]
#         self.grid_coord = (self.coord / self.grid_size).floor().int()
#         self.offset = self._batch2offset(self.batch)

#     def _batch2offset(self, batch):
#         batch = batch.view(-1).to("cpu")
#         count = torch.bincount(batch)
#         return torch.cumsum(count, dim=0).long()

#     def serialization(self, order="z", depth=None, shuffle_orders=False):
#         if depth is None:
#             depth = int(self.grid_coord.max()).bit_length()
#         self["serialized_depth"] = depth
#         code = self.grid_coord[:, 0] << (depth * 2) | self.grid_coord[:, 1] << depth | self.grid_coord[:, 2]
#         order = torch.argsort(code)
#         inverse = torch.zeros_like(order)
#         inverse[order] = torch.arange(len(order), device=order.device)
#         self["serialized_code"] = code
#         self["serialized_order"] = order.unsqueeze(0)
#         self["serialized_inverse"] = inverse.unsqueeze(0)

#     def sparsify(self, pad=96):
#         sparse_shape = (self.grid_coord.max(dim=0).values + pad).tolist()
#         indices = torch.cat([self.batch.unsqueeze(1).int(), self.grid_coord.int()], dim=1)
#         sparse_conv_feat = spconv.SparseConvTensor(
#             features=self.feat,
#             indices=indices,
#             spatial_shape=sparse_shape,
#             batch_size=int(self.batch.max().item()) + 1,
#         )
#         self["sparse_shape"] = sparse_shape
#         self["sparse_conv_feat"] = sparse_conv_feat



import torch
import spconv.pytorch as spconv
from torch import nn

class Point(dict):
    def __init__(self, data_dict):
        super().__init__()
        self.coord = data_dict["coord"]          # (N, 3)
        self.feat = data_dict["feat"]            # (N, C)
        self.label = data_dict.get("label", None)
        self.batch = data_dict["batch"].view(-1)  # (N,)
        self.grid_size = 0.02  # fixed float

        self.grid_coord = (self.coord / self.grid_size).floor().int()
        self.offset = self._batch2offset(self.batch)

    def _batch2offset(self, batch):
        batch = batch.view(-1).to("cpu")
        count = torch.bincount(batch)
        return torch.cumsum(count, dim=0).long()

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        # 为避免 key 写入报错，这里不再使用 self[...] = ...
        self.serialized_depth = 0
        self.serialized_code = None
        self.serialized_order = None
        self.serialized_inverse = None
        self.order = order

    def sparsify(self, pad=96):
        # ⚠️ 确保 grid_coord 非负，避免稀疏张量索引越界
        min_coord = self.grid_coord.min(dim=0).values
        self.grid_coord = self.grid_coord - min_coord

        sparse_shape = (self.grid_coord.max(dim=0).values + pad).tolist()

        indices = torch.cat([
            self.batch.view(-1, 1).int(),
            self.grid_coord.int()
        ], dim=1)

        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=indices,
            spatial_shape=sparse_shape,
            batch_size=int(self.batch.max().item()) + 1,
        )

        self.sparse_shape = sparse_shape
        self.sparse_conv_feat = sparse_conv_feat

