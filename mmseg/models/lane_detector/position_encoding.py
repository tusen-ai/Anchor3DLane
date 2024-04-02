# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn
import pdb
# from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats  # 128
        self.temperature = temperature  # 10000
        self.normalize = normalize   # True

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale  # 2pi

    def forward(self, x, mask=None):
        # x = tensor_list.tensors
        # mask = tensor_list.mask  # the image location which is padded with 0 is set to be 1 at the corresponding mask location
        # print(x.shape)  # b 128 8 8
        # print(mask.shape)  # b 8 8
        # exit()
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask  # image 0 -> 0 [B, H, W]

        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # 2 28 38
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # 2 28 38

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale  # [0~2pi]
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale  # [0~2pi]

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)

        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)  # [C]

        pos_x = x_embed[:, :, :, None] / dim_t   # [B, H, W, C//2]
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


class PositionEmbeddingSine3D(nn.Module):
    """
    This class extends positional embedding to 3d spaces.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, norm=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats  # 128
        self.temperature = temperature  # 10000
        self.normalize = normalize   # True
        self.norm = norm

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale  # 2pi

    def forward(self, x_embed, y_embed, z_embed):
        if self.normalize:
            eps = 1e-6
            z_embed = (z_embed - self.norm[2]) / (self.norm[5] - self.norm[2] + eps) * self.scale
            y_embed = (y_embed - self.norm[1]) / (self.norm[4] - self.norm[1] + eps) * self.scale
            x_embed = (x_embed - self.norm[0]) / (self.norm[3] - self.norm[0] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x_embed.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        dim_t_z = torch.arange((self.num_pos_feats * 2), dtype=torch.float32, device=x_embed.device)
        dim_t_z = self.temperature ** (2 * (dim_t_z // 2) / (self.num_pos_feats * 2))

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_z = z_embed[..., None] / dim_t_z

        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)  # [B, N, C//2]
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)  # [B, N, C]

        pos = torch.cat((pos_y, pos_x), dim=-1) + pos_z  # [B, N, C]

        return pos

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos