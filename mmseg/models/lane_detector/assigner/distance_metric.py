# --------------------------------------------------------
# Source code for Anchor3DLane
# Copyright (c) 2023 TuSimple
# @Time    : 2023/04/05
# @Author  : Shaofei Huang
# nowherespyfly@gmail.com
# --------------------------------------------------------

from tarfile import LENGTH_LINK
import torch
import pdb
INFINITY = 987654.

def projection_transform(Matrix, xs, ys, zs):
    # Matrix: [3, 4], x, y, z: [Nl]
    ones = torch.ones_like(zs)   # [Nl]
    coordinates = torch.stack([xs, ys, zs, ones], dim=0)   # [4, Nl]
    trans = torch.matmul(Matrix, coordinates)   # [3, Nl]

    u_vals = trans[0, :] / trans[2, :]   # [Nl]
    v_vals = trans[1, :] / trans[2, :]   # [Nl]
    return u_vals, v_vals

def Euclidean_dis(proposals, targets, num_pro, num_tar, anchor_len=10):
    target_vis = targets[:, 5 + anchor_len * 2:5 + anchor_len * 3]   # [Np * Nt, 10], -1e5, if non-lane
    lengths = target_vis.sum(dim=1)  # [Np * Nt]
    distances_all = torch.abs((targets - proposals)) # [Np * Nt, 35]
    distances_x = distances_all[:, 5:5 + anchor_len]
    distances_z = distances_all[:, 5 + anchor_len:5 + anchor_len * 2]
    distances = (((distances_x ** 2  + distances_z ** 2) ** 0.5) * target_vis).sum(dim=1) / (lengths + 1e-9)   # [Np * Nt]
    distances[lengths < 0] = INFINITY
    distances = distances.reshape(num_pro, num_tar)   # [Np, Nt]
    return distances

def Partial_Euclidean_dis(proposals, targets, num_pro, num_tar, anchor_len=10, close_weight=0.7):
    target_vis = targets[:, 5 + anchor_len * 2:5 + anchor_len * 3]   # [Np * Nt, 10], -1e5, if non-lane
    lengths = target_vis.sum(dim=1)  # [Np * Nt]
    distances_all = torch.abs((targets - proposals)) # [Np * Nt, 35]
    distances_x_close = distances_all[:, 5:5 + anchor_len // 2] * close_weight
    distances_x_far = distances_all[:, 5 + anchor_len // 2:5 + anchor_len] * (1 - close_weight)
    distances_z_close= distances_all[:, 5 + anchor_len:5 + anchor_len + anchor_len // 2] * close_weight
    distances_z_far = distances_all[:, 5 + anchor_len + anchor_len // 2:5 + anchor_len * 2] * (1 - close_weight)
    distances_x = torch.cat([distances_x_close, distances_x_far], dim=1)
    distances_z = torch.cat([distances_z_close, distances_z_far], dim=1)
    distances = (((distances_x ** 2  + distances_z ** 2) ** 0.5) * target_vis).sum(dim=1) / (lengths + 1e-9)   # [Np * Nt]
    distances[lengths < 0] = INFINITY
    distances = distances.reshape(num_pro, num_tar)   # [Np, Nt]
    return distances

def Manhattan_dis(proposals, targets, num_pro, num_tar, anchor_len=10):
    target_vis = targets[:, 5 + anchor_len * 2:5 + anchor_len * 3]   # [Np * Nt, 10], -1e5, if non-lane
    lengths = target_vis.sum(dim=1)  # [Np * Nt]
    distances_all = torch.abs((targets - proposals)) # [Np * Nt, 35]
    distances_x = distances_all[:, 5:5 + anchor_len]
    distances_z = distances_all[:, 5 + anchor_len:5 + anchor_len * 2]
    distances = ((abs(distances_x)  + abs(distances_z)) * target_vis).sum(dim=1) / (lengths + 1e-9)   # [Np * Nt]
    distances[lengths < 0] = INFINITY
    distances = distances.reshape(num_pro, num_tar)   # [Np, Nt]
    return distances

def Height_dis(proposals, targets, num_pro, num_tar, anchor_len=10):
    target_vis = targets[:, 5 + anchor_len * 2:5 + anchor_len * 3]   # [Np * Nt, 10], -1e5, if non-lane
    lengths = target_vis.sum(dim=1)  # [Np * Nt]
    distances_all = torch.abs((targets - proposals)) # [Np * Nt, 35]
    # distances_x = distances_all[:, 5:5 + anchor_len]
    distances_z = distances_all[:, 5 + anchor_len:5 + anchor_len * 2]
    distances = (torch.abs(distances_z) * target_vis).sum(dim=1) / (lengths + 1e-9)   # [Np * Nt]
    # distances = (((distances_x ** 2  + distances_z ** 2) ** 0.5) * target_vis).sum(dim=1) / (lengths + 1e-9)   # [Np * Nt]
    distances[lengths < 0] = INFINITY
    distances = distances.reshape(num_pro, num_tar)   # [Np, Nt]
    return distances

def FV_Euclidean(proposals, targets, num_pro, num_tar, anchor_len=10, y_steps_3d=None, P_g2im=None, anchor_len_2d=72):
    targets_x = targets[:, :anchor_len_2d]
    y_steps_2d = targets[:, anchor_len_2d:anchor_len_2d*2]   # [N, 10]
    targets_vis = targets[:, anchor_len_2d*2:anchor_len_2d*3]
    lengths = targets_vis.sum(dim=1)  # [Np * Nt]

    proposals_x = proposals[:, 5 : 5 + anchor_len]
    proposals_z = proposals[:, 5 + anchor_len : 5 + anchor_len * 2]
    y_steps_3d = torch.from_numpy(y_steps_3d).to(targets.device)
    y_steps_3d = y_steps_3d.repeat(num_pro * num_tar, 1)
    proposals_u, proposals_v = projection_transform(P_g2im, proposals_x.reshape(-1), y_steps_3d.reshape(-1), proposals_z.reshape(-1))
    proposals_u = proposals_u.reshape(-1, anchor_len)   # [Np * Nt, 10]
    proposals_v = proposals_v.reshape(-1, anchor_len)
    tans = (proposals_u[:, 1] - proposals_u[:, 0]) / (proposals_v[:, 1] - proposals_v[:, 0] + 1e-11)  # [N]
    proposals_x_2d = proposals_u[:, 0:1] + tans.unsqueeze(1) * (y_steps_2d - proposals_v[:, 0:1])   # [N, 10]
    
    distances = (torch.abs(targets_x - proposals_x_2d) * targets_vis).sum(dim=1) / (lengths + 1e-9)  # [Np, Nt]
    distances[lengths < 0] = INFINITY
    distances = distances.reshape(num_pro, num_tar)   # [Np, Nt]
    return distances