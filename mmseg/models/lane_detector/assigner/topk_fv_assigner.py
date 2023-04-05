# --------------------------------------------------------
# Source code for Anchor3DLane
# Copyright (c) 2023 TuSimple
# @Time    : 2023/04/05
# @Author  : Shaofei Huang
# nowherespyfly@gmail.com
# --------------------------------------------------------

import torch
import numpy as np
from ...builder import ASSIGNER
from .distance_metric import *

INFINITY = 987654.

@ASSIGNER.register_module()
class TopkFVAssigner(object):
    def __init__(self, pos_k=30, neg_k=100, anchor_len=10, y_steps_3d=None, w2d=0.5, w3d=0.5, **kwargs):
        self.pos_k = pos_k
        self.neg_k = neg_k
        self.anchor_len = anchor_len
        self.y_steps_3d = np.array(y_steps_3d, dtype=np.float32)
        self.w2d = w2d
        self.w3d = w3d

    def match_proposals_with_targets(self, proposals, targets_3d, targets_2d, P_g2im, anchor_len_2d=72):

        num_proposals = proposals.shape[0]   # [Np, 35], [pos_score, neg_score, start_y, end_y, length, x_coord, z_coord, vis]
        num_targets = targets_3d.shape[0]   # [Nt, 35], [1, 0, start_y, end_y, length, x_coord, z_coord, vis]
        proposals = torch.repeat_interleave(proposals, num_targets, dim=0)  # [Np * Nt, 35], [a, b] -> [a, a, b, b]
        targets_3d = torch.cat(num_proposals * [targets_3d])   # [Nt * Np, 10, 35], [c, d] -> [c, d, c, d]
        targets_2d = torch.cat(num_proposals * [targets_2d])   # [Nt * Np, 10, 35], [c, d] -> [c, d, c, d]
        distances_fv = FV_Euclidean(proposals, targets_2d, num_proposals, num_targets, anchor_len=self.anchor_len, 
            y_steps_3d=self.y_steps_3d, P_g2im=P_g2im, anchor_len_2d=anchor_len_2d) / 360
        # [Nt, Np]
        distances_fv = distances_fv / distances_fv[distances_fv < INFINITY].max()
        distances_tv = Euclidean_dis(proposals, targets_3d, num_proposals, num_targets, anchor_len=self.anchor_len)
        distances_tv = distances_tv / distances_tv[distances_tv < INFINITY].max()
        distances = distances_fv * self.w2d + distances_tv * self.w3d
        # in case the same anchor been assigned twice
        min_indices = distances.min(dim=1)[1]   # [Np]
        range_indices = torch.arange(num_proposals).long().to(distances.device)
        invalid_mask = distances.new_ones(num_proposals, num_targets).to(torch.bool)   # [Np, Nt]
        invalid_mask[range_indices, min_indices] = False
        distances[invalid_mask] = INFINITY

        # select topk anchors for each gt
        topk_distances, topk_indices = distances.topk(self.pos_k, dim=0, largest=False)   # [pos_k, Nt]
        all_pos_indices = topk_indices.view(-1)   # [pos_k * Nt]
        positives = distances.new_zeros(num_proposals).to(torch.bool)   # [Np]
        positives[all_pos_indices] = True
        negatives = ~positives
        all_neg_indices = negatives.nonzero().view(-1)    # [Num_neg]
        perm = torch.randperm(all_neg_indices.shape[0])    # [Num_neg]
        negative_indices = all_neg_indices[perm[:self.neg_k]]  # [neg_k]
        negatives = distances.new_zeros(num_proposals).to(torch.bool)   # [Np]
        negatives[negative_indices] = True

        if positives.sum() == 0:
            target_positives_indices = torch.tensor([], device=positives.device, dtype=torch.long)
        else:
            target_positives_indices = distances[positives].argmin(dim=1)   # [N_pos]

        return positives, negatives, target_positives_indices