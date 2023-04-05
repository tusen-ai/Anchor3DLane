# --------------------------------------------------------
# Source code for Anchor3DLane
# Copyright (c) 2023 TuSimple
# @Time    : 2023/04/05
# @Author  : Shaofei Huang
# nowherespyfly@gmail.com
# --------------------------------------------------------

import torch
from ...builder import ASSIGNER
from .distance_metric import Euclidean_dis, Manhattan_dis, Partial_Euclidean_dis
import pdb

INFINITY = 987654.

@ASSIGNER.register_module()
class TopkAssigner(object):
    def __init__(self, pos_k=30, neg_k=100, anchor_len=10, t_pos=INFINITY, t_neg=0, neg_scale=None, metric='Euclidean', **kwargs):
        self.pos_k = pos_k
        self.neg_k = neg_k
        self.t_neg = t_neg
        self.t_pos = t_pos
        self.neg_scale = neg_scale
        self.anchor_len = anchor_len
        self.metric = metric

    def match_proposals_with_targets(self, proposals, targets, return_dis=False, **kwargs):
        valid_targets = targets[targets[:, 1] > 0]
        num_proposals = proposals.shape[0]   # [Np, 35], [pos_score, neg_score, start_y, end_y, length, x_coord, z_coord, vis]
        num_targets = valid_targets.shape[0]   # [Nt, 35], [1, 0, start_y, end_y, length, x_coord, z_coord, vis]

        proposals = torch.repeat_interleave(proposals, num_targets, dim=0)  # [Np * Nt, 35], [a, b] -> [a, a, b, b]
        valid_targets = torch.cat(num_proposals * [valid_targets])   # [Nt * Np, 10, 35], [c, d] -> [c, d, c, d]

        if self.metric == 'Euclidean':
            distances = Euclidean_dis(proposals, valid_targets, num_proposals, num_targets, anchor_len=self.anchor_len)
        elif self.metric == 'Manhattan':
            distances = Manhattan_dis(proposals, valid_targets, num_proposals, num_targets, anchor_len=self.anchor_len)
        elif self.metric == 'Partial_Euclidean':
            distances = Partial_Euclidean_dis(proposals, valid_targets, num_proposals, num_targets, anchor_len=self.anchor_len)
        else:
            raise Exception("No metrics as ", self.metric)

        topk_distances, topk_indices = distances.topk(self.pos_k, dim=0, largest=False)   # [pos_k, Nt]
        # in case the same anchor been assigned twice
        min_indices = distances.min(dim=1)[1]   # [Np]
        range_indices = torch.arange(num_proposals).long().to(distances.device)
        invalid_mask = distances.new_ones(num_proposals, num_targets).to(torch.bool)   # [Np, Nt]
        invalid_mask[range_indices, min_indices] = False
        distances[invalid_mask] = INFINITY
        proposal_distances = distances.min(dim=1)[0]   # [Np]

        # select topk anchors for each gt
        topk_distances, topk_indices = distances.topk(self.pos_k, dim=0, largest=False)   # [pos_k, Nt]

        all_pos_indices = topk_indices.view(-1)   # [pos_k * Nt]
        all_pos_distances = topk_distances.view(-1)
        all_pos_indices = all_pos_indices[all_pos_distances < self.t_pos]

        positives = distances.new_zeros(num_proposals).to(torch.bool)   # [Np]
        positives[all_pos_indices] = True
        negatives = ~positives
        all_neg_indices = negatives.nonzero().view(-1)    # [Num_neg]
        if self.neg_scale is not None:
            t_neg = all_pos_distances.max() * self.neg_scale
        else:
            t_neg = self.t_neg
        all_neg_indices = all_neg_indices[proposal_distances[negatives] > t_neg]
        perm = torch.randperm(all_neg_indices.shape[0])    # [Num_neg]
        neg_k = min(self.neg_k, len(all_neg_indices))
        negative_indices = all_neg_indices[perm[:neg_k]]  # [neg_k]
        negatives = distances.new_zeros(num_proposals).to(torch.bool)   # [Np]
        negatives[negative_indices] = True

        if positives.sum() == 0:
            target_positives_indices = torch.tensor([], device=positives.device, dtype=torch.long)
            target_positives_distances = torch.tensor([], device=positives.device, dtype=torch.float32)
        else:
            target_positives_distances, target_positives_indices = distances[positives].min(dim=1)   # [N_pos]

        if return_dis:
            return positives, negatives, target_positives_indices, target_positives_distances  
        return positives, negatives, target_positives_indices
