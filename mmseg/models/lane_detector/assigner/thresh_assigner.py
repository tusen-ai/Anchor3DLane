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

INFINITY = 987654.

@ASSIGNER.register_module()
class ThreshAssigner(object):
    def __init__(self, t_pos=3.5, t_neg=4.5, anchor_len=10, pos_k=5, neg_k=2000, metric='Euclidean', **kwargs):
        self.t_pos = t_pos
        self.t_neg = t_neg
        self.anchor_len = anchor_len
        self.metric = metric
        self.pos_k = pos_k
        self.neg_k = neg_k

    def match_proposals_with_targets(self, proposals, targets, **kwargs):
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

        positives = distances.min(dim=1)[0] < self.t_pos   # [Np], [True, True, False, False, ...]
        all_negatives = distances.min(dim=1)[0] > self.t_neg   # [Np]  [False, False, True, False, ...]
        
        # randomly select negatives
        all_neg_indices = all_negatives.nonzero().view(-1)    # [Num_neg]
        perm = torch.randperm(all_neg_indices.shape[0])    # [Num_neg]
        neg_k = min(self.neg_k, len(all_neg_indices))
        negative_indices = all_neg_indices[perm[:neg_k]]  # [neg_k]
        negatives = distances.new_zeros(num_proposals).to(torch.bool)   # [Np]
        negatives[negative_indices] = True
        if positives.sum() == 0:
            target_positives_indices = torch.tensor([], device=positives.device, dtype=torch.long)
        else:
            target_positives_indices = distances[positives].argmin(dim=1)   # [N_pos]

        return positives, negatives, target_positives_indices
