# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from mmseg.models.lane_detector.assigner.distance_metric import Manhattan_dis
from torch import nn
import torch.nn.functional as F


class HSFHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1,
                 poly_weight: float = 1, lower_weight: float = 1, upper_weight: float = 1,
                 seq_len: int = 0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super(HSFHungarianMatcher, self).__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        threshold = 15 / 720.
        self.threshold = nn.Threshold(threshold**2, 0.)

        self.poly_weight = poly_weight
        self.lower_weight = lower_weight
        self.upper_weight = upper_weight

        self.seq_len = seq_len

    @torch.no_grad()
    def forward(self, outputs, targets, targets_flag):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        tgt_ids  = torch.cat([tgt[:, 0] for tgt in targets]).long()  # 0 ~ b*nq
        # print('tgt_ids.shape: {}'.format(tgt_ids.shape))
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]
        # print('cost_class.shape: {}'.format(cost_class.shape))

        out_bbox = outputs["pred_boxes"]
        # print('out_bbox.shape: {}'.format(out_bbox.shape)) # B, num_queries, kps_dim
        self.seq_len = (targets[0].shape[1] - 5) // 8
        gflat_targets = [tgt[:, 3 + self.seq_len * 2:5 + self.seq_len * 5] for tgt in targets]
        # print(self.seq_len);exit()

        tgt_gflatlowers = torch.cat([tgt[:, 0] for tgt in gflat_targets])
        tgt_gflatuppers = torch.cat([tgt[:, 1] for tgt in gflat_targets])
        cost_gflatlower = torch.cdist(out_bbox[:, :, 2].view((-1, 1)), tgt_gflatlowers.unsqueeze(-1), p=1)
        cost_gflatupper = torch.cdist(out_bbox[:, :, 3].view((-1, 1)), tgt_gflatuppers.unsqueeze(-1), p=1)

        # targets       = [tgt[:, :3+self.seq_len*2] for tgt in targets]
        # tgt_lowers      = torch.cat([tgt[:, 1] for tgt in targets])
        # tgt_uppers      = torch.cat([tgt[:, 2] for tgt in targets])
        # cost_lower = torch.cdist(out_bbox[:, :, 0].view((-1, 1)), tgt_lowers.unsqueeze(-1), p=1)
        # cost_upper = torch.cdist(out_bbox[:, :, 1].view((-1, 1)), tgt_uppers.unsqueeze(-1), p=1)
        # print('cost_lower.shape: {}'.format(cost_lower.shape))
        # print('cost_upper.shape: {}'.format(cost_upper.shape))

        # # Compute the poly cost
        out_polys = out_bbox[:, :, 4:4+4].view((-1,4))  # [BN, 4]
        out_2poly = out_bbox[:, :, 4+4:4+4+4].view((-1,4))
        # out_heights = out_bbox[:, :, -2].view((-1))
        # out_pitches = out_bbox[:, :, -1].view((-1))

        # filter out non-lane
        valid_lane = targets[:, :, 0].flatten(0, 1)   # B * numlane (5)
        tgt_flags = torch.cat([tgt for tgt in targets_flag])
        valid_xs = tgt_flags > 0
        weights = (torch.sum(valid_xs, dtype=torch.float32) / torch.sum(valid_xs, dim=1, dtype=torch.float32)) ** 0.5
        weights[valid_lane == 0] = 0
        weights = weights / torch.max(weights)

        # TODO gflat fitting
        tgt_gflatpoints = torch.cat([tgt[:, 2:] for tgt in gflat_targets])
        tgt_gflatxs = tgt_gflatpoints[:, :self.seq_len]  # [BT, L]
        tgt_gflatys = tgt_gflatpoints[:, self.seq_len:self.seq_len*2]
        tgt_gflatzs = tgt_gflatpoints[:, self.seq_len*2:]
        tgt_gflatys = tgt_gflatys.repeat(out_polys.shape[0], 1, 1)  # [BT*BN, L, 1]
        tgt_gflatys = tgt_gflatys.transpose(0, 2)   # [1, L, BT*BN]
        tgt_gflatys = tgt_gflatys.transpose(0, 1)   # [L, 1, BT*BN]

        # Calculate the predicted xs
        out_gflatxs = out_polys[:, 0] * tgt_gflatys ** 3 + \
                      out_polys[:, 1] * tgt_gflatys ** 2 + \
                      out_polys[:, 2] * tgt_gflatys + \
                      out_polys[:, 3]
        tgt_gflatxs = tgt_gflatxs.repeat(out_polys.shape[0], 1, 1)
        tgt_gflatxs = tgt_gflatxs.transpose(0, 2)
        tgt_gflatxs = tgt_gflatxs.transpose(0, 1)

        # Calculate the predicted ys
        out_gflatzs = out_2poly[:, 0] * tgt_gflatys ** 3 + \
                      out_2poly[:, 1] * tgt_gflatys ** 2 + \
                      out_2poly[:, 2] * tgt_gflatys + \
                      out_2poly[:, 3]
        tgt_gflatzs = tgt_gflatzs.repeat(out_2poly.shape[0], 1, 1)
        tgt_gflatzs = tgt_gflatzs.transpose(0, 2)
        tgt_gflatzs = tgt_gflatzs.transpose(0, 1)

        # # TODO 2d fitting
        # tgt_points = torch.cat([tgt[:, 3:] for tgt in targets])
        # tgt_xs = tgt_points[:, :tgt_points.shape[1] // 2]
        # # print('tgt_xs.shape: {}'.format(tgt_xs.shape))
        # tgt_ys = tgt_points[:, tgt_points.shape[1] // 2:]
        # # print('tgt_ys.shape: {}'.format(tgt_ys.shape))
        # tgt_ys = tgt_ys.repeat(out_polys.shape[0], 1, 1)
        # tgt_ys = tgt_ys.transpose(0, 2)
        # tgt_ys = tgt_ys.transpose(0, 1)
        # # print('tgt_ys.shape: {}'.format(tgt_ys.shape))
        #
        # # Calculate the predicted xs
        # # out_xs = out_polys[:, 0] * tgt_ys**3 + out_polys[:, 1] * tgt_ys**2 + out_polys[:, 2] * tgt_ys + out_polys[:, 3]
        # # out_xs = out_polys[:, 0] / (tgt_ys - out_polys[:, 1]) + out_polys[:, 2] + out_polys[:, 3] * tgt_ys - out_polys[:, 4]
        # # out_xs = out_polys[:, 0] / (tgt_ys - out_polys[:, 1]) ** 2 + out_polys[:, 2] / (tgt_ys - out_polys[:, 1]) + out_polys[:, 3] + out_polys[:, 4] * tgt_ys - out_polys[:, 5]
        # # out_xs = (out_polys[:, 0] * out_heights ** 2 / (1 / 2015) ** 3) * torch.cos(out_pitches) ** 2 / (tgt_ys - 2015 * torch.sin(out_pitches)) ** 2 + \
        # #          (out_polys[:, 1] * out_heights / (1 / 2015) ** 2) * torch.cos(out_pitches) / (tgt_ys - 2015 * torch.sin(out_pitches)) + \
        # #          out_polys[:, 2] / (1 / 2015) + \
        # #          out_polys[:, 3] * tgt_ys / (out_heights * torch.cos(out_pitches)) - \
        # #          out_polys[:, 3] * 2015 * torch.tan(out_pitches) / out_heights
        # tgt_xs = tgt_xs.repeat(out_polys.shape[0], 1, 1)
        # tgt_xs = tgt_xs.transpose(0, 2)
        # tgt_xs = tgt_xs.transpose(0, 1)
        # # print('tgt_xs.shape: {}'.format(tgt_xs.shape))

        # cost_polys = torch.stack([torch.sum(torch.abs(tgt_x[valid_x] - out_x[valid_x]) +
        #                                     torch.abs(tgt_gflatx[valid_x] - out_gflatx[valid_x]), dim=0)
        #                           for tgt_x, out_x, valid_x, tgt_gflatx, out_gflatx in
        #                           zip(tgt_xs, out_xs, valid_xs, tgt_gflatxs, out_gflatxs)], dim=-1)
        cost_polys = torch.stack([torch.sum(torch.abs(tgt_gflatx[valid_x] - out_gflatx[valid_x]), dim=0)
                                  for valid_x, tgt_gflatx, out_gflatx in zip(valid_xs, tgt_gflatxs, out_gflatxs)], dim=-1)

        cost_polys = cost_polys + torch.stack([torch.sum(torch.abs(tgt_gflatz[valid_x] - out_gflatz[valid_x]), dim=0)
                                  for valid_x, tgt_gflatz, out_gflatz in zip(valid_xs, tgt_gflatzs, out_gflatzs)], dim=-1)

        cost_polys = cost_polys * weights


        # # Final cost matrix
        C = self.cost_class * cost_class + self.poly_weight * cost_polys + \
            self.lower_weight * cost_gflatlower + self.upper_weight * cost_gflatupper

        C = C.view(bs, num_queries, -1).cpu()

        sizes = [tgt.shape[0] for tgt in targets]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_dist: float = 3, anchor_len: int = 10, start_idx: int = 5):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
        """
        super(HungarianMatcher, self).__init__()
        self.cost_class = cost_class
        self.cost_dist = cost_dist
        self.anchor_len = anchor_len
        self.start_idx = start_idx

    @torch.no_grad()
    def forward(self, proposals, targets, use_sigmoid=False, focal_alpha=0.25, focal_gamma=2.0):
        """ Performs the matching

        Params:
            outputs: [num_anchors, 5+anchor_len*3+num_classes]
            targets: [num_anchors, 5+anchor_len*3+num_classes]
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        num_proposals = proposals.shape[0]
        valid_targets = targets[targets[:, 1] > 0]
        assert valid_targets.shape[0] == targets.shape[0]
        num_targets = targets.shape[0]
        target_ids  = targets[:, 1].long()   # [num_pred, num_classes] # 0 ~ b*nq

        if use_sigmoid:
            pred = (proposals[:, self.start_idx + 3 * self.anchor_len:]).sigmoid()
            # cost_class = pred[:, target_ids-1] # debug
            neg_cost_class = (1 - focal_alpha) * (pred ** focal_gamma) * (-(1 - pred + 1e-8).log())
            pos_cost_class = focal_alpha * ((1 - pred) ** focal_gamma) * (-(pred + 1e-8).log())
            cost_class = pos_cost_class[:, target_ids-1] - neg_cost_class[:, target_ids-1]
        else:
            # We flatten to compute the cost matrices in a batch
            proposal_prob = proposals[:, self.start_idx + 3 * self.anchor_len:].softmax(-1)  # 
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -proposal_prob[:, target_ids]  # [num_pred, num_target]

        proposal_regs = proposals[:, self.start_idx:self.start_idx + 2 * self.anchor_len]  # [Np, 30]
        target_regs = targets[:, self.start_idx:self.start_idx + 3 * self.anchor_len:]    # [Nt, 30]

        proposal_regs = torch.repeat_interleave(proposal_regs, num_targets, dim=0)  # [Np * Nt, 35], [a, b] -> [a, a, b, b]
        target_regs = torch.cat(num_proposals * [target_regs])   # [Nt * Np, 10, 35], [c, d] -> [c, d, c, d]
        cost_distances = Manhattan_dis(proposal_regs, target_regs, num_proposals, num_targets, anchor_len=self.anchor_len, start_idx=0)   # [Np, Nt]

        # cost_distances = cost_distances / (torch.max(cost_distances) + 1e-6)  # normalize the distance  # debug

        # # Final cost matrix
        C = self.cost_class * cost_class + self.cost_dist * cost_distances

        C = C.cpu()

        indices = linear_sum_assignment(C)
        return torch.as_tensor(indices[0], dtype=torch.int64), torch.as_tensor(indices[1], dtype=torch.int64)