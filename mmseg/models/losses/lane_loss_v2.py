# --------------------------------------------------------
# Source code for Anchor3DLane
# Copyright (c) 2025 TuSimple
# @Time    : 2025/05/07
# @Author  : Shaofei Huang
# nowherespyfly@gmail.com
# --------------------------------------------------------

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES, build_assigner
from .kornia_focal import FocalLoss
from .utils import get_class_weight, weight_reduce_loss
from .matcher import HungarianMatcher
from .focal_loss import FocalLossSigmoid

@LOSSES.register_module()
class LaneLossV2(nn.Module):
    def __init__(self,
                 focal_alpha=0.25,
                 focal_gamma=2.,
                 anchor_len=10,
                 gt_anchor_len=200,
                 anchor_steps=[],
                 weighted_ce=False,
                 use_sigmoid=False,
                 loss_weights=None,
                 anchor_assign=False,
                 delta = 0.2,
                 ds = 10,
                 assign_cfg=None):
        super(LaneLossV2, self).__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.anchor_len = anchor_len
        self.anchor_steps = np.array(anchor_steps) - 1
        self.gt_anchor_len = gt_anchor_len
        self.use_sigmoid = use_sigmoid

        self.weighted_ce = weighted_ce
        self.loss_weights = loss_weights
        self.anchor_assign = anchor_assign
        self.lane_prior = 'reg_losses_prior' in loss_weights.keys()
        self.consist = ('consist_losses' in loss_weights.keys())
        self.fp16_enabled = False
        self.delta = delta
        self.ds = ds
        self.assigner = HungarianMatcher(anchor_len=self.anchor_len, **assign_cfg)
        
    def forward(self, proposals_list, targets):
        if self.use_sigmoid:
            focal_loss = FocalLossSigmoid(alpha=self.focal_alpha, gamma=self.focal_gamma, reduction='none')
        else:
            focal_loss = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
        smooth_l1_loss = nn.SmoothL1Loss(reduction='none')
        cls_losses = 0
        reg_losses_x = 0
        reg_losses_z = 0
        reg_losses_vis = 0
        if self.lane_prior:
            reg_losses_prior = 0
        if self.consist:
            consist_losses = 0 
        valid_imgs = len(targets)
        total_positives = 0
        total_negatives = 0
        for idx in range(len(proposals_list)):
            proposals = proposals_list[idx][0]
            num_clses = proposals.shape[1] - 5 - self.anchor_len * 3
            anchors = proposals_list[idx][1]
            target = targets[idx]
            # Filter lanes that do not exist (confidence == 0)
            target = target[target[:, 1] > 0]   # [N, 605]
            if len(target) == 0:
                # If there are no targets, all proposals have to be negatives (i.e., 0 confidence)
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, 5+self.anchor_len*3:]
                cls_losses += focal_loss(cls_pred, cls_target).sum()
                reg_losses_x += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                reg_losses_z += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                reg_losses_vis += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                if self.lane_prior:
                    reg_losses_prior += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                if self.consist:
                    consist_losses += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                continue
            # Gradients are also not necessary for the positive & negative matching
            x_indices = torch.tensor(self.anchor_steps).to(torch.long).to(target.device) + 5
            z_indices = x_indices + self.gt_anchor_len
            vis_indices = x_indices + self.gt_anchor_len * 2
            x_target = target.index_select(1, x_indices)
            z_target = target.index_select(1, z_indices)
            vis_target = target.index_select(1, vis_indices)   # [N, 10]
            target = torch.cat((target[:, :5], x_target, z_target, vis_target), dim=1)   # [N, 35]
            with torch.no_grad():
                if self.anchor_assign:
                    anchor_assign = torch.cat([anchors, proposals[:, 65:]], 1)
                    indices_src, indices_tgt = self.assigner(anchor_assign, target, use_sigmoid=self.use_sigmoid)
                else:
                    indices_src, indices_tgt = self.assigner(proposals, target, use_sigmoid=self.use_sigmoid)

            positives = proposals[indices_src]
            num_positives = len(positives)
            total_positives += num_positives
            negatives_mask = torch.ones(proposals.shape[0], dtype=torch.bool, device=proposals.device)
            negatives_mask[indices_src] = False
            negatives = proposals[negatives_mask]
            num_negatives = len(negatives)
            total_negatives += num_negatives

            # Handle edge case of no positives found
            if num_positives == 0:
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_losses += focal_loss(cls_pred, cls_target).sum()
                reg_losses_x += smooth_l1_loss(cls_pred, cls_pred).sum() * 0  # avoid dividing zeros
                reg_losses_z += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                reg_losses_vis += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                if self.lane_prior:
                    reg_losses_prior += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                if self.consist:
                    consist_losses += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                continue

            # Get classification targets
            all_proposals = torch.cat([positives, negatives], 0)
            cls_target = proposals.new_zeros(num_positives + num_negatives).long()
            cls_target[:num_positives] = target[indices_tgt][:, 1]
            cls_pred = all_proposals[:, 5+self.anchor_len*3:]  # [N, C]

            # Regression targets
            x_pred = positives[:, 5:5+self.anchor_len]   # [N, l]
            z_pred = positives[:, 5+self.anchor_len:5+self.anchor_len*2]   # [N, l]
            vis_pred = positives[:, 5+self.anchor_len*2:5+self.anchor_len*3]  # [N, l]
            prior_pred = positives[:, 2:5]

            with torch.no_grad():
                target = target[indices_tgt]
                x_target = target[:, 5:5+self.anchor_len]
                z_target = target[:, 5+self.anchor_len:5+self.anchor_len*2]
                vis_target = target[:, 5+self.anchor_len*2:5+self.anchor_len*3]
                prior_target = target[:, 2:5]
                valid_points = vis_target.sum()

            # Loss calc
            reg_loss_x = smooth_l1_loss(x_pred, x_target)
            reg_loss_x = reg_loss_x * vis_target  #  * scores # [N, l]
            reg_losses_x += reg_loss_x.sum() / valid_points
            reg_loss_z = smooth_l1_loss(z_pred, z_target)
            reg_loss_z = reg_loss_z * vis_target # * scores
            reg_losses_z += reg_loss_z.sum() / valid_points
            reg_loss_vis = smooth_l1_loss(vis_pred, vis_target)
            reg_losses_vis += reg_loss_vis.mean()
            cls_loss = focal_loss(cls_pred, cls_target)
            if self.lane_prior:
                prior_loss = smooth_l1_loss(prior_pred, prior_target).mean()
                reg_losses_prior += prior_loss
            if self.consist:
                xr = x_pred.unsqueeze(1)  # [N, 1, l]
                xl = x_pred.unsqueeze(0)  # [1, N, l]
                cos = x_pred.new_zeros(x_pred.shape[0], self.anchor_len - 1)
                cos = 5 / ((x_pred[..., 1:] - x_pred[..., :-1]) ** 2 + 25) ** 0.5
                cos = torch.cat([cos[..., 0:1].clone(), cos], -1)
                distance = (xr - xl) * cos.detach()  # [N, N, l]
                distance_mean = distance.mean(-1, keepdims=True)  # [N, N, 1]
                distance_delta = (distance - distance_mean).abs()[..., self.ds:]  # [N, N, l']
                distance_mask = distance_delta < self.delta
                consist_loss = (distance_delta * distance_mask).sum(-1) / (distance_mask.sum(-1) + 1e-6)
                consist_loss = consist_loss.triu(diagonal=1)
                consist_losses += consist_loss.sum() / (num_positives * (num_positives - 1) / 2 + 1e-6)
            
            if self.use_sigmoid:
                cls_losses += cls_loss.sum() / num_positives / num_clses
            else:
                cls_losses += cls_loss.sum() / num_positives

        # Batch mean
        cls_losses = cls_losses / valid_imgs
        reg_losses_x = reg_losses_x / valid_imgs
        reg_losses_z = reg_losses_z / valid_imgs
        reg_losses_vis = reg_losses_vis / valid_imgs

        losses = {'cls_loss': cls_losses, 'reg_losses_x': reg_losses_x, 'reg_losses_z': reg_losses_z, 'reg_losses_vis': reg_losses_vis}

        if self.lane_prior:
            reg_losses_prior = reg_losses_prior / valid_imgs
            losses['reg_losses_prior'] = reg_losses_prior

        if self.consist:
            consist_losses = consist_losses / valid_imgs
            losses['consist_losses'] = consist_losses

        for k in losses.keys():
            losses[k] = losses[k] * self.loss_weights[k]

        bs = len(proposals_list)
        return {'losses':losses, 'batch_positives': total_positives / bs, 'batch_negatives': total_negatives / bs}