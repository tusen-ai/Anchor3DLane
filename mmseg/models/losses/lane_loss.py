# --------------------------------------------------------
# Source code for Anchor3DLane
# Copyright (c) 2023 TuSimple
# @Time    : 2023/04/05
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

@LOSSES.register_module()
class LaneLoss(nn.Module):
    def __init__(self,
                 focal_alpha=0.25,
                 focal_gamma=2.,
                 anchor_len=10,
                 gt_anchor_len=200,
                 anchor_steps=[],
                 weighted_ce=False,
                 use_sigmoid=False,
                 loss_weights=None,
                 anchor_assign=True,
                 assign_cfg=None):
        super(LaneLoss, self).__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.anchor_len = anchor_len
        self.anchor_steps = np.array(anchor_steps) - 1
        self.gt_anchor_len = gt_anchor_len
        self.use_sigmoid = use_sigmoid

        self.weighted_ce = weighted_ce
        self.loss_weights = loss_weights
        self.anchor_assign = anchor_assign
        self.fp16_enabled = False
        self.assigner = build_assigner(assign_cfg)
        
    def forward(self, proposals_list, targets):
        focal_loss = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
        smooth_l1_loss = nn.SmoothL1Loss(reduction='none')
        cls_losses = 0
        reg_losses_x = 0
        reg_losses_z = 0
        reg_losses_vis = 0
        valid_imgs = len(targets)
        total_positives = 0
        total_negatives = 0
        for idx, ((proposals, anchors), target) in enumerate(zip(proposals_list, targets)):
            # Filter lanes that do not exist (confidence == 0)
            num_clses = proposals.shape[1] - 5 - self.anchor_len * 3
            target = target[target[:, 1] > 0]   # [N, 605]
            if len(target) == 0:
                # If there are no targets, all proposals have to be negatives (i.e., 0 confidence)
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, 5+self.anchor_len*3:]
                cls_losses += focal_loss(cls_pred, cls_target).sum()
                reg_losses_x += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                reg_losses_z += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                reg_losses_vis += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
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
                    positives_mask, negatives_mask, target_positives_indices = self.assigner.match_proposals_with_targets(
                        anchors, target)
                else:
                    positives_mask, negatives_mask, target_positives_indices = self.assigner.match_proposals_with_targets(
                        proposals[:, :5+self.anchor_len*3], target)

            positives = proposals[positives_mask]
            num_positives = len(positives)
            total_positives += num_positives
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
                continue

            # Get classification targets
            all_proposals = torch.cat([positives, negatives], 0)
            cls_target = proposals.new_zeros(num_positives + num_negatives).long()
            cls_target[:num_positives] = target[target_positives_indices][:, 1]
            cls_pred = all_proposals[:, 5+self.anchor_len*3:]  # [N, C]

            # Regression targets
            x_pred = positives[:, 5:5+self.anchor_len]   # [N, l]
            z_pred = positives[:, 5+self.anchor_len:5+self.anchor_len*2]   # [N, l]
            vis_pred = positives[:, 5+self.anchor_len*2:5+self.anchor_len*3]  # [N, l]
            with torch.no_grad():
                target = target[target_positives_indices]
                x_target = target[:, 5:5+self.anchor_len]
                z_target = target[:, 5+self.anchor_len:5+self.anchor_len*2]
                vis_target = target[:, 5+self.anchor_len*2:5+self.anchor_len*3]
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

        for k in losses.keys():
            losses[k] = losses[k] * self.loss_weights[k]

        bs = len(proposals_list)
        return {'losses':losses, 'batch_positives': total_positives / bs, 'batch_negatives': total_negatives / bs}