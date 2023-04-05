# --------------------------------------------------------
# Source code for Anchor3DLane
# Copyright (c) 2023 TuSimple
# @Time    : 2023/04/05
# @Author  : Shaofei Huang
# nowherespyfly@gmail.com
# --------------------------------------------------------

from random import sample
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import pdb
import math

import mmcv
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from ..builder import LANENET2S
from .tools import homography_crop_resize
from ..lane_detector.utils import nms_3d
from .anchor_3dlane import Anchor3DLane

@LANENET2S.register_module()
class Anchor3DLaneMF(Anchor3DLane):

    def __init__(self, 
                 backbone,
                 prev_num = 1,
                 is_detach = False,
                 **kwargs):
        super(Anchor3DLaneMF, self).__init__(backbone, **kwargs)
        self.prev_num = prev_num
        self.is_detach = is_detach
        self.temp_fuse = nn.ModuleList()
        self.temp_fuse.append(nn.TransformerDecoderLayer(self.anchor_feat_channels, 2, 256, batch_first=True))
        for iter in range(self.iter_reg):
            self.temp_fuse.append(nn.TransformerDecoderLayer(self.anchor_feat_channels, 2, 256, batch_first=True))

    def feature_extractor(self, img, mask):
        output = self.backbone(img)
        if self.neck is not None:
            output = self.neck(output)
            feat = output[0]
        else:
            feat = output[-1]
        feat = self.input_proj(feat)

        mask_interp = F.interpolate(mask[:, 0, :, :][None], size=feat.shape[-2:]).to(torch.bool)[0]  # [B, h, w]
        mask_interp = mask_interp.repeat(self.prev_num + 1, 1, 1)
        pos = self.position_embedding(feat, mask_interp)   # [B, 32, h, w]
        
        bs, c, h, w = feat.shape
        feat = feat.flatten(2).permute(2, 0, 1)  # [hw, bs, c]
        pos = pos.flatten(2).permute(2, 0, 1)     # [hw, bs, 32]
        mask_interp = mask_interp.flatten(1)      # [hw, bs]
        trans_feat = self.transformer_layer(feat, src_key_padding_mask=mask_interp, pos=pos)  
        trans_feat = trans_feat.permute(1, 2, 0).reshape(bs, c, h, w)  # [bs, c, h, w]
        return trans_feat

    @force_fp32()
    def get_proposals(self, project_matrixes, prev_project_matrixes, anchor_feat, iter_idx=0, proposals_prev=None):
        batch_size = project_matrixes.shape[0]
        batch_anchor_features_prev = []
        if proposals_prev is None:
            batch_anchor_features_cur, _ = self.cut_anchor_features(anchor_feat[0], project_matrixes, self.xs, self.ys, self.zs)   # [B, C, N, l]
            for i in range(self.prev_num):
                prev_anchor_features, _ = self.cut_anchor_features(anchor_feat[i+1], prev_project_matrixes[i], self.xs, self.ys, self.zs)
                batch_anchor_features_prev.append(prev_anchor_features)  # [B, C, N, l]
        else:
            sampled_anchor = torch.zeros(batch_size, len(self.anchors), 5 + self.anchor_feat_len * 3, device = project_matrixes.device)
            sampled_anchor[:, :, 5:5+self.anchor_feat_len] = proposals_prev[:, :, 5:5+self.anchor_len][:, :, self.feat_sample_index]
            sampled_anchor[:, :, 5+self.anchor_feat_len:5+self.anchor_feat_len*2] = proposals_prev[:, :, 5+self.anchor_len:5+self.anchor_len*2][:, :, self.feat_sample_index]
            xs, ys, zs = self.compute_anchor_cut_indices(sampled_anchor, self.feat_y_steps)
            batch_anchor_features_cur, _ = self.cut_anchor_features(anchor_feat[0], project_matrixes, xs, ys, zs)   # [B, C, N, l]
            for i in range(self.prev_num):
                prev_anchor_features, _ = self.cut_anchor_features(anchor_feat[i+1], prev_project_matrixes[i], xs, ys, zs)
                batch_anchor_features_prev.append(prev_anchor_features)  # [B, C, N, l]

        batch_anchor_features_cur = batch_anchor_features_cur.transpose(1, 2).transpose(2, 3).flatten(0, 1)  # [BN, l, C]
        for i in range(self.prev_num):
            batch_anchor_features_prev[i] = batch_anchor_features_prev[i].transpose(1, 2).transpose(2, 3).flatten(0, 1)  # [BN, l, C]

        if self.prev_num == 1:
            batch_anchor_features_prev = batch_anchor_features_prev[0]
        else:
            batch_anchor_features_prev = torch.cat(batch_anchor_features_prev, dim=1)  # [BN, pl, C]

        batch_anchor_features_fuse = self.temp_fuse[iter_idx](batch_anchor_features_cur, batch_anchor_features_prev)  # [BN, l, C]
        batch_anchor_features_fuse = batch_anchor_features_fuse.flatten(1, 2)  # [BN, lC]

        # Predict
        cls_logits = self.cls_layer[iter_idx](batch_anchor_features_fuse)   # [B * N, C]
        cls_logits = cls_logits.reshape(batch_size, -1, cls_logits.shape[1])   # [B, N, C]
        reg_x = self.reg_x_layer[iter_idx](batch_anchor_features_fuse)    # [B * N, l]
        reg_x = reg_x.reshape(batch_size, -1, reg_x.shape[1])   # [B, N, l]
        reg_z = self.reg_z_layer[iter_idx](batch_anchor_features_fuse)    # [B * N, l]
        reg_z = reg_z.reshape(batch_size, -1, reg_z.shape[1])   # [B, N, l]
        reg_vis = self.reg_vis_layer[iter_idx](batch_anchor_features_fuse)  # [B * N, l]
        reg_vis = torch.sigmoid(reg_vis)
        reg_vis = reg_vis.reshape(batch_size, -1, reg_vis.shape[1])   # [B, N, l]
        
        # Add offsets to anchors
        # [B, N, l]
        reg_proposals = torch.zeros(batch_size, len(self.anchors), 5 + self.anchor_len * 3 + self.num_category, device = project_matrixes.device)
        if proposals_prev is None:
            reg_proposals[:, :, :5+self.anchor_len*3] = reg_proposals[:, :, :5+self.anchor_len*3] + self.anchors
        else:
            reg_proposals[:, :, :5+self.anchor_len*3] = reg_proposals[:, :, :5+self.anchor_len*3] + proposals_prev[:, :, :5+self.anchor_len*3]
        
        reg_proposals[:, :, 5:5+self.anchor_len] += reg_x
        reg_proposals[:, :, 5+self.anchor_len:5+self.anchor_len*2] += reg_z
        reg_proposals[:, :, 5+self.anchor_len*2:5+self.anchor_len*3] = reg_vis
        reg_proposals[:, :, 5+self.anchor_len*3:5+self.anchor_len*3+self.num_category] = cls_logits   # [B, N, C]
        return reg_proposals

    def encoder_decoder(self, img, mask, gt_project_matrix, prev_poses=None, **kwargs):
        # img: [B, 3, inp_h, inp_w, Np+1]; mask: [B, 1, 36, 480]
        # prev_poses: [B, Np, 3, 4]
        batch_size = img.shape[0] 
        img = torch.cat(img.split(1, dim=4), dim=0).squeeze(4)  # [2B, 3, h, w]
        trans_feat = self.feature_extractor(img, mask)  # [B(Np+1), C, h, w]
        
        # anchor
        anchor_feat = self.anchor_projection(trans_feat)
        anchor_feat = anchor_feat.split(batch_size, dim=0)  # [B(Np+1), C, h, w]
        project_matrixes = self.obtain_projection_matrix(gt_project_matrix, feat_size=self.feat_size)
        project_matrixes = torch.stack(project_matrixes, dim=0)   # [B, 3, 4]

        prev_project_matrixes = []
        for i in range(self.prev_num):
            prev_matrix = self.obtain_projection_matrix(prev_poses[:, i, :, :], feat_size=self.feat_size)
            prev_matrix = torch.stack(prev_matrix, dim=0)   # [B, 3, 4]
            prev_project_matrixes.append(prev_matrix)

        reg_proposals_all = []
        anchors_all = []
        reg_proposals_s1 = self.get_proposals(project_matrixes, prev_project_matrixes, anchor_feat, 0)
        reg_proposals_all.append(reg_proposals_s1)
        anchors_all.append(torch.stack([self.anchors] * batch_size, dim=0))

        for iter in range(self.iter_reg):
            proposals_prev = reg_proposals_all[iter]
            reg_proposals_all.append(self.get_proposals(project_matrixes, prev_project_matrixes, anchor_feat, iter+1, proposals_prev))
            anchors_all.append(proposals_prev[:, :, :5+self.anchor_len*3])

        output = {'reg_proposals':reg_proposals_all[-1], 'anchors':anchors_all[-1]}
        if self.iter_reg > 0:
            output_aux = {'reg_proposals':reg_proposals_all[:-1], 'anchors':anchors_all[:-1]}
            return output, output_aux
        return output, None