# --------------------------------------------------------
# Source code for Anchor3DLane++
# Copyright (c) 2025 TuSimple
# @Time    : 2025/05/08
# @Author  : Shaofei Huang
# nowherespyfly@gmail.com
# --------------------------------------------------------


import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from random import sample

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16, force_fp32

from ..builder import LANENET2S, build_backbone, build_neck, build_reader
from .anchor_3dlane_pp import Anchor3DLanePP
from .utils import DecodeLayer


@LANENET2S.register_module()
class Anchor3DLanePPFuse(Anchor3DLanePP):

    def __init__(self, 
                 backbone,
                 reader,
                 scatter,
                 lidar_neck = None,
                 lidar_y_steps = [5,  10,  15,  20,  30,  40,  50,  60,  70,  75],
                 lidar_dims = [64, 128, 384],
                 sample_feat_lidar = ['conv2', 'conv3'],
                 voxel_size = None,
                 pc_range = None,
                 grid_shape = [468, 468, 1],
                 lidar_feat_size = [],
                 use_voxel = False,
                 **kwargs):
        super(Anchor3DLanePPFuse, self).__init__(backbone, **kwargs)

        self.reader = build_reader(reader)
        self.scatter = build_backbone(scatter)
        if lidar_neck is not None:
            self.lidar_neck = build_neck(lidar_neck)
        else:
            self.lidar_neck = None

         # lidar settings
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.lidar_dims = lidar_dims
        self.use_voxel = use_voxel
        self.grid_shape = np.array(grid_shape)
        self.lidar_y_steps = np.array(lidar_y_steps, dtype=np.float32)
        self.lidar_anchor_len = len(lidar_y_steps)
        self.sample_feat_lidar = sample_feat_lidar
        self.lidar_sample_index = torch.from_numpy(np.isin(self.y_steps, self.lidar_y_steps))
        self.lidar_feat_size = lidar_feat_size

        self.anchor_projection_lidar = nn.ModuleDict()
        self.model_fuse = nn.ModuleDict()
        for idx, lidar_dim in enumerate(self.lidar_dims):
            if self.use_voxel:
                self.anchor_projection_lidar[f'layer_{idx}'] = nn.Conv3d(lidar_dim, self.anchor_feat_channels, kernel_size=1)
            else:
                self.anchor_projection_lidar[f'layer_{idx}'] = nn.Conv2d(lidar_dim, self.anchor_feat_channels, kernel_size=1)
            self.model_fuse[f'layer_{idx}'] = nn.ModuleList([DecodeLayer(self.proj_channel * (self.lidar_anchor_len + self.anchor_feat_len), \
                                                                         512, self.proj_channel * self.anchor_feat_len) for i in range(self.iter_reg)])
            
    def point_feature_extractor(self, voxels, num_points, coordinates, batch_size, **kwargs):
        input_features = []
        for idx in range(batch_size):
            input_feature = self.reader(voxels[idx], num_points[idx], coordinates[idx])
            input_features.append(input_feature)
        if self.use_voxel:
            for idx in range(batch_size):
                batch_idx = coordinates[idx].new_empty([coordinates[idx].shape[0], 1])  # [N, 1]
                batch_idx.fill_(idx)
                coordinates[idx] = torch.cat([batch_idx, coordinates[idx]], dim=1)  # [N, 4]
            input_features = torch.cat(input_features, dim=0)  # [N1+N2+..., 5]
            coordinates = torch.cat(coordinates, dim=0)  # [N1+N2+..., 4]
        outputs = self.scatter(
            input_features, coordinates, batch_size, self.grid_shape)  # [B, 64, 468, 468]
        if isinstance(outputs, tuple):
            feat = outputs[0]
        else:
            feat = outputs
        if self.lidar_neck is not None:
            neck_feat = self.lidar_neck(feat)  # [B, 384, 468, 468]
            if self.use_voxel:
                feats = [feat,]
                for layer in self.sample_feat_lidar:
                    feats.append(neck_feat[layer])
                return feats
            else:
                return neck_feat

        feats = []
        for layer in self.sample_feat_lidar:  # ['conv2', 'conv3']
            feats.append(outputs[1][layer].dense())
        feats.append(feat)
        return feats
    
    def cut_anchor_bev_features(self, features, xs, ys, anchor_len):
        batch_size = features.shape[0]
        if len(xs.shape) == 1:
            batch_xs = xs.repeat(batch_size, 1)   # [B, Nl]
            batch_ys = ys.repeat(batch_size, 1)   # [B, Nl]
        else:
            batch_xs = xs
            batch_ys = ys

        batch_xs = (batch_xs - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        batch_ys = (batch_ys - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        batch_xs = (batch_xs - 0.5) * 2
        batch_ys = (batch_ys - 0.5) * 2

        batch_grid = torch.stack([batch_xs, batch_ys], dim=-1) 
        batch_grid = batch_grid.reshape(batch_size, -1, anchor_len, 2)  # [B, N, l, 2]
        batch_anchor_features = F.grid_sample(features, batch_grid, padding_mode='zeros')   # [B, C, N, l]

        valid_mask = (batch_xs > -1) & (batch_xs < 1) & (batch_ys > -1) & (batch_ys < 1)

        return batch_anchor_features, valid_mask.reshape(batch_size, -1, anchor_len)

    def cut_anchor_xyz_features(self, features, xs, ys, zs, anchor_len):
        batch_size = features.shape[0]  # [B, C, N, H, W]
        if len(xs.shape) == 1:
            batch_xs = xs.repeat(batch_size, 1)   # [B, Nl]
            batch_ys = ys.repeat(batch_size, 1)   # [B, Nl]
            batch_zs = zs.repeat(batch_size, 1)   # [B, Nl]
        else:
            batch_xs = xs
            batch_ys = ys
            batch_zs = zs
        
        batch_xs = (batch_xs - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        batch_ys = (batch_ys - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        batch_zs = (batch_zs - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
        batch_xs = (batch_xs - 0.5) * 2
        batch_ys = (batch_ys - 0.5) * 2
        batch_zs = (batch_zs - 0.5) * 2

        batch_grid = torch.stack([batch_xs, batch_ys, batch_zs], dim=-1)  #
        batch_grid = batch_grid.reshape(batch_size, 1, -1, anchor_len, 3)  # [B, 1, N, l, 3]
        batch_anchor_features = F.grid_sample(features, batch_grid, padding_mode='zeros')   # [B, C, 1, N, l]
        batch_anchor_features = batch_anchor_features.squeeze(2)  # [B, C, N, l]

        valid_mask = (batch_xs > -1) & (batch_xs < 1) & (batch_ys > -1) & (batch_ys < 1) & (batch_zs > -1) & (batch_zs < 1)

        return batch_anchor_features, valid_mask.reshape(batch_size, -1, anchor_len)

    def encode_position(self, batch_anchor_features, xs, ys, zs, batch_size, anchor_len, feat_idx=None, iter_idx=None):
        xs = xs / self.x_norm # [B, N*L]
        ys = ys / self.y_norm
        zs = zs / self.z_norm
        xyz = torch.stack([xs, ys, zs], -1)  # [B, NL, 3]
        batch_pos_features = self.position_encoder(xyz)  # [B, NL, C]
        batch_pos_features = batch_pos_features.transpose(1, 2).reshape(batch_size, self.anchor_feat_channels, self.anchor_num, anchor_len)
        if self.with_pos == 'add':
            batch_anchor_features = batch_anchor_features + batch_pos_features
        elif self.with_pos == 'pcat':
            batch_anchor_features = torch.cat([batch_anchor_features, batch_pos_features], 1)  # [B, C, N, l]
            batch_anchor_features = batch_anchor_features.permute(0, 2, 3, 1) # [B, N, l, C]
            batch_anchor_features = self.fuse_pos[f'layer_{feat_idx}'][iter_idx](batch_anchor_features)
            batch_anchor_features = batch_anchor_features.permute(0, 3, 1, 2)  # [B, C, N, l]
        else:
            batch_anchor_features = torch.cat([batch_anchor_features, batch_pos_features], 1)
        return batch_anchor_features
    
    @force_fp32()
    def get_proposals(self, project_matrixes, anchor_feat_i, anchor_feat_l, feat_idx, proposals_prev, feat_size, iter_idx, reg_prior=False):
        batch_size = project_matrixes.shape[0]
        xs_i, ys_i, zs_i = self.compute_anchor_cut_indices(proposals_prev, self.feat_y_steps)
        batch_anchor_features_i, _ = self.cut_anchor_features(anchor_feat_i, project_matrixes, xs_i, ys_i, zs_i, self.anchor_feat_len, feat_size)   # [B, C, N, l]
        # lidar
        proposals_prev_l = torch.zeros(batch_size, self.anchor_num, 5 + self.lidar_anchor_len * 3, device = anchor_feat_i.device)
        proposals_prev_l[:, :, 5:5+self.lidar_anchor_len] = proposals_prev[:, :, 5:5+self.anchor_len][:, :, self.lidar_sample_index]
        proposals_prev_l[:, :, 5+self.lidar_anchor_len:5+self.lidar_anchor_len*2] = proposals_prev[:, :, 5+self.anchor_len:5+self.anchor_len*2][:, :, self.lidar_sample_index]
        xs_l, ys_l, zs_l = self.compute_anchor_cut_indices(proposals_prev_l, self.lidar_y_steps)
        if self.use_voxel:
            batch_anchor_features_l, _ = self.cut_anchor_xyz_features(anchor_feat_l, xs_l, ys_l, zs_l, self.lidar_anchor_len)   # [B, C, N, l]
        else:
            batch_anchor_features_l, _ = self.cut_anchor_bev_features(anchor_feat_l, xs_l, ys_l, self.lidar_anchor_len)   # [B, C, N, l]

        if self.with_pos != 'none':
            batch_anchor_features_i = self.encode_position(batch_anchor_features_i, xs_i, ys_i, zs_i, batch_size, self.anchor_feat_len)
            batch_anchor_features_l = self.encode_position(batch_anchor_features_l, xs_l, ys_l, zs_l, batch_size, self.lidar_anchor_len)
        
        # model fuse
        batch_anchor_features_i = batch_anchor_features_i.permute(0, 2, 3, 1)  # [B, N, l, C]
        batch_anchor_features_i = batch_anchor_features_i.flatten(2, 3)  # [B, N, lC]    
        batch_anchor_features_l = batch_anchor_features_l.permute(0, 2, 3, 1)  # [B, N, l, C]
        batch_anchor_features_l = batch_anchor_features_l.flatten(2, 3)  # [B, N, lC]
        batch_anchor_features = self.model_fuse[f'layer_{feat_idx}'][iter_idx](torch.cat([batch_anchor_features_i, batch_anchor_features_l], -1))  # [B, N, lC]

        batch_anchor_features = self.dynamic_head[f'layer_{feat_idx}'][iter_idx](batch_anchor_features) # [B, N, Cl]
        batch_anchor_features = batch_anchor_features.flatten(0, 1)  # [B*N, C*l]

        # Predict
        cls_logits = self.cls_layer[f'layer_{feat_idx}'][iter_idx](batch_anchor_features)   # [B * N, C]
        cls_logits = cls_logits.reshape(batch_size, -1, cls_logits.shape[1])   # [B, N, C]
        reg_x = self.reg_x_layer[f'layer_{feat_idx}'][iter_idx](batch_anchor_features)    # [B * N, l]  
        reg_x = reg_x.reshape(batch_size, -1, reg_x.shape[1])   # [B, N, l]
        reg_z = self.reg_z_layer[f'layer_{feat_idx}'][iter_idx](batch_anchor_features)    # [B * N, l]
        reg_z = reg_z.reshape(batch_size, -1, reg_z.shape[1])   # [B, N, l]
        reg_vis = self.reg_vis_layer[f'layer_{feat_idx}'][iter_idx](batch_anchor_features)  # [B * N, l]
        reg_vis = torch.sigmoid(reg_vis)
        reg_vis = reg_vis.reshape(batch_size, -1, reg_vis.shape[1])   # [B, N, l]

        # lane prior regression
        if reg_prior:
            reg_lane_priors = self.reg_prior_layer[iter_idx](batch_anchor_features)   # [B * N, 3]
            reg_lane_priors = reg_lane_priors.reshape(batch_size, -1, 3)   # [B, N, C]
            reg_lane_priors = torch.tanh(reg_lane_priors) # yaws, pitch, xs, [-1, 1]
            
            # Add offsets to anchors
            # [B, N, l]
            lane_priors = reg_lane_priors + proposals_prev[..., 2:5]
            cur_anchors = self.anchor_generator.generate_anchors_batch(lane_priors[:, :, 2], lane_priors[:, :, 0], lane_priors[:, :, 1])
            reg_proposals = torch.zeros(batch_size, self.anchor_num, 5 + self.anchor_len * 3 + self.num_category, device = project_matrixes.device)
            reg_proposals[:, :, :5+self.anchor_len*3] = reg_proposals[:, :, :5+self.anchor_len*3] + cur_anchors[:, :, :5+self.anchor_len*3]
        else:
            # Add offsets to anchors
            # [B, N, l]
            reg_proposals = torch.zeros(batch_size, self.anchor_num, 5 + self.anchor_len * 3 + self.num_category, device = project_matrixes.device)
            reg_proposals[:, :, :5+self.anchor_len*3] = reg_proposals[:, :, :5+self.anchor_len*3] + proposals_prev[:, :, :5+self.anchor_len*3]
        
        reg_proposals[:, :, 5:5+self.anchor_len] += reg_x
        reg_proposals[:, :, 5+self.anchor_len:5+self.anchor_len*2] += reg_z
        reg_proposals[:, :, 5+self.anchor_len*2:5+self.anchor_len*3] = reg_vis
        reg_proposals[:, :, 5+self.anchor_len*3:5+self.anchor_len*3+self.num_category] = cls_logits   # [B, N, C]

        if reg_prior:
            return reg_proposals, cur_anchors
        else:
            return reg_proposals, None


    def encoder_decoder(self, img, mask, voxels, coordinates, num_points, num_voxels, gt_project_matrix, **kwargs):
        # img: [B, 3, inp_h, inp_w]; mask: [B, 1, 36, 480]
        batch_size = img.shape[0]
        anchor_feats_i = self.feature_extractor(img, mask) # [1, 2, 3]
        lidar_feats = self.point_feature_extractor(voxels, num_points, coordinates, batch_size)
        anchor_feats_l = []
        for feat_idx, dim in enumerate(self.lidar_dims):
            anchor_feats_l.append(self.anchor_projection_lidar[f'layer_{feat_idx}'](lidar_feats[feat_idx]))
            
        reg_proposals_all = []
        anchors_all = []

        for iter_idx in range(self.iter_reg):
            
            reg_proposals_layer = []
            anchors_layer = []
            for feat_idx, feat_size in enumerate(self.feat_sizes[::-1]):
                # [4, 3, 2]
                project_matrixes = self.obtain_projection_matrix(gt_project_matrix, feat_size)
                project_matrixes = torch.stack(project_matrixes, dim=0)   # [B, 3, 4]
                image_feat_i = self.feat_num - 1 - feat_idx
                if iter_idx == 0:
                    if feat_idx == 0:
                        yaw_weights, pitch_weights, x_weights = self.expert_layer(anchor_feats_i[self.expert_idx])
                        init_yaw = self.init_proposals_yaws.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, M, 1]
                        init_pitch = self.init_proposals_pitches.weight.unsqueeze(0).repeat(batch_size, 1, 1)
                        yaws = yaw_weights @ init_yaw  # [B, N, M] * [B, M, 1] -> [B, N, 1]
                        yaws = yaws.squeeze(-1)
                        pitches = pitch_weights @ init_pitch
                        pitches = pitches.squeeze(-1)
                        init_xs = self.init_proposals_xs.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, M, 1]
                        xs = x_weights @ init_xs
                        xs = xs.squeeze(-1)
                        anchors = self.anchor_generator.generate_anchors_batch(xs, yaws, pitches)
                        reg_proposals, update_anchors = self.get_proposals(project_matrixes, anchor_feats_i[image_feat_i], anchor_feats_l[feat_idx], feat_idx, anchors, feat_size, iter_idx, True)
                        reg_proposals_layer.append(reg_proposals)
                        anchors_layer.append(update_anchors)
                    else:
                        proposals_prev = reg_proposals_layer[feat_idx - 1]
                        reg_proposals, _ = self.get_proposals(project_matrixes, anchor_feats_i[image_feat_i], anchor_feats_l[feat_idx], feat_idx, proposals_prev, feat_size, iter_idx, False)
                        reg_proposals_layer.append(reg_proposals)
                        anchors_layer.append(proposals_prev[:, :, :5+self.anchor_len*3])
                else:
                    if feat_idx == 0:
                        proposals_prev = reg_proposals_all[iter_idx - 1][0]
                        reg_proposals, update_anchors = self.get_proposals(project_matrixes, anchor_feats_i[image_feat_i], anchor_feats_l[feat_idx], feat_idx, proposals_prev, feat_size, iter_idx, True)
                        reg_proposals_layer.append(reg_proposals)
                        anchors_layer.append(update_anchors)
                    else:
                        proposals_prev = reg_proposals_layer[feat_idx - 1]
                        reg_proposals, _ = self.get_proposals(project_matrixes, anchor_feats_i[image_feat_i], anchor_feats_l[feat_idx], feat_idx, proposals_prev, feat_size, iter_idx, False)
                        reg_proposals_layer.append(reg_proposals)
                        anchors_layer.append(proposals_prev[:, :, :5+self.anchor_len*3])
            
            reg_proposals_all.append(reg_proposals_layer)
            anchors_all.append(anchors_layer)
                    
        output = {'reg_proposals':reg_proposals_all, 'anchors':anchors_all}
        return output

    def forward_test(self, img, mask=None, img_metas=None, gt_project_matrix=None, voxels=None, 
                coordinates=None, num_points=None, num_voxels=None, **kwargs):
        gt_project_matrix = gt_project_matrix.squeeze(1)
        output = self.encoder_decoder(img, mask, voxels, coordinates, num_points, num_voxels, gt_project_matrix, **kwargs)

        proposals_list = self.nms(output['reg_proposals'][-1][-1], output['anchors'][-1][-1], self.test_cfg.nms_thres, 
                                  self.test_cfg.conf_threshold, refine_vis=self.test_cfg.refine_vis,
                                  vis_thresh=self.test_cfg.vis_thresh)
        output['proposals_list'] = proposals_list

        return output
    
    @auto_fp16(apply_to=('img', 'mask', ))
    def forward_train(self, img, mask, img_metas, gt_3dlanes=None, gt_project_matrix=None, voxels=None, 
                coordinates=None, num_points=None, num_voxels=None, **kwargs): 
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        gt_project_matrix = gt_project_matrix.squeeze(1)
        output = self.encoder_decoder(img, mask, voxels, coordinates, num_points, num_voxels, 
                                      gt_project_matrix, **kwargs)
        losses, other_vars = self.loss(output, gt_3dlanes)
        return losses, other_vars