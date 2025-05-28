# --------------------------------------------------------
# Source code for Anchor3DLane
# Copyright (c) 2025 TuSimple
# @Time    : 2025/05/07
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
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from ..builder import LANENET2S, build_backbone, build_loss, build_neck
from .tools import homography_crop_resize
from .utils import AnchorGenerator_torch, DecodeLayer, nms_3d


class ExpertDecode(nn.Module):
    def __init__(self, in_channel, x_out, yaw_out, pitch_out, anchor_num):
        super(ExpertDecode, self).__init__()
        self.x_expert_layer = DecodeLayer(in_channel, 512, x_out)
        self.pitch_expert_layer = DecodeLayer(in_channel, 512, pitch_out)
        self.yaw_expert_layer = DecodeLayer(in_channel, 512, yaw_out)
        self.anchor_num = anchor_num

    def forward(self, anchor_feat):
        # anchor_feat: [B, C, H, W] or [B, C, D, H, W]
        bs = anchor_feat.shape[0]
        feat = anchor_feat.mean(2).flatten(1, 2) if anchor_feat.dim() == 4 else anchor_feat.mean(2).mean(2).flatten(1, 2)  # [B, CW]
        yaw_weight = self.yaw_expert_layer(feat).reshape(bs, self.anchor_num, -1)
        yaw_weight = torch.softmax(yaw_weight, -1)
        pitch_weight = self.pitch_expert_layer(feat).reshape(bs, self.anchor_num, -1)
        pitch_weight = torch.softmax(pitch_weight, -1)
        x_weight = self.x_expert_layer(feat).reshape(bs, self.anchor_num, -1)
        x_weight = torch.softmax(x_weight, -1)
        return yaw_weight, pitch_weight, x_weight
        
@LANENET2S.register_module()
class Anchor3DLanePP(BaseModule):

    def __init__(self, 
                 backbone,
                 neck = None,
                 neck_aux = None,
                 pretrained = None,
                 y_steps = [  5.,  10.,  15.,  20.,  30.,  40.,  50.,  60.,  80.,  100.],
                 feat_y_steps = [  5.,  10.,  15.,  20.,  30.,  40.,  50.,  60.,  80.,  100.],
                 anchor_cfg = None,
                 db_cfg = None,
                 backbone_dim = 512,
                 drop_out = 0.1,
                 num_heads = None,
                 enc_layers = 1,
                 dim_feedforward = None,
                 pre_norm = None,
                 anchor_feat_channels = 64,
                 proj_channel = 128,
                 sample_feat = [1, 2, 3],
                 iter_reg = 2,
                 feat_sizes = [(48, 60), (48, 60), (48, 60)],
                 x_range = (-30, 30),
                 with_pos = 'none',
                 num_category = 21,
                 expert_idx = -1,
                 use_sigmoid=False,
                 loss_lane = None,
                 loss_aux = None,
                 init_cfg = None,
                 train_cfg = None,
                 test_cfg = None):
        super(Anchor3DLanePP, self).__init__(init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.db_cfg = db_cfg
        self.loss_aux = loss_aux
        self.anchor_feat_channels = anchor_feat_channels
        self.feat_sizes = feat_sizes
        self.iter_reg = iter_reg
        self.sample_feat = sample_feat
        self.use_sigmoid = use_sigmoid
        self.feat_num = len(self.feat_sizes)
        if self.use_sigmoid:
            self.num_category = num_category - 1
            test_cfg['use_sigmoid'] = True
        else:
            self.num_category = num_category
        self.enc_layers = enc_layers
        self.fp16_enabled = False
        self.with_pos = with_pos
        self.expert_idx = expert_idx

        # Anchor
        self.y_steps = np.array(y_steps, dtype=np.float32)
        self.feat_y_steps = np.array(feat_y_steps, dtype=np.float32)
        self.feat_sample_index = torch.from_numpy(np.isin(self.y_steps, self.feat_y_steps))
        self.x_norm = x_range[1]
        self.y_norm = 100.
        self.z_norm = 10.
        self.x_min = x_range[0]
        self.x_max = x_range[1]
        self.anchor_len = len(y_steps)
        self.anchor_feat_len = len(feat_y_steps)

        # Build Proposals
        self.anchor_num = anchor_cfg['anchor_num']
        self.yaw_num = len(anchor_cfg['yaws'])
        self.pitch_num = len(anchor_cfg['pitches'])
        self.x_num = anchor_cfg['num_x']
        xs = torch.linspace(-1, 1, self.x_num + 2, dtype=torch.float32)
        yaws = torch.tensor(anchor_cfg['yaws'], dtype=torch.float32) / 180.
        pitches = torch.tensor(anchor_cfg['pitches'], dtype=torch.float32) / 180.
        self.init_proposals_xs = nn.Embedding(self.x_num, 1)
        self.init_proposals_xs.weight[:, 0].data.copy_(xs[1:-1])
        self.init_proposals_yaws = nn.Embedding(self.yaw_num, 1)
        self.init_proposals_yaws.weight[:, 0].data.copy_(yaws)
        self.init_proposals_pitches = nn.Embedding(self.pitch_num, 1)
        self.init_proposals_pitches.weight[:, 0].data.copy_(pitches)

        self.anchor_generator = AnchorGenerator_torch(anchor_cfg, x_min=self.x_min, x_max=self.x_max, y_max=int(self.y_steps[-1]), y_steps=self.y_steps,
                                                norm=(self.x_norm, self.y_norm, self.z_norm)) 

        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        else:
            self.neck = None
        
        if neck_aux is not None:
            self.neck_aux = build_neck(neck_aux)
        else:
            self.neck_aux = None

        self.cls_layer = nn.ModuleDict()
        self.reg_x_layer = nn.ModuleDict()
        self.reg_z_layer = nn.ModuleDict()
        self.reg_vis_layer = nn.ModuleDict()
        self.dynamic_head = nn.ModuleDict()
        if self.with_pos == 'pcat':
            self.fuse_pos = nn.ModuleDict()

        for idx, layer in enumerate(range(self.feat_num)):
            self.proj_channel = proj_channel
            self.dynamic_head[f'layer_{idx}'] = nn.ModuleList([nn.TransformerEncoderLayer(self.proj_channel * self.anchor_feat_len, \
                                                                                           2, 256, batch_first=True) for i in range(self.iter_reg)])
            self.cls_layer[f'layer_{idx}'] = nn.ModuleList([DecodeLayer(self.proj_channel * self.anchor_feat_len, \
                self.anchor_feat_channels * self.anchor_feat_len, self.num_category) for i in range(self.iter_reg)])
            self.reg_x_layer[f'layer_{idx}'] = nn.ModuleList([DecodeLayer(self.proj_channel * self.anchor_feat_len, \
                self.anchor_feat_channels, self.anchor_len) for i in range(self.iter_reg)])
            self.reg_z_layer[f'layer_{idx}'] = nn.ModuleList([DecodeLayer(self.proj_channel * self.anchor_feat_len, \
                self.anchor_feat_channels, self.anchor_len) for i in range(self.iter_reg)])
            self.reg_vis_layer[f'layer_{idx}'] = nn.ModuleList([DecodeLayer(self.proj_channel * self.anchor_feat_len, \
                self.anchor_feat_channels, self.anchor_len) for i in range(self.iter_reg)])
            if self.with_pos == 'pcat':
                self.fuse_pos[f'layer_{idx}'] = nn.ModuleList([DecodeLayer(self.anchor_feat_channels * 2, self.anchor_feat_channels * 4, self.anchor_feat_channels) for i in range(self.iter_reg)])
            
        self.reg_prior_layer = nn.ModuleList([DecodeLayer(self.proj_channel * self.anchor_feat_len, \
            self.anchor_feat_channels, 3, bias=False) for i in range(self.iter_reg)])
        self.expert_layer = ExpertDecode(self.anchor_feat_channels * self.feat_sizes[-1][1], self.anchor_num * self.x_num, \
                                          self.anchor_num * self.yaw_num, self.anchor_num * self.pitch_num, self.anchor_num)
        for i in range(self.iter_reg):
            nn.init.zeros_(self.reg_prior_layer[i].layer[-1].weight.data)

        if self.with_pos != 'none':
            self.position_encoder = DecodeLayer(3, self.anchor_feat_channels * 4, self.anchor_feat_channels)
           
        # build loss function
        self.lane_loss = dict()
        if loss_lane is not None:
            for idx, layer in enumerate(range(self.feat_num)):
                if self.use_sigmoid:
                    for ridx in range(len(loss_lane[idx])):
                        loss_lane[idx][ridx]['use_sigmoid'] = True
                self.lane_loss[f'loss_{idx}'] = [build_loss(l) for l in loss_lane[idx]]

    def load_pretrained(self, ckpt, strict=True):
        pth = torch.load(ckpt, map_location='cpu')
        self.load_state_dict(pth['state_dict'])
        
    def sample_from_dense_anchors(self, sample_steps, dense_inds, dense_anchors):
        sample_index = np.isin(dense_inds, sample_steps)
        anchor_len = len(sample_steps)
        dense_anchor_len = len(sample_index)
        anchors = np.zeros((len(dense_anchors), 5 + anchor_len * 3), dtype=np.float32)
        anchors[:, :5] = dense_anchors[:, :5].copy()
        anchors[:, 5:5+anchor_len] = dense_anchors[:, 5:5+dense_anchor_len][:, sample_index]    # [N, 20]
        anchors[:, 5+anchor_len:5+2*anchor_len] = dense_anchors[:, 5+dense_anchor_len:5+2*dense_anchor_len][:, sample_index]    # [N, 20]
        anchors = torch.from_numpy(anchors)
        return anchors

    def compute_anchor_cut_indices(self, anchors, y_steps):
        # definitions
        if len(anchors.shape) == 2:
            n_proposals = len(anchors)
        else:
            batch_size, n_proposals = anchors.shape[:2]

        num_y_steps = len(y_steps)

        # indexing
        xs = anchors[..., 5:5 + num_y_steps]  # [N, l] or [B, N, l]
        xs = torch.flatten(xs, -2)  # [Nl] or [B, Nl]

        ys = torch.from_numpy(y_steps).to(anchors.device)   # [l]
        if len(anchors.shape) == 2:
            ys = ys.repeat(n_proposals)  # [Nl]
        else:
            ys = ys.repeat(batch_size, n_proposals)  # [B, Nl]

        zs = anchors[..., 5 + num_y_steps:5 + num_y_steps * 2]  # [N, l]
        zs = torch.flatten(zs, -2)  # [Nl] or [B, Nl]
        return xs, ys, zs

    def projection_transform(self, Matrix, xs, ys, zs):
        # Matrix: [B, 3, 4], x, y, z: [B, NCl]
        ones = torch.ones_like(zs)   # [B, NCl]
        coordinates = torch.stack([xs, ys, zs, ones], dim=1)   # [B, 4, NCl]
        trans = torch.bmm(Matrix, coordinates)   # [B, 3, NCl]

        u_vals = trans[:, 0, :] / trans[:, 2, :]   # [B, NCl]
        v_vals = trans[:, 1, :] / trans[:, 2, :]   # [B, NCl]
        return u_vals, v_vals

    def cut_anchor_features(self, features, h_g2feats, xs, ys, zs, anchor_feat_len, feat_size):
        # definitions
        batch_size = features.shape[0]

        if len(xs.shape) == 1:
            batch_xs = xs.repeat(batch_size, 1)   # [B, Nl]
            batch_ys = ys.repeat(batch_size, 1)   # [B, Nl]
            batch_zs = zs.repeat(batch_size, 1)   # [B, Nl]
        else:
            batch_xs = xs
            batch_ys = ys
            batch_zs = zs

        batch_us, batch_vs = self.projection_transform(h_g2feats, batch_xs, batch_ys, batch_zs)
        batch_us = (batch_us / feat_size[1] - 0.5) * 2
        batch_vs = (batch_vs / feat_size[0] - 0.5) * 2

        batch_grid = torch.stack([batch_us, batch_vs], dim=-1)  #
        batch_grid = batch_grid.reshape(batch_size, -1, anchor_feat_len, 2)  # [B, N, l, 2]
        batch_anchor_features = F.grid_sample(features, batch_grid, padding_mode='zeros')   # [B, C, N, l]

        valid_mask = (batch_us > -1) & (batch_us < 1) & (batch_vs > -1) & (batch_vs < 1)

        return batch_anchor_features, valid_mask.reshape(batch_size, -1, anchor_feat_len)

    def feature_extractor(self, img, mask):
        output = self.backbone(img)
        # effnet-b3: [B, 232/136/96, 45, 60]
        if self.neck is not None:
            fpn_features = [output[i] for i in self.sample_feat]
            output = self.neck(fpn_features, mask)
        if self.neck_aux is not None:
            output = self.neck_aux(output)
        return output

    @force_fp32()
    def get_proposals(self, project_matrixes, anchor_feat, feat_idx, proposals_prev, feat_size, iter_idx, reg_prior=False):
        batch_size = project_matrixes.shape[0]
        xs, ys, zs = self.compute_anchor_cut_indices(proposals_prev, self.feat_y_steps)
        batch_anchor_features, _ = self.cut_anchor_features(anchor_feat, project_matrixes, xs, ys, zs, self.anchor_feat_len, feat_size)   # [B, C, N, l]

        if self.with_pos != 'none':
            xs = xs / self.x_norm # [B, N*L]
            ys = ys / self.y_norm
            zs = zs / self.z_norm
            xyz = torch.stack([xs, ys, zs], -1)  # [B, NL, 3]
            batch_pos_features = self.position_encoder(xyz)  # [B, NL, C]
            batch_pos_features = batch_pos_features.transpose(1, 2).reshape(batch_size, self.anchor_feat_channels, self.anchor_num, self.anchor_feat_len)
            if self.with_pos == 'add':
                batch_anchor_features = batch_anchor_features + batch_pos_features
            elif self.with_pos == 'pcat':
                batch_anchor_features = torch.cat([batch_anchor_features, batch_pos_features], 1)  # [B, C, N, l]
                batch_anchor_features = batch_anchor_features.permute(0, 2, 3, 1) # [B, N, l, C]
                batch_anchor_features = self.fuse_pos[f'layer_{feat_idx}'][iter_idx](batch_anchor_features)
                batch_anchor_features = batch_anchor_features.permute(0, 3, 1, 2)  # [B, C, N, l]
            else:
                batch_anchor_features = torch.cat([batch_anchor_features, batch_pos_features], 1)

        batch_anchor_features = batch_anchor_features.transpose(1, 2)  # [B, N, C, l]
        batch_anchor_features = batch_anchor_features.flatten(2, 3)  # [B, N, C*l]
        batch_anchor_features = self.dynamic_head[f'layer_{feat_idx}'][iter_idx](batch_anchor_features) # [B, N, Cl]
        batch_anchor_features = batch_anchor_features.flatten(0, 1)  # [B*N, C*l]

        # Predict
        cls_logits = self.cls_layer[f'layer_{feat_idx}'][iter_idx](batch_anchor_features)   # [B * N, C]
        cls_logits = cls_logits.reshape(batch_size, -1, cls_logits.shape[1])   # [B, N, C]
        reg_x = self.reg_x_layer[f'layer_{feat_idx}'][iter_idx](batch_anchor_features)    # [B * N, l]  
        reg_x = reg_x.reshape(batch_size, -1, reg_x.shape[1])   # [B, N, l]
        reg_z = self.reg_z_layer[f'layer_{feat_idx}'][iter_idx](batch_anchor_features)    # [B * N, l]  # tanh
        reg_z = reg_z.reshape(batch_size, -1, reg_z.shape[1])   # [B, N, l]
        reg_vis = self.reg_vis_layer[f'layer_{feat_idx}'][iter_idx](batch_anchor_features)  # [B * N, l]
        reg_vis = torch.sigmoid(reg_vis)
        reg_vis = reg_vis.reshape(batch_size, -1, reg_vis.shape[1])   # [B, N, l]

        # lane prior regression
        if reg_prior:
            reg_lane_priors = self.reg_prior_layer[iter_idx](batch_anchor_features)   # [B * N, 3]
            reg_lane_priors = reg_lane_priors.reshape(batch_size, -1, 3)   # [B, N, C]
            reg_lane_priors = torch.tanh(reg_lane_priors) # / 30.  # yaws, pitch, xs, [-1, 1]
            
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


    def encoder_decoder(self, img, mask, gt_project_matrix, **kwargs):
        # img: [B, 3, inp_h, inp_w]; mask: [B, 1, 36, 480]
        batch_size = img.shape[0]
        anchor_feats = self.feature_extractor(img, mask)
            
        reg_proposals_all = []
        anchors_all = []

        for iter_idx in range(self.iter_reg):
            
            reg_proposals_layer = []
            anchors_layer = []
            
            for feat_idx, feat_size in enumerate(self.feat_sizes[::-1]):
                # [4, 3, 2]
                project_matrixes = self.obtain_projection_matrix(gt_project_matrix, feat_size)
                project_matrixes = torch.stack(project_matrixes, dim=0)   # [B, 3, 4]
                select_idx = self.feat_num - 1 - feat_idx
                if iter_idx == 0:
                    if feat_idx == 0:
                        yaw_weights, pitch_weights, x_weights = self.expert_layer(anchor_feats[-1])
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
                        reg_proposals, update_anchors = self.get_proposals(project_matrixes, anchor_feats[select_idx], \
                                                                                           feat_idx, anchors, feat_size, iter_idx, True)
                        reg_proposals_layer.append(reg_proposals)
                        anchors_layer.append(update_anchors)
                    else:
                        proposals_prev = reg_proposals_layer[feat_idx - 1]
                        reg_proposals, _ = self.get_proposals(project_matrixes, anchor_feats[select_idx], \
                                                                              feat_idx, proposals_prev, feat_size, iter_idx, False)
                        reg_proposals_layer.append(reg_proposals)
                        anchors_layer.append(proposals_prev[:, :, :5+self.anchor_len*3])
                else:
                    if feat_idx == 0:
                        proposals_prev = reg_proposals_all[iter_idx - 1][0]
                        reg_proposals, update_anchors = self.get_proposals(project_matrixes, anchor_feats[select_idx], \
                                                                                           feat_idx, proposals_prev, feat_size, iter_idx, True)
                        reg_proposals_layer.append(reg_proposals)
                        anchors_layer.append(update_anchors)
                    else:
                        proposals_prev = reg_proposals_layer[feat_idx - 1]
                        reg_proposals, _ = self.get_proposals(project_matrixes, anchor_feats[select_idx], \
                                                                              feat_idx, proposals_prev, feat_size, iter_idx, False)
                        reg_proposals_layer.append(reg_proposals)
                        anchors_layer.append(proposals_prev[:, :, :5+self.anchor_len*3])
            
            reg_proposals_all.append(reg_proposals_layer)
            anchors_all.append(anchors_layer)

                    
        output = {'reg_proposals':reg_proposals_all, 'anchors':anchors_all}
        return output
        

    def obtain_projection_matrix(self, project_matrix, feat_size):
        """
            Update transformation matrix based on ground-truth cam_height and cam_pitch
            This function is "Mutually Exclusive" to the updates of M_inv from network prediction
        :param args:
        :param cam_height:
        :param cam_pitch:
        :return:
        """
        h_g2feats = []
        device = project_matrix.device
        project_matrix = project_matrix.cpu().numpy()
        for i in range(len(project_matrix)):
            P_g2im = project_matrix[i]
            Hc = homography_crop_resize((self.db_cfg.org_h, self.db_cfg.org_w), 0, feat_size)
            h_g2feat = np.matmul(Hc, P_g2im)
            h_g2feats.append(torch.from_numpy(h_g2feat).type(torch.FloatTensor).to(device))
        return h_g2feats


    def nms(self, batch_proposals, batch_anchors, nms_thres=0, conf_threshold=None, refine_vis=False, vis_thresh=0.5):
        softmax = nn.Softmax(dim=1)
        proposals_list = []
        for proposals, anchors in zip(batch_proposals, batch_anchors):
            anchor_inds = torch.arange(batch_proposals.shape[1], device=proposals.device)
            # The gradients do not have to (and can't) be calculated for the NMS procedure
            # apply nms
            if self.use_sigmoid:
                scores = proposals[:, 5 + self.anchor_len * 3:5 + self.anchor_len * 3+self.num_category].sigmoid().max(dim=1)[0]
            else:
                scores = 1 - softmax(proposals[:, 5 + self.anchor_len * 3:5 + self.anchor_len * 3+self.num_category])[:, 0]  # pos_score
            if conf_threshold > 0:
                above_threshold = scores > conf_threshold
                proposals = proposals[above_threshold]
                scores = scores[above_threshold]
                anchor_inds = anchor_inds[above_threshold]
            if proposals.shape[0] == 0:
                proposals_list.append((proposals[[]], anchors[[]], None))
                continue
            if nms_thres > 0:
                # refine vises to ensure consistent lane
                vises = proposals[:, 5 + self.anchor_len * 2:5 + self.anchor_len * 3] >= vis_thresh  # need check  #[N, l]
                flag_l = vises.cumsum(dim=1)
                flag_r = vises.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
                refined_vises = (flag_l > 0) & (flag_r > 0)
                if refine_vis:
                    proposals[:, 5 + self.anchor_len * 2:5 + self.anchor_len * 3] = refined_vises
                keep = nms_3d(proposals, scores, refined_vises, thresh=nms_thres, anchor_len=self.anchor_len)
                proposals = proposals[keep]
                anchor_inds = anchor_inds[keep]
                proposals_list.append((proposals, anchors[anchor_inds], anchor_inds))
            else:
                proposals_list.append((proposals, anchors[anchor_inds], anchor_inds))
        return proposals_list

    @force_fp32()
    def get_proposals(self, project_matrixes, anchor_feat, feat_idx, proposals_prev, feat_size, iter_idx, reg_prior=False):
        batch_size = project_matrixes.shape[0]
        xs, ys, zs = self.compute_anchor_cut_indices(proposals_prev, self.feat_y_steps)
        batch_anchor_features, _ = self.cut_anchor_features(anchor_feat, project_matrixes, xs, ys, zs, self.anchor_feat_len, feat_size)   # [B, C, N, l]

        if self.with_pos != 'none':
            xs = xs / self.x_norm # [B, N*L]
            ys = ys / self.y_norm
            zs = zs / self.z_norm
            xyz = torch.stack([xs, ys, zs], -1)  # [B, NL, 3]
            batch_pos_features = self.position_encoder(xyz)  # [B, NL, C]
            batch_pos_features = batch_pos_features.transpose(1, 2).reshape(batch_size, self.anchor_feat_channels, self.anchor_num, self.anchor_feat_len)
            if self.with_pos == 'add':
                batch_anchor_features = batch_anchor_features + batch_pos_features
            elif self.with_pos == 'pcat':
                batch_anchor_features = torch.cat([batch_anchor_features, batch_pos_features], 1)  # [B, C, N, l]
                batch_anchor_features = batch_anchor_features.permute(0, 2, 3, 1) # [B, N, l, C]
                batch_anchor_features = self.fuse_pos[f'layer_{feat_idx}'][iter_idx](batch_anchor_features)
                batch_anchor_features = batch_anchor_features.permute(0, 3, 1, 2)  # [B, C, N, l]
            else:
                batch_anchor_features = torch.cat([batch_anchor_features, batch_pos_features], 1)

        batch_anchor_features = batch_anchor_features.transpose(1, 2)  # [B, N, C, l]
        batch_anchor_features = batch_anchor_features.flatten(2, 3)  # [B, N, C*l]
        batch_anchor_features = self.dynamic_head[f'layer_{feat_idx}'][iter_idx](batch_anchor_features) # [B, N, Cl]
        batch_anchor_features = batch_anchor_features.flatten(0, 1)  # [B*N, C*l]

        # Predict
        cls_logits = self.cls_layer[f'layer_{feat_idx}'][iter_idx](batch_anchor_features)   # [B * N, C]
        cls_logits = cls_logits.reshape(batch_size, -1, cls_logits.shape[1])   # [B, N, C]
        reg_x = self.reg_x_layer[f'layer_{feat_idx}'][iter_idx](batch_anchor_features)    # [B * N, l]  
        reg_x = reg_x.reshape(batch_size, -1, reg_x.shape[1])   # [B, N, l]
        reg_z = self.reg_z_layer[f'layer_{feat_idx}'][iter_idx](batch_anchor_features)    # [B * N, l]  # tanh
        reg_z = reg_z.reshape(batch_size, -1, reg_z.shape[1])   # [B, N, l]
        reg_vis = self.reg_vis_layer[f'layer_{feat_idx}'][iter_idx](batch_anchor_features)  # [B * N, l]
        reg_vis = torch.sigmoid(reg_vis)
        reg_vis = reg_vis.reshape(batch_size, -1, reg_vis.shape[1])   # [B, N, l]

        # lane prior regression
        if reg_prior:
            reg_lane_priors = self.reg_prior_layer[iter_idx](batch_anchor_features)   # [B * N, 3]
            reg_lane_priors = reg_lane_priors.reshape(batch_size, -1, 3)   # [B, N, C]
            reg_lane_priors = torch.tanh(reg_lane_priors) # / 30.  # yaws, pitch, xs, [-1, 1]
            
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



    def forward_test(self, img, mask=None, img_metas=None, gt_project_matrix=None, **kwargs):
        gt_project_matrix = gt_project_matrix.squeeze(1)
        output = self.encoder_decoder(img, mask, gt_project_matrix, **kwargs)

        proposals_list = self.nms(output['reg_proposals'][-1][-1], output['anchors'][-1][-1], self.test_cfg.nms_thres, 
                                  self.test_cfg.conf_threshold, refine_vis=self.test_cfg.refine_vis,
                                  vis_thresh=self.test_cfg.vis_thresh)

        output['proposals_list'] = proposals_list

        return output
        
    def forward(self, img, img_metas, mask=None, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, mask, img_metas, **kwargs)
        else:
            return self.forward_test(img, mask, img_metas, **kwargs)
    

    @force_fp32()
    def loss(self, output, gt_3dlanes):
        losses = dict()
        # postprocess
        for iter_idx in range(self.iter_reg):
            for feat_idx in range(self.feat_num):
                proposals_list = []
                for proposal, anchor in zip(output['reg_proposals'][iter_idx][feat_idx], output['anchors'][iter_idx][feat_idx]):
                    proposals_list.append((proposal, anchor))
                anchor_losses = self.lane_loss[f'loss_{feat_idx}'][iter_idx](proposals_list, gt_3dlanes)
                for k, v in anchor_losses['losses'].items():
                    if 'loss' in k:
                        if iter_idx == 0:
                            losses[k+f'_{feat_idx}'] = v
                        else:
                            losses[k+f'_{feat_idx}_{iter_idx}'] = v
                
        other_vars = {}
        other_vars['batch_positives'] = anchor_losses['batch_positives']
        other_vars['batch_negatives'] = anchor_losses['batch_negatives']
        return losses, other_vars

    @auto_fp16(apply_to=('img', 'mask', ))
    def forward_train(self, img, mask, img_metas, gt_3dlanes=None, gt_project_matrix=None, **kwargs): 
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
        output = self.encoder_decoder(img, mask, gt_project_matrix, **kwargs)
        losses, other_vars = self.loss(output, gt_3dlanes)
        return losses, other_vars

    def train_step(self, data_batch, optimizer=None, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses, other_vars = self(**data_batch)
        loss, log_vars = self._parse_losses(losses, other_vars)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=data_batch['img'].shape[0])

        return outputs

    def val_step(self, data_batch, optimizer=None, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        log_vars_ = dict()
        for loss_name, loss_value in log_vars.items():
            k = loss_name + '_val'
            log_vars_[k] = loss_value

        outputs = dict(
            loss=loss,
            log_vars=log_vars_,
            num_samples=len(data_batch['img_metas']))

        return outputs

    @staticmethod
    def _parse_losses(losses, other_vars=None):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for var_name, var_value in other_vars.items():
            log_vars[var_name] = var_value
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')
        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, raise assertion error
        # to prevent GPUs from infinite waiting.
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            # print("log_val_length before reduce:", log_var_length)
            dist.all_reduce(log_var_length)
            # print("log_val_length after reduce:", log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()) + '\n')
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            if isinstance(loss_value, int) or isinstance(loss_value, float):
                continue
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()
        return loss, log_vars

    def cuda(self, device=None):
        cuda_self = super().cuda(device)
        cuda_self.init_proposals_pitches = cuda_self.init_proposals_pitches.cuda(device)
        cuda_self.init_proposals_xs = cuda_self.init_proposals_xs.cuda(device)
        cuda_self.init_proposals_yaws = cuda_self.init_proposals_yaws.cuda(device)
        return cuda_self

    def to(self, *args, **kwargs):
        device_self = super().to(*args, **kwargs)
        device_self.init_proposals_pitches = device_self.init_proposals_pitches.to(*args, **kwargs)
        device_self.init_proposals_xs = device_self.init_proposals_xs.to(*args, **kwargs)
        device_self.init_proposals_yaws = device_self.init_proposals_yaws.to(*args, **kwargs)
        return device_self