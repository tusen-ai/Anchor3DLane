# --------------------------------------------------------
# Source code for Anchor3DLane
# Copyright (c) 2023 TuSimple
# @Time    : 2024/04/02
# @Author  : Shaofei Huang
# nowherespyfly@gmail.com
# --------------------------------------------------------


from random import sample
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import pdb
import math

import time
import mmcv
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from ..builder import build_loss, build_backbone, build_neck
from ..builder import LANENET2S
from .tools import homography_crop_resize
from .utils import AnchorGenerator, nms_3d
from .msda import MSDALayer

class DecodeLayer(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(DecodeLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channel, mid_channel),
            nn.ReLU6(),
            nn.Linear(mid_channel, mid_channel),
            nn.ReLU6(),
            nn.Linear(mid_channel, out_channel))
    def forward(self, x):
        return self.layer(x)

@LANENET2S.register_module()
class Anchor3DLaneDeform(BaseModule):

    def __init__(self, 
                 backbone,
                 neck = None,
                 pretrained = None,
                 y_steps = [  5.,  10.,  15.,  20.,  30.,  40.,  50.,  60.,  80.,  100.],
                 feat_y_steps = [  5.,  10.,  15.,  20.,  30.,  40.,  50.,  60.,  80.,  100.],
                 anchor_cfg = None,
                 db_cfg = None,
                 backbone_dim = 512,
                 attn_dim = None,
                 iter_reg = 0,
                 drop_out = 0.1,
                 num_heads = None,
                 enc_layers = 1,
                 dim_feedforward = None,
                 pre_norm = None,
                 anchor_feat_channels = 64,
                 feat_size = (48, 60),
                 num_category = 21,
                 loss_lane = None,
                 loss_aux = None,
                 init_cfg = None,
                 train_cfg = None,
                 test_cfg = None):
        super(Anchor3DLaneDeform, self).__init__(init_cfg)
        assert loss_aux is None or len(loss_aux) == iter_reg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.db_cfg = db_cfg
        hidden_dim = attn_dim
        self.iter_reg = iter_reg
        self.loss_aux = loss_aux
        self.anchor_feat_channels = anchor_feat_channels
        self.feat_size = feat_size
        self.num_category = num_category
        self.enc_layers = enc_layers
        self.fp16_enabled = False

        # Anchor
        self.y_steps = np.array(y_steps, dtype=np.float32)
        self.feat_y_steps = np.array(feat_y_steps, dtype=np.float32)
        self.feat_sample_index = torch.from_numpy(np.isin(self.y_steps, self.feat_y_steps))
        self.x_norm = 30.
        self.y_norm = 100.
        self.z_norm = 10.
        self.x_min = -30
        self.x_max = 30
        self.anchor_len = len(y_steps)
        self.anchor_feat_len = len(feat_y_steps)
        self.anchor_generator = AnchorGenerator(anchor_cfg, x_min=self.x_min, x_max=self.x_max, y_max=int(self.y_steps[-1]),
                                                norm=(self.x_norm, self.y_norm, self.z_norm))
        dense_anchors = self.anchor_generator.generate_anchors()  # [N, 65]
        anchor_inds = self.anchor_generator.y_steps   # [100]
        self.anchors = self.sample_from_dense_anchors(self.y_steps, anchor_inds, dense_anchors)
        self.feat_anchors = self.sample_from_dense_anchors(self.feat_y_steps, anchor_inds, dense_anchors)
        self.xs, self.ys, self.zs = self.compute_anchor_cut_indices(self.feat_anchors, self.feat_y_steps)

        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        # transformer layer
        self.input_proj = nn.Conv2d(backbone_dim, hidden_dim, kernel_size=1)
        self.transformer_layer = MSDALayer([64,], in_features=[0,], feature_strides=[1,], conv_dim=64, transformer_enc_layers=1)
                                        
        # decoder heads
        self.anchor_projection = nn.Conv2d(hidden_dim, self.anchor_feat_channels, kernel_size=1)

        # FPN
        if neck is not None:
            self.neck = build_neck(neck)
        else:
            self.neck = None

        self.cls_layer = nn.ModuleList()
        self.reg_x_layer = nn.ModuleList()
        self.reg_z_layer = nn.ModuleList()
        self.reg_vis_layer = nn.ModuleList()

        self.cls_layer.append(DecodeLayer(self.anchor_feat_channels * self.anchor_feat_len, self.anchor_feat_channels * self.anchor_feat_len, self.num_category))
        self.reg_x_layer.append(DecodeLayer(self.anchor_feat_channels * self.anchor_feat_len, self.anchor_feat_channels, self.anchor_len))
        self.reg_z_layer.append(DecodeLayer(self.anchor_feat_channels * self.anchor_feat_len, self.anchor_feat_channels, self.anchor_len))
        self.reg_vis_layer.append(DecodeLayer(self.anchor_feat_channels * self.anchor_feat_len, self.anchor_feat_channels, self.anchor_len))

        # build loss function
        self.lane_loss = build_loss(loss_lane)

        # build iterative regression layers
        self.build_iterreg_layers()

    def build_iterreg_layers(self):
        self.aux_loss = nn.ModuleList()
        for iter in range(self.iter_reg):
            self.cls_layer.append(DecodeLayer(self.anchor_feat_channels * self.anchor_feat_len, self.anchor_feat_channels * self.anchor_feat_len, self.num_category))
            self.reg_x_layer.append(DecodeLayer(self.anchor_feat_channels * self.anchor_feat_len, self.anchor_feat_channels, self.anchor_len))
            self.reg_z_layer.append(DecodeLayer(self.anchor_feat_channels * self.anchor_feat_len, self.anchor_feat_channels, self.anchor_len))
            self.reg_vis_layer.append(DecodeLayer(self.anchor_feat_channels * self.anchor_feat_len, self.anchor_feat_channels, self.anchor_len))
            self.aux_loss.append(build_loss(self.loss_aux[iter]))
        

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

    def cut_anchor_features(self, features, h_g2feats, xs, ys, zs):
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
        batch_us = (batch_us / self.feat_size[1] - 0.5) * 2
        batch_vs = (batch_vs / self.feat_size[0] - 0.5) * 2

        batch_grid = torch.stack([batch_us, batch_vs], dim=-1)  #
        batch_grid = batch_grid.reshape(batch_size, -1, self.anchor_feat_len, 2)  # [B, N, l, 2]
        batch_anchor_features = F.grid_sample(features, batch_grid, padding_mode='zeros')   # [B, C, N, l]

        valid_mask = (batch_us > -1) & (batch_us < 1) & (batch_vs > -1) & (batch_vs < 1)

        return batch_anchor_features, valid_mask.reshape(batch_size, -1, self.anchor_feat_len)

    def feature_extractor(self, img, mask):
        output = self.backbone(img)
        if self.neck is not None:
            output = self.neck(output)
            feat = output[0]
        else:
            feat = output[-1]
        feat = self.input_proj(feat)
        
        # transformer forward
        bs, c, h, w = feat.shape
        assert h == self.feat_size[0] and w == self.feat_size[1]
        trans_feat = self.transformer_layer([feat,])  
        trans_feat = trans_feat[0]

        return trans_feat

    @force_fp32()
    def get_proposals(self, project_matrixes, anchor_feat, iter_idx=0, proposals_prev=None):
        batch_size = project_matrixes.shape[0]
        if proposals_prev is None:
            batch_anchor_features, _ = self.cut_anchor_features(anchor_feat, project_matrixes, self.xs, self.ys, self.zs)   # [B, C, N, l]
        else:
            sampled_anchor = torch.zeros(batch_size, len(self.anchors), 5 + self.anchor_feat_len * 3, device = anchor_feat.device)
            sampled_anchor[:, :, 5:5+self.anchor_feat_len] = proposals_prev[:, :, 5:5+self.anchor_len][:, :, self.feat_sample_index]
            sampled_anchor[:, :, 5+self.anchor_feat_len:5+self.anchor_feat_len*2] = proposals_prev[:, :, 5+self.anchor_len:5+self.anchor_len*2][:, :, self.feat_sample_index]
            xs, ys, zs = self.compute_anchor_cut_indices(sampled_anchor, self.feat_y_steps)
            batch_anchor_features, _ = self.cut_anchor_features(anchor_feat, project_matrixes, xs, ys, zs)   # [B, C, N, l]
    
        batch_anchor_features = batch_anchor_features.transpose(1, 2)  # [B, N, C, l]
        batch_anchor_features = batch_anchor_features.reshape(-1, self.anchor_feat_channels * self.anchor_feat_len)  # [B * N, C * l]

        # Predict
        cls_logits = self.cls_layer[iter_idx](batch_anchor_features)   # [B * N, C]
        cls_logits = cls_logits.reshape(batch_size, -1, cls_logits.shape[1])   # [B, N, C]
        reg_x = self.reg_x_layer[iter_idx](batch_anchor_features)    # [B * N, l]
        reg_x = reg_x.reshape(batch_size, -1, reg_x.shape[1])   # [B, N, l]
        reg_z = self.reg_z_layer[iter_idx](batch_anchor_features)    # [B * N, l]
        reg_z = reg_z.reshape(batch_size, -1, reg_z.shape[1])   # [B, N, l]
        reg_vis = self.reg_vis_layer[iter_idx](batch_anchor_features)  # [B * N, l]
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

    def encoder_decoder(self, img, mask, gt_project_matrix, **kwargs):
        # img: [B, 3, inp_h, inp_w]; mask: [B, 1, 36, 480]
        batch_size = img.shape[0]
        trans_feat = self.feature_extractor(img, mask)

        # anchor
        anchor_feat = self.anchor_projection(trans_feat)
        project_matrixes = self.obtain_projection_matrix(gt_project_matrix, self.feat_size)
        project_matrixes = torch.stack(project_matrixes, dim=0)   # [B, 3, 4]

        reg_proposals_all = []
        anchors_all = []
        reg_proposals_s1 = self.get_proposals(project_matrixes, anchor_feat, 0)
        reg_proposals_all.append(reg_proposals_s1)
        anchors_all.append(torch.stack([self.anchors] * batch_size, dim=0))

        for iter in range(self.iter_reg):
            proposals_prev = reg_proposals_all[iter]
            reg_proposals_all.append(self.get_proposals(project_matrixes, anchor_feat, iter+1, proposals_prev))
            anchors_all.append(proposals_prev[:, :, :5+self.anchor_len*3])

        output = {'reg_proposals':reg_proposals_all[-1], 'anchors':anchors_all[-1]}
        if self.iter_reg > 0:
            output_aux = {'reg_proposals':reg_proposals_all[:-1], 'anchors':anchors_all[:-1]}
            return output, output_aux
        return output, None
        

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
            scores = 1 - softmax(proposals[:, 5 + self.anchor_len * 3:5 + self.anchor_len * 3+self.num_category])[:, 0]  # pos_score  # for debug
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


    def forward_dummy(self, img, mask=None, img_metas=None, gt_project_matrix=None, **kwargs):
        mask = img.new_zeros((img.shape[0], 1, img.shape[2], img.shape[3]))
        gt_project_matrix = img.new_zeros((img.shape[0], 3, 4))
        output, _ = self.encoder_decoder(img, mask, gt_project_matrix, **kwargs)
        return output

    def forward_test(self, img, mask=None, img_metas=None, gt_project_matrix=None, **kwargs):
        gt_project_matrix = gt_project_matrix.squeeze(1)
        output, _ = self.encoder_decoder(img, mask, gt_project_matrix, **kwargs)

        proposals_list = self.nms(output['reg_proposals'], output['anchors'], self.test_cfg.nms_thres, 
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
    def loss(self, output, gt_3dlanes, output_aux=None):
        losses = dict()

        # postprocess
        proposals_list = []
        for proposal, anchor in zip(output['reg_proposals'], output['anchors']):
            proposals_list.append((proposal, anchor))
        anchor_losses = self.lane_loss(proposals_list, gt_3dlanes)
        losses.update(anchor_losses['losses'])
        
        # auxiliary loss
        for iter in range(self.iter_reg):
            proposals_list_aux = []
            for proposal, anchor in zip(output_aux['reg_proposals'][iter], output_aux['anchors'][iter]):
                proposals_list_aux.append((proposal, anchor))
            anchor_losses_aux = self.aux_loss[iter](proposals_list_aux, gt_3dlanes)
            for k, v in anchor_losses_aux['losses'].items():
                if 'loss' in k:
                    losses[k+str(iter)] = v
                
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
        output, output_aux = self.encoder_decoder(img, mask, gt_project_matrix, **kwargs)
        losses, other_vars = self.loss(output, gt_3dlanes, output_aux)
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
        cuda_self.anchors = cuda_self.anchors.cuda(device)
        cuda_self.feat_anchors = cuda_self.feat_anchors.cuda(device)
        cuda_self.zs = cuda_self.zs.cuda(device)
        cuda_self.ys = cuda_self.ys.cuda(device)
        cuda_self.xs = cuda_self.xs.cuda(device)
        return cuda_self

    def to(self, *args, **kwargs):
        device_self = super().to(*args, **kwargs)
        device_self.anchors = device_self.anchors.to(*args, **kwargs)
        device_self.feat_anchors = device_self.feat_anchors.to(*args, **kwargs)
        device_self.zs = device_self.zs.to(*args, **kwargs)
        device_self.ys = device_self.ys.to(*args, **kwargs)
        device_self.xs = device_self.xs.to(*args, **kwargs)
        return device_self