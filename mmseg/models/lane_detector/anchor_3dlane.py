from random import sample
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import re 
import os 

import neptune
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from neptune.exceptions import NeptuneModelKeyAlreadyExistsError
from scipy.interpolate import interp1d

from configs.zod.anchor3dlane import model, checkpoint_config, work_dir,\
    api_token, project, model_id, log_to_neptune, optimizer, runner, data
from ..builder import build_loss, build_backbone, build_neck
from .transformer import TransformerEncoderLayer, TransformerEncoder
from .position_encoding import PositionEmbeddingSine
from ..builder import LANENET2S
from .tools import homography_crop_resize

from .utils import *


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
    
def postprocess(output, anchor_len=10):
            proposals = output[0]
            logits = F.softmax(proposals[:, 5 + 3 * anchor_len:], dim=1)
            score = 1 - logits[:, 0]  # [N]
            proposals[:, 5 + 3 * anchor_len:] = logits  # [N, 2]
            proposals[:, 1] = score
            results = {'proposals_list': proposals.cpu().numpy()}
            return results   # [1, 7, 16]

@LANENET2S.register_module()
class Anchor3DLane(BaseModule):

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
        super(Anchor3DLane, self).__init__(init_cfg)
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
        dense_anchors = self.anchor_generator.generate_anchors()  # [~4K, 305] ~(45*yaws*pitch)
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
        self.position_embedding = PositionEmbeddingSine(num_pos_feats=hidden_dim // 2, normalize=True)
        self.input_proj = nn.Conv2d(backbone_dim, hidden_dim, kernel_size=1)  # the same as channel of self.layer4
        if self.enc_layers == 1:
            self.transformer_layer = TransformerEncoderLayer(hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
                                    dropout=drop_out, normalize_before=pre_norm)
        else:
            transformer_layer = TransformerEncoderLayer(hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward, \
                dropout=drop_out, normalize_before=pre_norm)
            self.transformer_layer = TransformerEncoder(transformer_layer, self.enc_layers)
                                        
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
        self.save_check_iter = checkpoint_config['interval']
        self.iter = 0
        self.output_dir = work_dir 
        
        #Eval
        self.top_view_region = np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]])
        self.x_min = self.top_view_region[0, 0]
        self.x_max = self.top_view_region[1, 0]
        self.y_min = self.top_view_region[2, 1]
        self.y_max = self.top_view_region[0, 1]
        self.y_samples = np.linspace(self.y_min, self.y_max, num=100, endpoint=False)
        self.dist_th = 1.5
        self.ratio_th = 0.75
        self.close_range = 40
        
        if log_to_neptune:
            self.neptune_logger = neptune.init_run(project=project, api_token=api_token)
            run_id = self.neptune_logger["sys/id"].fetch()
            try:
                self.model = neptune.init_model(
                    name="Basic Anchor3dLane model",
                    key=model_id, 
                    project=project, 
                    api_token=api_token, # your credentials
                )
            except NeptuneModelKeyAlreadyExistsError:
                pass
            parameters = {"backbone_dim":backbone_dim, "iter_reg":iter_reg, "drop_out":drop_out, \
                "anchor_feat_channel":anchor_feat_channels, "feat_size":feat_size, "x_norm":self.x_norm, "y_norm":self.y_norm, \
                "z_norm":self.z_norm, "x_min":self.x_min, "x_max":self.x_max, "runner_type": runner['type'], "num_iters": runner['max_iters'], \
                "batch_size": data["samples_per_gpu"]}
            self.neptune_logger["model/parameters"] = parameters
            self.model_version = neptune.init_model_version(
            model=str(run_id[:2]+"-"+model_id),
            project=project,
            api_token=api_token, # your credentials
        )
            model_version_id = self.model_version["sys/id"].fetch()
            self.model_version['model/config'].upload('configs/openlane/anchor3dlane.py')
            self.model_version['model/parameters'] = parameters
            self.model_version['model/parameters/run_id'] = run_id
            self.neptune_logger["model/parameters/model_version_id"] = model_version_id
        else:
            self.neptune_logger = None
            self.model_version = None


    def build_iterreg_layers(self):
        self.aux_loss = nn.ModuleList()
        for iter in range(self.iter_reg):
            self.cls_layer.append(DecodeLayer(self.anchor_feat_channels * self.anchor_feat_len, self.anchor_feat_channels * self.anchor_feat_len, self.num_category))
            self.reg_x_layer.append(DecodeLayer(self.anchor_feat_channels * self.anchor_feat_len, self.anchor_feat_channels, self.anchor_len))
            self.reg_z_layer.append(DecodeLayer(self.anchor_feat_channels * self.anchor_feat_len, self.anchor_feat_channels, self.anchor_len))
            self.reg_vis_layer.append(DecodeLayer(self.anchor_feat_channels * self.anchor_feat_len, self.anchor_feat_channels, self.anchor_len))
            self.aux_loss.append(build_loss(self.loss_aux[iter]))
        

    def sample_from_dense_anchors(self, sample_steps, dense_inds, dense_anchors):
        # self.y_steps, anchor_inds, desne_anchors
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
        batch_us = (batch_us / self.feat_size[1] - 0.5) * 2 #Scaling the 2d anchors so that we can project it to the feature space 
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
        mask_interp = F.interpolate(mask[:, 0, :, :][None], size=feat.shape[-2:]).to(torch.bool)[0]  # [B, h, w]
        
        pos = self.position_embedding(feat, mask_interp)   # [B, 32, h, w]
        
        # transformer forward
        bs, c, h, w = feat.shape
        assert h == self.feat_size[0] and w == self.feat_size[1]
        feat = feat.flatten(2).permute(2, 0, 1)  # [hw, bs, c]
        pos = pos.flatten(2).permute(2, 0, 1)     # [hw, bs, 32]
        mask_interp = mask_interp.flatten(1)      # [hw, bs]
        trans_feat = self.transformer_layer(feat, src_key_padding_mask=mask_interp, pos=pos)  
        trans_feat = trans_feat.permute(1, 2, 0).reshape(bs, c, h, w)  # [bs, c, h, w]

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
        self.iter = self.iter + 1
        if self.neptune_logger:
            if self.iter % self.save_check_iter+1==0:
                max_file = max((f for f in os.listdir(self.output_dir) if re.match(r'iter_\d+\.pth', f)), 
                key=lambda x: int(re.search(r'(\d+)', x).group()), default='No matching files found.')
                max_file_path = os.path.join(self.output_dir, max_file)
                self.model_version[f"model/weights_iter_{self.iter-1}"].upload(max_file_path)
                
        losses['model_outputs'] = output['reg_proposals']
        losses['model_anchors'] = output['anchors']
            
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
        del losses['model_outputs']
        loss, log_vars = self._parse_losses(losses, other_vars)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=data_batch['img'].shape[0])
        if self.neptune_logger:
            self.neptune_logger["train/loss"].append(loss.item())
            for item, value in log_vars.items():
                self.neptune_logger["train/"+item].append(value)

        return outputs
    
    def pred2lanes(self, pred):
        ys = np.array(self.y_steps, dtype=np.float32)
        lanes = []
        logits = []
        probs = []
        for lane in pred:
            lane_xs = lane[5:5 + self.anchor_len]
            lane_zs = lane[5 + self.anchor_len : 5 + 2 * self.anchor_len]
            lane_vis = (lane[5 + self.anchor_len * 2 : 5 + 3 * self.anchor_len]) > 0
            if (lane_vis).sum() < 2:
                continue
            lane_xs = lane_xs[lane_vis]
            lane_ys = ys[lane_vis]
            lane_zs = lane_zs[lane_vis]
            lanes.append(np.stack([lane_xs, lane_ys, lane_zs], axis=-1).tolist())
            logits.append(lane[5 + 3 * self.anchor_len:])
            probs.append(lane[1])
        return lanes, probs, logits

    def pred2apollosimformat(self, idx, pred):
        json_line = dict()
        pred_proposals = pred['proposals_list']
        pred_lanes, prob_lanes, logits_lanes = self.pred2lanes(pred_proposals)
        json_line["laneLines"] = pred_lanes
        json_line["laneLines_prob"]  = prob_lanes
        json_line["laneLines_logit"] = logits_lanes
        return json_line

    def format_results(self, predictions):
        all_predictions = []
        for idx in range(len(predictions)):
            result = self.pred2apollosimformat(idx, predictions[idx])
            save_result = {}
            lane_lines = []
            for k in range(len(result['laneLines'])):
                cate = int(np.argmax(result['laneLines_logit'][k][1:])) + 1
                prob = float(result['laneLines_prob'][k])
                lane_lines.append({'xyz': result['laneLines'][k], 'category': cate, 'laneLines_prob': prob})
            save_result['lane_lines'] = lane_lines
            all_predictions.append(save_result)
        return all_predictions
    
    def bench(self, pred_lanes, pred_category, gt_lanes, gt_visibility, gt_category):
        """
            Matching predicted lanes and ground-truth lanes in their IPM projection, ignoring z attributes.
            x error, y_error, and z error are all considered, although the matching does not rely on z
            The input of prediction and ground-truth lanes are in ground coordinate, x-right, y-forward, z-up
            The fundamental assumption is: 1. there are no two points from different lanes with identical x, y
                                              but different z's
                                           2. there are no two points from a single lane having identical x, y
                                              but different z's
            If the interest area is within the current drivable road, the above assumptions are almost always valid.

        :param pred_lanes: N X 2 or N X 3 lists depending on 2D or 3D
        :param gt_lanes: N X 2 or N X 3 lists depending on 2D or 3D
        :param raw_file: file path rooted in dataset folder
        :param gt_cam_height: camera height given in ground-truth data
        :param gt_cam_pitch: camera pitch given in ground-truth data
        :return:
        """
        # change this properly
        close_range_idx = np.where(self.y_samples > self.close_range)[0][0]

        r_lane, p_lane, c_lane = 0., 0., 0.
        x_error_close = []
        x_error_far = []
        z_error_close = []
        z_error_far = []

        # ======================== Added in latest version ===============================
        # only keep the visible portion
        gt_lanes = [prune_3d_lane_by_visibility(np.array(gt_lane), np.array(gt_visibility[k])) for k, gt_lane in
                    enumerate(gt_lanes)]
        gt_category = [gt_category[k] for k, lane in enumerate(gt_lanes) if lane.shape[0] > 1]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]
        # ======================== Added in latest version ===============================

        # only consider those pred lanes overlapping with sampling range
        pred_category = [pred_category[k] for k, lane in enumerate(pred_lanes)
                        if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]
        pred_lanes = [lane for lane in pred_lanes if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]

        pred_lanes = [prune_3d_lane_by_range(np.array(lane), self.x_min, self.x_max) for lane in pred_lanes]

        pred_category = [pred_category[k] for k, lane in enumerate(pred_lanes) if lane.shape[0] > 1]
        pred_lanes = [lane for lane in pred_lanes if lane.shape[0] > 1]

        # only consider those gt lanes overlapping with sampling range
        gt_category = [gt_category[k] for k, lane in enumerate(gt_lanes)
                        if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]
        gt_lanes = [lane for lane in gt_lanes if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]

        gt_lanes = [prune_3d_lane_by_range(np.array(lane), self.x_min, self.x_max) for lane in gt_lanes]

        gt_category = [gt_category[k] for k, lane in enumerate(gt_lanes) if lane.shape[0] > 1]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]

        cnt_gt = len(gt_lanes)
        cnt_pred = len(pred_lanes)

        gt_visibility_mat = np.zeros((cnt_gt, 100))
        pred_visibility_mat = np.zeros((cnt_pred, 100))

        # resample gt and pred at y_samples
        for i in range(cnt_gt):
            min_y = np.min(np.array(gt_lanes[i])[:, 1])
            max_y = np.max(np.array(gt_lanes[i])[:, 1])
            x_values, z_values, visibility_vec = resample_laneline_in_y(np.array(gt_lanes[i]), self.y_samples, out_vis=True)
            gt_lanes[i] = np.vstack([x_values, z_values]).T
            gt_visibility_mat[i, :] = np.logical_and(x_values >= self.x_min, np.logical_and(x_values <= self.x_max,
                                                     np.logical_and(self.y_samples >= min_y, self.y_samples <= max_y)))
            gt_visibility_mat[i, :] = np.logical_and(gt_visibility_mat[i, :], visibility_vec)

        for i in range(cnt_pred):
            # # ATTENTION: ensure y mono increase before interpolation: but it can reduce size
            # pred_lanes[i] = make_lane_y_mono_inc(np.array(pred_lanes[i]))
            # pred_lane = prune_3d_lane_by_range(np.array(pred_lanes[i]), self.x_min, self.x_max)
            min_y = np.min(np.array(pred_lanes[i])[:, 1])
            max_y = np.max(np.array(pred_lanes[i])[:, 1])
            x_values, z_values, visibility_vec = resample_laneline_in_y(np.array(pred_lanes[i]), self.y_samples, out_vis=True)
            pred_lanes[i] = np.vstack([x_values, z_values]).T
            pred_visibility_mat[i, :] = np.logical_and(x_values >= self.x_min, np.logical_and(x_values <= self.x_max,
                                                       np.logical_and(self.y_samples >= min_y, self.y_samples <= max_y)))
            pred_visibility_mat[i, :] = np.logical_and(pred_visibility_mat[i, :], visibility_vec)
            # pred_visibility_mat[i, :] = np.logical_and(x_values >= self.x_min, x_values <= self.x_max)

        # at least two-points for both gt and pred
        gt_lanes = [gt_lanes[k] for k in range(cnt_gt) if np.sum(gt_visibility_mat[k, :]) > 1]
        gt_category = [gt_category[k] for k in range(cnt_gt) if np.sum(gt_visibility_mat[k, :]) > 1]
        gt_visibility_mat = gt_visibility_mat[np.sum(gt_visibility_mat, axis=-1) > 1, :]
        cnt_gt = len(gt_lanes)

        pred_lanes = [pred_lanes[k] for k in range(cnt_pred) if np.sum(pred_visibility_mat[k, :]) > 1]
        pred_category = [pred_category[k] for k in range(cnt_pred) if np.sum(pred_visibility_mat[k, :]) > 1]
        pred_visibility_mat = pred_visibility_mat[np.sum(pred_visibility_mat, axis=-1) > 1, :]
        cnt_pred = len(pred_lanes)

        adj_mat = np.zeros((cnt_gt, cnt_pred), dtype=int)
        cost_mat = np.zeros((cnt_gt, cnt_pred), dtype=int)
        cost_mat.fill(1000)
        num_match_mat = np.zeros((cnt_gt, cnt_pred), dtype=float)
        x_dist_mat_close = np.zeros((cnt_gt, cnt_pred), dtype=float)
        x_dist_mat_close.fill(1000.)
        x_dist_mat_far = np.zeros((cnt_gt, cnt_pred), dtype=float)
        x_dist_mat_far.fill(1000.)
        z_dist_mat_close = np.zeros((cnt_gt, cnt_pred), dtype=float)
        z_dist_mat_close.fill(1000.)
        z_dist_mat_far = np.zeros((cnt_gt, cnt_pred), dtype=float)
        z_dist_mat_far.fill(1000.)

        # compute curve to curve distance
        for i in range(cnt_gt):
            for j in range(cnt_pred):
                x_dist = np.abs(gt_lanes[i][:, 0] - pred_lanes[j][:, 0])
                z_dist = np.abs(gt_lanes[i][:, 1] - pred_lanes[j][:, 1])

                # apply visibility to penalize different partial matching accordingly
                both_visible_indices = np.logical_and(gt_visibility_mat[i, :] >= 0.5, pred_visibility_mat[j, :] >= 0.5)
                both_invisible_indices = np.logical_and(gt_visibility_mat[i, :] < 0.5, pred_visibility_mat[j, :] < 0.5)
                other_indices = np.logical_not(np.logical_or(both_visible_indices, both_invisible_indices))
                
                euclidean_dist = np.sqrt(x_dist ** 2 + z_dist ** 2)
                euclidean_dist[both_invisible_indices] = 0
                euclidean_dist[other_indices] = self.dist_th

                # if np.average(euclidean_dist) < 2*self.dist_th: # don't prune here to encourage finding perfect match
                num_match_mat[i, j] = np.sum(euclidean_dist < self.dist_th) - np.sum(both_invisible_indices)
                adj_mat[i, j] = 1
                # ATTENTION: use the sum as int type to meet the requirements of min cost flow optimization (int type)
                # using num_match_mat as cost does not work?
                # make sure cost is not set to 0 when it's smaller than 1
                cost_ = np.sum(euclidean_dist)
                if cost_<1 and cost_>0:
                    cost_ = 1
                else:
                    cost_ = (cost_).astype(int)
                cost_mat[i, j] = cost_

                # use the both visible portion to calculate distance error
                if np.sum(both_visible_indices[:close_range_idx]) > 0:
                    x_dist_mat_close[i, j] = np.sum(
                        x_dist[:close_range_idx] * both_visible_indices[:close_range_idx]) / np.sum(
                        both_visible_indices[:close_range_idx])
                    z_dist_mat_close[i, j] = np.sum(
                        z_dist[:close_range_idx] * both_visible_indices[:close_range_idx]) / np.sum(
                        both_visible_indices[:close_range_idx])
                else:
                    x_dist_mat_close[i, j] = -1
                    z_dist_mat_close[i, j] = -1
                    

                if np.sum(both_visible_indices[close_range_idx:]) > 0:
                    x_dist_mat_far[i, j] = np.sum(
                        x_dist[close_range_idx:] * both_visible_indices[close_range_idx:]) / np.sum(
                        both_visible_indices[close_range_idx:])
                    z_dist_mat_far[i, j] = np.sum(
                        z_dist[close_range_idx:] * both_visible_indices[close_range_idx:]) / np.sum(
                        both_visible_indices[close_range_idx:])
                else:
                    x_dist_mat_far[i, j] = -1
                    z_dist_mat_far[i, j] = -1

        # solve bipartite matching vis min cost flow solver
        match_results = SolveMinCostFlow(adj_mat, cost_mat)
        match_results = np.array(match_results)

        # only a match with avg cost < self.dist_th is consider valid one
        match_gt_ids = []
        match_pred_ids = []
        match_num = 0
        if match_results.shape[0] > 0:
            for i in range(len(match_results)):
                if match_results[i, 2] < self.dist_th * self.y_samples.shape[0]:
                    match_num += 1
                    gt_i = match_results[i, 0]
                    pred_i = match_results[i, 1]
                    # consider match when the matched points is above a ratio
                    if num_match_mat[gt_i, pred_i] / np.sum(gt_visibility_mat[gt_i, :]) >= self.ratio_th:
                        r_lane += 1
                        match_gt_ids.append(gt_i)
                    if num_match_mat[gt_i, pred_i] / np.sum(pred_visibility_mat[pred_i, :]) >= self.ratio_th:
                        p_lane += 1
                        match_pred_ids.append(pred_i)
                    if pred_category != []:
                        if pred_category[pred_i] == gt_category[gt_i] or (pred_category[pred_i]==20 and gt_category[gt_i]==21):
                            c_lane += 1    # category matched num
                    x_error_close.append(x_dist_mat_close[gt_i, pred_i])
                    x_error_far.append(x_dist_mat_far[gt_i, pred_i])
                    z_error_close.append(z_dist_mat_close[gt_i, pred_i])
                    z_error_far.append(z_dist_mat_far[gt_i, pred_i])
        return r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num, x_error_close, x_error_far, z_error_close, z_error_far
    def eval(self, outputs, gt_3d_lanes, prob_th = 0.02):
        results = []
        for output in outputs:
                result = postprocess(output, anchor_len=self.anchor_len)
                results.append(result)
        results = self.format_results(results)
                
        laneline_stats = []
        laneline_x_error_close = []
        laneline_x_error_far = []
        laneline_z_error_close = []
        laneline_z_error_far = []
        for i, pred in enumerate(results):
            pred_lanelines = pred['lane_lines'].copy()
            pred_lanes = [np.array(lane['xyz']) for i, lane in enumerate(pred_lanelines)]
            pred_category = [int(lane['category']) for i, lane in enumerate(pred_lanelines)]
            pred_laneLines_prob = [np.array(lane['laneLines_prob']) for i, lane in enumerate(pred_lanelines)]

            # filter out probability
            pred_lanes = [pred_lanes[ii] for ii in range(len(pred_laneLines_prob)) if
                    pred_laneLines_prob[ii] > prob_th]
            pred_category = [pred_category[ii] for ii in range(len(pred_laneLines_prob)) if
                    pred_laneLines_prob[ii] > prob_th]
            pred_laneLines_prob = [pred_laneLines_prob[ii] for ii in range(len(pred_laneLines_prob)) if
                    pred_laneLines_prob[ii] > prob_th]

            gt = gt_3d_lanes
            if gt['original_xyz_lanes'][i] ==0:
                continue

            
            # evaluate lanelines
            cam_extrinsics = np.array(gt['original_extrinsics'])[i].cpu()
            cam_intrinsics = np.array(gt['original_instrinsics'])[i].cpu()
            # Re-calculate extrinsic matrix based on ground coordinate
            R_vg = np.array([[0, 1, 0],
                                [-1, 0, 0],
                                [0, 0, 1]], dtype=float)
            R_gc = np.array([[1, 0, 0],
                                [0, 0, 1],
                                [0, -1, 0]], dtype=float)
            cam_extrinsics[:3, :3] = np.matmul(np.matmul(
                                        np.matmul(np.linalg.inv(R_vg), cam_extrinsics[:3, :3]),
                                            R_vg), R_gc)
            gt_cam_height = cam_extrinsics[2, 3]
            gt_cam_pitch = 0

            cam_extrinsics[0:2, 3] = 0.0
            # cam_extrinsics[2, 3] = gt_cam_height

            try:
                gt_lanes_packed = gt['original_xyz_lanes'][i]
            except:
                print("error 'lane_lines' in gt: ", gt['file_path'])

            gt_lanes, gt_visibility, gt_category = [], [], []
            for j, gt_lane_packed in enumerate(gt_lanes_packed):
                # A GT lane can be either 2D or 3D
                # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
                lane = gt_lane_packed.cpu()
                lane_visibility = np.array(gt['original_visibility'][i][j].cpu())

                lane = np.vstack((lane, np.ones((1, lane.shape[1]))))
                cam_representation = np.linalg.inv(
                                        np.array([[0, 0, 1, 0],
                                                  [-1, 0, 0, 0],
                                                  [0, -1, 0, 0],
                                                  [0, 0, 0, 1]], dtype=float))
                lane = np.matmul(cam_extrinsics, np.matmul(cam_representation, lane))
                lane = lane[0:3, :].T

                gt_lanes.append(lane)
                gt_visibility.append(lane_visibility)
                gt_category.append(gt['original_categoties'][i][j].cpu())
            
            P_g2im = projection_g2im_extrinsic(cam_extrinsics, cam_intrinsics)


            # N to N matching of lanelines
            r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num, \
            x_error_close, x_error_far, \
            z_error_close, z_error_far = self.bench(pred_lanes,
                                                    pred_category, 
                                                    gt_lanes,
                                                    gt_visibility,
                                                    gt_category)
            laneline_stats.append(np.array([r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num]))
            # consider x_error z_error only for the matched lanes
            # if r_lane > 0 and p_lane > 0:
            laneline_x_error_close.extend(x_error_close)
            laneline_x_error_far.extend(x_error_far)
            laneline_z_error_close.extend(z_error_close)
            laneline_z_error_far.extend(z_error_far)
            recall = r_lane / (cnt_gt + 1e-6)
            precision = p_lane / (cnt_pred + 1e-6)
            f_score = 2 * recall * precision / (recall + precision + 1e-6)
            cate_acc = c_lane / (match_num + 1e-6)

        output_stats = []
        laneline_stats = np.array(laneline_stats)
        laneline_x_error_close = np.array(laneline_x_error_close)
        laneline_x_error_far = np.array(laneline_x_error_far)
        laneline_z_error_close = np.array(laneline_z_error_close)
        laneline_z_error_far = np.array(laneline_z_error_far)

        if np.sum(laneline_stats[:, 3])!= 0:
            R_lane = np.sum(laneline_stats[:, 0]) / (np.sum(laneline_stats[:, 3]))
        else:
            R_lane = np.sum(laneline_stats[:, 0]) / (np.sum(laneline_stats[:, 3]) + 1e-6)   # recall = TP / (TP+FN)
        if np.sum(laneline_stats[:, 4]) != 0:
            P_lane = np.sum(laneline_stats[:, 1]) / (np.sum(laneline_stats[:, 4]))
        else:
            P_lane = np.sum(laneline_stats[:, 1]) / (np.sum(laneline_stats[:, 4]) + 1e-6)   # precision = TP / (TP+FP)
        if np.sum(laneline_stats[:, 5]) != 0:
            C_lane = np.sum(laneline_stats[:, 2]) / (np.sum(laneline_stats[:, 5]))
        else:
            C_lane = np.sum(laneline_stats[:, 2]) / (np.sum(laneline_stats[:, 5]) + 1e-6)   # category_accuracy
        if R_lane + P_lane != 0:
            F_lane = 2 * R_lane * P_lane / (R_lane + P_lane)
        else:
            F_lane = 2 * R_lane * P_lane / (R_lane + P_lane + 1e-6)
        x_error_close_avg = np.average(laneline_x_error_close[laneline_x_error_close > -1 + 1e-6])
        x_error_far_avg = np.average(laneline_x_error_far[laneline_x_error_far > -1 + 1e-6])
        z_error_close_avg = np.average(laneline_z_error_close[laneline_z_error_close > -1 + 1e-6])
        z_error_far_avg = np.average(laneline_z_error_far[laneline_z_error_far > -1 + 1e-6])

        output_stats.append(F_lane)
        output_stats.append(R_lane)
        output_stats.append(P_lane)
        output_stats.append(C_lane)
        output_stats.append(x_error_close_avg)
        output_stats.append(x_error_far_avg)
        output_stats.append(z_error_close_avg)
        output_stats.append(z_error_far_avg)
        output_stats.append(np.sum(laneline_stats[:, 0]))   # 8
        output_stats.append(np.sum(laneline_stats[:, 1]))   # 9
        output_stats.append(np.sum(laneline_stats[:, 2]))   # 10
        output_stats.append(np.sum(laneline_stats[:, 3]))   # 11
        output_stats.append(np.sum(laneline_stats[:, 4]))   # 12
        output_stats.append(np.sum(laneline_stats[:, 5]))   # 13
        
        output_dict = {"F_lane": F_lane, "R_lane": R_lane, "P_lane": P_lane, "C_lane": C_lane, "x_error_close_avg": x_error_close_avg, \
            "x_error_far_avg": x_error_far_avg, "z_error_close_avg":z_error_close_avg, "z_error_far_avg":z_error_far_avg}

        return output_dict
        
        
    def val_step(self, data_batch, optimizer=None, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data_batch)
        nms_outputs = self.nms(losses[0]['model_outputs'], losses[0]['model_anchors'], self.test_cfg.nms_thres, 
                                  self.test_cfg.conf_threshold, refine_vis=self.test_cfg.refine_vis,
                                  vis_thresh=self.test_cfg.vis_thresh)
        del losses[0]['model_outputs'], losses[0]['model_anchors']
        loss, log_vars = self._parse_losses(losses)

        log_vars_ = dict()
        for loss_name, loss_value in log_vars.items():
            k = loss_name + '_val'
            log_vars_[k] = loss_value
            
        eval_metrics = self.eval(nms_outputs, data_batch)
        
        if self.neptune_logger:
            for item, value in eval_metrics.items():
                self.neptune_logger["val/"+item].append(value)

        outputs = dict(
            loss=loss,
            log_vars=log_vars_,
            num_samples=len(data_batch['img_metas']))
        
        if self.neptune_logger:
            self.neptune_logger["val/loss"].append(loss.item())
            for item, value in log_vars.items():
                self.neptune_logger["val/"+item].append(value)

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
        # for var_name, var_value in other_vars.items():
        #     log_vars[var_name] = var_value
        if type(losses)!=dict:
            losses = losses[0]
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