# --------------------------------------------------------
# Source code for Anchor3DLane
# Copyright (c) 2023 TuSimple
# @Time    : 2023/04/05
# @Author  : Shaofei Huang
# nowherespyfly@gmail.com
# --------------------------------------------------------

import json
import math
import os

import cv2
import mmcv
import numpy as np
import torch

from mmseg.datasets.tools.utils import projection_g2im, resample_laneline_in_y


class AnchorGenerator(object):
    """Normalized anchor coords"""
    def __init__(self, anchor_cfg, y_steps=None, x_min=None, x_max=None, y_max=100, norm=None):
        self.y_steps = y_steps
        if self.y_steps is None:
            self.y_steps = np.linspace(1, y_max, y_max)
        self.pitches = anchor_cfg['pitches']
        self.yaws = anchor_cfg['yaws']
        self.num_x = anchor_cfg['num_x']
        self.anchor_len = len(self.y_steps)
        self.x_min = x_min
        self.x_max = x_max
        self.y_max = y_max
        self.norm = norm
        self.start_z = anchor_cfg.get('start_z', 0)

    def generate_anchors(self):
        anchors = []
        starts = [x for x in np.linspace(self.x_min, self.x_max, num=self.num_x, dtype=np.float32)]
        idx = 0
        for start_x in starts:
            for pitch in self.pitches:
                for yaw in self.yaws:
                    anchor = self.generate_anchor(start_x, pitch, yaw, start_z = self.start_z)
                    if anchor is not None:
                        anchors.append(anchor)
                        idx += 1
        self.anchor_num = len(anchors)
        print("anchor:", len(anchors))
        anchors = np.array(anchors)
        return anchors

    def generate_anchor(self, start_x, pitch, yaw, start_z=0, cut=True):
        # anchor [pos_score, neg_score, start_y, end_y, d, x_coords * 10, z_coords * 10, vis_coords * 10]
        anchor = np.zeros(2 + 2 + 1 + self.anchor_len * 3, dtype=np.float32)
        pitch = pitch * math.pi / 180.  # degrees to radians
        yaw = yaw * math.pi / 180.
        anchor[2] = 0
        anchor[3] = 1
        anchor[5:5+self.anchor_len] = start_x + self.y_steps * math.tan(yaw)
        anchor[5+self.anchor_len:5+self.anchor_len*2] = start_z + self.y_steps * math.tan(pitch)
        anchor_vis = np.logical_and(anchor[5:5+self.anchor_len] > self.x_min, anchor[5:5+self.anchor_len] < self.x_max)
        if cut:
            if sum(anchor_vis) / self.anchor_len < 0.5:
                return None
        return anchor
    
class AnchorGenerator_torch(object):
    def __init__(self, anchor_cfg, y_steps=None, x_min=None, x_max=None, y_max=100, norm=None):
        self.y_steps = y_steps
        if y_steps is None:
            self.y_steps = torch.linspace(1, y_max, y_max, dtype=torch.float32)
        else:
            self.y_steps = torch.tensor(y_steps, dtype=torch.float32)
        self.pitches = anchor_cfg['pitches']
        self.yaws = anchor_cfg['yaws']
        self.num_x = anchor_cfg['num_x']
        self.anchor_len = len(self.y_steps)
        if 'anchor_num' not in anchor_cfg.keys():
            self.num_yaws = len(self.yaws)
            self.num_pitches = len(self.pitches)
            self.anchor_num = self.num_x * self.num_yaws * self.num_pitches
        self.xs = torch.linspace(-1, 1, self.num_x, dtype=torch.float32)
        self.y_steps = torch.tensor(self.y_steps, dtype=torch.float32)
        self.pitches = torch.tensor(self.pitches, dtype=torch.float32) / 180.
        self.yaws = torch.tensor(self.yaws, dtype=torch.float32) / 180.

        self.x_min = x_min
        self.x_max = x_max
        self.y_max = y_max
        self.norm = norm
        self.start_z = anchor_cfg.get('start_z', 0)

    def generate_anchors(self, xs=None, yaws=None, pitches=None, cut=True):
        # xs: [B, N]
        anchors = torch.zeros(self.anchor_num, 5 + self.anchor_len * 3)  # [N, L']
        if xs is not None:
            anchors = anchors.to(xs.device)
            y_steps = self.y_steps[None, :].repeat(self.anchor_num, 1).to(xs.device)
            xs = torch.clamp(xs, -1, 1)
            yaws = torch.clamp(yaws, -1, 1)
            pitches = torch.clamp(pitches, -1, 1)
        else:
            xs = self.xs[:, None, None].repeat(1, self.num_pitches, self.num_yaws).flatten(0, 2)    # [X, 1, 1, 1]
            pitches = self.pitches[None, :, None].repeat(self.num_x, 1, self.num_yaws).flatten(0, 2)  # [1, P, 1, 1]
            yaws = self.yaws[None, None, :].repeat(self.num_x, self.num_pitches, 1).flatten(0, 2)   # [1, 1, Y, 1]
            y_steps = self.y_steps[None, :].repeat(self.num_x * self.num_pitches * self.num_yaws, 1)  # [1, 1, 1, L]
        anchors[..., 2] = yaws   # yaw
        anchors[..., 3] = pitches   # pitch
        anchors[..., 4] = xs   # xs
        anchors[..., 5:5+self.anchor_len] = (xs[..., None] + 1) / 2 * (self.x_max - self.x_min) + self.x_min  + (y_steps - 1) * torch.tan(yaws[..., None] * math.pi)
        anchors[..., 5+self.anchor_len:5+self.anchor_len*2] = self.start_z + (y_steps - 1) * torch.tan(pitches[..., None] * math.pi)
        if cut:
            anchor_vis = torch.logical_and(anchors[:, 5:5+self.anchor_len] > self.x_min, anchors[:, 5:5+self.anchor_len] < self.x_max)
            valid = (anchor_vis.sum(1) / self.anchor_len) > 0.5  # [N]
            anchors = anchors[valid]
            self.anchor_num = anchors.shape[0]
            self.y_steps = self.y_steps[:self.anchor_num]
        # anchors[:, 0] = valid.float()
        return anchors

    def generate_anchors_batch(self, xs=None, yaws=None, pitches=None):
        # xs: [B, N]
        bs, N, = xs.shape
        anchors = torch.zeros(bs, N, 5 + self.anchor_len * 3, device=xs.device)  # [N, L']
        y_steps = self.y_steps[None, None, :].repeat(bs, N, 1).to(xs.device)
        xs = torch.clamp(xs, -1, 1)
        yaws = torch.clamp(yaws, -1, 1)
        pitches = torch.clamp(pitches, -1, 1)
        anchors[..., 2] = yaws   # yaw
        anchors[..., 3] = pitches   # pitch
        anchors[..., 4] = xs   # xs
        anchors[..., 5:5+self.anchor_len] = (xs[..., None] + 1) / 2 * (self.x_max - self.x_min) + self.x_min  + (y_steps - 1) * torch.tan(yaws[..., None] * math.pi)
        anchors[..., 5+self.anchor_len:5+self.anchor_len*2] = self.start_z + (y_steps - 1) * torch.tan(pitches[..., None] * math.pi)
        return anchors


