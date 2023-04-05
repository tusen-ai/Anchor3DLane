# --------------------------------------------------------
# Source code for Anchor3DLane
# Copyright (c) 2023 TuSimple
# @Time    : 2023/04/05
# @Author  : Shaofei Huang
# nowherespyfly@gmail.com
# --------------------------------------------------------

import os
import numpy as np
import json
import math
import torch
import cv2
import mmcv
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

