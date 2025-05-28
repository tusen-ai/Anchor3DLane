# --------------------------------------------------------
# Source code for Anchor3DLane
# Copyright (c) 2023 TuSimple
# @Time    : 2023/04/05
# @Author  : Shaofei Huang
# nowherespyfly@gmail.com
# --------------------------------------------------------

import pdb
from collections.abc import Sequence
from os import path as osp

import cv2
import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from ..builder import PIPELINES
from .formatting import to_tensor


@PIPELINES.register_module()
class LaneFormat(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img"
    and other lane data. These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if len(img.shape) > 3:
                # [H, W, 3, N] -> [3, H, W, N]
                img = np.ascontiguousarray(img.transpose(2, 0, 1, 3))
            else:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        if 'gt_3dlanes' in results:
            results['gt_3dlanes'] = DC(to_tensor(results['gt_3dlanes'].astype(np.float32)))
        if 'gt_2dlanes' in results:
            results['gt_2dlanes'] = DC(to_tensor(results['gt_2dlanes'].astype(np.float32)))
        if 'gt_camera_extrinsic' in results:
            results['gt_camera_extrinsic'] = DC(to_tensor(results['gt_camera_extrinsic'][None, ...].astype(np.float32)), stack=True)
        if 'gt_camera_intrinsic' in results:
            results['gt_camera_intrinsic'] = DC(to_tensor(results['gt_camera_intrinsic'][None, ...].astype(np.float32)), stack=True)
        if 'gt_project_matrix' in results:
            results['gt_project_matrix'] = DC(to_tensor(results['gt_project_matrix'][None, ...].astype(np.float32)), stack=True)
        if 'gt_homography_matrix' in results:
            results['gt_homography_matrix'] = DC(to_tensor(results['gt_homography_matrix'][None, ...].astype(np.float32)), stack=True)
        if 'gt_camera_pitch' in results:
            results['gt_camera_pitch'] = DC(to_tensor([results['gt_camera_pitch']]))
        if 'gt_camera_height' in results:
            results['gt_camera_height'] = DC(to_tensor([results['gt_camera_height']]))
        if 'prev_poses' in results:
            results['prev_poses'] = DC(to_tensor(np.stack(results['prev_poses'], axis=0).astype(np.float32)), stack=True)  # [Np, 3, 4]
        if 'mask' in results:
            results['mask'] = DC(to_tensor(results['mask'][None, ...].astype(np.float32)), stack=True)
        if 'voxels' in results:
            results['num_voxels'] = DC(to_tensor(results['voxels']['num_voxels']))
            results['num_points'] = DC(to_tensor(results['voxels']['num_points']))
            results['shape'] = DC(to_tensor(results['voxels']['shape']))
            results['coordinates'] = DC(to_tensor(results['voxels']['coordinates']))
            results['voxels'] = DC(to_tensor(results['voxels']['voxels']))
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class MaskGenerate(object):
    def __init__(self, input_size):
        self.input_size = input_size 

    def __call__(self, results):
        mask  = np.ones((self.input_size[0], self.input_size[1]), dtype=np.bool)
        mask = np.logical_not(mask)
        results['mask'] = mask
        return results
    
@PIPELINES.register_module()
class AngleCalculate(object):
    def __init__(self, start_idx=5, anchor_gt_len=200, anchor_len=100):
        self.start_idx = start_idx
        self.anchor_len = anchor_len
        self.anchor_gt_len = anchor_gt_len
        self.y_steps = np.linspace(1, anchor_len, anchor_len)
        self.x_range = 30.0
    def __call__(self, results):
        lanes = results['gt_3dlanes']
        for idx, l in enumerate(lanes):
            if l[1] > 0:
                # TODO: replace anchor len
                vis = l[405:505]
                if (vis > 0.5).sum() < 2:
                    lanes[idx][2] = 0
                    lanes[idx][3] = 0
                    lanes[idx][4] = min(max(l[5] / self.x_range, -1), 1)
                else:
                    xs = l[5:105][vis > 0.5] 
                    ys = self.y_steps[vis > 0.5]
                    zs = l[205:305][vis > 0.5]
                    yaw = (np.arctan((xs[1:] - xs[0]) / (ys[1:] - ys[0])) / np.pi).mean()
                    pitch = (np.arctan((zs[1:] - zs[0]) / (ys[1:] - ys[0])) / np.pi).mean()
                    lanes[idx][2] = yaw
                    lanes[idx][3] = pitch
                    lanes[idx][4] = min(max(l[5] / self.x_range, -1), 1)
        results['gt_3dlanes'] = lanes
        return results