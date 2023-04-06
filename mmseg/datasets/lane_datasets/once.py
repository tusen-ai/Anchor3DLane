# --------------------------------------------------------
# Source code for Anchor3DLane
# Copyright (c) 2023 TuSimple
# @Time    : 2023/04/05
# @Author  : Shaofei Huang
# nowherespyfly@gmail.com
# --------------------------------------------------------

import os
import json
import sys
import random
import warnings
import pickle
import tqdm


import cv2
import numpy as np
from tabulate import tabulate
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from copy import deepcopy

import mmcv

from ..tools.utils import *
from ..tools import eval_once
from ..builder import DATASETS
from ..pipelines import Compose

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

DARK_GREEN = (115, 181, 34)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
PINK = (180, 105, 255)
CYAN = (255, 128, 0)

CHOCOLATE = (30, 105, 210)
PEACHPUFF = (185, 218, 255)
STATEGRAY = (255, 226, 198)

GT_COLOR = [PINK, CYAN, ORANGE, YELLOW, BLUE]
PRED_COLOR = [RED, GREEN, DARK_GREEN, PURPLE, CHOCOLATE, PEACHPUFF, STATEGRAY]
PRED_HIT_COLOR = GREEN
PRED_MISS_COLOR = RED
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


@DATASETS.register_module()
class ONCEDataset(Dataset):
    def __init__(self, 
                 pipeline,
                 data_root,
                 img_dir='raw_data', 
                 img_suffix='.jpg',
                 data_list='train.txt',
                 y_steps = [  2,  5,  8,  10,  15,  20,  25,  30,  40,  50],
                 test_mode=False,
                 dataset_config=None,
                 test_config=None,
                 is_resample=True):
        self.pipeline = Compose(pipeline)
        self.data_root = data_root
        self.img_dir = os.path.join(data_root, img_dir)
        self.img_suffix = img_suffix
        self.test_mode = test_mode
        self.metric = 'default'
        self.is_resample = is_resample
        self.dataset_config = dataset_config
        self.test_config = test_config
        self.data_list = os.path.join(data_root, 'data_lists', data_list)
        self.cache_dir = os.path.join(data_root, 'cache_dense')
        self.eval_dir = os.path.join(data_root, 'annotations', 'val')
        self.eval_file = os.path.join(data_root, 'data_splits', 'val.json')
        
        print('is_resample: {}'.format(is_resample))
        inp_h, inp_w = dataset_config['input_size']

        # dataset parameters
        self.no_3d = False
        self.no_centerline = True

        self.h_org  = 1020
        self.w_org  = 1920
        self.org_h  = 1020
        self.org_w  = 1920
        self.h_crop = 0
        self.crop_y = 0

        # parameters related to service network
        self.h_net = inp_h
        self.w_net = inp_w
        self.resize_h = inp_h
        self.resize_w = inp_w
        self.ipm_h = 416 #208
        self.ipm_w = 284 #128

        self.top_view_region = np.array([[-45, 50], [45, 50], [-45, 0], [45, 0]])
        self.H_crop_ipm = homography_crop_resize([self.h_org, self.w_org], self.h_crop, [self.h_net, self.w_net])
        self.H_crop_im  = homography_crop_resize([self.h_org, self.w_org], self.h_crop, [self.h_org, self.w_org])
        self.H_crop_resize_im = homography_crop_resize([self.h_org, self.w_org], self.h_crop, [self.resize_h, self.resize_w])
        self.H_ipm2g = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [self.ipm_w - 1, 0], [0, self.ipm_h - 1], [self.ipm_w - 1, self.ipm_h - 1]]),
            np.float32(self.top_view_region))
        self.fix_cam = False

        x_min = self.top_view_region[0, 0] 
        x_max = self.top_view_region[1, 0] 
        self.x_min = x_min  
        self.x_max = x_max  
        self.anchor_y_steps = np.array(y_steps, dtype=np.float)
        self.anchor_len = len(self.anchor_y_steps)
        self.y_min = self.top_view_region[2, 1]
        self.y_max = self.top_view_region[0, 1]

        self.num_types = 1
        self.num_categories = 2

        self.img_w, self.img_h = self.h_org, self.w_org
        self.max_lanes = 8  
        self.to_tensor = ToTensor()

        self.R_c2g = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
        self.R_g2c = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)

        self.load_annotations()


    def load_annotations(self):
        print('Now loading annotations...')
        self.img_infos = []
        with open(self.data_list, 'r') as anno_obj:
            all_ids = [s.strip() for s in anno_obj.readlines()]
            for k, id in enumerate(all_ids):
                if id.startswith('/'):
                    id = id[1:]
                anno = {'filename': os.path.join(self.img_dir, id),
                        'anno_file': os.path.join(self.cache_dir, id.replace('.jpg', '.pkl'))}
                self.img_infos.append(anno)
        print("after load annotation")
        print("find {} samples in {}.".format(len(self.img_infos), self.data_list))

    def __getitem__(self, idx, transform=False):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        results = self.img_infos[idx].copy()
        results['img_info'] = {}
        results['img_info']['filename'] = results['filename']
        results['ori_filename'] = results['filename']
        results['ori_shape'] = (self.h_org, self.w_org)
        results['flip'] = False
        results['flip_direction'] = None
        with open(results['anno_file'], 'rb') as f:
            obj = pickle.load(f)
            results.update(obj)

        extrinsic = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
        results['gt_project_matrix'] = np.matmul(results['gt_camera_intrinsic'], extrinsic)
        results['img_metas'] = {'ori_shape':results['ori_shape']}
        results = self.pipeline(results)
        return results

    def pred2lanes(self, pred, is_resample=False):
        ys = np.array(self.anchor_y_steps, dtype=np.float32)
        lanes = []
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
            if is_resample:
                min_y = np.min(lane_ys)
                max_y = np.max(lane_ys)
                y_resamples = np.linspace(1, 50, 50)
                lane_vis = np.zeros(50, dtype=np.bool)
                lane_xs, lane_zs, visibility_vec = resample_laneline_in_y(np.stack([lane_xs, ys, lane_zs], axis=-1), y_resamples, out_vis=True)
                lane_vis = np.logical_and(lane_xs >= self.x_min, np.logical_and(lane_xs <= self.x_max,
                                                        np.logical_and(y_resamples >= min_y, y_resamples <= max_y)))
                lane_vis = np.logical_and(lane_vis, visibility_vec)
        
            lane_stack = np.stack([lane_xs, lane_ys, lane_zs], axis=-1)   # [N, 3]
            lane_stack = np.matmul(self.R_g2c, lane_stack.T)   # [3, num_p]
            lane_stack = lane_stack.T
            lane_stack = lane_stack.tolist()   # [nump, 3]
            lane_stack = lane_stack[::-1]
            lanes.append(lane_stack)
            probs.append(lane[5 + 3 * self.anchor_len:])
        return lanes, probs

    def pred2format(self, idx, pred):
        old_anno = self.img_infos[idx]
        json_line = dict()
        pred_proposals = pred['proposals_list']
        pred_lanes, prob_lanes = self.pred2lanes(pred_proposals)
        json_line["laneLines"] = pred_lanes
        json_line["laneLines_prob"]  = prob_lanes
        filename = old_anno['filename']
        json_line['raw_file'] = '/'.join(filename.split('/')[-3:])
        return json_line

    def format_results(self, predictions, result_dir, prob_th=0.1):
        merge_results = []
        print("saving results...")
        for idx in tqdm.tqdm(range(len(predictions))):
            result = self.pred2format(idx, predictions[idx])
            lane_lines = []
            save_results = {}
            for k in range(len(result['laneLines'])):
                cate = int(np.argmax(result['laneLines_prob'][k]))
                cate = 1
                prob = result['laneLines_prob'][k][1]
                if cate == 0 or prob < prob_th:
                    continue
                lane_lines.append({'points': result['laneLines'][k], 'score': float(prob)})
            filename = '/'.join(self.img_infos[idx]['filename'].split('/')[-3:]).replace('.jpg', '.json')
            filename = os.path.join(result_dir, filename)
            mmcv.mkdir_or_exist(os.path.dirname(filename))
            with open(filename, 'w') as f:
                json.dump({'lanes':lane_lines}, f)
            save_results['laneLines'] = lane_lines
            save_results['file_path'] = result['raw_file']
            merge_results.append(save_results)
        with open(os.path.join(result_dir, 'prediction_3d.json'), 'w') as f:
            for item in merge_results:
                s = json.dumps(item)
                f.write(s+'\n')
        print("save merged results at ", os.path.join(result_dir, 'prediction_3d.json'))

    def eval(self, pred_dir):
        evaluator = eval_once.LaneEval()
        eval_stats = evaluator.lane_evaluation(pred_dir, self.eval_dir, self.test_config)
        eval_string = eval_stats.get_string()
        return eval_string

    def __len__(self):
        return len(self.img_infos)

    def project_with_intricit(self, P, x, y, z):
        u, v, dep = np.matmul(P, np.vstack((x, y, z)))
        u = u / dep
        v = v / dep
        return u, v