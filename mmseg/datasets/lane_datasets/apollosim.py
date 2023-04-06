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
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from ..tools.utils import *
from ..tools import eval_apollosim
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
class APOLLOSIMDataset(Dataset):
    def __init__(self, 
                 pipeline,
                 data_root,
                 img_dir='images', 
                 img_suffix='.jpg',
                 data_list='train.txt',
                 y_steps=[  5,  10,  15,  20,  30,  40,  50,  60,  80,  100],
                 split='standard',
                 test_mode=False,
                 dataset_config=None,
                 is_resample=True):
        self.pipeline = Compose(pipeline)
        self.data_root = data_root
        self.img_dir = os.path.join(data_root, img_dir)
        self.img_suffix = img_suffix
        self.test_mode = test_mode
        self.metric = 'default'
        self.is_resample = is_resample
        self.dataset_config = dataset_config
        self.data_list = os.path.join(data_root, 'data_lists', split, data_list)
        self.eval_file = os.path.join(data_root, 'data_splits', split, 'test.json') 
        self.cache_dir = os.path.join(data_root, 'cache_dense')
        
        inp_h, inp_w = dataset_config['input_size']

        # dataset parameters
        self.no_3d = False
        self.no_centerline = True

        self.h_org  = 1080
        self.w_org  = 1920
        self.org_h  = 1080
        self.org_w  = 1920
        self.h_crop = 0
        self.crop_y = 0

        # parameters related to service network
        self.h_net = inp_h
        self.w_net = inp_w
        self.resize_h = inp_h
        self.resize_w = inp_w
        self.ipm_h = 416  # 208
        self.ipm_w = 284  # 128
        self.top_view_region = np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]])
        self.K = np.array([[2015., 0., 960.], [0., 2015., 540.], [0., 0., 1.]])
        self.H_crop_ipm = homography_crop_resize([self.h_org, self.w_org], self.h_crop, [self.h_net, self.w_net])
        self.H_crop_im  = homography_crop_resize([self.h_org, self.w_org], self.h_crop, [self.h_org, self.w_org])
        self.H_crop_resize_im = homography_crop_resize([self.h_org, self.w_org], self.h_crop, [self.resize_h, self.resize_w])
        self.H_g2side = cv2.getPerspectiveTransform(
            np.float32([[0, -10], [0, 10], [100, -10], [100, 10]]),
            np.float32([[0, 300], [0, 0], [300, 300], [300, 0]]))
        # org2resized+cropped
        self.H_ipm2g = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [self.ipm_w - 1, 0], [0, self.ipm_h - 1], [self.ipm_w - 1, self.ipm_h - 1]]),
            np.float32(self.top_view_region))
        self.fix_cam = False

        x_min = self.top_view_region[0, 0]  # -10
        x_max = self.top_view_region[1, 0]  # 10
        self.x_min = x_min  # -10
        self.x_max = x_max  # 10
        self.anchor_y_steps = y_steps
        self.anchor_len = len(self.anchor_y_steps)
        self.y_min = self.top_view_region[2, 1]
        self.y_max = self.top_view_region[0, 1]
        self.gflatYnorm = self.anchor_y_steps[-1]
        self.gflatZnorm = 10
        self.gflatXnorm = 30

        self.pitch = 3  # pitch angle of camera to ground in centi degree
        self.cam_height = 1.55  # height of camera in meters

        if self.no_centerline:  # False
            self.num_types = 1
        else:
            self.num_types = 3
        if self.is_resample:
            self.sample_hz = 1
        else:
            self.sample_hz = 4

        self.max_lanes = 7
        self.img_w, self.img_h = self.h_org, self.w_org  # apollo sim original image resolution
        self.normalize = True
        self.to_tensor = ToTensor()

        self.load_annotations()

    def load_annotations(self):
        print('Now loading annotations...')
        self.img_infos = []
        with open(self.data_list, 'r') as anno_obj:
            all_ids = [s.strip() for s in anno_obj.readlines()]  # images/03/0000712.jpg
            for k, id in tqdm.tqdm(enumerate(all_ids)):
                anno = {'filename': os.path.join(self.data_root, id),
                        'anno_file': os.path.join(self.cache_dir, id[7:].replace('.jpg', '.pkl'))}
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
        # intrinsic and extrinsic
        results['gt_project_matrix'] = projection_g2im(results['gt_camera_pitch'], results['gt_camera_height'], self.K)
        results['img_metas'] = {'ori_shape':results['ori_shape']}
        return self.pipeline(results)

    def pred2lanes(self, pred):
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
            lanes.append(np.stack([lane_xs, lane_ys, lane_zs], axis=-1).tolist())
            probs.append(float(lane[-1]))

        return lanes, probs

    def pred2apollosimformat(self, idx, pred, proposal_key = 'proposals_list'):
        old_anno = self.img_infos[idx]
        filename = old_anno['filename']
        json_line = dict()
        pred_proposals = pred[proposal_key]
        pred_lanes, prob_lanes = self.pred2lanes(pred_proposals)
        json_line['raw_file'] = '/'.join(filename.split('/')[-3:])
        json_line["laneLines"] = pred_lanes
        json_line["laneLines_prob"]  = prob_lanes
        return json_line

    def format_results(self, predictions, filename):
        print("Writing results to", filename)
        with open(filename, 'w') as jsonFile:
            for idx in tqdm.tqdm(range(len(predictions))):
                json_line = self.pred2apollosimformat(idx, predictions[idx])
                json.dump(json_line, jsonFile)
                jsonFile.write('\n')

    def eval(self, pred_filename):
        evaluator = eval_apollosim.LaneEval(self)
        eval_stats_pr = evaluator.bench_one_submit_varying_probs(pred_filename, self.eval_file)
        max_f_prob = eval_stats_pr['max_F_prob_th']
        eval_stats = evaluator.bench_one_submit(pred_filename, self.eval_file, prob_th=max_f_prob)
        print("Metrics: F-score,    AP, x error (close), x error (far), z error (close), z error (far), Rec  , Pre")
        print("Laneline:{:.3}, {:.3},   {:.3},           {:.3},         {:.3},           {:.3},     {:.3},    {:.3}".format(
            eval_stats[0], eval_stats_pr['laneline_AP'], eval_stats[3], eval_stats[4], eval_stats[5], eval_stats[6], 
            eval_stats[1], eval_stats[2]))
        result = {
            'AP':  eval_stats_pr['laneline_AP'],
            'F_score': eval_stats[0],
            'x_error_close': eval_stats[3],
            'x_error_far': eval_stats[4],
            'z_error_close': eval_stats[5],
            'z_error_far': eval_stats[6],
            'x_error_close_all': eval_stats[7],
            'x_error_far_all': eval_stats[8],
            'z_error_close_all': eval_stats[9],
            'z_error_far_all': eval_stats[10],
        }
        return result

    def __len__(self):
        return len(self.img_infos)

    def _get_img_heigth(self, path):
        return 1080

    def _get_img_width(self, path):
        return 1920