# --------------------------------------------------------
# Source code for Anchor3DLane
# Copyright (c) 2023 TuSimple
# @Time    : 2023/04/05
# @Author  : Shaofei Huang
# nowherespyfly@gmail.com
# --------------------------------------------------------

from re import L
import os
import json
import pickle

import tqdm
import pdb

import cv2
import numpy as np
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from copy import deepcopy
from scipy.interpolate import interp1d

from ..tools.utils import *
from ..tools import eval_openlane
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
class OpenlaneDataset(Dataset):
    def __init__(self, 
                 pipeline,
                 data_root,
                 img_dir='images', 
                 img_suffix='.jpg',
                 data_list='training.txt',
                 test_list=None,
                 test_mode=False,
                 dataset_config=None,
                 y_steps = [  5,  10,  15,  20,  30,  40,  50,  60,  80,  100],
                 is_resample=True, 
                 visibility=False,
                 no_cls=False):
        self.pipeline = Compose(pipeline)
        self.data_root = data_root
        self.img_dir = os.path.join(data_root, img_dir)
        self.img_suffix = img_suffix
        self.test_mode = test_mode
        self.metric = 'default'
        self.is_resample = is_resample
        self.dataset_config = dataset_config
        self.data_list = os.path.join(data_root, 'data_lists', data_list)
        self.cache_dir = os.path.join(data_root, 'cache_dense')
        self.eval_file = os.path.join(data_root, 'data_splits', 'validation.json')  
        self.visibility = visibility
        self.no_cls = no_cls
        
        print('is_resample: {}'.format(is_resample))
        inp_h, inp_w = dataset_config['input_size']

        self.h_org  = 1280
        self.w_org  = 1920
        self.org_h  = 1280
        self.org_w  = 1920
        self.h_crop = 0
        self.crop_y = 0

        # parameters related to service network
        self.h_net = inp_h
        self.w_net = inp_w
        self.resize_h = inp_h
        self.resize_w = inp_w
        self.ipm_h = 208  # 26
        self.ipm_w = 128  # 16

        self.top_view_region = np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]])
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

        x_min = self.top_view_region[0, 0]  # -10
        x_max = self.top_view_region[1, 0]  # 10
        self.x_min = x_min  # -10
        self.x_max = x_max  # 10
        self.anchor_y_steps = np.array(y_steps, dtype=np.float)
        self.anchor_len = len(self.anchor_y_steps)

        self.y_min = self.top_view_region[2, 1]
        self.y_max = self.top_view_region[0, 1]
        if self.is_resample:
            self.gflatYnorm = self.anchor_y_steps[-1]
            self.gflatZnorm = 10
            self.gflatXnorm = 30
        else:
            self.gflatYnorm = 200
            self.gflatZnorm = 1
            self.gflatXnorm = 20

        self.num_types = 1
        self.num_categories = 21
        if self.is_resample:
            self.sample_hz = 1
        else:
            self.sample_hz = 4

        self.img_w, self.img_h = self.h_org, self.w_org
        self.max_lanes = 25
        self.normalize = True
        self.to_tensor = ToTensor()

        if test_list is not None:
            self.test_list =  os.path.join(self.data_root, test_list)
        else:
            self.test_list = None

        self.load_annotations()

    def load_annotations(self):
        print('Now loading annotations...')
        self.img_infos = []
        with open(self.data_list, 'r') as anno_obj:
            all_ids = [s.strip() for s in anno_obj.readlines()]
            for k, id in enumerate(all_ids):
                anno = {'filename': os.path.join(self.img_dir, id + self.img_suffix),
                        'anno_file': os.path.join(self.cache_dir, id + '.pkl')}
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
        if self.no_cls:
            results['gt_3dlanes'][:, 1] = results['gt_3dlanes'][:, 1] > 0
        results['img_metas'] = {'ori_shape':results['ori_shape']}
        results['gt_project_matrix'] = projection_g2im_extrinsic(results['gt_camera_extrinsic'], results['gt_camera_intrinsic'])
        results['gt_homography_matrix'] = homography_g2im_extrinsic(results['gt_camera_extrinsic'], results['gt_camera_intrinsic'])
        results = self.pipeline(results)
        return results

    def pred2lanes(self, pred):
        ys = np.array(self.anchor_y_steps, dtype=np.float32)
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
        old_anno = self.img_infos[idx]
        filename = old_anno['filename']
        json_line = dict()
        pred_proposals = pred['proposals_list']
        pred_lanes, prob_lanes, logits_lanes = self.pred2lanes(pred_proposals)
        json_line['raw_file'] = '/'.join(filename.split('/')[-3:])
        json_line["laneLines"] = pred_lanes
        json_line["laneLines_prob"]  = prob_lanes
        json_line["laneLines_logit"] = logits_lanes
        return json_line

    def format_results(self, predictions, filename):
        with open(filename, 'w') as jsonFile:
            for idx in tqdm.tqdm(range(len(predictions))):
                result = self.pred2apollosimformat(idx, predictions[idx])
                save_result = {}
                save_result['file_path'] = result['raw_file']
                lane_lines = []
                for k in range(len(result['laneLines'])):
                    cate = int(np.argmax(result['laneLines_logit'][k][1:])) + 1
                    prob = float(result['laneLines_prob'][k])
                    lane_lines.append({'xyz': result['laneLines'][k], 'category': cate, 'laneLines_prob': prob})
                save_result['lane_lines'] = lane_lines
                json.dump(save_result, jsonFile)
                jsonFile.write('\n')
        print("save results to ", filename)

    def eval(self, pred_filename, prob_th=0.5):
        evaluator = eval_openlane.OpenLaneEval(self)
        pred_lines = open(pred_filename).readlines()
        json_pred = [json.loads(line) for line in pred_lines]
        json_gt = [json.loads(line) for line in open(self.eval_file).readlines()]
        if len(json_gt) != len(json_pred):
            print("gt len:", len(json_gt))
            print("pred len:", len(json_pred))
            # raise Exception('We do not get the predictions of all the test tasks')
        if self.test_list is not None:
            test_list = [s.strip().split('.')[0] for s in open(self.test_list, 'r').readlines()]
            json_pred = [s for s in json_pred if s['file_path'][:-4] in test_list]
            json_gt = [s for s in json_gt if s['file_path'][:-4] in test_list]
        gts = {l['file_path']: l for l in json_gt}
        eval_stats = evaluator.bench_one_submit(json_pred, gts, prob_th=prob_th)
        eval_results = {}
        eval_results['F_score'] = eval_stats[0]
        eval_results['recall'] = eval_stats[1]
        eval_results['precision'] = eval_stats[2]
        eval_results['cate_acc'] = eval_stats[3]
        eval_results['x_error_close'] = eval_stats[4]
        eval_results['x_error_far'] = eval_stats[5]
        eval_results['z_error_close'] = eval_stats[6]
        eval_results['z_error_far'] = eval_stats[7]
        return eval_results

    def __len__(self):
        return len(self.img_infos)

    def _get_img_heigth(self, path):
        return 1280

    def _get_img_width(self, path):
        return 1920
