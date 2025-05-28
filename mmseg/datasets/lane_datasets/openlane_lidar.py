# --------------------------------------------------------
# Source code for Anchor3DLane
# Copyright (c) 2025 TuSimple
# @Time    : 2025/05/07
# @Author  : Shaofei Huang
# nowherespyfly@gmail.com
# --------------------------------------------------------

import json
import os
import pdb
import pickle
from copy import deepcopy
from re import L

import cv2
import numpy as np
import torchvision.transforms.functional as F
import tqdm
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from ..builder import DATASETS
from ..pipelines import Compose
from ..tools import eval_openlane
from ..tools.utils import *
from .openlane import OpenlaneDataset

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
class OpenlaneLidarDataset(OpenlaneDataset):
    def __init__(self, 
                 pipeline,
                 data_root,
                 nsweeps=1,
                 lidar_dir='lidar',
                 **kwargs):
        self.nsweeps = nsweeps
        self.lidar_dir = os.path.join(data_root, lidar_dir)
        super(OpenlaneLidarDataset, self).__init__(pipeline, data_root, **kwargs)

    def load_annotations(self):
        print('Now loading annotations...')
        self.img_infos = []
        with open(self.data_list, 'r') as anno_obj:
            all_ids = [s.strip() for s in anno_obj.readlines()]
            for k, id in enumerate(all_ids):
                seg_id = '/'.join(id.split('/')[1:])
                anno = {'filename': os.path.join(self.img_dir, id + self.img_suffix),
                        'anno_file': os.path.join(self.cache_dir, id + '.pkl'),
                        'lidar_file': os.path.join(self.lidar_dir, seg_id[:-2] + '.pkl')}
                self.img_infos.append(anno)
        print("after load annotation")
        print("find {} samples in {}.".format(len(self.img_infos), self.data_list))