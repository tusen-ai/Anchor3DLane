# --------------------------------------------------------
# Source code for Anchor3DLane
# Copyright (c) 2023 TuSimple
# @Time    : 2023/04/05
# @Author  : Shaofei Huang
# nowherespyfly@gmail.com
# --------------------------------------------------------

from re import L
import pickle

import tqdm
import numpy as np

from ..tools.utils import *
from .openlane import OpenlaneDataset
from ..builder import DATASETS

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
class OpenlaneMFDataset(OpenlaneDataset):
    def __init__(self, 
                 pipeline,
                 data_root,
                 prev_dir=None,
                 prev_num=2,
                 prev_step=1,  # for test
                 prev_range=5,   # for train
                 is_prev=False,
                 **kwargs):
        self.prev_dir = os.path.join(data_root, prev_dir)
        self.prev_num = prev_num
        self.prev_step = prev_step
        self.prev_range = prev_range
        self.is_prev = is_prev
        super(OpenlaneMFDataset, self).__init__(pipeline, data_root, **kwargs)

    def load_annotations(self):
        print('Now loading annotations...')
        self.img_infos = []
        with open(self.data_list, 'r') as anno_obj:
            all_ids = [s.strip() for s in anno_obj.readlines()]
            for k, id in tqdm.tqdm(enumerate(all_ids)):
                anno = {'filename': os.path.join(self.img_dir, id + self.img_suffix),
                        'anno_file': os.path.join(self.cache_dir, id + '.pkl'), 
                        'prev_file': os.path.join(self.prev_dir, id + '.pkl')}
                self.img_infos.append(anno)
        print("after load annotation")
        print("found {} samples in total".format(len(self.img_infos)))

    def sample_prev_frame_train(self, prev_datas, cur_project_matrix, cur_filename):
        if len(prev_datas['prev_data']) < self.prev_range:
            ori_len = len(prev_datas['prev_data'])
            for i in range(ori_len, self.prev_range):
                prev_datas['prev_data'].append({'file_path':cur_filename, 'project_matrix':cur_project_matrix.copy()})
        select_prev_datas = np.random.choice(prev_datas['prev_data'][-self.prev_range:], self.prev_num, replace=False)
        prev_images = [os.path.join(self.data_root, p['file_path']) for p in select_prev_datas]
        prev_poses = [p['project_matrix'].copy() for p in select_prev_datas]
        return prev_images, prev_poses

    def sample_prev_frame_test(self, prev_datas, cur_project_matrix, cur_filename):
        if len(prev_datas['prev_data']) < self.prev_num * self.prev_step:
            ori_len = len(prev_datas['prev_data'])
            for i in range(ori_len, self.prev_num * self.prev_step):
                prev_datas['prev_data'].append({'file_path':cur_filename, 'project_matrix':cur_project_matrix.copy()})
        prev_images = [os.path.join(self.data_root, p['file_path']) for p in prev_datas['prev_data'][-self.prev_num*self.prev_step::self.prev_step]]
        prev_poses = [p['project_matrix'].copy() for p in prev_datas['prev_data'][-self.prev_num*self.prev_step::self.prev_step]]  
        return prev_images, prev_poses

    def sample_post_frame_train(self, post_datas, cur_project_matrix, cur_filename):
        if len(post_datas['post_data']) < self.prev_range:
            ori_len = len(post_datas['post_data'])
            for i in range(ori_len, self.prev_range):
                post_datas['post_data'].insert(0, {'file_path':cur_filename, 'project_matrix':cur_project_matrix.copy()})
        select_post_datas = np.random.choice(post_datas['post_data'][:self.prev_range], self.prev_num, replace=False)
        post_images = [os.path.join(self.data_root, p['file_path']) for p in select_post_datas]
        post_poses = [p['project_matrix'].copy() for p in select_post_datas]
        return post_images, post_poses

    def sample_post_frame_test(self, post_datas, cur_project_matrix, cur_filename):
        if len(post_datas['post_data']) < self.prev_num * self.prev_step:
            ori_len = len(post_datas['post_data'])
            for i in range(ori_len, self.prev_num * self.prev_step):
                post_datas['post_data'].insert(0, {'file_path':cur_filename, 'project_matrix':cur_project_matrix.copy()})
        post_images = [os.path.join(self.data_root, p['file_path']) for p in post_datas['post_data'][self.prev_step-1:self.prev_num*self.prev_step:self.prev_step]]
        post_poses = [p['project_matrix'].copy() for p in post_datas['post_data'][self.prev_step-1:self.prev_num*self.prev_step:self.prev_step]]  
        return post_images, post_poses

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
        with open(results['prev_file'], 'rb') as f:
            prev_datas = pickle.load(f)
        
        if self.test_mode:
            if self.is_prev:
                prev_images, prev_poses = self.sample_prev_frame_test(prev_datas, results['gt_project_matrix'], results['filename'])
            else:
                prev_images, prev_poses = self.sample_post_frame_test(prev_datas, results['gt_project_matrix'], results['filename'])
        else:
            if self.is_prev:
                prev_images, prev_poses = self.sample_prev_frame_train(prev_datas, results['gt_project_matrix'], results['filename'])
            else:
                prev_images, prev_poses = self.sample_post_frame_train(prev_datas, results['gt_project_matrix'], results['filename'])
        results['prev_images'] = prev_images
        results['prev_poses'] = prev_poses
        results = self.pipeline(results)
        return results