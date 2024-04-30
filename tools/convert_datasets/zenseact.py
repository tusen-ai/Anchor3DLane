import os
import json
import re
import cv2
from mmseg.datasets.tools.utils import *
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import numpy as np
import gc
import tqdm
import matplotlib.pyplot as plt
import tqdm
import mmcv
import pickle
import argparse
import glob
from mmcv.utils import Config

def moving_average(interval, windowsize=5):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'valid')
    return re

def smooth_data_and_find_peak(data_raw, windowsize=5, threshold=1, influence=0.5):
    # Smoothed Z-Score Algorithm
    res_peak = []
    res_data_smoothed = data_raw
    len_data = len(data_raw)
    filter_avg = np.zeros(len_data)
    filter_std = np.zeros(len_data)
    filter_avg[windowsize - 1] = np.mean(data_raw[:windowsize])
    filter_std[windowsize - 1] = np.std(data_raw[windowsize])
    for i in range(windowsize, len_data):
        if abs(data_raw[i] - filter_avg[i - 1]) > threshold * filter_std[i - 1]:
            if data_raw[i] > filter_avg[i-1]:
                res_peak.append(i)
        res_data_smoothed[i] = influence * data_raw[i] + (1-influence) * res_data_smoothed[i-1]
        filter_avg[i] = np.mean(res_data_smoothed[(i - windowsize): i])
        filter_std[i] = np.std(res_data_smoothed[(i - windowsize): i])
    return res_peak

def lane_smoothing(lane, y_end=200, interp_step=0.5):
    k_size = 11
    padding = (k_size - 1) // 2
    
    f_x = interp1d(lane[:, 1], lane[:, 0], fill_value='extrapolate')
    f_z = interp1d(lane[:, 1], lane[:, 2], fill_value='extrapolate')
    min_y = max(lane[:, 1].min(), 0)
    max_y = min(lane[:, 1].max(), y_end)
    y_sample = np.linspace(min_y, max_y - interp_step, int((max_y - min_y) / interp_step))
    x_interp = f_x(y_sample)
    z_interp = f_z(y_sample)
    lane_interp = np.vstack([x_interp, y_sample, z_interp]).T

    if len(lane_interp) <= k_size:
        return lane_interp
    
    del_peak1 = smooth_data_and_find_peak(lane_interp[:, 0], windowsize=k_size, threshold=5)
    lane_interp = np.delete(lane_interp, del_peak1, 0)
    if len(lane_interp) <= k_size:
        return lane_interp
    del_peak2 = smooth_data_and_find_peak(lane_interp[:, 2], windowsize=k_size, threshold=5)
    lane_interp = np.delete(lane_interp, del_peak2, 0)
    if len(lane_interp) <= k_size:
        return lane_interp

    x = moving_average(lane_interp[:, 0], windowsize=k_size)
    x = np.insert(x, 0, lane_interp[0, 0])
    x = np.append(x, lane_interp[-1, 0])
    z = moving_average(lane_interp[:, 2], windowsize=k_size)
    z = np.insert(z, 0, lane_interp[0, 2])
    z = np.append(z, lane_interp[-1, 2])
    y = lane_interp[:, 1][padding:-padding]
    y = np.insert(y, 0, lane_interp[0, 1])
    y = np.append(y, lane_interp[-1, 1])
    return np.vstack([x, y, z]).T


def generate_annotation(lane1, dev, distances):
    new_lane = np.zeros(lane1.shape)
    new_lane[:, 0] = lane1[:, 0] + distances / dev[:, 0]
    new_lane[:, 1] = lane1[:, 1]
    new_lane[:, 2] = lane1[:, 2]
    return new_lane

def generate_annotation2(lane1, dev, distances):
    # import pdb
    # pdb.set_trace()
    new_lane = np.zeros(lane1.shape)
    new_lane[:, 0] = lane1[:, 0] + np.abs(distances * dev[:, 0])
    new_lane[:, 1] = lane1[:, 1] + np.abs(distances * dev[:, 1])
    new_lane[:, 2] = lane1[:, 2]
    return new_lane

def make_lane_y_mono_inc(lane):
    """
        Due to lose of height dim, projected lanes to flat ground plane may not have monotonically increasing y.
        This function trace the y with monotonically increasing y, and output a pruned lane
    :param lane:
    :return:
    """
    idx2del = []
    max_y = lane[0, 1]
    end_y = lane[-1, 1]
    for i in range(1, lane.shape[0]):
        # hard-coded a smallest step, so the far-away near horizontal tail can be pruned
        if lane[i, 1] <= max_y or lane[i, 1] > end_y:
            idx2del.append(i)
        else:
            max_y = lane[i, 1]
    lane = np.delete(lane, idx2del, 0)
    return lane

def transform_annotation(anno, max_lanes, anchor_len=20, index_begin=5, anchor_len_2d=72):
    gflatYnorm = 100
    gflatXnorm = 30
    gflatZnorm = 10
    gt_3dlanes = anno['gt_3dlanes']
    categories = anno['categories']
    lanes3d     = np.ones((max_lanes, index_begin + anchor_len * 3), dtype=np.float32) * -1e5
    lanes3d[:, 1] = 0
    lanes3d[:, 0] = 1
    lanes3d_norm = np.ones((max_lanes, index_begin + anchor_len * 3), dtype=np.float32) * -1e5
    lanes3d_norm[:, 1] = 0
    lanes3d_norm[:, 0] = 1
    lanes2d = np.ones((max_lanes, anchor_len_2d * 3), dtype=np.float32) * -1e5
    
    for lane_pos, (gt_3dlane, cate) in enumerate(zip(gt_3dlanes, categories)):
        gt_3dlane = gt_3dlanes[lane_pos]
        Xs = np.array([p[0] for p in gt_3dlane])
        Zs = np.array([p[1] for p in gt_3dlane])
        vis = np.array([p[2] for p in gt_3dlane])
        lanes3d[lane_pos, 0] = 0
        lanes3d[lane_pos, 1] = cate

        lanes3d_norm[lane_pos, 0] = 0
        lanes3d_norm[lane_pos, 1] = cate


        lanes3d[lane_pos, index_begin:(index_begin+anchor_len)] = Xs
        lanes3d_norm[lane_pos, index_begin:(index_begin+anchor_len)] = Xs / gflatXnorm

        lanes3d[lane_pos, (index_begin+anchor_len):(index_begin+anchor_len*2)] = Zs
        lanes3d_norm[lane_pos, (index_begin+anchor_len):(index_begin+anchor_len*2)] = Zs / gflatZnorm
        
        lanes3d[lane_pos, (index_begin+anchor_len*2):(index_begin+anchor_len*3)] = vis  # vis_flag
        lanes3d_norm[lane_pos, (index_begin+anchor_len*2):(index_begin+anchor_len*3)] = vis

    new_anno = {
        'path': anno['path'],
        'gt_3dlanes': lanes3d,
        'gt_3dlanes_norm': lanes3d_norm,
        'old_anno': anno,
        'gt_camera_extrinsic': anno['gt_camera_extrinsic'],
        'gt_camera_intrinsic': anno['gt_camera_intrinsic'],
    }

    return new_anno

import pandas as pd

def extract_data_with_smoothing(data_root, cfg, anno_file, tar_path, test_mode=False, sample_step=1, prune_vis=True, smooth=True):
    anchor_y_steps = np.linspace(1, 200, 200 // sample_step)
    image_id  = 0
    cnt_idx = 0
    with open(anno_file, 'r') as anno_obj:
        # Initialize an empty DataFrame for annotations with predefined columns

        # Temporary storage for data to be appended
        temp_annotations = []

        for i, line in enumerate(tqdm(anno_obj)):
        # Your processing code here
        # Example: time.sleep(0.1) # Simulate work

        # Print progress message
            
            info_dict = json.loads(line)
            image_path = os.path.join('data', info_dict['file_path'])
            pickle_file = os.path.join(tar_path, '/'.join(image_path.split('/')[-3:]).replace('.jpg', '.pkl'))
            
            cnt_idx += 1

            if not os.path.exists(image_path):
                continue
            print(f'Processing item {i+1}', end='\r', flush=True)

            cam_extrinsics = np.array(info_dict['extrinsic'])
            cam_intrinsics = np.array(info_dict['intrinsic'])

            # Re-calculate extrinsic matrix based on ground coordinate
            R_vg = np.array([[0, 1, 0],
                                [-1, 0, 0],
                                [0, 0, 1]], dtype=np.float32)
            R_gc = np.array([[1, 0, 0],
                                [0, 0, 1],
                                [0, -1, 0]], dtype=np.float32)
            cam_extrinsics[:3, :3] = np.matmul(np.matmul(
                                        np.matmul(np.linalg.inv(R_vg), cam_extrinsics[:3, :3]),
                                            R_vg), R_gc)
            cam_extrinsics[0:2, 3] = 0.0

            gt_lanes_packed = info_dict['lane_lines']

            if len(gt_lanes_packed) < 1:
                    temp_annotations.append({'path': image_path,
                                                        'gt_3dlanes': [],
                                                        'categories': [],
                                                        'gt_distances': [],
                                                        'gt_next_indices': [],
                                                        'aug': False,
                                                        'relative_path': info_dict['file_path'],
                                                        'gt_camera_extrinsic': cam_extrinsics,
                                                        'gt_camera_intrinsic': cam_intrinsics,})

                    new_anno = transform_annotation(temp_annotations[0], max_lanes=cfg.dataset_config.max_lanes, anchor_len=200)
                    anno = {}
                    anno['filename'] = new_anno['path']
                    anno['gt_3dlanes'] = new_anno['gt_3dlanes']
                    anno['gt_camera_extrinsic'] = new_anno['gt_camera_extrinsic']
                    anno['gt_camera_intrinsic'] = new_anno['gt_camera_intrinsic']
                    anno['old_anno'] = new_anno['old_anno']
                    pickle_path = os.path.join(tar_path, '/'.join(anno['filename'].split('/')[-3:-1]))
                    mmcv.mkdir_or_exist(pickle_path)
                    pickle_file = os.path.join(tar_path, '/'.join(anno['filename'].split('/')[-3:]).replace('.jpg', '.pkl'))
                    w = open(pickle_file, 'wb')
                    pickle.dump({'image_id':anno['filename'],
                                'gt_3dlanes':anno['gt_3dlanes'],
                                'gt_camera_extrinsic':anno['gt_camera_extrinsic'],
                                'gt_camera_intrinsic':anno['gt_camera_intrinsic']}, w)
                    w.close()
                    image_id += 1

            all_lanes = []
            lane_cates = []
            for i, gt_lane_packed in enumerate(gt_lanes_packed):
                lane_results = {}
                # A GT lane can be either 2D or 3D
                # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
                lane = np.array(gt_lane_packed['xyz'])   # [3, num_p]
                lane_visibility = np.array(gt_lane_packed['visibility'])  # [num_p]
                # Coordinate convertion
                lane = np.vstack((lane, np.ones((1, lane.shape[1]))))
                cam_representation = np.linalg.inv(
                                        np.array([[0, 0, 1, 0],
                                                    [-1, 0, 0, 0],
                                                    [0, -1, 0, 0],
                                                    [0, 0, 0, 1]], dtype=np.float32))  # transformation from apollo camera to openlane camera
                lane = np.matmul(cam_extrinsics, np.matmul(cam_representation, lane))
                lane = lane[0:3, :].T   # [N, 3]

                # pruned unvisible points
                if prune_vis:
                    lane = prune_3d_lane_by_visibility(lane, lane_visibility)

                pruned_lane = prune_3d_lane_by_range(lane, -30, 30)
                if pruned_lane.shape[0] < 2:
                    continue
            
                pruned_lane = make_lane_y_mono_inc(pruned_lane)

                if pruned_lane.shape[0] < 2:
                    continue

                if smooth:
                    pruned_lane = lane_smoothing(pruned_lane)
                    if pruned_lane.shape[0] < 2:
                        continue

                if (pruned_lane[-1, 1] - pruned_lane[0, 1]) < 5:  # pruned short lanes
                    continue

                x_values, z_values, visibility_vec = resample_laneline_in_y(pruned_lane, anchor_y_steps, out_vis=True)
                if sum(visibility_vec) <= 1:
                    continue
                resample_lane = np.stack([x_values, z_values, visibility_vec], axis=-1)  # [N, 3]

                lane_results['gt_lane'] = resample_lane
                lane_results['category'] = gt_lane_packed['category']
                
                all_lanes.append(lane_results)
                lane_cates.append(lane_results['category'])

            if len(all_lanes) == 0:
                    temp_annotations.append({'path': image_path,
                                    'gt_3dlanes': [],
                                    'categories': [],
                                    'aug': False,
                                    'relative_path': info_dict['file_path'],
                                    'gt_camera_extrinsic': cam_extrinsics,
                                    'gt_camera_intrinsic': cam_intrinsics,})
                    
                    new_anno = transform_annotation(temp_annotations[0], max_lanes=max_lanes, anchor_len=200)
                    anno = {}
                    anno['filename'] = new_anno['path']
                    anno['gt_3dlanes'] = new_anno['gt_3dlanes']
                    anno['gt_camera_extrinsic'] = new_anno['gt_camera_extrinsic']
                    anno['gt_camera_intrinsic'] = new_anno['gt_camera_intrinsic']
                    anno['old_anno'] = new_anno['old_anno']
                    pickle_path = os.path.join(tar_path, '/'.join(anno['filename'].split('/')[-3:-1]))
                    mmcv.mkdir_or_exist(pickle_path)
                    pickle_file = os.path.join(tar_path, '/'.join(anno['filename'].split('/')[-3:]).replace('.jpg', '.pkl'))
                    w = open(pickle_file, 'wb')
                    pickle.dump({'image_id':anno['filename'],
                                'gt_3dlanes':anno['gt_3dlanes'],
                                'gt_camera_extrinsic':anno['gt_camera_extrinsic'],
                                'gt_camera_intrinsic':anno['gt_camera_intrinsic']}, w)
                    w.close()
                
                        # After processing each item or batch of items, append data to the DataFrame
                    if len(temp_annotations) > 0:  # Or use a larger threshold for efficiency
                            temp_annotations.clear()  # Clear temporary storage to free up memory

                    image_id += 1

            gt_3dlanes = [p['gt_lane'] for p in all_lanes]
            gt_laneline_category = [p['category'] for p in all_lanes]
            
            temp_annotations.append({
                'image_id': image_id,
                'path': image_path,
                'gt_3dlanes': gt_3dlanes,
                'categories': gt_laneline_category,
                'aug': False,
                'relative_path': info_dict['file_path'],
                'gt_camera_extrinsic': cam_extrinsics,
                'gt_camera_intrinsic': cam_intrinsics,
            })
            
            new_anno = transform_annotation(temp_annotations[0], max_lanes=max_lanes, anchor_len=200)
            anno = {}
            anno['filename'] = new_anno['path']
            anno['gt_3dlanes'] = new_anno['gt_3dlanes']
            anno['gt_camera_extrinsic'] = new_anno['gt_camera_extrinsic']
            anno['gt_camera_intrinsic'] = new_anno['gt_camera_intrinsic']
            anno['old_anno'] = new_anno['old_anno']
            pickle_path = os.path.join(tar_path, '/'.join(anno['filename'].split('/')[-3:-1]))
            mmcv.mkdir_or_exist(pickle_path)
            pickle_file = os.path.join(tar_path, '/'.join(anno['filename'].split('/')[-3:]).replace('.jpg', '.pkl'))
            w = open(pickle_file, 'wb')
            
            pickle.dump({'image_id':anno['filename'],
                        'gt_3dlanes':anno['gt_3dlanes'],
                        'gt_camera_extrinsic':anno['gt_camera_extrinsic'],
                        'gt_camera_intrinsic':anno['gt_camera_intrinsic'],
                        'original_anno':info_dict}, w)
            w.close()
                
                # After processing each item or batch of items, append data to the DataFrame
            if len(temp_annotations) > 0:  # Or use a larger threshold for efficiency
                    temp_annotations.clear()  # Clear temporary storage to free up memory

            image_id += 1
            if test_mode and image_id != cnt_idx:
                raise Exception("missing test files")
            gc.collect()
            


from tqdm import tqdm 
def merge_annotations(anno_path, json_file):
    with open(anno_path, 'r') as file:
        all_files = file.read().splitlines()
    w = open(json_file, 'w')
    for idx, file_name in enumerate(tqdm(all_files)):
        directory_path = f'data/zod_dataset/single_frames/{file_name}/annotations'
        file_name, full_path = next(((f, os.path.join(directory_path, f)) for f in os.listdir(directory_path) if "ol_format_lane_markings" in f and os.path.isfile(os.path.join(directory_path, f))), (None, None))
        with open(full_path, 'r') as f:
            s = f.readline()
        w.write(s+'\n')
    w.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process Openlane dataset')
    parser.add_argument('--config', default= 'configs/zod/anchor3dlane.py', help='train config file path')
    parser.add_argument('--merge', action='store_true', default=True, help='whether to merge the annotation json files')
    parser.add_argument('--generate', action='store_true', default=False, help='whether to pickle files')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    
    if args.merge:
        print("merging data")
        print("new_directory", os.path.join(cfg.data_root, 'data_splits'))
        mmcv.mkdir_or_exist(os.path.join(cfg.data_root, 'data_splits'))
        merge_annotations('data/zod_dataset/data_lists/training.txt', os.path.join(cfg.data_root, 'data_splits', 'training.json'))
        merge_annotations('data/zod_dataset/data_lists/validation.txt', os.path.join(cfg.data_root, 'data_splits', 'validation.json'))
    elif args.generate:
        ori_json = os.path.join(cfg.data_root, 'data_splits', 'training.json')
        tar_path = os.path.join(cfg.data_root, 'cache_dense')
        data_list_path = os.path.join(cfg.data_root, 'data_lists')
        os.makedirs(data_list_path, exist_ok=True)
        extract_data_with_smoothing(data_root = cfg.data_root, cfg = cfg, anno_file = ori_json, tar_path=tar_path, test_mode=False)
        ori_json = os.path.join(args.data_root, 'data_splits', 'validation.json')
        extract_data_with_smoothing(data_root = cfg.data_root, cfg = cfg, anno_file = ori_json, tar_path=tar_path, test_mode=True)