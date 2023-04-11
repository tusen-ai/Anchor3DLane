import pdb
import os
import numpy as np
import json
import pickle
import cv2
import glob
import tqdm
from matplotlib import pyplot as plt
import mmcv
from mmseg.datasets.tools.utils import *
from multiprocessing import Pool


def generate_proj_matrix(last_info, last_pose, cur_info, cur_pose, cur_v2g_extrinsic):
    cam_representation = np.array([[0, -1, 0, 0], [0, 0, -1, 0], \
        [1, 0, 0, 0], [0, 0, 0, 1]], dtype=float)  # waymo cam to normal cam
    proj_matrix = np.matmul(cur_pose, cur_info['extrinsic'])   # current vehicle to global frame
    proj_matrix = np.matmul(np.linalg.inv(last_pose), proj_matrix)  # cam2 to vehicle1
    proj_matrix = np.matmul(np.linalg.inv(last_info['extrinsic']), proj_matrix)   # last vehicle to last camera
    proj_matrix = np.matmul(cam_representation, proj_matrix)
    cam_intrinsics = np.array(last_info['intrinsic'])
    P_g2im_last = np.matmul(np.linalg.inv(cam_representation), np.linalg.inv(cur_v2g_extrinsic))
    P_g2im_last = np.matmul(proj_matrix, P_g2im_last)
    P_g2im_last = np.matmul(cam_intrinsics, P_g2im_last[0:3, :])  # cur ground to last pixel
    return P_g2im_last

def warp_one(data_path, pose_path, target_path):
    data_list = os.listdir(data_path)
    data_list = sorted(data_list, key=lambda k: int(k.split('.')[0]))
    seq_len = len(data_list)
    for idx in tqdm.tqdm(range(len(data_list))):
        data_save = {}
        cur_info = json.load(open(os.path.join(data_path, data_list[idx]), 'r'))
        cur_pose = np.loadtxt(os.path.join(pose_path, data_list[idx].replace('00.json', '.txt')))
        # cur lane cam to ground
        cam_extrinsics = np.array(cur_info['extrinsic'])
        R_vg = np.array([[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 1]], dtype=float)
        R_gc = np.array([[1, 0, 0],
                            [0, 0, 1],
                            [0, -1, 0]], dtype=float)
        cam_extrinsics[:3, :3] = np.matmul(np.matmul(
                                    np.matmul(np.linalg.inv(R_vg), cam_extrinsics[:3, :3]),
                                        R_vg), R_gc)
        cam_extrinsics[0:2, 3] = 0.0

        data_save['pose'] = cur_pose
        prev_datas = []
        post_datas = []
        if idx > 0:
            for prev_idx in range(max(0, idx - 10), idx):
                prev_data = {}
                prev_info = json.load(open(os.path.join(data_path, data_list[prev_idx]), 'r'))
                prev_pose = np.loadtxt(os.path.join(pose_path, data_list[prev_idx].replace('00.json', '.txt')))
                prev_path = prev_info['file_path']
                P_g2im_last = generate_proj_matrix(prev_info, prev_pose, cur_info, cur_pose, cam_extrinsics)
                prev_data['file_path'] = os.path.join(data_root, 'images', prev_path)
                prev_data['project_matrix'] = P_g2im_last
                prev_data['pose'] = prev_pose
                prev_datas.append(prev_data)

        if idx < seq_len - 1:
            for post_idx in range(idx+1, min(seq_len, idx + 11)):
                post_data = {}
                post_info = json.load(open(os.path.join(data_path, data_list[post_idx]), 'r'))
                post_pose = np.loadtxt(os.path.join(pose_path, data_list[post_idx].replace('00.json', '.txt')))
                post_path = post_info['file_path']
                P_g2im_last = generate_proj_matrix(post_info, post_pose, cur_info, cur_pose, cam_extrinsics)
                post_data['file_path'] = os.path.join(data_root, 'images', post_path)
                post_data['project_matrix'] = P_g2im_last
                post_data['pose'] = post_pose
                post_datas.append(post_data)
        data_save['prev_data'] = prev_datas
        data_save['post_data'] = post_datas
        w = open(os.path.join(target_path,  data_list[idx].replace('.json', '.pkl')), 'wb')
        pickle.dump(data_save, w)
        w.close()

def warp_prev_frames(path_lists, raw_pose_path, target_pose_path, pid):
    for idx, item in enumerate(path_lists):
        split, segment = item.split('/')[-2:]
        pose_path = os.path.join(raw_pose_path, segment)
        target_path = os.path.join(target_pose_path, split, segment)
        mmcv.mkdir_or_exist(target_path)
        warp_one(item, pose_path, target_path)
        print("finish {} batch".format(idx))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process Openlane dataset')
    parser.add_argument('data_root', help='root path of openlane dataset')
    args = parser.parse_args()
    path_lists = glob.glob(os.path.join(data_root, 'lane3d_1000/*/segment-*'))
    p = Pool(8)
    interval = len(path_lists) // 8
    for pid in range(8):
        if pid == 7:
            cur_list = path_lists[pid * interval:]
        else:
            cur_list = path_lists[pid * interval:(pid+1)*interval]
        print("add process {}".format(pid))
        p.apply_async(warp_prev_frames, args=(cur_list, pid))
    p.close()
    p.join()