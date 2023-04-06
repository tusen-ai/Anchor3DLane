import numpy as np
import cv2
import os
import os.path as ops
import copy
import math
import ujson as json
from scipy.interpolate import interp1d
import matplotlib
from tqdm import tqdm
import warnings

import mmcv
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .utils import *
from .MinCostFlow import SolveMinCostFlow

plt.rcParams['figure.figsize'] = (35, 30)
plt.rcParams.update({'font.size': 25})
plt.rcParams.update({'font.weight': 'semibold'})

color = [[0, 0, 255],  # red
         [0, 255, 0],  # green
         [255, 0, 255],  # purple
         [255, 255, 0]]  # cyan

vis_min_y = 5
vis_max_y = 80

class LaneVis(object):
    def __init__(self, db):
        self.K = db.K
        self.resize_h = db.resize_h
        self.resize_w = db.resize_w
        self.H_crop = homography_crop_resize([db.org_h, db.org_w], db.crop_y, [db.resize_h, db.resize_w])
        self.H_g2side = cv2.getPerspectiveTransform(
            np.float32([[0, -5], [0, 5], [100, -5], [100, 5]]),
            np.float32([[0, 300], [0, 0], [300, 300], [300, 0]]))
        self.top_view_region = db.top_view_region
        self.ipm_h = db.ipm_h
        self.ipm_w = db.ipm_w
        self.org_h = db.org_h
        self.org_w = db.org_w
        self.crop_y = db.crop_y
        self.x_min = db.top_view_region[0, 0]
        self.x_max = db.top_view_region[1, 0]
        self.y_min = db.top_view_region[2, 1]
        self.y_max = db.top_view_region[0, 1]
        self.y_samples = np.linspace(self.y_min, self.y_max, num=100, endpoint=False)
        self.dist_th = 1.5
        self.ratio_th = 0.75
        self.close_range = 40

    def vis(self, gt, pred, save_dir, img_dir, img_name, prob_th=0.5):
        img_path = os.path.join(img_dir, img_name)
        fig = plt.figure(figsize=(35,20))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133, projection='3d')

        gt_cam_height = gt['cam_height']
        gt_cam_pitch = gt['cam_pitch']
        gt_lanelines = gt['laneLines']
        gt_visibility = gt['laneLines_visibility']

        # only keep the visible portion
        gt_lanes = [prune_3d_lane_by_visibility(np.array(gt_lane), np.array(gt_visibility[k])) for k, gt_lane in
                    enumerate(gt_lanelines)]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]
        # only consider those gt lanes overlapping with sampling range
        gt_lanes = [lane for lane in gt_lanes if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]
        gt_lanes = [prune_3d_lane_by_range(np.array(gt_lane), 3 * self.x_min, 3 * self.x_max) for gt_lane in gt_lanes]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]

        pred_lanes = pred['laneLines']
        pred_lanes_prob = pred['laneLines_prob']
        pred_lanes = [pred_lanes[ii] for ii in range(len(pred_lanes_prob)) if
                            pred_lanes_prob[ii] > prob_th]
        
        cnt_gt = len(gt_lanes)
        cnt_pred = len(pred_lanes)

        P_g2im = projection_g2im(gt_cam_pitch, gt_cam_height, self.K)
        P_gt = np.matmul(self.H_crop, P_g2im)
        img = cv2.imread(img_path)
        img = cv2.warpPerspective(img, self.H_crop, (self.resize_w, self.resize_h))
        img = img.astype(np.float) / 255
        
        H_ipm2g = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [self.ipm_w - 1, 0], [0, self.ipm_h - 1], [self.ipm_w - 1, self.ipm_h - 1]]), 
            np.float32(self.top_view_region))
        H_g2ipm = np.linalg.inv(H_ipm2g)
        H_g2im = homograpthy_g2im(gt_cam_pitch, gt_cam_height, self.K)
        H_im2g = np.linalg.inv(H_g2im)
        H_im2ipm = np.linalg.inv(np.matmul(H_g2im, H_ipm2g))
        raw_img = cv2.imread(img_path)
        raw_img = raw_img.astype(np.float) / 255
        im_ipm = cv2.warpPerspective(raw_img, H_im2ipm, (self.ipm_w, self.ipm_h))
        im_ipm = np.clip(im_ipm, 0, 1)

        gt_visibility_mat = np.zeros((cnt_gt, 100))
        pred_visibility_mat = np.zeros((cnt_pred, 100))

        for i in range(cnt_gt):
            min_y = np.min(np.array(gt_lanes[i])[:, 1])
            max_y = np.max(np.array(gt_lanes[i])[:, 1])
            x_values, z_values, visibility_vec = resample_laneline_in_y(np.array(gt_lanes[i]), self.y_samples,
                                                                        out_vis=True)
            gt_lanes[i] = np.vstack([x_values, z_values]).T
            gt_visibility_mat[i, :] = np.logical_and(x_values >= self.x_min,
                                                     np.logical_and(x_values <= self.x_max,
                                                                    np.logical_and(self.y_samples >= min_y,
                                                                                   self.y_samples <= max_y)))
            gt_visibility_mat[i, :] = np.logical_and(gt_visibility_mat[i, :], visibility_vec)

        for i in range(cnt_pred):
            min_y = np.min(np.array(pred_lanes[i])[:, 1])
            max_y = np.max(np.array(pred_lanes[i])[:, 1])
            x_values, z_values, visibility_vec = resample_laneline_in_y(np.array(pred_lanes[i]), self.y_samples,
                                                                        out_vis=True)
            pred_lanes[i] = np.vstack([x_values, z_values]).T
            pred_visibility_mat[i, :] = np.logical_and(x_values >= self.x_min,
                                                       np.logical_and(x_values <= self.x_max,
                                                                      np.logical_and(self.y_samples >= min_y,
                                                                                     self.y_samples <= max_y)))
            pred_visibility_mat[i, :] = np.logical_and(pred_visibility_mat[i, :], visibility_vec)


        for i in range(cnt_gt):
            x_values = gt_lanes[i][:, 0]
            z_values = gt_lanes[i][:, 1]
            x_2d, y_2d = projective_transformation(P_gt, x_values, self.y_samples, z_values)
            x_2d = x_2d.astype(np.int)
            y_2d = y_2d.astype(np.int)
            x_ipm, y_ipm = transform_lane_g2gflat(gt_cam_height, x_values, self.y_samples, z_values)
            x_ipm, y_ipm = homographic_transformation(H_g2ipm, x_ipm, y_ipm)
            x_ipm = x_ipm.astype(np.int)
            y_ipm = y_ipm.astype(np.int)

            color = [0, 0, 1]   # tp: blue
            for k in range(1, x_2d.shape[0]):
                # only draw the visible portion
                if gt_visibility_mat[i, k - 1] and gt_visibility_mat[i, k]:
                    img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), color[-1::-1], 3)
                    im_ipm = cv2.line(im_ipm, (x_ipm[k - 1], y_ipm[k - 1]), (x_ipm[k], y_ipm[k]), color[-1::-1], 2)
            ax3.plot(x_values[np.where(gt_visibility_mat[i, :])],
                        self.y_samples[np.where(gt_visibility_mat[i, :])],
                        z_values[np.where(gt_visibility_mat[i, :])], color=color, linewidth=5)

        for i in range(cnt_pred):
            x_values = pred_lanes[i][:, 0]
            z_values = pred_lanes[i][:, 1]
            x_2d, y_2d = projective_transformation(P_gt, x_values, self.y_samples, z_values)
            x_2d = x_2d.astype(np.int)
            y_2d = y_2d.astype(np.int)
            x_ipm, y_ipm = transform_lane_g2gflat(gt_cam_height, x_values, self.y_samples, z_values)
            x_ipm, y_ipm = homographic_transformation(H_g2ipm, x_ipm, y_ipm)
            x_ipm = x_ipm.astype(np.int)
            y_ipm = y_ipm.astype(np.int)

            color = [1, 0, 0]    # tp: red
            for k in range(1, x_2d.shape[0]):
                # only draw the visible portion
                if pred_visibility_mat[i, k - 1] and pred_visibility_mat[i, k]:
                    img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), color[-1::-1], 2)
                    im_ipm = cv2.line(im_ipm, (x_ipm[k - 1], y_ipm[k - 1]), (x_ipm[k], y_ipm[k]), color[-1::-1], 2)
            ax3.plot(x_values[np.where(pred_visibility_mat[i, :])],
                        self.y_samples[np.where(pred_visibility_mat[i, :])],
                        z_values[np.where(pred_visibility_mat[i, :])], color=color, linewidth=5)

        ax1.imshow(img[:, :, [2, 1, 0]])
        ax2.imshow(im_ipm[:, :, [2, 1, 0]])

        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        bottom, top = ax3.get_zlim()
        left, right = ax3.get_xlim()
        ax3.set_zlim(min(bottom, -0.1), max(top, 0.1))
        ax3.set_xlim(left, right)
        ax3.set_ylim(vis_min_y, vis_max_y)
        ax3.locator_params(nbins=5, axis='x')
        ax3.locator_params(nbins=10, axis='z')
        ax3.tick_params(pad=18, labelsize=15)
        fig.savefig(ops.join(save_dir, img_name.replace("/", "_")))
        plt.close(fig)
        print('processed sample: {}'.format(img_name))

    def visualize(self, pred_file, gt_file, img_dir=None, save_dir=None, prob_th=0.5, vis_step=10):
        mmcv.mkdir_or_exist(save_dir)
        json_pred = [json.loads(line) for line in open(pred_file).readlines()]
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        if len(json_gt) != len(json_pred):
            warnings.warn('We do not get the predictions of all the test tasks')
        gts = {l['raw_file']: l for l in json_gt}
        for i, pred in tqdm(enumerate(json_pred)):
            if i % vis_step != 0:
                continue
            if 'raw_file' not in pred or 'laneLines' not in pred:
                raise Exception('raw_file or lanelines not in some predictions.')
            raw_file = pred['raw_file']
            gt = gts[raw_file]
            self.vis(gt, pred, save_dir, img_dir, raw_file, prob_th=prob_th)

