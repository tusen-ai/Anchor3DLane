import numpy as np
import cv2
import os
import os.path as ops
import copy
import math
import ujson as json
import matplotlib
from tqdm import tqdm
import warnings
import random

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
        self.resize_h = db.resize_h
        self.resize_w = db.resize_w
        # H_crop: [[rx, 0, 0], [0, ry, 0], [0, 0, 1]]
        self.H_crop = homography_crop_resize([db.org_h, db.org_w], db.crop_y, [db.resize_h, db.resize_w])
        self.top_view_region = db.top_view_region
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
        self.R_c2g = db.R_c2g
        self.R_g2c = db.R_g2c

    def vis(self, gt, pred, save_dir, img_dir, img_name, prob_th=0.5):
        img_path = os.path.join(img_dir, 'raw_data', 'data', img_name)
        print("img_path", img_path)
        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133, projection='3d')

        gt_camera_intrinsics = np.matmul(np.array(gt['calibration'], dtype=np.float32)[:3, :3], self.R_g2c)  # 3*4
        gt_lanes = gt['lanes']

        gt_lanes = gt_lanes

        # only keep the visible portion
        gt_lanes = [np.matmul(self.R_c2g, np.array(lane[::-1]).T).T for lane in gt_lanes if len(lane) > 1]
        # only consider those gt lanes overlapping with sampling range
        gt_lanes = [lane for lane in gt_lanes if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]

        predictions = pred['laneLines']
        pred_lanes = [np.matmul(self.R_c2g, np.array(predictions[ii]['points'][::-1]).T).T for ii in range(len(predictions)) if
                            predictions[ii]['score'] > prob_th]
        pred_lanes_prob = [predictions[ii]['score'] for ii in range(len(predictions)) if
                            predictions[ii]['score'] > prob_th]
        cnt_gt = len(gt_lanes)
        cnt_pred = len(pred_lanes)

        P_gt = np.matmul(self.H_crop, gt_camera_intrinsics)
        img = cv2.imread(img_path)
        img = cv2.warpPerspective(img, self.H_crop, (self.resize_w, self.resize_h))
        img = img.astype(np.float) / 255

        raw_img = cv2.imread(img_path)
        raw_img = raw_img.astype(np.float) / 255

        gt_visibility_mat = np.zeros((cnt_gt, 100))
        pred_visibility_mat = np.zeros((cnt_pred, 100))

        for i in range(cnt_pred):
            # # ATTENTION: ensure y mono increase before interpolation: but it can reduce size
            min_y = np.min(np.array(pred_lanes[i])[:, 1])
            max_y = np.max(np.array(pred_lanes[i])[:, 1])
            x_values, z_values, visibility_vec = resample_laneline_in_y(pred_lanes[i], self.y_samples,
                                                                        out_vis=True)
            pred_lanes[i] = np.vstack([x_values, z_values]).T
            pred_visibility_mat[i, :] = np.logical_and(x_values >= self.x_min,
                                                       np.logical_and(x_values <= self.x_max,
                                                                      np.logical_and(self.y_samples >= min_y,
                                                                                     self.y_samples <= max_y)))
            pred_visibility_mat[i, :] = np.logical_and(pred_visibility_mat[i, :], visibility_vec)

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
            x_values = pred_lanes[i][:, 0]
            z_values = pred_lanes[i][:, 1]
            x_2d, y_2d = self.projective_transformation(P_gt, x_values, self.y_samples, z_values)
            x_2d = x_2d.astype(np.int)
            y_2d = y_2d.astype(np.int)

            prob = pred_lanes_prob[i]
            # if i in match_pred_ids:
            color = [0, 0, 1]    # tp: red
            # else:
                # color = [1, 0, 1]    # fp: red, blue
            for k in range(1, x_2d.shape[0]):
                # only draw the visible portion
                if pred_visibility_mat[i, k - 1] and pred_visibility_mat[i, k]:
                    img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), color[-1::-1], 1)
            ax2.plot(x_values[np.where(pred_visibility_mat[i, :])],
                        self.y_samples[np.where(pred_visibility_mat[i, :])], color=color, linewidth=2)
            ax3.plot(x_values[np.where(pred_visibility_mat[i, :])],
                        self.y_samples[np.where(pred_visibility_mat[i, :])],
                        z_values[np.where(pred_visibility_mat[i, :])], color=color, linewidth=2)

            x_text = None
            y_text = None
            for k in range(3, x_2d.shape[0]):
                if x_2d[k] > 0 and x_2d[k] < img.shape[1] and y_2d[k] > 0 and y_2d[k] < img.shape[0]:
                    x_text, y_text = x_2d[k], y_2d[k]
                    break
            if x_text is not None:
                cv2.putText(img, str(prob)[:5], (int(x_text), int(y_text)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2, color=(1, 0, 0))

        for i in range(cnt_gt):
            x_values = gt_lanes[i][:, 0]
            z_values = gt_lanes[i][:, 1]
            x_2d, y_2d = self.projective_transformation(P_gt, x_values, self.y_samples, z_values)
            x_2d = x_2d.astype(np.int)
            y_2d = y_2d.astype(np.int)
            color = [1, 0, 0]   # tp: blue
            for k in range(1, x_2d.shape[0]):
                # only draw the visible portion
                if gt_visibility_mat[i, k - 1] and gt_visibility_mat[i, k]:
                    img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), color[-1::-1], 1)
            ax2.plot(x_values[np.where(gt_visibility_mat[i, :])],
                        self.y_samples[np.where(gt_visibility_mat[i, :])], color=color, linewidth=2)
            ax3.plot(x_values[np.where(gt_visibility_mat[i, :])],
                        self.y_samples[np.where(gt_visibility_mat[i, :])],
                        z_values[np.where(gt_visibility_mat[i, :])], color=color, linewidth=2)

        ax1.imshow(img[:, :, [2, 1, 0]])

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

    def visualize(self, pred_file, gt_file, prob_th=0.5, img_dir=None, save_dir=None, vis_step=20):
        mmcv.mkdir_or_exist(save_dir)
        pred_lines = open(pred_file).readlines()
        json_pred = [json.loads(line) for line in pred_lines]
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        if len(json_gt) != len(json_pred):
            warnings.warn('We do not get the predictions of all the test tasks')
        gts = {l['filename']: l for l in json_gt}
        for i, pred in tqdm(enumerate(json_pred)):
            if i % vis_step != 0:
                continue
            raw_file = pred['file_path']
            gt = gts[raw_file.split('.')[0]]
            self.vis(gt, pred, save_dir, img_dir, raw_file)

    def projective_transformation(self, Matrix, x, y, z):
        """
        Helper function to transform coordinates defined by transformation matrix

        Args:
                Matrix (multi dim - array): 3x4 projection matrix
                x (array): original x coordinates
                y (array): original y coordinates
                z (array): original z coordinates
        """
        coordinates = np.vstack((x, y, z))
        trans = np.matmul(Matrix, coordinates)

        x_vals = trans[0, :]/(trans[2, :] + 1e-8)
        y_vals = trans[1, :]/(trans[2, :] + 1e-8)
        return x_vals, y_vals