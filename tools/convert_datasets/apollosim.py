import os
import json
import cv2
from mmseg.datasets.tools.utils import *
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import mmcv
import pickle

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
        
def extract_data(ori_json, data_root, tar_path, test_mode=False, sample_step=1):
    x_min, x_max = -10, 10
    anchor_y_steps = np.linspace(1, 200, 200 // sample_step)
    image_id  = 0
    old_annotations = {}
    cnt_idx = 0
    with open(ori_json, 'r') as anno_obj:
        for line in anno_obj:
            cnt_idx += 1
            info_dict = json.loads(line)
            gt_lane_pts = info_dict['laneLines']
            image_path = info_dict['raw_file']
            if len(gt_lane_pts) < 1:
                if test_mode:
                    old_annotations[image_id] = {
                        'path': image_path,
                        'gt_3dlanes': [],
                        'aug': False,
                        'relative_path': info_dict['raw_file'],
                        'gt_camera_pitch': gt_cam_pitch,
                        'gt_camera_height': gt_cam_height,
                        'json_line': info_dict,}
                    image_id += 1
                continue
            gt_lane_visibility = info_dict['laneLines_visibility']

            assert os.path.exists(os.path.join(data_root, image_path)), '{:s} not exist'.format(os.path.join(data_root, image_path))

            gt_cam_height = info_dict['cam_height']
            gt_cam_pitch = info_dict['cam_pitch']

            gt_lanes = []
            for i, lane in enumerate(gt_lane_pts):
                lane = np.array(lane)
                # A GT lane can be either 2D or 3D
                # if a lane is 3D, the height is intact from 3D GT, so keep it intact here too
                lane_visibility = np.array(gt_lane_visibility[i], dtype=np.float32)
                # prune gt lanes by visibility labels
                pruned_lane = prune_3d_lane_by_visibility(lane, lane_visibility)
                # prune out-of-range points are necessary before transformation -30~30
                pruned_lane = prune_3d_lane_by_range(pruned_lane, 3*x_min, 3*x_max)

                # Resample
                if pruned_lane.shape[0] < 2:
                    continue
                resample_lane = make_lane_y_mono_inc(pruned_lane)

                if resample_lane.shape[0] < 2:
                    continue
                x_values, z_values, visibility_vec = resample_laneline_in_y(resample_lane, anchor_y_steps, out_vis=True)
                resample_lane = np.stack([x_values, z_values, visibility_vec], axis=-1)   # [200, 3]
                if sum(resample_lane[:, -1]) >= 1:
                    gt_lanes.append(resample_lane)

            if len(gt_lanes) == 0 and test_mode:
                old_annotations[image_id] = {
                    'path': image_path,
                    'gt_3dlanes': [],
                    'aug': False,
                    'relative_path': info_dict['raw_file'],
                    'gt_camera_pitch': gt_cam_pitch,
                    'gt_camera_height': gt_cam_height,
                    'json_line': info_dict,
                }
                image_id += 1
                continue
            
            old_annotations[image_id] = {
                'path': image_path,
                'gt_3dlanes': gt_lanes,
                'aug': False,
                'relative_path': info_dict['raw_file'],
                'gt_camera_pitch': gt_cam_pitch,
                'gt_camera_height': gt_cam_height,
                'json_line': info_dict,

            }
            image_id += 1
            print("image_id", image_id, cnt_idx)
            if test_mode and image_id != cnt_idx:
                raise Exception("missing test files")

    print('Now transforming annotations...')
    print("total_len of old anno", len(old_annotations))

    for image_id, old_anno in old_annotations.items():
        new_anno = transform_annotation(old_anno)
        anno = {}
        anno['filename'] = new_anno['path']
        anno['gt_3dlanes'] = new_anno['gt_3dlanes']
        anno['gt_camera_pitch'] = new_anno['gt_camera_pitch']
        anno['gt_camera_height'] = new_anno['gt_camera_height']
        anno['old_anno'] = new_anno['old_anno']
        pickle_path = os.path.join(tar_path, anno['filename'].split('/')[-2])
        mmcv.mkdir_or_exist(pickle_path)
        pickle_file = os.path.join(tar_path, '/'.join(anno['filename'].split('/')[-2:]).replace('.jpg', '.pkl'))
        print("path:", pickle_path)
        print("file:", pickle_file)
        w = open(pickle_file, 'wb')
        pickle.dump({'filename':anno['filename'],
                     'gt_3dlanes':anno['gt_3dlanes'],
                     'gt_camera_pitch':anno['gt_camera_pitch'],
                     'gt_camera_height':anno['gt_camera_height']}, w)
        w.close()

def transform_annotation(anno, max_lanes=7, anchor_len=200, begin_idx=5):
    gflatXnorm = 30
    gflatZnorm = 10
    gt_3dlanes = anno['gt_3dlanes']

    lanes3d     = np.ones((max_lanes, begin_idx + anchor_len * 3), dtype=np.float32) * -1e5
    lanes3d[:, 1] = 0
    lanes3d[:, 0] = 1
    lanes3d_norm = np.ones((max_lanes, begin_idx + anchor_len * 3), dtype=np.float32) * -1e5
    lanes3d_norm[:, 1] = 0
    lanes3d_norm[:, 0] = 1

    for lane_pos, gt_3dlane in enumerate(gt_3dlanes):
        gt_3dlane = gt_3dlanes[lane_pos]
        Xs = np.array([p[0] for p in gt_3dlane])
        Zs = np.array([p[1] for p in gt_3dlane])
        vis = np.array([p[2] for p in gt_3dlane])
        lanes3d[lane_pos, 0] = 0
        lanes3d[lane_pos, 1] = 1

        lanes3d_norm[lane_pos, 0] = 0
        lanes3d_norm[lane_pos, 1] = 1

        lanes3d[lane_pos, begin_idx:(begin_idx+anchor_len)] = Xs
        lanes3d_norm[lane_pos, begin_idx:(begin_idx+anchor_len)] = Xs / gflatXnorm

        lanes3d[lane_pos, (begin_idx+anchor_len):(begin_idx+anchor_len*2)] = Zs
        lanes3d_norm[lane_pos, (begin_idx+anchor_len):(begin_idx+anchor_len*2)] = Zs / gflatZnorm
        
        lanes3d[lane_pos, (begin_idx+anchor_len*2):(begin_idx+anchor_len*3)] = vis  # vis_flag
        lanes3d_norm[lane_pos, (begin_idx+anchor_len*2):(begin_idx+anchor_len*3)] = vis

    new_anno = {
        'path': anno['path'],
        'gt_3dlanes': lanes3d,
        'gt_3dlanes_norm': lanes3d_norm,
        'old_anno': anno,
        'gt_camera_pitch': anno['gt_camera_pitch'],
        'gt_camera_height': anno['gt_camera_height'],
    }

    return new_anno

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process Openlane dataset')
    parser.add_argument('data_root', help='root path of openlane dataset')
    args = parser.parse_args()
    split = ['standard', 'illus_chg', 'rare_subset']
    for s in split:
        tar_path = os.path.join(args.data_root, 'cache_dense')
        ori_json = os.path.join(args.data_root, 'data_splits/{}/train.json'.format(s))
        extract_data(ori_json, args.data_root, tar_path, False)
        ori_json = os.path.join(args.data_root, 'data_splits/{}/test.json'.format(s))
        extract_data(ori_json, args.data_root, tar_path, True)