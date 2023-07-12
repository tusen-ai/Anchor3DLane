import os
import json
import numpy as np
import mmcv
import tqdm
import pickle
import argparse
import glob
from mmseg.datasets.tools.utils import *

def make_lane_y_mono_inc(lane):
    """
        Due to lose of height dim, projected lanes to flat ground plane may not have monotonically increasing y.
        This function trace the y with monotonically increasing y, and output a pruned lane
    :param lane:
    :return:
    """
    idx2del = []
    max_y = lane[0, 1]
    for i in range(1, lane.shape[0]):
        # hard-coded a smallest step, so the far-away near horizontal tail can be pruned
        if lane[i, 1] <= max_y:
            idx2del.append(i)
        else:
            max_y = lane[i, 1]
    lane = np.delete(lane, idx2del, 0)
    return lane

def extract_data(anno_file, data_root, target_path, max_lanes=8, test_mode=False):
    R_c2g = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
    R_g2c = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    anchor_y_steps = np.linspace(1, 50, 50)
    anchor_len = len(anchor_y_steps)
    image_id  = 0
    old_annotations = {}
    cnt_idx = 0
    with open(anno_file, 'r') as anno_obj:
        for line in anno_obj:
            cnt_idx += 1
            info_dict = json.loads(line)
            image_path = os.path.join('raw_data', info_dict['filename'])+'.jpg'
            assert os.path.exists(os.path.join(data_root, image_path)), '{:s} not exist'.format(os.path.join(data_root, image_path))
            if not test_mode and info_dict['lane_num'] > max_lanes:
                continue
            cam_intrinsics = np.array(info_dict['calibration'], dtype=np.float32)[:3, :3]
            cam_intrinsics = np.matmul(cam_intrinsics, R_g2c)

            gt_lanes_packed = info_dict['lanes']   # [N, l, 3]
            if len(gt_lanes_packed) < 1:
                if test_mode:
                    old_annotations[image_id] = {'path': image_path,
                                                'gt_3dlanes': [],
                                                'categories': [],
                                                'aug': False,
                                                'relative_path': info_dict['filename'],
                                                'gt_camera_intrinsic': cam_intrinsics,
                                                'json_line': info_dict,}
                    image_id += 1
                continue

            gt_lane_pts, gt_lane_visibility, gt_laneline_category = [], [], []
            for i, gt_lane_packed in enumerate(gt_lanes_packed):
                # A GT lane can be either 2D or 3D
                # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
                # [N, 3]
                gt_lane_packed = gt_lane_packed[::-1]
                lane = np.array(gt_lane_packed, dtype=np.float32)   # [num_p, 3]
                lane_visibility = np.ones(len(lane), dtype=np.float32)  # [num_p]
                lane = np.matmul(R_c2g, lane.T)   # [3, num_p]
                lane = lane.T
                gt_lane_pts.append(lane)
                gt_lane_visibility.append(lane_visibility)
                lane_cate = 1
                gt_laneline_category.append(lane_cate)

            gt_lanes = []
            for i, lane in enumerate(gt_lane_pts):
                # A GT lane can be either 2D or 3D
                # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
                # Resample
                pruned_lane = make_lane_y_mono_inc(lane)
                if pruned_lane.shape[0] < 2:
                    continue
                x_values, z_values, visibility_vec = resample_laneline_in_y(pruned_lane, anchor_y_steps, out_vis=True)
                pruned_lane = np.stack([x_values, z_values, visibility_vec], axis=-1)
                if sum(pruned_lane[:, -1]) > 1:
                    gt_lanes.append(pruned_lane)

            old_annotations[image_id] = {
                'path': image_path,
                'gt_3dlanes': gt_lanes,
                'categories': gt_laneline_category,
                'aug': False,
                'relative_path': info_dict['filename'],
                'gt_camera_intrinsic': cam_intrinsics,
                'json_line': info_dict
            }
            image_id += 1
            print("image_id", image_id, cnt_idx)
            if test_mode and image_id != cnt_idx:
                raise Exception("missing test files")
    print("total_len of old anno", len(old_annotations))
    
    print('Now transforming annotations...')
    for image_id, old_anno in tqdm.tqdm(old_annotations.items()):
        new_anno = transform_annotation(old_anno, max_lanes, anchor_len, test_mode)
        anno = {}
        anno['filename'] = new_anno['path']
        anno['gt_3dlanes'] = new_anno['gt_3dlanes']
        anno['gt_camera_intrinsic'] = new_anno['gt_camera_intrinsic']
        anno['old_anno'] = new_anno['old_anno']
        pickle_path = os.path.join(target_path, '/'.join(anno['filename'].split('/')[-3:-1]))
        mmcv.mkdir_or_exist(pickle_path)
        pickle_file = os.path.join(target_path, '/'.join(anno['filename'].split('/')[-3:]).replace('.jpg', '.pkl'))
        print("path:", pickle_path)
        print("file:", pickle_file)
        w = open(pickle_file, 'wb')
        pickle.dump({'image_id':anno['filename'],
                        'gt_3dlanes':anno['gt_3dlanes'],
                        'gt_camera_intrinsic':anno['gt_camera_intrinsic']}, w)
        w.close()


def transform_annotation(anno, max_lanes=8, anchor_len=50, test_mode=False):
    gt_3dlanes = anno['gt_3dlanes']
    categories = anno['categories']
    lanes3d     = np.ones((max_lanes, 5 + anchor_len * 3), dtype=np.float32) * -1e5
    lanes3d[:, 1] = 0
    lanes3d[:, 0] = 1

    for lane_pos, (gt_3dlane, cate) in enumerate(zip(gt_3dlanes, categories)):
        if lane_pos >= 8 and test_mode:
            break
        gt_3dlane = gt_3dlanes[lane_pos]

        lower, upper = gt_3dlane[0][1], gt_3dlane[-1][1]
        Xs = np.array([p[0] for p in gt_3dlane])
        Zs = np.array([p[1] for p in gt_3dlane])
        vis = np.array([p[2] for p in gt_3dlane])
        lanes3d[lane_pos, 0] = 0
        lanes3d[lane_pos, 1] = cate
        lanes3d[lane_pos, 2] = lower    
        lanes3d[lane_pos, 3] = upper
        lanes3d[lane_pos, 4] = (upper - lower)
        lanes3d[lane_pos, 5:(5+anchor_len)] = Xs
        lanes3d[lane_pos, (5+anchor_len):(5+anchor_len*2)] = Zs      
        lanes3d[lane_pos, (5+anchor_len*2):(5+anchor_len*3)] = vis

    new_anno = {
        'path': anno['path'],
        'gt_3dlanes': lanes3d,
        'old_anno': anno,
        'gt_camera_intrinsic': anno['gt_camera_intrinsic'],
    }

    return new_anno

def merge_annotations(anno_path, json_file):
    all_files = glob.glob(os.path.join(anno_path, '*', 'cam01', '*.json'))
    w = open(json_file, 'w')
    for idx, file_name in enumerate(all_files):
        with open(file_name, 'r') as f:
            data = json.load(f)
        data['filename'] = '/'.join(file_name.split('/')[-3:])[:-5]
        s = json.dumps(data)
        w.write(s+'\n')
    w.close()

def generate_datalist(cache_path, data_list, annotation):
    all_cache_file = glob.glob(os.path.join(cache_path, '*', 'cam01', '*.pkl'))
    select_files = []
    with open(annotation, 'r') as r:
        select_files = [json.loads(s)['filename'] for s in r.readlines()]
    with open(data_list, 'w') as w:
        for item in all_cache_file:
            file_name = '/'.join(item[:-4].split('/')[-3:])
            if file_name in select_files:
                w.write(file_name+'.jpg' + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process ONCE-3DLane dataset')
    parser.add_argument('data_root', help='root path of once dataset')
    parser.add_argument('--merge', action='store_true', default=False, help='whether to merge the annotation json files')
    parser.add_argument('--generate', action='store_true', default=False, help='whether to pickle files')
    args = parser.parse_args()
    if args.merge:
        mmcv.mkdir_or_exist(os.path.join(args.data_root, 'data_splits'))
        merge_annotations(os.path.join(args.data_root, 'annotations', 'train'), os.path.join(args.data_root, 'data_splits', 'train.json'))
        merge_annotations(os.path.join(args.data_root, 'annotations', 'val'), os.path.join(args.data_root, 'data_splits', 'val.json'))
    elif args.generate:
        tar_path = os.path.join(args.data_root, 'cache_dense')
        anno_file = os.path.join(args.data_root, 'data_splits/train.json')
        data_list_path = os.path.join(args.data_root, 'data_lists')
        mmcv.mkdir_or_exist(data_list_path, exist_ok=True)
        extract_data(anno_file, args.data_root, tar_path, test_mode=False)
        generate_datalist(tar_path, os.path.join(data_list_path, 'train.txt', anno_file))
        anno_file = os.path.join(args.data_root, 'data_splits/val.json')
        extract_data(anno_file, args.data_root, tar_path, test_mode=True)
        generate_datalist(tar_path, os.path.join(data_list_path, 'val.txt', anno_file))