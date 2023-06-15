import os
import os.path as osp
import torch
import cv2
import json
import time
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import mmcv
from mmcv.runner import get_dist_info
from mmcv.engine import collect_results_cpu
import tqdm

from mmseg.datasets.tools.vis_apollosim import LaneVis


def postprocess(output, anchor_len=10):
    proposals = output[0]
    logits = F.softmax(proposals[:, 5 + 3 * anchor_len:], dim=1)
    score = 1 - logits[:, 0] 
    proposals[:, 5 + 3 * anchor_len:] = logits   # [N, 2]
    proposals[:, 1] = score
    results = {'proposals_list': proposals.cpu().numpy()}
    return results   # [1, 7, 16]


def test_apollosim(model,
                  data_loader,
                  eval=True,
                  show=False,
                  out_dir=None,
                  **kwargs):
    """Test with single GPU by progressive mode.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        eval (bool): Whether evaluate results. Defalut: True.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
    """
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    model.eval()
    results = []
    dataset = data_loader.dataset
    loader_indices = data_loader.batch_sampler

    pred_file = osp.join(out_dir, 'lane3d_prediction.json')
    print("testing model...")
    for batch_indices, data in tqdm.tqdm(zip(loader_indices, data_loader)):
        with torch.no_grad():
            outputs = model(return_loss=False, **data)
            for output in outputs['proposals_list']:
                result = postprocess(output, anchor_len=dataset.anchor_len)
                results.append(result)  # [7, 16]

    dataset.format_results(results, pred_file)

    # evaluating
    if eval:
        print("evaluating results...")
        test_result = dataset.eval(pred_file)
        json_result = {}
        json_result['AP'] = test_result['AP']
        json_result['F_score'] = test_result['F_score']
        json_result['x_error_close'] = test_result['x_error_close']
        json_result['x_error_far'] = test_result['x_error_far']
        json_result['z_error_close'] = test_result['z_error_close']
        json_result['z_error_far'] = test_result['z_error_far']
        print('F-score:', test_result['F_score'])
        print('AP:', test_result['AP'])
        print("x error close / far:", test_result['x_error_close'], test_result['x_error_far'])
        print("z error close / far:", test_result['z_error_close'], test_result['z_error_far'])
        print("save test result to", osp.join(out_dir, 'evaluation_result.json'))
        with open(osp.join(out_dir, 'evaluation_result.json'), 'w') as f:
            json.dump(test_result, f)

    # visualizing
    if show:
        save_dir = osp.join(out_dir, 'vis')
        mmcv.mkdir_or_exist(save_dir)
        print("visualizing results at", save_dir)
        visualizer = LaneVis(dataset)
        visualizer.visualize(pred_file, gt_file = dataset.eval_file, img_dir = dataset.data_root, 
                             save_dir = save_dir, prob_th=model.module.test_cfg.test_conf)

def test_apollosim_multigpu(model,
                            data_loader,
                            eval=True,
                            show=False,
                            out_dir=None,
                            **kwargs):
    """Test with single GPU by progressive mode.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        eval (bool): Whether evaluate results. Defalut: True.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
    """
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    rank, world_size = get_dist_info()
    tmpdir = os.path.join(out_dir, 'tmp')

    model.eval()
    results = []
    dataset = data_loader.dataset
    loader_indices = data_loader.batch_sampler

    pred_file = osp.join(out_dir, 'lane3d_prediction.json')
    print("testing model...")
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for batch_indices, data in tqdm.tqdm(zip(loader_indices, data_loader)):
        with torch.no_grad():
            outputs = model(return_loss=False, **data)
            for output in outputs['proposals_list']:
                result = postprocess(output, anchor_len=dataset.anchor_len)
                results.append(result)  # [7, 16]

        if rank == 0:
            batch_size = len(result) * world_size
            for _ in range(batch_size):
                prog_bar.update()
    results = collect_results_cpu(results, len(dataset), tmpdir)

    if rank == 0:
        dataset.format_results(results, pred_file)
    else:
        return

    # evaluating
    if eval:
        print("evaluating results...")
        test_result = dataset.eval(pred_file)
        json_result = {}
        json_result['AP'] = test_result['AP']
        json_result['F_score'] = test_result['F_score']
        json_result['x_error_close'] = test_result['x_error_close']
        json_result['x_error_far'] = test_result['x_error_far']
        json_result['z_error_close'] = test_result['z_error_close']
        json_result['z_error_far'] = test_result['z_error_far']
        print('F-score:', test_result['F_score'])
        print('AP:', test_result['AP'])
        print("x error close / far:", test_result['x_error_close'], test_result['x_error_far'])
        print("z error close / far:", test_result['z_error_close'], test_result['z_error_far'])
        print("save test result to", osp.join(out_dir, 'evaluation_result.json'))
        with open(osp.join(out_dir, 'evaluation_result.json'), 'w') as f:
            json.dump(test_result, f)

    # visualizing
    if show:
        save_dir = osp.join(out_dir, 'vis')
        mmcv.mkdir_or_exist(save_dir)
        print("visualizing results at", save_dir)
        visualizer = LaneVis(dataset)
        visualizer.visualize(pred_file, gt_file = dataset.eval_file, img_dir = dataset.data_root, 
                             save_dir = save_dir, prob_th=model.module.test_cfg.test_conf)