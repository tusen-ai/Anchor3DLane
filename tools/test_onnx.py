# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import time
import pdb
import warnings
import cv2
import numpy as np
import torch
import tracemalloc
import copy


import mmcv
from mmengine.model import revert_sync_batchnorm
# from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, HOOKS, DistSamplerSeedHook,
                        #  wrap_fp16_model, DistSamplerSeedHook, build_runner, Fp16OptimizerHook, OptimizerHook)
from mmcv.utils import Config, DictAction, get_git_hash

# from mmseg.core import DistEvalHook, EvalHook, build_optimizer
# from mmseg import digit_version
# from mmseg.apis import init_random_seed, set_random_seed
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_lanedetector
# from mmseg.utils import build_ddp, build_dp, get_device, setup_multi_processes, get_root_logger, collect_env


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--config', default= 'configs/openlane/anchor3dlane.py', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff_seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=None,
        help="--options is deprecated in favor of --cfg_options' and it will "
        'not be supported in version v0.22.0. Override some settings in the '
        'used config, the key-value pair in xxx=yyy format will be merged '
        'into config file. If the value to be overwritten is a list, it '
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        'marks are necessary and that no white space is allowed.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=None,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options

    return args

def main():
    args = parse_args() 
    cfg = Config.fromfile(args.config)
    
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
        
    model = build_lanedetector(cfg.model)
    print('MODEL: ', model)
    model.init_weights()

    distributed = False
    # SyncBN is not support for DP
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        model = revert_sync_batchnorm(model)

    dataset = build_dataset(cfg.data.test)
    
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=1,
        dist=distributed,
        # seed=cfg.seed,
        drop_last=True,
        shuffle=cfg.data_shuffle,
        persistent_workers=False)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })

    # The specific dataloader settings
    train_loader_cfg = {**loader_cfg, **cfg.data.get('test_dataloader', {})}
    data_loaders = build_dataloader(dataset, **train_loader_cfg)
    input_ = next(iter(data_loaders))
    import pickle 
    with open('data.pickle', 'rb') as handle:
        data = pickle.load(handle)
    for i in range(len(data['img_metas']['ori_shape'])):
        data['img_metas']['ori_shape'][i] = data['img_metas']['ori_shape'][i].cpu()
    for i in range(len(data['gt_3dlanes'])):
        data['gt_3dlanes'][i] = data['gt_3dlanes'][i].cpu()

    model.eval()
    from functools import partial

    data_with_metas = (
    data['img'].cpu(),  # Include img_metas
    data['mask'].cpu(),
    data['img_metas'],
    data['gt_3dlanes'],
    data['gt_project_matrix'].cpu())
    # Include other necessary keys from your original data dictionary


    with torch.no_grad():
        torch.onnx.export(
                model, data_with_metas,
                "test.onnx",
                input_names=['input'],
                output_names=['output'],
                export_params=True,
                keep_initializers_as_inputs=False,
                verbose=True,
                opset_version=16,)
                # dynamic_axes=dynamic_axes)
    # breakpoint()



if __name__ == '__main__':
    main()