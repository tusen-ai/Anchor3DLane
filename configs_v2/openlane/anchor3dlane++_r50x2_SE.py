feat_y_steps = [5,  10,  15,  20,  30,  40,  50,  60,  80,  100]
anchor_y_steps = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
lidar_y_steps = [5,  10,  15,  20,  25,  30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
anchor_len = len(anchor_y_steps)

voxel_size = (0.32, 0.32, 0.15)
pc_range = (-30.4, -1.6, -4, 30.4, 75.2, 4)

# dataset settings
dataset_type = 'OpenlaneLidarDataset'
data_root = './data/OpenLane'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
input_size = (720, 960)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(input_size[1], input_size[0]), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='MaskGenerate', input_size=input_size),
    dict(type='LoadPointCloudFromFile', rotate_xy=True),
    dict(type='Voxelization', range=pc_range, 
         voxel_size=voxel_size, max_points_in_voxel=20, max_voxel_num=[32000, 60000], shuffle_points=False),
    dict(type='AngleCalculate'),
    dict(type='LaneFormat'),
    dict(type='Collect', keys=['img', 'img_metas','gt_3dlanes', 'gt_project_matrix', 'mask',
                               'voxels', 'coordinates', 'num_points', 'num_voxels', 'shape']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(input_size[1], input_size[0]), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='MaskGenerate', input_size=input_size),
    dict(type='LoadPointCloudFromFile'),
    dict(type='Voxelization', range=pc_range, 
         voxel_size=voxel_size, max_points_in_voxel=20, max_voxel_num=[32000, 60000], shuffle_points=False),
    dict(type='LaneFormat'),
    dict(type='Collect', keys=['img', 'img_metas', 'gt_3dlanes', 'gt_project_matrix', 'mask',
                               'voxels', 'coordinates', 'num_points', 'num_voxels', 'shape']),
]

dataset_config = dict(
    max_lanes = 25,
    input_size = input_size,
)

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        data_list='training.txt',
        data_list_dir='lane3d_1000_v1.3/data_lists',
        cache_dir='lane3d_1000_v1.3/cache_dense',
        eval_dir='lane3d_1000_v1.3/data_splits',
        dataset_config=dataset_config,
        y_steps=anchor_y_steps,
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        y_steps=anchor_y_steps,
        data_list='validation.txt',
        data_list_dir='lane3d_1000_v1.3/data_lists',
        cache_dir='lane3d_1000_v1.3/cache_dense',
        eval_dir='lane3d_1000_v1.3/data_splits',
        dataset_config=dataset_config, 
        test_mode=True,
        pipeline=test_pipeline))

expert_idx = -1
hidden_dim = 64
lane_loss_prior = dict(
    type = 'LaneLossV2',
    loss_weights = dict(cls_loss = 1,
                        reg_losses_x = 1,
                        reg_losses_z = 1,
                        reg_losses_vis = 1,
                        reg_losses_prior = 20,
                        consist_losses = 0.1),
    assign_cfg = dict(
        cost_class = 3.0,
        cost_dist = 1.0,
    ),
    anchor_len = anchor_len,
    anchor_steps=anchor_y_steps,
    anchor_assign=False,
    delta = 0.1,
    ds = 10
)

lane_loss = dict(
    type = 'LaneLossV2',
    loss_weights = dict(cls_loss = 1,
                        reg_losses_x = 1,
                        reg_losses_z = 1,
                        reg_losses_vis = 1),
    assign_cfg = dict(
        cost_class = 3.0,
        cost_dist = 1.0,
    ),
    anchor_len = anchor_len,
    anchor_steps=anchor_y_steps,
    anchor_assign=False,
    delta = 0.1,
    ds = 10
)

# model setting
model = dict(
    type = 'Anchor3DLanePPFuse',
        backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        with_cp=False,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
    ),
    reader=dict(
        type="VoxelFeatureExtractorV3",
        num_input_features=5,
    ),
    scatter=dict(
        type="SpMiddleResNetFHD", num_input_features=5, ds_factor=8, xy_strides=[2, 1, 1], z_strides=[2, 2, 1]),
    pretrained = 'pretrained/resnet50_v1c-2cccc1ad.pth',
    iter_reg=2,
    neck = dict(type='FPN',
            in_channels=[512, 1024, 2048],
            out_channels=hidden_dim,
            num_outs=3,
            attention=False),
    neck_aux=dict(
        type='MSDA_neck',
        in_channels=[64, 64, 64],
        in_features=[0, 1, 2],
        feature_strides=[1, 1, 1],
        conv_dim=64,
        transformer_enc_layers=3),
    y_steps = anchor_y_steps,
    feat_y_steps = anchor_y_steps,
    lidar_y_steps = lidar_y_steps,
    anchor_cfg = dict(pitches = [5, 2, 1, 0, -1, -2, -5],
        yaws = [30, 20, 15, 10, 7, 5, 3, 1, 0, -1, -3, -5, -7, -10, -15, -20, -30],
        num_x = 30, distances=[3,], anchor_num=30),
    db_cfg = dict(
        org_h = 1280,
        org_w = 1920,
        resize_h = 720,
        resize_w = 960,
        ipm_h = 208,
        ipm_w = 128,
        pitch = 3,
        cam_height = 1.55,
        crop_y = 0,
        K = [[2015., 0., 960.], [0., 2015., 540.], [0., 0., 1.]],
        top_view_region = [[-10, 103], [10, 103], [-10, 3], [10, 3]],
        max_2dpoints = 10,
    ),
    voxel_size = voxel_size,
    grid_shape = (190, 240, 53),
    use_voxel=True,
    pc_range = pc_range,
    anchor_feat_channels = hidden_dim,
    lidar_dims = [32, 64, 128],
    sample_feat_lidar = ['conv2', 'conv3'],
    drop_out=0.,
    num_heads = 2,
    dim_feedforward = 128,
    pre_norm = False,
    feat_sizes=[(90, 120), (90, 120), (90, 120)],
    num_category = 21,
    use_sigmoid = False,
    with_pos = 'cat',
    expert_idx = expert_idx,
    loss_lane = [[lane_loss_prior, lane_loss_prior],
    [lane_loss, lane_loss],
    [lane_loss, lane_loss],],
    train_cfg = dict(
        nms_thres = 0,
        conf_threshold = 0),
    test_cfg = dict(
        nms_thres = 0.1,
        conf_threshold = 0.2,
        test_conf = 0.5,
        refine_vis = True,
        vis_thresh = 0.7
    )
)

# training setting
data_shuffle = True
optimizer = dict(type='Adam', lr=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=[50000,], by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=60000)
checkpoint_config = dict(by_epoch=False, interval=5000)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 10000000)]
cudnn_benchmark = True
work_dir = 'output/openlane/anchor3dlane++_r50x2_se'