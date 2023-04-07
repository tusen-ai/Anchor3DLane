anchor_y_steps = [  2,  5,  8,  10,  15,  20,  25,  30,  40,  50]
feat_y_steps = [  2,  5,  8,  10,  15,  20,  25,  30,  40,  50]
anchor_len = len(anchor_y_steps)

# dataset settings
dataset_type = 'ONCEDataset'
data_root = './data/ONCE/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
input_size = (360, 480)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(input_size[1], input_size[0]), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='MaskGenerate', input_size=input_size),
    dict(type='LaneFormat'),
    dict(type='Collect', keys=['img', 'img_metas', 'gt_3dlanes', 'gt_project_matrix', 'mask']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(input_size[1], input_size[0]), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='MaskGenerate', input_size=input_size),
    dict(type='LaneFormat'),
    dict(type='Collect', keys=['img', 'img_metas', 'gt_3dlanes', 'gt_project_matrix', 'mask']),
]

dataset_config = dict(
    max_lanes = 8,
    input_size = input_size,
)

test_config = dict(
    side_range_l = -10,
    side_range_h = 10,
    fwd_range_l = 0,
    fwd_range_h = 50,
    height_range_l = 0,
    height_range_h = 5,
    res = 0.05,
    lane_width_x = 30,
    lane_width_y = 10,
    iou_thresh = 0.3,
    distance_thresh = 0.3,
    process_num = 10,
    score_l = 0.10,
    score_h = 1,
    score_step = 0.05,
    exp_name = 'evaluation')

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        data_list='train.txt',
        y_steps = anchor_y_steps,
        dataset_config=dataset_config,
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        data_list='val.txt',
        dataset_config=dataset_config,
        test_config=test_config,
        y_steps=anchor_y_steps,
        test_mode=True,
        pipeline=test_pipeline))

# model setting
model = dict(
    type='Anchor3DLane',
    backbone=dict(
        type='EfficientNet',
        arch='b3',
        lv6=False,
        lv5=True,
        lv4=True,
        lv3=True,
        stride=1),
    backbone_dim=232,
    enc_layers=1,
    y_steps=anchor_y_steps,
    feat_y_steps=feat_y_steps,
    anchor_cfg=dict(
        pitches=[5, 2, 1, 0, -1, -2, -5],
        yaws=[
            30, 20, 15, 10, 7, 5, 3, 1, 0, -1, -3, -5, -7, -10, -15, -20, -30
        ],
        num_x=45,
        start_z=-1.5),
    db_cfg=dict(
        org_h=1020,
        org_w=1920,
        resize_h=360,
        resize_w=480,
        ipm_h=208,
        ipm_w=128,
        pitch=3,
        cam_height=1.55,
        crop_y=0,
        top_view_region=[[-10, 103], [10, 103], [-10, 3], [10, 3]]),
    attn_dim=64,
    drop_out=0.0,
    num_heads=2,
    dim_feedforward=128,
    pre_norm=False,
    feat_size=(45, 60),
    num_category=2,
    loss_lane=dict(
        type='LaneLoss',
        loss_weights=dict(
            cls_loss=1, reg_losses_x=1, reg_losses_z=1, reg_losses_vis=1),
        assign_cfg=dict(
            type='TopkAssigner',
            pos_k=3,
            neg_k=450,
            anchor_len=anchor_len,
            metric='Euclidean'),
        anchor_len=anchor_len,
        anchor_steps=anchor_y_steps,
        gt_anchor_len=50),
    train_cfg=dict(nms_thres=0, conf_threshold=0),
    test_cfg=dict(
        nms_thres=2,
        conf_threshold=0.3,
        refine_vis=True,
        vis_thresh=0.5,
        debug=False,
        use_sigmoid=False))

# training setting
data_shuffle = True
optimizer = dict(type='Adam', lr=2e-4)
optimizer_config = dict()

# learning policy
lr_config = dict(policy='step', step=[50000,], by_epoch=False)

# runtime settings
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
work_dir = 'output/once/anchor3dlane_effb3'