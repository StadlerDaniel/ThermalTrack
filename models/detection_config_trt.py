# data_root = 'TMOT_DATASET_ROOT'
import os
import inspect
root_dir = os.path.dirname(os.path.dirname(inspect.getfile(lambda: None)))
ann_file = os.path.join(root_dir, 'val-thermal.json')

backend_args = None
batch_shapes_cfg = None
bs = 8
class_name = ('person', )
dataset_type = 'YOLOv5CocoDataset'
deepen_factor = 0.33
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmyolo'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
last_stage_out_channels = 1024
launcher = 'none'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
loss_bbox_weight = 7.5
loss_cls_weight = 0.75
loss_dfl_weight = 0.375
lr = 5e-05
metainfo = dict(
    classes=('person', ), palette=[
        (
            0,
            255,
            0,
        ),
    ])
model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        arch='P5',
        deepen_factor=0.33,
        last_stage_out_channels=1024,
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        type='YOLOv8CSPDarknet',
        widen_factor=0.5),
    bbox_head=dict(
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        head_module=dict(
            act_cfg=dict(inplace=True, type='SiLU'),
            featmap_strides=[
                8,
                16,
                32,
            ],
            in_channels=[
                256,
                512,
                1024,
            ],
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            num_classes=1,
            reg_max=16,
            type='YOLOv8HeadModule',
            widen_factor=0.5),
        loss_bbox=dict(
            bbox_format='xyxy',
            iou_mode='ciou',
            loss_weight=7.5,
            reduction='sum',
            return_iou=False,
            type='IoULoss'),
        loss_cls=dict(
            loss_weight=0.75,
            reduction='none',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        loss_dfl=dict(
            loss_weight=0.375,
            reduction='mean',
            type='mmdet.DistributionFocalLoss'),
        prior_generator=dict(
            offset=0.5, strides=[
                8,
                16,
                32,
            ], type='mmdet.MlvlPointGenerator'),
        type='YOLOv8Head'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            0.0,
            0.0,
            0.0,
        ],
        std=[
            255.0,
            255.0,
            255.0,
        ],
        type='YOLOv5DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        deepen_factor=0.33,
        in_channels=[
            256,
            512,
            1024,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_csp_blocks=3,
        out_channels=[
            256,
            512,
            1024,
        ],
        type='YOLOv8PAFPN',
        widen_factor=0.5),
    test_cfg=dict(
        max_per_img=300,
        multi_label=True,
        nms=dict(iou_threshold=0.7, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    train_cfg=dict(
        assigner=dict(
            alpha=0.5,
            beta=6.0,
            eps=1e-09,
            num_classes=1,
            topk=10,
            type='BatchTaskAlignedAssigner',
            use_ciou=True)),
    type='YOLODetector')
model_test_cfg = dict(
    max_per_img=300,
    multi_label=True,
    nms=dict(iou_threshold=0.7, type='nms'),
    nms_pre=30000,
    score_thr=0.001)
norm_cfg = dict(eps=0.001, momentum=0.03, type='BN')
num_classes = 1
num_det_layers = 3
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=5e-05, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
]
persistent_workers = True
resume = False
strides = [
    8,
    16,
    32,
]
tal_alpha = 0.5
tal_beta = 6.0
tal_topk = 10
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=64,
    dataset=dict(
        ann_file='val-thermal-n1.json',
        batch_shapes_cfg=None,
        data_prefix=dict(img=''),
        data_root=data_root,
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                1875,
                1500,
            ), type='Resize'),
            dict(size_divisor=32, type='Pad'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=ann_file,
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        1875,
        1500,
    ), type='Resize'),
    dict(size_divisor=32, type='Pad'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
tmot_train_ann_file = ann_file
tmot_train_data_prefix = ''
tmot_train_data_root = '/net/vid-ssd1/storage/deeplearning/datasets/TP-MOT/tmot_dataset/'
tmot_train_dataset = dict(
    ann_file=ann_file,
    data_prefix=dict(img=''),
    data_root=
    '/net/vid-ssd1/storage/deeplearning/datasets/TP-MOT/tmot_dataset/',
    filter_cfg=dict(filter_empty_gt=False, min_size=0),
    metainfo=dict(classes=('person', ), palette=[
        (
            0,
            255,
            0,
        ),
    ]),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            max_rotate_degree=0.0,
            max_shear_degree=0.0,
            scaling_ratio_range=(
                0.5,
                1.5,
            ),
            type='YOLOv5RandomAffine'),
        dict(
            bbox_params=dict(
                format='pascal_voc',
                label_fields=[
                    'gt_bboxes_labels',
                    'gt_ignore_flags',
                ],
                type='BboxParams'),
            keymap=dict(gt_bboxes='bboxes', img='image'),
            transforms=[
                dict(p=0.01, type='Blur'),
                dict(p=0.01, type='MedianBlur'),
                dict(p=0.01, type='ToGray'),
                dict(p=0.01, type='CLAHE'),
            ],
            type='mmdet.Albu'),
        dict(type='mmdet.YOLOXHSVRandomAug'),
        dict(prob=0.5, type='RandomFlip'),
        dict(keep_ratio=True, scale=(
            1875,
            1500,
        ), type='Resize'),
        dict(size_divisor=32, type='Pad'),
        dict(type='PackDetInputs'),
    ],
    type='YOLOv5CocoDataset')
tmot_val_ann_file = ann_file
tmot_val_data_prefix = ''
tmot_val_data_root = '/net/vid-ssd1/storage/deeplearning/datasets/TP-MOT/tmot_dataset/'
train_batch_size_per_gpu = 8
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=8,
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        datasets=[
            dict(
                ann_file=ann_file,
                data_prefix=dict(img=''),
                data_root=data_root,
                filter_cfg=dict(filter_empty_gt=False, min_size=0),
                metainfo=dict(classes=('person', ), palette=[
                    (
                        0,
                        255,
                        0,
                    ),
                ]),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        max_rotate_degree=0.0,
                        max_shear_degree=0.0,
                        scaling_ratio_range=(
                            0.5,
                            1.5,
                        ),
                        type='YOLOv5RandomAffine'),
                    dict(
                        bbox_params=dict(
                            format='pascal_voc',
                            label_fields=[
                                'gt_bboxes_labels',
                                'gt_ignore_flags',
                            ],
                            type='BboxParams'),
                        keymap=dict(gt_bboxes='bboxes', img='image'),
                        transforms=[
                            dict(p=0.01, type='Blur'),
                            dict(p=0.01, type='MedianBlur'),
                            dict(p=0.01, type='ToGray'),
                            dict(p=0.01, type='CLAHE'),
                        ],
                        type='mmdet.Albu'),
                    dict(type='mmdet.YOLOXHSVRandomAug'),
                    dict(prob=0.5, type='RandomFlip'),
                    dict(keep_ratio=True, scale=(
                        1875,
                        1500,
                    ), type='Resize'),
                    dict(size_divisor=32, type='Pad'),
                    dict(type='PackDetInputs'),
                ],
                type='YOLOv5CocoDataset'),
            dict(
                ann_file=ann_file,
                data_prefix=dict(img=''),
                data_root=data_root,
                filter_cfg=dict(filter_empty_gt=False, min_size=0),
                metainfo=dict(classes=('person', ), palette=[
                    (
                        0,
                        255,
                        0,
                    ),
                ]),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        max_rotate_degree=0.0,
                        max_shear_degree=0.0,
                        scaling_ratio_range=(
                            0.5,
                            1.5,
                        ),
                        type='YOLOv5RandomAffine'),
                    dict(
                        bbox_params=dict(
                            format='pascal_voc',
                            label_fields=[
                                'gt_bboxes_labels',
                                'gt_ignore_flags',
                            ],
                            type='BboxParams'),
                        keymap=dict(gt_bboxes='bboxes', img='image'),
                        transforms=[
                            dict(p=0.01, type='Blur'),
                            dict(p=0.01, type='MedianBlur'),
                            dict(p=0.01, type='ToGray'),
                            dict(p=0.01, type='CLAHE'),
                        ],
                        type='mmdet.Albu'),
                    dict(type='mmdet.YOLOXHSVRandomAug'),
                    dict(prob=0.5, type='RandomFlip'),
                    dict(keep_ratio=True, scale=(
                        1875,
                        1500,
                    ), type='Resize'),
                    dict(size_divisor=32, type='Pad'),
                    dict(type='PackDetInputs'),
                ],
                type='YOLOv5CocoDataset'),
            dict(
                ann_file=ann_file,
                data_prefix=dict(img=''),
                data_root=data_root,
                filter_cfg=dict(filter_empty_gt=False, min_size=0),
                metainfo=dict(classes=('person', ), palette=[
                    (
                        0,
                        255,
                        0,
                    ),
                ]),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        max_rotate_degree=0.0,
                        max_shear_degree=0.0,
                        scaling_ratio_range=(
                            0.5,
                            1.5,
                        ),
                        type='YOLOv5RandomAffine'),
                    dict(
                        bbox_params=dict(
                            format='pascal_voc',
                            label_fields=[
                                'gt_bboxes_labels',
                                'gt_ignore_flags',
                            ],
                            type='BboxParams'),
                        keymap=dict(gt_bboxes='bboxes', img='image'),
                        transforms=[
                            dict(p=0.01, type='Blur'),
                            dict(p=0.01, type='MedianBlur'),
                            dict(p=0.01, type='ToGray'),
                            dict(p=0.01, type='CLAHE'),
                        ],
                        type='mmdet.Albu'),
                    dict(type='mmdet.YOLOXHSVRandomAug'),
                    dict(prob=0.5, type='RandomFlip'),
                    dict(keep_ratio=True, scale=(
                        1875,
                        1500,
                    ), type='Resize'),
                    dict(size_divisor=32, type='Pad'),
                    dict(type='PackDetInputs'),
                ],
                type='YOLOv5CocoDataset'),
        ],
        type='ConcatDataset'),
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_num_workers = 8
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.5,
            1.5,
        ),
        type='YOLOv5RandomAffine'),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(prob=0.5, type='RandomFlip'),
    dict(keep_ratio=True, scale=(
        1875,
        1500,
    ), type='Resize'),
    dict(size_divisor=32, type='Pad'),
    dict(type='PackDetInputs'),
]
val_batch_size_per_gpu = 64
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=64,
    dataset=dict(
        ann_file=ann_file,
        batch_shapes_cfg=None,
        data_prefix=dict(img=''),
        data_root=data_root,
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                1875,
                1500,
            ), type='Resize'),
            dict(size_divisor=32, type='Pad'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=ann_file,
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
val_num_workers = 8
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
widen_factor = 0.5
