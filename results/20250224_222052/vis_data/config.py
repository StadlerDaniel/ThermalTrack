ann_file = '/home/dan97494/Projekte/Code_PBVS_Challenge/val-thermal.json'
batch_size = 1
data_root = '/net/vid-ssd1/storage/deeplearning/datasets/TP-MOT/tmot_dataset'
default_scope = 'mmyolo'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = '/home/dan97494/Projekte/Code_PBVS_Challenge/models/detector_weights.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    module=dict(
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
                loss_weight=0.5,
                reduction='none',
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True),
            loss_dfl=dict(
                loss_weight=0.375,
                reduction='mean',
                type='mmdet.DistributionFocalLoss'),
            prior_generator=dict(
                offset=0.5,
                strides=[
                    8,
                    16,
                    32,
                ],
                type='mmdet.MlvlPointGenerator'),
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
            max_per_img=1000,
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
        type='YOLODetector'),
    tta_cfg=dict(max_per_img=1000, nms=dict(iou_threshold=0.5, type='nms')),
    type='mmdet.DetTTAModel')
root_dir = '/home/dan97494/Projekte/Code_PBVS_Challenge'
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='/home/dan97494/Projekte/Code_PBVS_Challenge/val-thermal.json',
        batch_shapes_cfg=None,
        data_prefix=dict(img=''),
        data_root=
        '/net/vid-ssd1/storage/deeplearning/datasets/TP-MOT/tmot_dataset',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                transforms=[
                    [
                        dict(
                            transforms=[
                                dict(
                                    keep_ratio=True,
                                    scale=(
                                        1875,
                                        1500,
                                    ),
                                    type='Resize'),
                                dict(size_divisor=32, type='Pad'),
                            ],
                            type='Compose'),
                        dict(
                            transforms=[
                                dict(
                                    keep_ratio=True,
                                    scale=(
                                        2125,
                                        1700,
                                    ),
                                    type='Resize'),
                                dict(size_divisor=32, type='Pad'),
                            ],
                            type='Compose'),
                        dict(
                            transforms=[
                                dict(
                                    keep_ratio=True,
                                    scale=(
                                        2375,
                                        1900,
                                    ),
                                    type='Resize'),
                                dict(size_divisor=32, type='Pad'),
                            ],
                            type='Compose'),
                        dict(
                            transforms=[
                                dict(
                                    keep_ratio=True,
                                    scale=(
                                        2625,
                                        2100,
                                    ),
                                    type='Resize'),
                                dict(size_divisor=32, type='Pad'),
                            ],
                            type='Compose'),
                        dict(
                            transforms=[
                                dict(
                                    keep_ratio=True,
                                    scale=(
                                        2875,
                                        2300,
                                    ),
                                    type='Resize'),
                                dict(size_divisor=32, type='Pad'),
                            ],
                            type='Compose'),
                    ],
                    [
                        dict(prob=0.0, type='mmdet.RandomFlip'),
                    ],
                    [
                        dict(type='mmdet.LoadAnnotations', with_bbox=True),
                    ],
                    [
                        dict(
                            meta_keys=(
                                'img_id',
                                'img_path',
                                'ori_shape',
                                'img_shape',
                                'scale_factor',
                                'pad_param',
                                'flip',
                                'flip_direction',
                            ),
                            type='mmdet.PackDetInputs'),
                    ],
                ],
                type='TestTimeAug'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='/home/dan97494/Projekte/Code_PBVS_Challenge/val-thermal.json',
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
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
work_dir = '/home/dan97494/Projekte/Code_PBVS_Challenge/output'
