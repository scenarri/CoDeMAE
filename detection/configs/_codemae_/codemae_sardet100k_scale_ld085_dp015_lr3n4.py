_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_costum_dataset_/sardet100k_scale.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

pretrained = '/root/autodl-tmp/weights/INP_dec8share_dino_pixC_CA_800E/checkpoint-800.pth'
# auto_scale_lr = dict(enable=True, base_batch_size=16)

model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[35.8299, 35.8299, 35.8299],
        std=[51.4447, 51.4447, 51.4447],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        _delete_=True,
        type='iTPN_pixel_dualembed',
        modality='SAR',
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=512,
        mlp_depth=3,
        fpn_dim=256,
        fpn_depth=1,
        depth=24,
        num_heads=8,
        bridge_mlp_ratio=3.,
        mlp_ratio=4.,
        num_outs=5,
        out_embed_dim=256,
        drop_path_rate=0.15,
        ape=True,
        rpe=False,
        patch_norm=True,
        use_checkpoint=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=None,
    roi_head=dict(
        bbox_head=dict(
            type='ConvFCBBoxHead',
            num_shared_convs=4,
            num_shared_fcs=1,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=6,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0))),
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    clip_grad=dict(max_norm=35, norm_type=2),
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_type': 'layer_wise',
        'num_layers': 31,
        'decay_rate': 0.85,
        'absolute_pos_embed': dict(decay_mult=0.),
        'relative_position_bias_table': dict(decay_mult=0.),
        'norm': dict(decay_mult=0.)
    },
    optimizer=dict(
        type='AdamW',
        lr=0.0003,
        betas=(0.9, 0.999),
        weight_decay=0.05))


default_hooks = dict(
    logger=dict(type='LoggerHook', interval=250),
    checkpoint=dict(
        type='CheckpointHook',
        save_best='auto',
        max_keep_ckpts=1))