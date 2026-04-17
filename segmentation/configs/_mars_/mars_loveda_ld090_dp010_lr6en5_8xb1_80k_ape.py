_base_ = [
    '../_base_/models/upernet_swin.py',
    '../_custom_dataset_/loveda_b1.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

pretrained = '/root/autodl-tmp/weights/mars_base_rgb_encoder_only_trans_swinofficial.pth'

# auto_scale_lr = dict(enable=True, base_batch_size=16)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        size=(512, 512),
        pad_val=0,
        seg_pad_val=255),
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='SwinTransformerV2_seg',
        img_size=512,
        drop_path_rate=0.1,
        ape=True,
        use_checkpoint=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    decode_head=dict(
        in_channels=[128, 256, 512, 1024],
        num_classes=7,
    ),
    auxiliary_head=dict(
        in_channels=512,
        num_classes=7
    ),
    test_cfg=dict(mode='slide', stride=(256, 256), crop_size=(512, 512)))

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    clip_grad=dict(max_norm=35, norm_type=2),
    constructor='MaRSLayerDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_type': 'layer_wise',
        'num_layers': 24,
        'decay_rate': 0.9,
        'absolute_pos_embed': dict(decay_mult=0.),
        'relative_position_bias_table': dict(decay_mult=0.),
        'norm': dict(decay_mult=0.)
    },
    optimizer=dict(
        type='AdamW',
        lr=6e-5,
        betas=(0.9, 0.999),
        weight_decay=0.05))


train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=1000000)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(
        type='PolyLR',
        eta_min=0.,
        power=1.0,
        begin=500,
        end=80000,
        by_epoch=False,
    )
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='auto',
        max_keep_ckpts=1,
        by_epoch=False, interval=2000))