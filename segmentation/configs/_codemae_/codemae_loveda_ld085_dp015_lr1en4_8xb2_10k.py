_base_ = [
    '../_base_/models/upernet_beit.py',
    '../_custom_dataset_/loveda.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

pretrained = '/root/autodl-tmp/weights/INP_dec8share_dino_pixC_CA_800E/checkpoint-800.pth'

# auto_scale_lr = dict(enable=True, base_batch_size=16)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[73.1018, 79.1335, 75.9548],
        std=[39.2882, 35.3748, 34.0695],
        bgr_to_rgb=True,
        size=(512, 512),
        pad_val=0,
        seg_pad_val=255),
    pretrained=None,
    backbone=dict(
        type='iTPN_pixel_dualembed',
        modality='RGB',
        img_size=224,
        patch_size=16,
        embed_dim=512,
        mlp_depth1=3,
        mlp_depth2=3,
        depth=24,
        num_heads=8,
        mlp_ratio=4,
        fpn_dim=256,
        fpn_depth=1,
        qkv_bias=True,
        ape=True,
        rpe=False,
        drop_path_rate=0.15,
        num_outs=4,
        use_checkpoint=False,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=None,
    decode_head=dict(
        in_channels=[256, 256, 256, 256],
        num_classes=7,
        channels=768,
    ),
    auxiliary_head=dict(
        in_channels=256,
        num_classes=7
    ),
    test_cfg=dict(mode='slide', stride=(256, 256), crop_size=(512, 512)))

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    clip_grad=dict(max_norm=35, norm_type=2),
    constructor='itpn_LayerDecayOptimizerConstructor',
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
        lr=3e-5,
        betas=(0.9, 0.999),
        weight_decay=0.05))


train_cfg = dict(type='IterBasedTrainLoop', max_iters=10000, val_interval=1000000)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(
        type='PolyLR',
        eta_min=0.,
        power=1.0,
        begin=500,
        end=10000,
        by_epoch=False,
    )
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='auto',
        max_keep_ckpts=1,
        by_epoch=False, interval=2000))