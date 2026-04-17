import torch
import json
import torch.nn as nn
from functools import partial

from .swin_transformer import SwinTransformer, SharedDSwin
from .moby import MoBY
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def dotdictify(d):
    """recursively wrap a dictionary and
    all the dictionaries that it contains
    with the dotdict class
    """
    d = dotdict(d)
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dotdictify(v)
    return d

def build_model(config):
    model_type = config.MODEL.TYPE
    encoder_type = config.MODEL.MOBY.ENCODER

    if encoder_type == "swin":
        enc = partial(
            SwinTransformer,
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            norm_befor_mlp=config.MODEL.SWIN.NORM_BEFORE_MLP,
        )
    else:
        raise NotImplementedError(f"--> Unknown encoder_type: {encoder_type}")

    if model_type == "moby":
        encoder = enc(
            num_classes=0,
            drop_path_rate=config.MODEL.MOBY.ONLINE_DROP_PATH_RATE,
        )
        encoder_k = enc(
            num_classes=0,
            drop_path_rate=config.MODEL.MOBY.TARGET_DROP_PATH_RATE,
        )
        model = MoBY(
            cfg=config,
            encoder=encoder,
            encoder_k=encoder_k,
            contrast_momentum=config.MODEL.MOBY.CONTRAST_MOMENTUM,
            contrast_temperature=config.MODEL.MOBY.CONTRAST_TEMPERATURE,
            contrast_num_negative=config.MODEL.MOBY.CONTRAST_NUM_NEGATIVE,
            proj_num_layers=config.MODEL.MOBY.PROJ_NUM_LAYERS,
            pred_num_layers=config.MODEL.MOBY.PRED_NUM_LAYERS,
        )
    elif model_type == "linear":
        model = enc(
            num_classes=config.MODEL.NUM_CLASSES,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
        )
    elif model_type == "d-swin":
        model = enc(
            num_classes=config.MODEL.NUM_CLASSES,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
        )
    elif model_type == "shared-d-swin":
        """a d-swin trained swin transformer where blocks 2 and beyond are shared across views"""
        block1_s1 = SwinTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.DATA.S1_INPUT_CHANS,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            norm_befor_mlp=config.MODEL.SWIN.NORM_BEFORE_MLP,
            num_classes=config.MODEL.NUM_CLASSES,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
        )
        block1_s2 = SwinTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.DATA.S2_INPUT_CHANS,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            norm_befor_mlp=config.MODEL.SWIN.NORM_BEFORE_MLP,
            num_classes=config.MODEL.NUM_CLASSES,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
        )
        shared = SwinTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.DATA.S2_INPUT_CHANS,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            norm_befor_mlp=config.MODEL.SWIN.NORM_BEFORE_MLP,
            num_classes=config.MODEL.NUM_CLASSES,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
        )

        s1_encoder = get_block1_encoder(block1_s1)
        s2_encoder = get_block1_encoder(block1_s2)
        shared_backbone = get_model_shared_backbone(shared)

        model = SharedDSwin(s1_encoder, s2_encoder, shared_backbone)

    else:
        raise NotImplementedError(f"--> Unknown model_type: {model_type}")

    return model

def forward_block1(self, x):
    x = self.patch_embed(x)
    if self.ape:
        x = x + self.absolute_pos_embed
    x = self.pos_drop(x)

    for layer in self.layers:
        x = layer(x)
    # x = self.layers[0](x)

    return x

def forward_shared_features(self, x):
    for layer in self.layers:
        x = layer(x)

    x = self.norm(x)  # B L C
    x = self.avgpool(x.transpose(1, 2))  # B C 1
    x = torch.flatten(x, 1)
    return x

def get_block1_encoder(t_swin_model):
    del t_swin_model.layers[-1]
    del t_swin_model.layers[-1]
    del t_swin_model.layers[-1]
    del t_swin_model.norm
    del t_swin_model.avgpool
    del t_swin_model.head

    t_swin_model.forward = partial(forward_block1, t_swin_model)

    return t_swin_model

def get_model_shared_backbone(t_swin_model):
    # del t_swin_model.patch_embed

    del t_swin_model.pos_drop
    del t_swin_model.layers[0]
    del t_swin_model.head

    t_swin_model.forward_features = partial(forward_shared_features, t_swin_model)

    return t_swin_model

class DoubleSwinTransformerDownstream(nn.Module):
    def __init__(self, encoder1, encoder2, out_dim, freeze_layers=True):
        super(DoubleSwinTransformerDownstream, self).__init__()

        self.backbone1 = encoder1
        self.backbone2 = encoder2

        # add final linear layer
        self.fc = nn.Linear(
            self.backbone2.num_features + self.backbone1.num_features,
            out_dim,
            bias=True,
        )

        # freeze all layers but the last fc
        if freeze_layers:
            for name, param in self.named_parameters():
                if name not in ["fc.weight", "fc.bias"]:
                    param.requires_grad = False

    def forward(self, s1, s2):
        x1, _, _ = self.backbone1.forward_features(s1)
        x2, _, _ = self.backbone2.forward_features(s2)

        z = torch.cat([x1, x2], dim=1)
        z = self.fc(z)
        return z
    

class SwinSSL(nn.Module):
    def __init__(self, modality='RGB', num_classes=19):
        super().__init__()
        self.modality = modality
        
        with open("./model/SwinSSL/swinsslconfig.json", "r") as fp:
            config = dotdictify(json.load(fp))
        
        if self.modality == 'SAR':
            config.model_config.MODEL.SWIN.IN_CHANS = 2
            self.backbone = build_model(config.model_config)
            checkpoint = torch.load('./weights/SwinSSL/swin_t.pth', map_location='cpu')
            weights = checkpoint['state_dict']
            weights = {k[len("backbone1."):]: v for k, v in weights.items() if "backbone1" in k}
            self.backbone.load_state_dict(weights)
        elif self.modality == 'RGB':
            config.model_config.MODEL.SWIN.IN_CHANS = 13
            self.backbone = build_model(config.model_config)
            checkpoint = torch.load('./weights/SwinSSL/swin_t.pth', map_location='cpu')
            weights = checkpoint['state_dict']
            weights = {k[len("backbone2."):]: v for k, v in weights.items() if "backbone2" in k}
            self.backbone.load_state_dict(weights)
        self.backbone.head = nn.Identity()
        
        self.head = torch.nn.Linear(self.backbone.num_features, num_classes)
    
    def forward(self, s2, s1):
        if self.modality == 'SAR':
            ft = self.backbone(s1)
        elif self.modality == 'RGB':
            ft = self.backbone(s2)
        return self.head(ft)
