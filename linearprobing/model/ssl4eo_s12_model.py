import torch
from torchvision import models
import os
import math
import timm
import torch.nn as nn
import model.models_vit as mae_vit
from timm.models.vision_transformer import PatchEmbed, Block, Attention, DropPath, Mlp, trunc_normal_
from functools import partial


# vanllina MAE
class vit(nn.Module):
    def __init__(self, ckpt_dir=None, modality='RGB'):
        super().__init__()

        self.backbone = mae_vit.vit_base_patch16(img_size=224, in_chans=13 if modality == 'RGB' else 2)
        if ckpt_dir is not None:
            ckpt = torch.load(ckpt_dir, map_location='cpu')
        else:
            if modality == 'RGB':
                path = './weights/SSL4EO-S12/B13_vitb16_mae_ep99.pth'
            elif modality == 'SAR':
                path = './weights/SSL4EO-S12/B2_vitb16_mae_ep99.pth'
            else:
                raise NotImplementedError
            ckpt = torch.load(path, map_location='cpu')
        msg = self.backbone.load_state_dict(ckpt['model'], strict=False)
        # print(msg)
        self.backbone.head = nn.Identity()
    
    def forward(self, x):
        B = x.shape[0]
        x = self.backbone.patch_embed(x)
        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)

        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm(x)

        return x[:, 0, :]
        # return x[:, 1:, :].mean(dim=1)

class SS4EOS12(torch.nn.Module):
    def __init__(self, modality, num_classes):
        super().__init__()

        self.modality = modality
        self.backbone = vit(modality=modality)
        self.head = torch.nn.Linear(768, num_classes)

    def forward(self, rgb, sar):
        if self.modality == 'RGB':
            out = self.backbone(rgb)
        elif self.modality == 'SAR':
            out = self.backbone(sar)
        return self.head(out)
