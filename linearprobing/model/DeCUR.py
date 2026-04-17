import torch
from torchvision import models
import os
import math
import timm
import model.models_vit as mae_vit
import torch.nn as nn

def split_weights(state_dict, prefix):
    """提取指定前缀的权重并去除前缀"""
    return {key[len(prefix):]: val
            for key, val in state_dict.items()
            if key.startswith(prefix)}


class vit(nn.Module):
    def __init__(self, ckpt_dir=None, modality='RGB'):
        super().__init__()
        
        self.backbone = mae_vit.vit_small_patch16(img_size=224, in_chans=13 if modality == 'RGB' else 2)
        if ckpt_dir is not None:
            ckpt = torch.load(ckpt_dir, map_location='cpu')
        else:
            if modality == 'RGB':
                path = 'G:\project\mycls\weights/DeCUR/vits16_ssl4eo-s12_ms_decur_ep100.pth'
            elif modality == 'SAR':
                path = 'G:\project\mycls\weights/DeCUR/vits16_ssl4eo-s12_sar_decur_ep100.pth'
            else:
                raise NotImplementedError
            ckpt = torch.load(path, map_location='cpu')
        msg = self.backbone.load_state_dict(ckpt, strict=False)
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
    
class DeCUR(torch.nn.Module):
    def __init__(self,modality='RGB', num_classes=19):
        super().__init__()
        
        self.modality = modality
        self.backbone = vit(modality=modality)
        self.backbone.head = torch.nn.Identity()
        
        self.head = torch.nn.Linear(384, num_classes)
        

    def forward(self,rgb, sar):
        if self.modality=='RGB':
            out = self.backbone(rgb)
        else:
            out = self.backbone(sar)
        return self.head(out)
    