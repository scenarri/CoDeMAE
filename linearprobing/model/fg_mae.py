import torch
from torchvision import models
import os
import math
import timm
import model.models_vit as mae_vit
import torch.nn as nn

class vit(nn.Module):
    def __init__(self, ckpt_dir=None, modality='RGB'):
        super().__init__()
        
        self.backbone = mae_vit.vit_base_patch16(img_size=224, in_chans=13 if modality == 'RGB' else 2)
        if ckpt_dir is not None:
            ckpt = torch.load(ckpt_dir, map_location='cpu')
        else:
            if modality == 'RGB':
                path = 'G:\project\mycls\weights/fgmae/B13_vitb16_fgmae_ep99.pth'
            elif modality == 'SAR':
                path = 'G:\project\mycls\weights/fgmae/B2_vitb16_fgmae_ep99.pth'
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
        
        # return x[:, 0, :]
        return x[:, 1:, :].mean(dim=1)
    
class FGMAE(torch.nn.Module):
    def __init__(self, modality='RGB', num_classes=10):
        super().__init__()
        
        self.modality = modality
        self.backbone = vit(modality=modality)
        self.backbone.head = torch.nn.Identity()
        
        self.head = torch.nn.Linear(768, num_classes)
        
    
    def forward(self, rgb, sar):
        if self.modality == 'RGB':
            out = self.backbone(rgb)
        else:
            out = self.backbone(sar)
        return self.head(out)