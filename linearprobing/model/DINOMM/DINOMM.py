import torch
import torch.nn as nn
import model.DINOMM.dinomm_vits as vits

class dinomm(nn.Module):
    def __init__(self,modality='RGB', num_classes=19):
        super(dinomm, self).__init__()
        self.modality = modality
        self.backbone = vits.__dict__['vit_small'](patch_size=8, num_classes=0, in_chans=14)
        
        state_dict = torch.load(r'G:\project\mycls\weights/DINOMM/B14_vits8_dinomm_ep99.pth', map_location='cpu')['student']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = self.backbone.load_state_dict(state_dict, strict=False)
        # print(msg)
        self.backbone.head = torch.nn.Identity()
        
        self.head = torch.nn.Linear(self.backbone.num_features, num_classes)
        
        
    def forward(self, rgb, sar):
        if self.modality == 'SAR':
            rgb = torch.zeros_like(rgb)
        if self.modality == 'RGB':
            sar = torch.zeros_like(sar)
        sar = sar[:, [1, 0], :, :]
        x = torch.cat((rgb, sar), dim=1)
        out = self.backbone(x)
        return self.head(out)
    