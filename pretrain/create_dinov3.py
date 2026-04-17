import torch.nn as nn
import torch
from dinov3.dinov3 import DINOV3

class dino_distill(nn.Module):
    def __init__(self, model_name="./dinov3", **kwargs):
        super().__init__()
        self.scaling_layer = ScalingLayerForDINOv3()
        self.teacher_model = DINOV3()
        self.LN = nn.LayerNorm(1024, elementwise_affine=False)
        
    @torch.no_grad()
    def get_target(self, x):
        norm_imgs = self.scaling_layer(x)
        target = self.teacher_model(norm_imgs)
        target = self.LN(target)
        patch_tokens = target[:, 5:, :]
        return patch_tokens

    def forward(self, x, **kwargs):
        """
        x: shape [B, 3, H, W] in [0, 1]
        """
        target = self.get_target(x, **kwargs)
        return target


class ScalingLayerForDINOv3(nn.Module):
    def __init__(self):
        super(ScalingLayerForDINOv3, self).__init__()
        self.register_buffer('shift', torch.Tensor([0.430, 0.411, 0.296])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.213, 0.156, 0.143])[None, :, None, None])

    def forward(self, inp):
        out = (inp - self.shift) / self.scale
        return out
    
if __name__ == '__main__':
    import albumentations as A
    import cv2
    from PIL import Image
    import numpy as np
    
    transform = A.Compose([
                A.Resize(
                    224, 224, interpolation=Image.BILINEAR
                )])
    image = Image.open(r'F:\TESTDATASETS\LoveDA\img\trainval\31.png').convert('RGB')
    # a = np.array(image)
    # b = a * 0.00392156862745098
    data = transform(image=np.array(image))["image"]
    x = torch.from_numpy(data.astype(np.float32) /255).permute(2, 0, 1).float().unsqueeze(0)
    model = dino_distill()
    output = model(x)
    print()