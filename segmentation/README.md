# CoDe-MAE Segmentation

We evaluate segmentation capabilities using UperNet as the segmentation head for the LoveDA dataset (RGB) .

This is done based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

<div align="center">
Detailed configurations of fine-tuning on dense prediction tasks.
</div>
<p align="center">
  <img src="../assets/det.png" alt="CoDe-MAE" width="80%">
</p>

## Get Started

Prepare the environment for both detection and segmentation evaluations:
```bash
conda create --name det python=3.8 -y
conda activate det

pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
pip install mmengine==0.7.1
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

pip install yapf==0.40.0
pip install "setuptools>=69.0.3"
pip install opencv-python==4.9.0.80
pip install einops
pip install cython==0.29.28
pip install timm==0.5.4
pip install six==1.16.0

cd detection 
pip install -v -e .

cd ..
cd segmentation
pip install -v -e .
pip install ftfy

```
> **1. MMSeg Tutorial**: See [MMSeg_README.md](MMDet_README.md)

> **2. Configs, logs, and checkpoints of CoDe-MAE and MaRS**: See [benchmark_configs_weights](https://pan.baidu.com/s/1MNFYW0sU5Nkla67Aw0EL3g?pwd=vkpg)
