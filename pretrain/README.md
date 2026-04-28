# Pretrain CoDe-MAE

CoDe-MAE is based on [iTPN](https://github.com/sunsmarterjie/iTPN/tree/main) and [HiViT](https://github.com/zhangxiaosong18/hivit).

In established practices, pre-trained models commonly adopt ImageNet statistics for data standardization during both pre-training and downstream tasks. OSPretrain-1M is composed of patches from 15 datasets, encompassing diverse geographical distributions, land-cover types, imaging bands, and post-processing algorithms. To accommodate this heterogeneity, we apply separate normalization for each constituent dataset and modality during input preprocessing. Consequently, when adapting CoDe-MAE to downstream tasks, we compute and apply the normalization statistics specifically for each target dataset.

During training, we sample optical and SAR data at a 1:1 ratio. Recall that OSPretrain-1M consists of both paired and unpaired portions. In distributed training, we ensure that each GPU receives batches that are entirely either paired or unpaired. When a batch is unpaired, the losses <i>L</i><sub>CCL</sub> and <i>L</i><sub>CDR</sub> are multiplied by zero to avoid confusion. In this case, only the shared encoder and decoder contribute to implicit cross-modal alignment.


## Requiments
* Python 3.8
* Pytorch/torchvision 2.0/0.15.1
* timm 0.3.2
* albumentations 1.4.18
* kornia 0.7.3


First download the [iTPN](https://github.com/sunsmarterjie/iTPN/tree/main) and 
[DINOv3-Sat-L](https://github.com/facebookresearch/dinov3) weights. 

To pre-train CoDe-MAE-B with a batch size of 2048 (8*128 pairs), run the following on 8 GPUs:

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=6667 main_pretrain.py \
        --batch_size 128 \
        --model itpn_base_dec512d8b \
        --decoder_depth 8 \
        --epochs 800 \
        --blr 1.5e-4 \
        --mask_ratio 0.75
        --warmup_epochs 15 \
        --modality Both \
        --INP \
        --dino \
        --cr_layers 8 \
        --dec_share \
        --target pixel \
        --reduction channel \
        --contrast \
        --conditioned \
```
