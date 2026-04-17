# CoDe-MAE Linear Probing

To enable a fair comparison with existing optical–SAR foundation models, we evaluate on six optical–SAR registered datasets for land-cover classification, including PIE-RGB-SAR, DDHR-SK, WHU-OPT-SAR, DFC20, BigEarthNet-MM, and EuroSat-MM. Among these, the first four provide pixel-level annotations, BigEarthNet-MM provides multi-label classification labels, and EuroSat-MM provides single-label classification labels. Following SwinSSL, we retain categories that account for more than 10% of the total pixels as valid labels. Accordingly, we use micro mAP as the evaluation metric for the first five datasets, while EuroSat-MM is evaluated using overall accuracy (OA). We freeze the backbone and perform linear probing by adding a Batch Normalization layer before the classifier. Following FG-MAE, we adopt the SGD optimizer with a learning rate of 0.5, momentum of 0.9, and weight decay of 0 for 50-epoch training. All input images are resized to match the pretraining resolution of each compared model. The final results are averaged over 5 distinct seeds.

## Requiments
* Python 3.10
* Pytorch/torchvision 2.1/0.16.0
* timm 0.5.4
* albumentations 2.0.7
* torchmetrics 1.7.1

> **1. Configurations**: See [run_trainings.py](run_trainings.py)

> **2. Logs**: See [./output](ouput)

> **3. Aggregate results**: See [result_stats.py](result_stats.py)
