import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--dir', type=str)
    return parser

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    
    data_root = f'/root/det/mmseg/{args.dir}'

    for img in os.listdir(data_root):
        img_path = os.path.join(data_root, img)
        img = Image.open(img_path)
        x = np.array(img)

        x = x - 1
        img_ = Image.fromarray(x)
        img_.save(img_path)
        print(f'{img_path} processed')