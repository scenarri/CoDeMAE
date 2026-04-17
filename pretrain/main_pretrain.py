# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models

from engine_pretrain import train_one_epoch

import torch.utils.data as data
from opt_sar_dataset import PAIRED_Dataset, MixedBatchSampler

import create_dinov3


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    
    # Model parameters
    parser.add_argument('--model', default='itpn_base_dec512d8b', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=True)
    
    parser.add_argument('--modality', default='Both', type=str)
    
    parser.add_argument('--decoder_depth', default=8, type=int)
    parser.add_argument('--INP', action='store_true', default=True)
    
    # Cross reconstruction parameters
    parser.add_argument('--cr_layers', default=0, type=int)
    parser.add_argument('--target', default='pixel', type=str)  # pixel hog mgf
    parser.add_argument('--reduction', default='channel', type=str)  # channel spatial dual
    parser.add_argument('--dec_share', action='store_true', default=False)  # sharing decoders
    
    # Contrastive learning parameters
    parser.add_argument('--dino', action='store_true', default=True)  # enabling distillation from dinov3
    # parser.set_defaults(dino=True)
    parser.add_argument('--contrast', action='store_true', default=False)
    parser.add_argument('--conditioned', action='store_true', default=False)
    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    
    parser.add_argument('--warmup_epochs', type=int, default=15, metavar='N',
                        help='epochs to warmup LR')
    
    # Dataset parameters
    parser.add_argument('--data_path', default='G:/OSPretrain_1M', type=str,
                        help='dataset path')
    parser.add_argument('--data_resume', default=None, type=str, help='load fixed part of the whole dataset')
    parser.add_argument('--exclude_test', default=['WHU-OPT-SAR', 'OGSOD-2', 'DDHR', 'DFC23_Track1'], type=str, help='')
    parser.add_argument('--exclude_dataset', default=['PIE-RGB-SAR', 'YESeg-OPT-SAR', 'SS4EO-S12'], type=str, help='')
    
    parser.add_argument('--output_dir', default='baseline',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    return parser


def load_INP(args, model):
    if 'base' in args.model:
        ckpt = torch.load('./IN_weights/itpn_base_fpn256.pth', map_location='cpu')
    elif 'large' in args.model:
        ckpt = torch.load('./IN_weights/itpn_large_fpn256.pth', map_location='cpu')
    else:
        raise ValueError()
    new_state_dict = {}
    for key, value in ckpt.items():
        if not key.startswith('mask') and not key.startswith('decoder'):
            if key.startswith('patch_embed'):
                new_key = key.replace('patch_embed', 'patch_embed_rgb')
                new_state_dict[new_key] = value
                new_key = key.replace('patch_embed', 'patch_embed_sar')
                new_state_dict[new_key] = value
            elif key.startswith('decoder_norm'):
                new_key = key.replace('decoder_norm', 'decoder_norm_rgb')
                new_state_dict[new_key] = value
                new_key = key.replace('decoder_norm', 'decoder_norm_sar')
                new_state_dict[new_key] = value
            elif key.startswith('decoder_pred'):
                new_key = key.replace('decoder_pred', 'decoder_pred_rgb')
                new_state_dict[new_key] = value
                new_key = key.replace('decoder_pred', 'decoder_pred_sar')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(msg)


def main(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()
    
    if args.log_dir is None:
        args.log_dir = args.output_dir
    
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    
    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    cudnn.benchmark = True
    
    # simple augmentation
    dataset_train = PAIRED_Dataset(args.data_path, mode='pair', exclude_dataset=args.exclude_dataset,
                                   exclude_test=args.exclude_test, img_size=args.input_size, use_aug=True,
                                   resume=args.data_resume)
    print(len(dataset_train))
    
    num_tasks = misc.get_world_size()
    sampler_train = MixedBatchSampler(dataset_train, batch_size=args.batch_size,
                                      num_replicas=num_tasks, shuffle=True, drop_last=True)
    print("Sampler_train = %s" % str(sampler_train))
    
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    
    data_loader_train = data.DataLoader(
        dataset=dataset_train,
        batch_sampler=sampler_train,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem
    )
    
    # define the model
    model = models.__dict__[args.model](norm_pix_loss=args.norm_pix_loss,
                                        modality=args.modality,
                                        cr_layers=args.cr_layers,
                                        dec_share=args.dec_share,
                                        dino=args.dino,
                                        decoder_depth=args.decoder_depth,
                                        contrast=args.contrast,
                                        conditioned=args.conditioned,
                                        target=args.target,
                                        reduction=args.reduction)
    if args.INP:
        load_INP(args, model)
    
    model.to(device)
    
    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))
    
    if args.modality == 'RGB' or args.modality == 'SAR':
        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    elif args.modality == 'Both':
        eff_batch_size = (2 * args.batch_size) * args.accum_iter * misc.get_world_size()
    else:
        raise NotImplementedError
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    
    if args.dino:
        dino_tea = create_dinov3.dino_distill().to(device)
    else:
        dino_tea = None
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()
    
    # if len(args.resume) == 0:
    #     try:
    #         last_epoch = -1
    #         for checkpoint in os.listdir(args.output_dir):
    #             if checkpoint[-4:] == '.pth':
    #                 epoch = int(checkpoint[:-4].split('-')[1])
    #                 last_epoch = max(last_epoch, epoch)
    #         if last_epoch >= 0:
    #             args.resume = f'{args.output_dir}/checkpoint-{last_epoch}.pth'
    #     except:
    #         pass
    
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.batch_sampler.set_epoch(epoch)
        
        data_loader_train.dataset.set_epoch(epoch)
        data_loader_train.dataset.shuffle()
        
        train_stats = train_one_epoch(
            model, dino_tea, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        
        if args.output_dir and (epoch % 50 == 0 or epoch + 1 == args.epochs or epoch + 5 == args.epochs) and (
                epoch != 0):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }
        
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args.output_dir = os.path.join('./output', args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
