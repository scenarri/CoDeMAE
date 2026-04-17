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
from typing import Iterable, List, Tuple
from collections import OrderedDict
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

assert timm.__version__ == "0.3.2"  # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEmaV2

import util.lr_decay as lrd
import util.mars_lr_decay as mars_lrd
import util.misc as misc
# from util.flop_count import flop_count
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models

from engine_finetune import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    
    # Dataset parameters
    parser.add_argument('--linprob', action='store_true')
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.add_argument('--model', default='itpn_base_dual', type=str, metavar='MODEL')
    parser.add_argument('--finetune', default='./output/INP_dec8share_dino_pixC_CA_800E/checkpoint-800.pth')
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--layer_decay', type=float, default=0.85)
    parser.add_argument('--drop_path', type=float, default=0.15, metavar='PCT')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR')
    parser.add_argument('--epochs', default=100, type=int)
    
    parser.add_argument('--dataset', default='NWPURESISC45', type=str)
    parser.add_argument('--data_path', default='F:/TESTDATASETS/NWPU-RESISC45', type=str)
    parser.add_argument('--data_split', default='10', type=str)
    parser.add_argument('--nb_classes', default=45, type=int)
    parser.add_argument('--modality', default='RGB', type=str)
    
    # parser.add_argument('--dataset', default='MSTAR', type=str)
    # parser.add_argument('--data_path', default='F:\\TESTDATASETS\\SARCls\\MSTAR_SOC', type=str)
    # parser.add_argument('--data_split', default='40', type=str)
    # parser.add_argument('--nb_classes', default=10, type=int)
    # parser.add_argument('--modality', default='SAR', type=str)
    

    parser.add_argument('--reg', action='store_true')
    
    # Model Exponential Moving Average
    parser.add_argument('--model-ema', action='store_true', default=False,
                        help='Enable tracking moving average of model weights')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=True,
                        help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    parser.add_argument('--model-ema-decay', type=float, default=0.995,
                        help='decay factor for model weights moving average (default: 0.9999)')  # 0.995 0.999
    
    
    # Model parameters
    

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')


    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--opt_beta2', type=float, default=0.999,
                        help='AdamW beta2 (default: 0.999)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
   

    parser.add_argument('--output_dir', default='INP_dec8share_dino_pixC_CA_800E', type=str,
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=8, type=int)
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


def main(args):
    misc.init_distributed_mode(args)

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

    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    model = models.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
        modality=args.modality,
        use_checkpoint=args.use_checkpoint,
    )

    args.finetune = args.finetune[0] if len(args.finetune) == 1 else args.finetune
    if args.finetune and not args.eval:
        if isinstance(args.finetune, (List, Tuple)):
            for finetune, submodel in zip(args.finetune, model):
                checkpoint = torch.load(finetune, map_location='cpu')
                print("Load pre-trained checkpoint from: %s" % finetune)
                if 'model' in checkpoint:
                    checkpoint_model = checkpoint['model']
                else:
                    checkpoint_model = checkpoint
                msg = submodel.load_state_dict(checkpoint_model, strict=False)
                print(msg.missing_keys)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

            print("Load pre-trained checkpoint from: %s" % args.finetune)
            if 'model' in checkpoint:
                checkpoint_model = checkpoint['model']
            else:
                checkpoint_model = checkpoint
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=0.01)
        
        if args.dataset == 'FUSAR' or args.dataset == 'MSTAR' or args.dataset == 'SARACD':
            print(f'BN layer appened')
            bn = torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6)
            model.head = torch.nn.Sequential(bn, model.head)
        

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    model.to(device)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else None)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    if args.model == 'mars':
        param_groups = mars_lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                            no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                            layer_decay=args.layer_decay
                                            )
    else:
        param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                            no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                            layer_decay=args.layer_decay
                                            )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, args.opt_beta2))
    
    # optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, betas=(0.9, args.opt_beta2))
    
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    if len(args.resume) == 0:
        try:
            last_epoch = -1
            for checkpoint in os.listdir(args.output_dir):
                if checkpoint[-4:] == '.pth':
                    epoch = int(checkpoint[:-4].split('-')[1])
                    last_epoch = max(last_epoch, epoch)
            if last_epoch >= 0:
                args.resume = f'{args.output_dir}/checkpoint-{last_epoch}.pth'
        except:
            pass
        
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler,
                    model_ema=model_ema)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        if model_ema is not None:
            state_dict_ema = OrderedDict()
            for k, v in model_ema.module.state_dict().items():
                state_dict_ema.update({k: v.to(device)})
            model.load_state_dict(model_ema.module.state_dict())
            test_stats = evaluate(data_loader_val, model, device)
            print(f"Accuracy of the ema network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        
        train_stats = train_one_epoch(
            model, model_ema, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )
        if (epoch+1) % 5 == 0 or epoch == 0:
            test_stats = evaluate(data_loader_val, model, device)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters,
                     'max perf': max_accuracy}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if model_ema is None:
        return
    state_dict_ema = OrderedDict()
    for k, v in model_ema.module.state_dict().items():
        state_dict_ema.update({k: v.to(device)})
    model_without_ddp.load_state_dict(model_ema.module.state_dict())
    test_stats = evaluate(data_loader_val, model, device)
    print(f"Accuracy of the ema network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    log_stats = {
        **{f'test_ema_{k}': v for k, v in test_stats.items()},
        'epoch': epoch,
        'n_parameters': n_parameters,
    }

    if log_writer is not None:
        log_writer.add_scalar('perf_ema/test_acc1', test_stats['acc1'], epoch)
        log_writer.add_scalar('perf_ema/test_acc5', test_stats['acc5'], epoch)
        log_writer.add_scalar('perf_ema/test_loss', test_stats['loss'], epoch)
    if args.output_dir and misc.is_main_process():
        if log_writer is not None:
            log_writer.flush()
        with open(os.path.join(args.output_dir, f"{str(round(max_accuracy,4))}_log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # if args.seed == 1:
    #     args.seed = np.random.randint(10000)
    args.output_dir = os.path.join('./log_finetune', f'{args.dataset}_{args.output_dir}_lr-{str(args.layer_decay)}_dp-{str(args.drop_path)}_blr-{str(args.blr)}_ep-{args.epochs}_bs-{args.batch_size}_seed-{args.seed}_split-{args.data_split}')
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
