# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ELECTRA https://github.com/google-research/electra
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import json
from DamageFormer_iTPN import PatchEmbed, PatchMerge

def get_layer_id(var_name, num_max_layer):
    if var_name.endswith("cls_token") or var_name.endswith("mask_token") or var_name.endswith("pos_embed"):
        return 0
    elif var_name.startswith("encoder.patch_embed") or var_name.startswith("encoder_pre.patch_embed") or var_name.startswith("encoder_post.patch_embed"):
        return 0
    elif var_name.startswith("encoder.blocks") or var_name.startswith("encoder_pre.blocks") or var_name.startswith("encoder_post.blocks"):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return num_max_layer - 1

def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.85, base_lr=1e-5):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}
    
    num_layers = 33
    decay_rate = layer_decay
    
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
            
        if (
                len(p.shape) == 1 or n.endswith(".bias") or
                n.endswith("_token") or n.endswith("pos_embed")
        ):
            g_decay = 'no_decay'
            this_decay = 0.
        else:
            g_decay = 'decay'
            this_decay = weight_decay
        
        layer_id = get_layer_id(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)
        
        if group_name not in param_group_names:
            this_scale = decay_rate ** (num_layers - layer_id - 1)
            
            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
                "lr": base_lr * this_scale,
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
                "lr": base_lr * this_scale,
            }
        
        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)
    
    print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))
    
    return list(param_groups.values())


class LinearWarmupLR:
    def __init__(self, optimizer, warmup_iters, warmup_ratio, last_iter=-1):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.last_iter = last_iter
        # 保存每个参数组的base_lr（注意：这里保存的是配置时的初始lr）
        self.base_lrs = []
        for group in optimizer.param_groups:
            # 如果参数组有lr_scale，需要保存原始的lr, 是不是optim已经处理了lr_scale?
            if 'lr_scale' in group:
                self.base_lrs.append(group['lr'] / group['lr_scale'])
            else:
                self.base_lrs.append(group['lr'])
    
    def get_lr(self, cur_iter):
        if cur_iter < self.warmup_iters:
            # 线性warmup
            k = (1 - cur_iter / self.warmup_iters) * (1 - self.warmup_ratio)
            base_lrs = [base_lr * (1 - k) for base_lr in self.base_lrs]
        else:
            base_lrs = self.base_lrs
        
        # 应用每个参数组的lr_scale（如果有的话）
        lrs = []
        for i, (base_lr, param_group) in enumerate(zip(base_lrs, self.optimizer.param_groups)):
            if 'lr_scale' in param_group:
                lrs.append(base_lr * param_group['lr_scale'])
            else:
                lrs.append(base_lr)
        return lrs
    
    def step(self, cur_iter=None):
        if cur_iter is None:
            cur_iter = self.last_iter + 1
        self.last_iter = cur_iter
        
        lrs = self.get_lr(cur_iter)
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = lrs[i]
        
        return lrs


class PolyLR:
    def __init__(self, optimizer, power, total_iters, min_lr=1e-7, last_iter=-1):
        self.optimizer = optimizer
        self.power = power
        self.total_iters = total_iters
        self.min_lr = min_lr
        self.last_iter = last_iter
        # 保存每个参数组的base_lr
        self.base_lrs = []
        for group in optimizer.param_groups:
            if 'lr_scale' in group:
                self.base_lrs.append(group['lr'] / group['lr_scale'])
            else:
                self.base_lrs.append(group['lr'])
    
    def get_lr(self, cur_iter):
        if cur_iter < 0 or cur_iter >= self.total_iters:
            base_lrs = [self.min_lr for _ in self.base_lrs]
        else:
            decay_factor = (1 - cur_iter / self.total_iters) ** self.power
            base_lrs = [max(base_lr * decay_factor, self.min_lr)
                        for base_lr in self.base_lrs]
        
        # 应用每个参数组的lr_scale
        lrs = []
        for i, (base_lr, param_group) in enumerate(zip(base_lrs, self.optimizer.param_groups)):
            if 'lr_scale' in param_group:
                lrs.append(base_lr * param_group['lr_scale'])
            else:
                lrs.append(base_lr)
        return lrs
    
    def step(self, cur_iter=None):
        if cur_iter is None:
            cur_iter = self.last_iter + 1
        self.last_iter = cur_iter
        
        lrs = self.get_lr(cur_iter)
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = lrs[i]
        
        return lrs


class StepLR:
    def __init__(self, optimizer, milestones, gamma=0.1, min_lr=1e-7, last_iter=-1):
        """
        阶梯式学习率调度器

        Args:
            optimizer: 优化器
            milestones: 迭代步数列表，在这些步数时对学习率乘以gamma
            gamma: 衰减系数
            min_lr: 最小学习率
            last_iter: 上一次迭代的步数
        """
        self.optimizer = optimizer
        self.milestones = sorted(milestones)  # 确保milestones是升序的
        self.gamma = gamma
        self.min_lr = min_lr
        self.last_iter = last_iter
        
        # 计算当前处于哪个衰减阶段
        self.current_stage = 0
        for i, m in enumerate(self.milestones):
            if self.last_iter >= m:
                self.current_stage = i + 1
        
        # 保存每个参数组的base_lr
        self.base_lrs = []
        for group in optimizer.param_groups:
            if 'lr_scale' in group:
                self.base_lrs.append(group['lr'] / group['lr_scale'])
            else:
                self.base_lrs.append(group['lr'])
    
    def get_lr(self, cur_iter):
        # 计算衰减阶段
        stage = 0
        for i, m in enumerate(self.milestones):
            if cur_iter >= m:
                stage = i + 1
        
        # 计算衰减因子
        decay_factor = self.gamma ** stage
        
        # 应用衰减因子
        if cur_iter < 0:
            base_lrs = [max(base_lr * (self.gamma ** 0), self.min_lr)
                        for base_lr in self.base_lrs]
        else:
            base_lrs = [max(base_lr * decay_factor, self.min_lr)
                        for base_lr in self.base_lrs]
        
        # 应用每个参数组的lr_scale
        lrs = []
        for i, (base_lr, param_group) in enumerate(zip(base_lrs, self.optimizer.param_groups)):
            if 'lr_scale' in param_group:
                lrs.append(base_lr * param_group['lr_scale'])
            else:
                lrs.append(base_lr)
        return lrs
    
    def step(self, cur_iter=None):
        if cur_iter is None:
            cur_iter = self.last_iter + 1
        self.last_iter = cur_iter
        
        lrs = self.get_lr(cur_iter)
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = lrs[i]
        
        return lrs