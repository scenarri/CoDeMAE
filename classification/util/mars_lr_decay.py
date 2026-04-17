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

def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}
    
    num_layers = 25
    decay_rate = layer_decay
    
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        
        # no decay: all 1D parameters and model specific ones
        if (
                len(p.shape) == 1 or n.endswith(".bias") or
                n.endswith("_token") or n.endswith("pos_embed") or n.endswith("bias_table")
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
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
        
        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)
    
    print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))
    
    return list(param_groups.values())


downsample_id = [1, 1, 17, 1]
num_layers = [0, 2, 4, 22]


def get_layer_id(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed", "absolute_pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("layers"):
        layer_id = int(var_name.split('.')[1])
        if var_name.split('.')[2] == 'downsample':
            block_id = downsample_id[layer_id]
        else:
            block_id = int(var_name.split('.')[3])
        return num_layers[layer_id] + block_id + 1
    else:
        return num_max_layer - 1