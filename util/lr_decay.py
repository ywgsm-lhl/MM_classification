# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import json

import pdb

def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75, modality='single'):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}
    #pdb.set_trace()
    
    #对于拼接特征而言，两个模型最后的head都没有使用到，而是多模态模型自己的head，因此不算层数，和其他模型层数相同
    #if modality == 'multi_cat':
    #    num_layers = len(model.blocks) + 2
    if modality == 'multi_mil_cat_OCT' or modality == 'multi_mil_add_OCT' or modality == 'multi_mil_add_base_OCT': #mil里两层，分类head一层不算
        num_layers = len(model.blocks) + 3
    elif 'multi_mil_head' in modality:   #和单模态mil相比，多了一层proj
        num_layers = len(model.blocks) + 4
    elif 'multi_SA_CA' in modality:   #提取特征的24层，concat后的SA 3层，CA3层，head 1层
        num_layers = 24 + 3 + 3 +1
    elif 'multi_base_SA_CA' in modality or 'multi_base_3SA_3CA' in modality or 'multi_base_multi_head_baseline' in modality:  
        num_layers = 12 + 3 + 3 +1
    elif 'multi_base_CA' in modality or 'multi_base_multi_head_wo_SA' in modality:  
        num_layers = 12 + 3 +1
    elif 'multi_base_6SA_3CA' in modality:  
        num_layers = 12 + 6 + 3 +1
    else:
        num_layers = len(model.blocks) + 1
    

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        
        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
        
        if modality == 'multi_base_SA_CA_SA':
            layer_id = num_layers - 6 + int(n[0])
        elif modality == 'multi_base_SA_CA_CA':
            layer_id = num_layers - 3 + int(n[0])
        elif modality == 'multi_base_6SA_3CA_SA':
            num = int(n[0])
            layer_id = num_layers + 3*(num//2) + num%2 + 1 - 10
        elif modality == 'multi_base_6SA_3CA_CA':
            layer_id = num_layers + int(n[0])*3 - 10
        elif modality == 'multi_base_3SA_3CA_SA':
            layer_id = num_layers + int(n[0])*2 - 6
        elif modality == 'multi_base_3SA_3CA_CA':
            layer_id = num_layers + int(n[0])*2 - 5
        elif modality == 'multi_base_multi_head_baseline_SA':
            layer_id = num_layers + int(n[0]) - 6
        else:
            layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

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
    
    #pdb.set_trace()
    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))
    return list(param_groups.values())

def MM_param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75, modality='multi_cat'):
    fundus_param_groups = param_groups_lrd(model.fundus_ViT, weight_decay,
        no_weight_decay_list=model.fundus_ViT.no_weight_decay(),                                
        layer_decay=layer_decay, modality=modality
    )
    OCT_param_groups = param_groups_lrd(model.OCT_ViT, weight_decay,
        no_weight_decay_list=model.OCT_ViT.no_weight_decay(),                                
        layer_decay=layer_decay, modality=modality+'_OCT'
    )
    #tt = list(model.named_parameters())
    #pdb.set_trace()
    #加入最后一层线性分类层
    param_groups = fundus_param_groups + OCT_param_groups
    
    if modality == 'multi_cat':
        temp = {
                "lr_scale": 1,
                "weight_decay": weight_decay,
                "params": list(model.named_parameters())[0][1],   #0是weight，1是bias
            }
        param_groups.append(temp)
    
    elif modality == 'multi_mil_cat':
        temp = {
                "lr_scale": 1,
                "weight_decay": weight_decay,
                "params": list(model.named_parameters())[0][1],   #0是weight，1是bias
            }
        temp0 = {
                "lr_scale": 1,
                "weight_decay": 0,
                "params": list(model.named_parameters())[1][1],   #0是weight，1是bias
            }
        
        param_groups.append(temp)
        param_groups.append(temp0)
        
    elif 'multi_mil_head' in modality:
        num_layers = len(model.fundus_ViT.blocks) + 4
        layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))
        
        for n, p in model.named_parameters():
            #if 'OCT' in n or 'fundus' in n:
            #    continue
            if 'ViT' in n:
                continue
            elif 'fundus_proj.0.weight' in n or 'oct_proj.0.weight' in n:    #proj.1是LayerNorm，dim=1，不用管
                this_scale = layer_scales[num_layers-3]
                this_decay = weight_decay
            elif 'fundus_proj.0.bias' in n or 'oct_proj.0.bias' in n:    #proj.1是LayerNorm，dim=1，不用管
                this_scale = layer_scales[num_layers-3]
                this_decay = 0
            elif 'fundus_proj.1' in n or 'oct_proj.1' in n:    #proj.1是LayerNorm，dim=1，不用管
                this_scale = layer_scales[num_layers-3]
                this_decay = 0
            elif 'ins_attn.attn.0.weight' in n:
                this_scale = layer_scales[num_layers-2]
                this_decay = weight_decay
            elif 'ins_attn.attn.2.weight' in n:
                this_scale = layer_scales[num_layers-1]
                this_decay = weight_decay
            elif 'fc.weight' in n:
                this_scale = layer_scales[num_layers]
                this_decay = weight_decay
            else:
                this_scale = layer_scales[num_layers]
                this_decay = 0
            
            temp = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [p],
            }
            param_groups.append(temp)
    
    elif modality == 'multi_base_SA_CA':
        SA_param_groups = param_groups_lrd(model.Self_attn_blocks, weight_decay,
            no_weight_decay_list=[],                                
            layer_decay=layer_decay, modality=modality+'_SA'
        )
        CA_param_groups = param_groups_lrd(model.Co_attn_blocks, weight_decay,
            no_weight_decay_list=[],                                
            layer_decay=layer_decay, modality=modality+'_CA'
        )
        param_groups += SA_param_groups
        param_groups += CA_param_groups
        
        temp = {
                "lr_scale": 1,
                "weight_decay": weight_decay,
                "params": list(model.named_parameters())[0][1],   #0是weight，1是bias
            }
        param_groups.append(temp)
    
    elif 'multi_base_6SA_3CA' in modality or 'multi_base_3SA_3CA' in modality:
        SA_param_groups = param_groups_lrd(model.Self_attn_blocks_fundus, weight_decay,
            no_weight_decay_list=[],                                
            layer_decay=layer_decay, modality=modality+'_SA'
        )
        param_groups += SA_param_groups
        
        SA_param_groups = param_groups_lrd(model.Self_attn_blocks_OCT, weight_decay,
            no_weight_decay_list=[],                                
            layer_decay=layer_decay, modality=modality+'_SA'
        )
        param_groups += SA_param_groups
        
        CA_param_groups = param_groups_lrd(model.Co_attn_blocks, weight_decay,
            no_weight_decay_list=[],                                
            layer_decay=layer_decay, modality=modality+'_CA'
        )
        param_groups += CA_param_groups
        
        temp = {
                "lr_scale": 1,
                "weight_decay": weight_decay,
                "params": list(model.named_parameters())[0][1],   #0是weight，1是bias
            }
        param_groups.append(temp)
    
    elif 'multi_base_multi_head_baseline' in modality:
        SA_param_groups = param_groups_lrd(model.Self_attn_blocks_fundus, weight_decay,
            no_weight_decay_list=[],                                
            layer_decay=layer_decay, modality=modality+'_SA'
        )
        param_groups += SA_param_groups
        
        SA_param_groups = param_groups_lrd(model.Self_attn_blocks_OCT, weight_decay,
            no_weight_decay_list=[],                                
            layer_decay=layer_decay, modality=modality+'_SA'
        )
        param_groups += SA_param_groups
        
        num_layers = 12 + 3 + 3 +1
        layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))
        
        #MM-MIL
        for n, p in model.named_parameters():
            #ViT以及两个的模态SA已经初始化过
            if 'OCT_ViT' in n or 'fundus_ViT' in n or 'Self_attn_blocks' in n:
                continue
            elif 'fundus_proj.0.weight' in n or 'oct_proj.0.weight' in n:
                this_scale = layer_scales[num_layers-3]
                this_decay = weight_decay
            elif 'ins_attn.attn.0.weight' in n:
                this_scale = layer_scales[num_layers-2]
                this_decay = weight_decay
            elif 'ins_attn.attn.2.weight' in n:
                this_scale = layer_scales[num_layers-1]
                this_decay = weight_decay
            elif 'weight' in n and 'head' in n:  #最后一层有3个head
                this_scale = layer_scales[num_layers]
                this_decay = weight_decay
            else:
                this_scale = layer_scales[num_layers]
                this_decay = 0
            
            temp = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [p],
            }
            param_groups.append(temp)
            
    elif 'multi_base_multi_head_wo_SA' in modality:
        num_layers = 12 + 3 +1
        layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))
        
        #MM-MIL
        for n, p in model.named_parameters():
            #ViT已经初始化过
            if 'OCT_ViT' in n or 'fundus_ViT' in n:
                continue
            elif 'fundus_proj.0.weight' in n or 'oct_proj.0.weight' in n:
                this_scale = layer_scales[num_layers-3]
                this_decay = weight_decay
            elif 'ins_attn.attn.0.weight' in n:
                this_scale = layer_scales[num_layers-2]
                this_decay = weight_decay
            elif 'ins_attn.attn.2.weight' in n:
                this_scale = layer_scales[num_layers-1]
                this_decay = weight_decay
            elif 'weight' in n and 'head' in n:  #最后一层有3个head
                this_scale = layer_scales[num_layers]
                this_decay = weight_decay
            else:
                this_scale = layer_scales[num_layers]
                this_decay = 0
            
            temp = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [p],
            }
            param_groups.append(temp)
    
    elif modality == 'multi_base_CA':
        CA_param_groups = param_groups_lrd(model.Co_attn_blocks, weight_decay,
            no_weight_decay_list=[],                                
            layer_decay=layer_decay, modality=modality+'_CA'
        )
        param_groups += CA_param_groups
        
        temp = {
                "lr_scale": 1,
                "weight_decay": weight_decay,
                "params": list(model.named_parameters())[0][1],   #0是weight，1是bias
            }
        param_groups.append(temp)
    
    return param_groups

def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    elif 'mil.ins_attn.attn.0.weight' in name:
        return num_layers-2
    elif 'mil.ins_attn.attn.2.weight' in name:
        return num_layers-1
    else:
        return num_layers