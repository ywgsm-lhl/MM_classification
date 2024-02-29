# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,3'
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy#, BinaryCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.cross_attention import load_vit_to_co_attn

import models_vit
import models_vit_me

from engine_finetune import train_one_epoch, evaluate


import warnings
warnings.filterwarnings("ignore")
import pdb

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')    
    parser.add_argument('--modality', default='fundus', type=str, 
                        help='modality of model to train')    
    parser.add_argument('--freeze', action='store_true', 
                        help='whether to freeze the pretrained model')    
    parser.add_argument('--num_block', default=24, type=int, 
                        help='the number of transformer blocks used')
    parser.add_argument('--wo_cls_token', action='store_true', 
                        help='whether use cls token in head')
    #parser.set_defaults(wo_cls_token=False)
    
    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
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
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
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
    parser.add_argument('--finetune', default='',type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--finetune_fundus', default='',type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--finetune_OCT', default='',type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--task', default='',type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    #parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/home/jupyter/Mor_DR_data/data/data/IDRID/Disease_Grading/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
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
    parser.add_argument('--class_wise', action='store_true',
                        help='evaluate on class-wise')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
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


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_dataset(is_train='train', args=args)
    dataset_val = build_dataset(is_train='val', args=args)
    dataset_test = build_dataset(is_train='test', args=args)
    
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            
        if args.dist_eval:
            if len(dataset_test) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
            
    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir+args.task)
    else:
        log_writer = None
    
    
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

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
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
    
    model = models_vit_me.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
        modality=args.modality,
        num_block=args.num_block,
        #wo_cls_token=args.wo_cls_token
    )
    
    #pdb.set_trace()
    
    if args.finetune and not args.eval:
        
        if 'multi' in args.modality:
            checkpoint_fundus = torch.load(args.finetune_fundus, map_location='cpu')
            checkpoint_OCT = torch.load(args.finetune_OCT, map_location='cpu')
            #checkpoint = torch.load('./pretrained_weights/jx_vit_base_p16_224-80ecf9dd.pth', map_location='cpu')
            
            
            print("Load pre-trained checkpoint_fundus from: %s" % args.finetune_fundus)
            if args.finetune_fundus.startswith('./pretrained_weights/jx_vit'):
                checkpoint_model_fundus = checkpoint_fundus
            else:
                checkpoint_model_fundus = checkpoint_fundus['model']
            #checkpoint_model_fundus = checkpoint_fundus['model']
            
            print("Load pre-trained checkpoint_OCT from: %s" % args.finetune_OCT)
            if args.finetune_OCT.startswith('./pretrained_weights/jx_vit'):
                checkpoint_model_OCT = checkpoint_OCT
            else:
                checkpoint_model_OCT = checkpoint_OCT['model']
            #checkpoint_model_OCT = checkpoint_OCT['model']
            
            fundus_state_dict = model.fundus_ViT.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model_fundus and checkpoint_model_fundus[k].shape != fundus_state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint_fundus")
                    del checkpoint_model_fundus[k]
            OCT_state_dict = model.OCT_ViT.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model_OCT and checkpoint_model_OCT[k].shape != OCT_state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint_OCT")
                    del checkpoint_model_OCT[k]
            
            # interpolate position embedding
            interpolate_pos_embed(model.fundus_ViT, checkpoint_model_fundus)
            interpolate_pos_embed(model.OCT_ViT, checkpoint_model_OCT)

            # load pre-trained model
            if args.modality == 'multi_base_SA_CA' or args.modality == 'multi_base_CA' or args.modality == 'multi_base_6SA_3CA' or args.modality == 'multi_base_3SA_3CA' or args.modality == 'multi_base_multi_head_baseline' or args.modality == 'multi_base_multi_head_wo_SA':
                fundus_state_dict = load_vit_to_co_attn(checkpoint_model_fundus, fundus_state_dict)
                print("load_vit_to_co_attn")
                msg = model.fundus_ViT.load_state_dict(fundus_state_dict, strict=False)
            else:
                msg = model.fundus_ViT.load_state_dict(checkpoint_model_fundus, strict=False)
            print(msg)

            #if args.global_pool:
            #    assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            #else:
            #    assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

            if args.modality == 'multi_base_SA_CA' or args.modality == 'multi_base_CA' or args.modality == 'multi_base_6SA_3CA' or args.modality == 'multi_base_3SA_3CA' or args.modality == 'multi_base_multi_head_baseline' or args.modality == 'multi_base_multi_head_wo_SA':
                OCT_state_dict = load_vit_to_co_attn(checkpoint_model_OCT, OCT_state_dict)
                print("load_vit_to_co_attn")
                msg = model.OCT_ViT.load_state_dict(OCT_state_dict, strict=False)
            else:
                msg = model.OCT_ViT.load_state_dict(checkpoint_model_OCT, strict=False)
            print(msg)
            
            if args.num_block == 24:
                if args.modality == 'multi_mil_cat':
                    assert set(msg.missing_keys) == {'mil.ins_attn.attn.0.weight', 'head.weight', 'head.bias', 'fc_norm.bias', 'mil.ins_attn.attn.2.weight', 'fc_norm.weight'}
                elif args.modality == 'multi_mil_add':
                    assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'mil.ins_attn.attn.0.weight', 'mil.ins_attn.attn.2.weight', 'mil.fc.weight', 'mil.fc.bias', 'fc_norm.weight', 'fc_norm.bias'}
                elif args.global_pool:
                    assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
                #else:
                #    assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
            
            #pdb.set_trace()
            
            if args.modality == 'multi_base_SA_CA':   #初始化cat后的self attention
                state_dict = model.Self_attn_blocks.state_dict()
                for k,v in state_dict.items():
                    new_query = 'blocks.' + k
                    if new_query in checkpoint_OCT and v.size() == checkpoint_OCT[new_query].size():
                        state_dict.update({k: checkpoint_OCT[new_query]})
                msg = model.Self_attn_blocks.load_state_dict(state_dict, strict=False)
                print(msg)
            
            if args.modality == 'multi_base_6SA_3CA' or args.modality == 'multi_base_3SA_3CA' or args.modality == 'multi_base_multi_head_baseline':   #初始化self attention
                state_dict = model.Self_attn_blocks_fundus.state_dict()
                for k,v in state_dict.items():
                    new_query = 'blocks.' + k
                    if new_query in checkpoint_fundus and v.size() == checkpoint_fundus[new_query].size():
                        state_dict.update({k: checkpoint_fundus[new_query]})
                msg = model.Self_attn_blocks_fundus.load_state_dict(state_dict, strict=False)
                print(msg)
                
                state_dict = model.Self_attn_blocks_OCT.state_dict()
                for k,v in state_dict.items():
                    new_query = 'blocks.' + k
                    if new_query in checkpoint_OCT and v.size() == checkpoint_OCT[new_query].size():
                        state_dict.update({k: checkpoint_OCT[new_query]})
                msg = model.Self_attn_blocks_OCT.load_state_dict(state_dict, strict=False)
                print(msg)
                
            #pdb.set_trace()
            
            if args.freeze:
                if args.num_block == 24:
                    for (name, param) in model.named_parameters():
                        #if 'head' not in name and 'fc_norm' not in name:
                        if 'OCT' in name or 'fundus' in name:
                            name = '.'.join(name.split('.')[1:])
                        if name not in set(msg.missing_keys):
                            param.requires_grad = False
                else:     #提取特征部分全部冻结，只保留mil模块
                    for (name, param) in model.named_parameters():
                        if name not in {'mil.ins_attn.attn.0.weight', 'fc_norm.bias', 'mil.fc.bias', 'mil.ins_attn.attn.2.weight', 'fc_norm.weight', 'mil.fc.weight'}:
                            param.requires_grad = False
                            
            
            # manually initialize fc layer
            if args.modality == 'multi_base_multi_head_baseline' or args.modality == 'multi_base_multi_head_wo_SA':
                trunc_normal_(model.MM_head.weight, std=2e-5)
                trunc_normal_(model.CFP_head.weight, std=2e-5)
                trunc_normal_(model.OCT_head.weight, std=2e-5)
            elif args.modality[:14] != 'multi_mil_head':  #用mm-mil的话就不需要这些head，head在mil里已经初始化好了
                if 'mil' not in args.modality:
                    trunc_normal_(model.OCT_ViT.head.weight, std=2e-5)
                trunc_normal_(model.fundus_ViT.head.weight, std=2e-5)
                if 'multi_mil_add' not in args.modality:
                    trunc_normal_(model.head.weight, std=2e-5)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

            print("Load pre-trained checkpoint from: %s" % args.finetune)
            if args.finetune == './jx_vit_large_p16_224-4ee7a4dc.pth':
                checkpoint_model = checkpoint
            else:
                checkpoint_model = checkpoint['model']
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
            
            #if 'mil' in args.modality and args.num_block == 24:
            #    assert set(msg.missing_keys) == {'mil.ins_attn.attn.0.weight', 'head.weight', 'head.bias', 'fc_norm.bias', 'mil.fc.bias', 'mil.ins_attn.attn.2.weight', 'fc_norm.weight', 'mil.fc.weight'}
            #elif args.global_pool and args.num_block == 24:
            #    assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            #elif args.num_block == 24:
            #    assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
            
            if args.freeze:
                if args.num_block == 24:
                    for (name, param) in model.named_parameters():
                        if name not in set(msg.missing_keys):
                            param.requires_grad = False
                else:     #提取特征部分全部冻结，只保留mil模块
                    for (name, param) in model.named_parameters():
                        if name not in {'mil.ins_attn.attn.0.weight', 'fc_norm.bias', 'mil.fc.bias', 'mil.ins_attn.attn.2.weight', 'fc_norm.weight', 'mil.fc.weight'}:
                            param.requires_grad = False
            
            # manually initialize fc layer
            trunc_normal_(model.head.weight, std=2e-5)
    
    elif args.eval and args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        
        print("Load pre-trained checkpoint from: %s" % args.resume)
        
        checkpoint_model = checkpoint['model']
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
    
    model.to(device)
    
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # build optimizer with layer-wise lr decay (lrd)
    if 'multi' in args.modality:
        param_groups = lrd.MM_param_groups_lrd(model_without_ddp, args.weight_decay,
            layer_decay=args.layer_decay, modality=args.modality
        )
    else:
        param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
            no_weight_decay_list=model_without_ddp.no_weight_decay(),
            layer_decay=args.layer_decay
        )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()
    
    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        #criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        #criterion = BinaryCrossEntropy(smoothing=args.smoothing)
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        #criterion = torch.nn.CrossEntropyLoss()
        criterion = torch.nn.BCEWithLogitsLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    
    if args.eval:
        test_stats,auc_roc,ap = evaluate(data_loader_test, model, device, args.task, epoch=0, mode='test',num_class=args.nb_classes,class_wise=args.class_wise)
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_auc = 0.0
    max_ap = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )

        val_stats,val_auc_roc,val_ap = evaluate(data_loader_val, model, device,args.task,epoch, mode='val',num_class=args.nb_classes)
        #if max_auc<val_auc_roc:
        #    max_auc = val_auc_roc
        if max_ap<val_ap:
            max_ap = val_ap
            
            if args.output_dir:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
        

        if epoch==(args.epochs-1):
            test_stats,auc_roc,ap = evaluate(data_loader_test, model, device,args.task,epoch, mode='test', num_class=args.nb_classes)

        
        if log_writer is not None:
            log_writer.add_scalar('perf/val_ap', val_ap, epoch)
            log_writer.add_scalar('perf/val_auc', val_auc_roc, epoch)
            log_writer.add_scalar('perf/val_loss', val_stats['loss'], epoch)
            
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

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

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
