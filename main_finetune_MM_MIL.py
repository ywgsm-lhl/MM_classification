# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import argparse
import datetime
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import time
from pathlib import Path

import torch
import torch.nn as nn
import timm

assert timm.__version__ == "0.3.2" # version check

import util.misc as misc
from util.datasets import build_dataset

from multi_modal_mil import MultiModalMIL
from engine_finetune_MM_MIL import train_one_epoch, evaluate

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
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    
    dataset_train = build_dataset(is_train='train', args=args)
    dataset_val = build_dataset(is_train='val', args=args)
    dataset_test = build_dataset(is_train='test', args=args)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, shuffle=True,
        batch_size=args.batch_size,
        num_workers=8,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, shuffle=False,
        batch_size=args.batch_size,
        num_workers=8,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, shuffle=False,
        batch_size=args.batch_size,
        num_workers=8,
    )
    
    model = MultiModalMIL(pretrained=True, num_classes=args.nb_classes, num_mil=4, mixed_mil=False,
                              att_type='normal', csra=False)
    model.cuda()
    
    if args.eval and args.resume:
        checkpoint = torch.load(args.resume)
        
        print("Load pre-trained checkpoint from: %s" % args.resume)
        
        checkpoint_model = checkpoint['model']
        
        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model)
        print(msg)
    
    #model.to(device)
    
    if len(os.environ["CUDA_VISIBLE_DEVICES"]) > 1:
        model = nn.DataParallel(model)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    print("actual lr: %.2e" % args.lr)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)

    criterion = torch.nn.BCEWithLogitsLoss()

    print("criterion = %s" % str(criterion))
    
    #没有args.resume就不会用到这个函数
    #misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    
    if args.eval:
        test_stats,auc_roc,ap = evaluate(data_loader_test, model, device, args.task, epoch=0, mode='test',num_class=args.nb_classes,class_wise=args.class_wise)
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_auc = 0.0
    max_ap = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch,
            args.clip_grad, mixup_fn=None,
            log_writer=None,
            args=args
        )

        val_stats,val_auc_roc,val_ap = evaluate(data_loader_val, model, device, args.task, epoch, mode='val',num_class=args.nb_classes)
        #if max_auc<val_auc_roc:
        #    max_auc = val_auc_roc
        if max_ap<val_ap:
            max_ap = val_ap
            
            if args.output_dir:
                misc.save_model(
                    args=args, model=model, model_without_ddp=None, optimizer=optimizer,
                    loss_scaler=None, epoch=epoch)        

        if epoch==(args.epochs-1):
            test_stats,auc_roc,ap = evaluate(data_loader_test, model, device,args.task,epoch, mode='test', num_class=args.nb_classes)
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
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
