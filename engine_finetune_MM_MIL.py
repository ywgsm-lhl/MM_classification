# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import math
import sys
import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

from timm.data import Mixup
from timm.utils import accuracy
from typing import Iterable, Optional
import util.misc as misc
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score, average_precision_score
from pycm import *
import matplotlib.pyplot as plt
import numpy as np

import pdb
import json

def misc_measures(confusion_matrix):
    
    acc = []
    sensitivity = []
    specificity = []
    precision = []
    G = []
    F1_score_2 = []
    mcc_ = []
    
    for i in range(1, confusion_matrix.shape[0]):
        cm1=confusion_matrix[i]
        acc.append(1.*(cm1[0,0]+cm1[1,1])/np.sum(cm1))
        sensitivity_ = 1.*cm1[1,1]/(cm1[1,0]+cm1[1,1])
        sensitivity.append(sensitivity_)
        specificity_ = 1.*cm1[0,0]/(cm1[0,1]+cm1[0,0])
        specificity.append(specificity_)
        precision_ = 1.*cm1[1,1]/(cm1[1,1]+cm1[0,1])
        precision.append(precision_)
        G.append(np.sqrt(sensitivity_*specificity_))
        F1_score_2.append(2*precision_*sensitivity_/(precision_+sensitivity_))
        mcc = (cm1[0,0]*cm1[1,1]-cm1[0,1]*cm1[1,0])/np.sqrt((cm1[0,0]+cm1[0,1])*(cm1[0,0]+cm1[1,0])*(cm1[1,1]+cm1[1,0])*(cm1[1,1]+cm1[0,1]))
        mcc_.append(mcc)
        
    acc = np.array(acc).mean()
    sensitivity = np.array(sensitivity).mean()
    specificity = np.array(specificity).mean()
    precision = np.array(precision).mean()
    G = np.array(G).mean()
    F1_score_2 = np.array(F1_score_2).mean()
    mcc_ = np.array(mcc_).mean()
    
    return acc, sensitivity, specificity, precision, G, F1_score_2, mcc_

def confusion_matrix(y_true, y_pred, num_classes=None):
    """
    Confusion matrix such like:
                  predict
                0    1
    target  0   TN  FN
            1   FP  TP

    :param y_true: target, list or np.ndarray
    :param y_pred: predict, list or np.ndarray
    :return:
    """
    y_true = y_true.astype(np.int)
    y_pred = y_pred.astype(np.int)
    if num_classes is None:
        num_classes = max(max(y_true), max(y_pred)) + 1
    # bin的数量是x+1, weights参数： out[n] += weight[i], minlength参数： bin的最小值
    conf_mat = np.bincount((num_classes * y_true.astype(int) + y_pred).flatten(),
                           minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return conf_mat

def _each_score(y_true, y_score, score_func, **kwargs):
    """
        :param y_true: one hot targets, np.ndarray or list
        :param y_score: predict scores, np.ndarray or list
        :param score_func: function of calculate single score
        :return: scores(score for each class), np.ndarray
    """
    scores = []
    for i in range(y_true.shape[1]):
        score = score_func(y_true[:, i], y_score[:, i], **kwargs)
        scores.append(score)
    return np.array(scores)

def f1_ss_score(y_true, y_score, threshold=0.5):
    """
    F1 score of sensitivity and specificity

    :param y_true: target, list or np.ndarray
    :param y_score: predict, list or np.ndarray
    :param threshold: float or list or np.ndarray, default 0.5, e.g. 0.5 or  [0.2, 0.3]
    :return:
    """
    #y_pred = _score2pred(y_score, threshold)
    sens = recall_score(y_true, y_score)
    conf_mat = confusion_matrix(y_true, y_score)
    spec = float(conf_mat[0, 0]) / float(conf_mat[0, :].sum() + 1e-10)
    return (2 * sens * spec) / (sens + spec + 1e-10)

def specificity_score(y_true, y_score, threshold=0.5):
    """
    Specificity score

    :param y_true: target, list or np.ndarray
    :param y_score: output score, list or np.ndarray
    :param threshold: float or list or np.ndarray, default 0.5, e.g. 0.5 or  [0.2, 0.3]
    :return:
    """
    #y_pred = _score2pred(y_score, threshold)
    conf_mat = confusion_matrix(y_true, y_score)
    return float(conf_mat[0, 0]) / float(conf_mat[0, :].sum() + 1e-10)


def get_metric(all_targets, all_scores, threshold=0.5, class_wise=False):
    predict_b = np.where(all_scores >= threshold, 1, 0)
    
    sensitivities = _each_score(all_targets, predict_b, recall_score)
    specificities = _each_score(all_targets, predict_b, specificity_score)
    
    f1s = _each_score(all_targets, predict_b, f1_ss_score)   
    aps = _each_score(all_targets, all_scores, average_precision_score)
    aucs = _each_score(all_targets, all_scores, roc_auc_score)
    
    if class_wise:
        return np.round(sensitivities, 4), np.round(specificities, 4), np.round(f1s, 4), np.round(aps, 4), np.round(aucs, 4)
    
    return np.round(np.mean(sensitivities), 4), np.round(np.mean(specificities), 4), np.round(np.mean(f1s), 4), np.round(np.mean(aps), 4), np.round(np.mean(aucs), 4)
    

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    exp_lr_scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(data_loader), epochs=50)

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        samples[0] = samples[0].cuda()
        samples[1] = samples[1].cuda()
        targets = targets.cuda()
        
        outputs = model(samples[0], samples[1])
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        
        #loss.backward()和optimizer.step()都在loss_scaler这里做了
        if exp_lr_scheduler is not None:
            exp_lr_scheduler.step()
        
        #torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, task, epoch, mode, num_class, class_wise=False, feature_only=True):
    criterion = torch.nn.BCEWithLogitsLoss()
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    if not os.path.exists(task):
        os.makedirs(task, exist_ok=True)

    # switch to evaluation mode
    model.eval()
    
    with torch.no_grad():
        all_targets = []
        all_scores = []
        for batch in metric_logger.log_every(data_loader, 10, header):
            images = batch[0]
            target = batch[1]

            images[0] = images[0].to(device, non_blocking=True)
            images[1] = images[1].to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(images[0], images[1])

            loss = criterion(output, target)

            all_targets.append(target.cpu().numpy().copy())
            all_scores.append(torch.sigmoid(output).cpu().numpy().copy())

            metric_logger.update(loss=loss.item())   #中位数（平均数）

        all_targets = np.concatenate(all_targets)
        all_scores = np.concatenate(all_scores)

        label2dia = {
          "0": "未见异常",
          "1": "黄斑前膜",
          "2": "黄斑水肿",
          "3": "DR",
          "4": "干性AMD",
          "5": "湿性AMD",
          "6": "病理性近视"
        }

        sensitivities, specificities, f1s, aps, aucs = get_metric(all_targets, all_scores, class_wise=class_wise)

        metric_logger.synchronize_between_processes()

        if class_wise:
            for i in range(7):
                print(label2dia[str(i)]+': ',aps[i])
            return {k: meter.global_avg for k, meter in metric_logger.meters.items()},aucs

        print('Sklearn Metrics - Sen: {:.4f} Spe: {:.4f} AUC-roc: {:.4f} AUC-pr: {:.4f} F1-score: {:.4f} '.format(
            sensitivities, specificities, aucs, aps, f1s))
        results_path = task+'_metrics_{}.csv'.format(mode)
        with open(results_path,mode='a',newline='',encoding='utf8') as cfa:
            wf = csv.writer(cfa)
            data2=[[sensitivities, specificities, aucs, aps, f1s, metric_logger.loss]]
            for i in data2:
                wf.writerow(i)

        if mode=='test':
            predict_b = np.where(all_scores >= 0.5, 1, 0)
            for i in range(all_targets.shape[1]):
                cm = ConfusionMatrix(actual_vector=np.array(all_targets[i],dtype=np.int64), predict_vector = np.array(predict_b[i],dtype=np.int64))
                cm.plot(cmap=plt.cm.Blues,number_label=True,normalized=True,plot_lib="matplotlib")
                plt.savefig(task+label2dia[str(i)]+'_confusion_matrix_test.jpg',dpi=600,bbox_inches ='tight')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()},aucs,aps

