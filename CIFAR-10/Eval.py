# 在这里用不同的扰动大小，验证这个诱导效果

import argparse
import copy
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from apex import amp

#特征压缩
from advertorch.defenses import MedianSmoothing2D
from advertorch.defenses import BitSqueezing
from advertorch.defenses import JPEGFilter

from preact_resnet import PreActResNet18
from utils_cifar import (upper_limit, lower_limit, std, clamp, get_loaders,
                   attack_pgd, evaluate_pgd_change, evaluate_standard_change, evaluate_pgd, evaluate_standard,makeRandom,LabelSmoothingLoss)

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar10_data', type=str)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--max', default=1, type=int)
    parser.add_argument('--min', default=-1, type=int)
    parser.add_argument('--channel', default=3, type=int)
    parser.add_argument('--mean', default=[0.4914, 0.4822, 0.4465], type=float)
    parser.add_argument('--std', default=[0.2471, 0.2435, 0.2616], type=float)
    parser.add_argument('--alpha', default=10, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous'],
        help='Perturbation initialization method')
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--early-stop', action='store_true', default=False,help='Early stop if overfitting occurs')
    parser.add_argument('--opt-level', default='O1', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true', default=True,
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.info('begin')
    print('1')
    args = get_args()

    #设置随机因子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #获取数据
    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)


    # Evaluation 模型评估。
    # model_test = torch().cuda()
    model_test = torchvision.models.resnet18(num_classes=10).cuda()
    model_test.load_state_dict(torch.load(Path('train_pgd_output/model.pth')))
    model_test.float()
    model_test.eval()

    # pgd_loss, pgd_acc, defense_acc = evaluate_pgd_change(test_loader, model_test, 10, 5,squeezing=True)
    # test_loss, test_acc,defense_clean_acc = evaluate_standard_change(test_loader, model_test)
    #
    # logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc\t Defense Acc \t Clean_Defense Acc')
    # logger.info('%.4f \t \t %.4f \t %.4f \t %.4f \t%.4f\t%.4f', test_loss, test_acc, pgd_loss, pgd_acc,defense_acc,defense_clean_acc)

    pgd_loss, before_acc,after_acc = evaluate_pgd(test_loader, model_test, 10, 5,True)
    test_loss, test_acc = evaluate_standard(test_loader, model_test)

    logging.info('Test Loss \t Test Acc \t PGD Loss \t PGD Before_Acc \t PGD After_Acc')
    logging.info('%.4f \t \t %.4f \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, before_acc, after_acc)

if __name__ == "__main__":
    main()
