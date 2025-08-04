# 在这里用不同的扰动大小，验证这个诱导效果
from device_config import device
import argparse
import copy
import logging
import os
import time
from pathlib import Path
import torchattacks
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from vgg import vgg16_bn, vgg19_bn
# from apex import amp
# from autoattack import AutoAttack
# #特征压缩
# from advertorch.defenses import MedianSmoothing2D
# from advertorch.defenses import BitSqueezing
# from advertorch.defenses import JPEGFilter
from GoogleNet import GoogLeNet
from mobilevit import mobile_vit_small
from preact_resnet import PreActResNet18
from utils_cifar_out import (upper_limit, lower_limit, std, clamp, get_loaders,
                             attack_pgd, evaluate_pgd_change, evaluate_standard_change, evaluate_pgd, evaluate_standard, makeRandom, LabelSmoothingLoss)

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--data-dir', default='cifar10_data', type=str)
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
    # model_test = torch().to(device)
    # model_test = torchvision.models.resnet18(num_classes=11).to(device)
    # model_test = torchvision.models.vgg16(num_classes=10).to(device)
    # model_test = mobile_vit_small(num_classes=10).to(device)
    # model_test = GoogLeNet(num_classes=11).to(device)
    model_test = PreActResNet18().to(device)
    # model_test = torchvision.models.resnet50(num_classes=11).to(device)
    # model_test = vgg16_bn(num_classes=11).to(device)
    # model_test = vgg19_bn(num_classes=11).to(device)
    model_test.load_state_dict(torch.load(Path("/root/CIFAR10/save/ZZ.pth")))
    # model_test.load_state_dict(torch.load(Path("/root/CIFAR10/save/ZRpgdpll.pth")))
    # model_test.load_state_dict(torch.load(Path('/root/fast_adversarial-master/MNIST/train_fgsm_output_pre/resnet18_fgsm_ll.pth')))
    # model_test.load_state_dict(torch.load(Path('/root/fast_adversarial-master/MNIST/train_fgsm_output_pre/mobile_vit_small_trapcl_fgsm.pth')))
    # 20.6 AA 15.62  0.18  0.0
    # model_test.load_state_dict(torch.load(Path('/root/fast_adversarial-master/MNIST/train_fgsm_output_pre/mobile_vit_small_inil_fgsm.pth')))
    # model_test.load_state_dict(torch.load(Path('/root/fast_adversarial-master/MNIST/train_fgsm_output/mobile_vit_small_trapll330_fgsm.pth')))
    # 0.2606  0.0588 0.09 AA 21.18  0.0  0.0 0.0
    # 86.16 :26.91/27.31 12.64/13.14 0.0 72.52  AA  18.75/19.42  1.04/3.12 0.0 86.64
    # model_test.load_state_dict(torch.load(Path('/root/fast_adversarial-master/MNIST/train_fgsm_output_pre/vgg16_fgsm_cc.pth'))) 
    # model_test.load_state_dict(torch.load(Path('/root/fast_adversarial-master/MNIST/train_fgsm_output_pre/GoogLeNet_fgsm_ll.pth'))) 
    model_test.float()
    model_test.eval()

    # pgd_loss, pgd_acc, defense_acc = evaluate_pgd_change(test_loader, model_test, 10, 5,squeezing=True)
    # test_loss, test_acc,defense_clean_acc = evaluate_standard_change(test_loader, model_test)
    #
    # logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc\t Defense Acc \t Clean_Defense Acc')
    # logger.info('%.4f \t \t %.4f \t %.4f \t %.4f \t%.4f\t%.4f', test_loss, test_acc, pgd_loss, pgd_acc,defense_acc,defense_clean_acc)
    print('ccsa')
    
    total_loss = 0
    total_loss_clean = 0
    total_acc = 0
    total_acc_clean = 0
    total_acc_before = 0
    n = 0

    for i, (x,y) in enumerate(test_loader):
            x,y = x.to(device),y.to(device)  
            # pat-pgd:15.07/15.16 (targeted) 98.43 pat-fgsm 20.98/21.48  96.61   0.6 fgsmat   pgd at 0  trade 0
            # untarget pat-pgd:0/82.35 (targeted) 98.43 pat-fgsm 0/95.4  96.61   0 fgsmat   pgd at 0  trade 0
            # resnet-19
            # atk = torchattacks.EADEN(model,max_iterations=1000)
            # atk = torchattacks.JSMA(model,theta=1)
            # atk = torchattacks.DeepFool(model)
            atk = torchattacks.AutoAttack(model_test,eps=8/255,n_classes=10,seed=0)
            # atk = torchattacks.LGV(model_test,train_loader,eps=0.6, alpha=0.02, steps=50, verbose=True)
            # atk = torchattacks.PGD(model_test,eps=128/255,alpha=2/255,steps=10,loss="")

            # perturbation = pretrained_G(x)
            # perturbation = torch.clamp(perturbation, -0.3, 0.3)
            # X_adv = perturbation + x
            # X_adv = torch.clamp(X_adv, 0, 1)
            # atk = torchattacks.Pixle(model)
            # atk.set_mode_targeted_by_function(target_map_function=lambda images, labels:(labels+1)%10)
            # atk.set_mode_targeted_random()

            atk.set_normalization_used(mean=[0.4914, 0.4822, 0.4465],std=[0.2471, 0.2435, 0.2616])
            X_adv = atk(x,y)
            with torch.no_grad():
                output_clean = model_test(x)
                total_acc_clean += (output_clean.max(1)[1] == y).sum().item()
                output = model_test(X_adv)
                pred_ = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

                total_acc_before += (output.max(1)[1] == y).sum().item()

                clean = [i for i, x in enumerate(pred_) if x == 10]
                for j in range(len(clean)):
                    pred_[clean[j]] = y[clean[j]]

                total_acc += pred_.eq(y.view_as(pred_)).sum().item()
                n += y.size(0)
                logger.info(' Clean_Acc: %.4f -- Before_Acc: %.4f,Acc: %.4f',
                        total_acc_clean / n, total_acc_before / n,
                        total_acc / n)
    # pgd_loss, before_acc,after_acc = evaluate_pgd(test_loader, model_test, 10, 5,True)
    # test_loss, test_acc = evaluate_standard(test_loader, model_test)

    # logging.info('Test Loss \t Test Acc \t PGD Loss \t PGD Before_Acc \t PGD After_Acc')
    # logging.info('%.4f \t \t %.4f \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, before_acc, after_acc)

if __name__ == "__main__":
    main()
