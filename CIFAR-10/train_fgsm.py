import argparse
import copy
import logging
import os
import time

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
    # 特征压缩
    bits_squeezing = BitSqueezing(bit_depth=3)
    median_filter = MedianSmoothing2D(kernel_size=3)
    # 对准确率损失太大。
    jpeg_filter = JPEGFilter(90)

    defense = nn.Sequential(
        median_filter,
        bits_squeezing,
    )
    #获取参数
    args = get_args()

    #建立相关文件夹
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    #日志文件路径
    logfile = os.path.join(args.out_dir, 'output27.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    #创建 StreamHandler
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(args.out_dir, 'output27.log'))
    logger.info(args)

    #设置随机因子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #获取数据
    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    #设置超参数：/255.是归一化到[0,1]。/std是消除量级
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    pgd_alpha = (2 / 255.) / std

    #引入模型
    # model = PreActResNet18().cuda()
    model = torchvision.models.resnet18(num_classes=11).cuda()
    #训练模型
    model.train()

    # 设置SGD优化器，权重衰减：正则化，优化更小的权重，降低网络复杂度，增加数据拟合(奥卡姆剃刀)--过拟合要求高系数引发高梯度。
    #  动量：每次更新考虑上次的更新值，方向相同则加速，不同则抵消，方便收敛。通过梯度加上上一次动量乘以一定比例系数β，loss进行下一步梯度下降不仅要考虑到函数现在的梯度方向，还要考虑到函数之前的下降方向，相当于引入了物理中的惯性。有效避免了loss训练过程中抖动太大，受困于局部极小值点等问题。
    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)

    # 混合精度训练（Mixed Precision） 混合精度训练的精髓在于“在内存中用FP16做储存和乘法从而加速计算，用FP32做累加避免舍入误差”。混合精度训练的策略有效地缓解了舍入误差的问题。
    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = args.master_weights
        print(args.early_stop,'**')
    model, opt = amp.initialize(model, opt, **amp_args)
    criterion = nn.CrossEntropyLoss()
    loss_func = LabelSmoothingLoss(classes_target=10, classes=11, smoothing=0.35)

    if args.delta_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()

    lr_steps = args.epochs * len(train_loader)

    # 动态更新学习率
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # Training
    prev_robust_acc = 0.
    start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc \t Train_true Loss \t Train_true Acc\t\t Train_defense Acc')
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_loss_true = 0
        train_acc_true = 0
        train_n = 0
        train_acc_defense = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            if i == 0:
                first_batch = (X, y)
            if args.delta_init != 'previous':
                delta = torch.zeros_like(X).cuda()
                # 这个是创新点？随机初始化？
            if args.delta_init == 'random':
                for j in range(len(epsilon)):
                    delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.requires_grad = True
            output = model(X + delta[:X.size(0)])
            # loss = F.cross_entropy(output, y)
            loss = loss_func(output,y)
            # 准备对抗样本生成梯度
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            grad = delta.grad.detach()
            delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            # 这里是和传统不一样的，应该也只是防止他超过上下界。那随机化其实就只是上面那一步。把PGD里面的摘出来？WC
            delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            delta = delta.detach()
            # output = model(X + delta[:X.size(0)])
            X = X + delta[:X.size(0)]
            # padding装配
            # data_padding = makeRandom(channel=args.channel, data=X, max=args.max, min=args.min, mean=args.mean,
            #                           std=args.std, Epoch=args.epochs, epoch=epoch)
            # data_padding = data_padding.cuda()
            # y_padding = torch.ones_like(y) * 10
            # y_padding = y_padding.cuda()
            # X = torch.cat((X, data_padding[:10,:,:,:]), 0)
            # y = torch.cat((y, y_padding[:10]), 0)
            # X, y = X.cuda(), y.cuda()

            output = model(X)

            # loss = criterion(output, y)
            loss = loss_func(output,y)
            opt.zero_grad()

            # 对抗训练，这里的train_acc并不是真正的训练样本的acc
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            opt.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

            # 真正的训练样本acc
            # with torch.no_grad():
            #     output_true = model(X)
            #     loss_true = criterion(output_true, y)
            #     train_acc_true += (output_true.max(1)[1] == y).sum().item()
            #     train_loss_true += loss_true.item() * y.size(0)
                # defense_x =defense(X + delta[:X.size(0)]).cuda()
                # output_defense = model(defense_x)
                # train_acc_defense += (output_defense.max(1)[1] == y).sum().item()


            # 学习率更新
            scheduler.step()

            # 早停机制。阈值为0.1
        if args.early_stop:
            # Check current PGD robustness of model using random minibatch
            X_t, y_t = first_batch
            pgd_delta = attack_pgd(model, X_t, y_t, epsilon, pgd_alpha, 5, 1, opt)
            with torch.no_grad():
                output = model(clamp(X_t + pgd_delta[:X_t.size(0)], lower_limit, upper_limit))
            robust_acc = (output.max(1)[1] == y_t).sum().item() / y_t.size(0)
            print(robust_acc)
            if robust_acc - prev_robust_acc < -0.1:
                break
            prev_robust_acc = robust_acc
            best_state_dict = copy.deepcopy(model.state_dict())
        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
                    epoch, epoch_time - start_epoch_time, lr, train_loss / train_n, train_acc / train_n)

        test_output = model(X[:10])
        pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
        print(pred_y, 'prediction number')
        print(y[:10].cpu().numpy(), 'real number')

        test_output = model(X[-10:])
        pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
        print(pred_y, 'prediction number')
        print(y[-10:].cpu().numpy(), 'real number')

        # logger.info('%d \t\t %.1f \t \t %.4f \t\t %.4f \t %.4f\t\t %.4f\t %.4f\t\t %.4f',
        #     epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n,train_loss_true/train_n, train_acc_true/train_n,train_acc_defense/train_n)
    train_time = time.time()
    if not args.early_stop:
        best_state_dict = model.state_dict()
    torch.save(best_state_dict, os.path.join(args.out_dir, 'model_10.pth'))
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    # Evaluation 模型评估。
    # model_test = torch().cuda()
    model_test = torchvision.models.resnet18(num_classes=11).cuda()
    model_test.load_state_dict(best_state_dict)
    model_test.float()
    model_test.eval()

    # pgd_loss, pgd_acc, defense_acc = evaluate_pgd_change(test_loader, model_test, 10, 5,squeezing=True)
    # test_loss, test_acc,defense_clean_acc = evaluate_standard_change(test_loader, model_test)
    #
    # logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc\t Defense Acc \t Clean_Defense Acc')
    # logger.info('%.4f \t \t %.4f \t %.4f \t %.4f \t%.4f\t%.4f', test_loss, test_acc, pgd_loss, pgd_acc,defense_acc,defense_clean_acc)

    pgd_loss, before_acc,after_acc = evaluate_pgd(test_loader, model_test, 10, 5,True)
    test_loss, test_acc = evaluate_standard(test_loader, model_test)

    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Before_Acc \t PGD After_Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, before_acc, after_acc)

if __name__ == "__main__":
    main()
