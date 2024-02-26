import argparse
import logging
import time
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import utils
from model import trip_3_5_20_noT
from model import trip_3_5_20
from mnist_net import mnist_net



logger = logging.getLogger(__name__)


logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).cuda()
        #确保添加扰动后图片仍在0，1之间
        delta.data = clamp(delta, 0-X, 1-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)

            index = torch.where(output.max(1)[1] == y)[0]
            if len(index) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            d = clamp(d, 0-X, 1-X)
            delta.data[index] = d[index]
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--data-dir', default='../mnist-data', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--attack', default='fgsm', type=str, choices=['none', 'pgd', 'fgsm'])
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--epsilon', default=0.3, type=float)
    parser.add_argument('--alpha', default=0.375, type=float)
    parser.add_argument('--attack-iters', default=40, type=int)
    parser.add_argument('--lr-max', default=5e-3, type=float)
    parser.add_argument('--lr-type', default='cyclic')
    parser.add_argument('--fname', default='./adv_trip/test_3_5_20_ep3.pth', type=str)
    parser.add_argument('--seed', default=0, type=int)
    return parser.parse_args()


def main():
    args = get_args()

    # # 建立相关文件夹
    # if not os.path.exists(args.out_dir):
    #     os.mkdir(args.out_dir)
    # # 日志文件路径
    # logfile = os.path.join(args.out_dir, 'output.log')
    # if os.path.exists(logfile):
    #     os.remove(logfile)
    #
    # # 创建 StreamHandler
    # logging.basicConfig(
    #     format='[%(asctime)s] - %(message)s',
    #     datefmt='%Y/%m/%d %H:%M:%S',
    #     level=logging.INFO,
    #     filename=os.path.join(args.out_dir, 'output.log'))

    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    mnist_train = datasets.MNIST("../mnist-data", train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True)

    model = trip_3_5_20().cuda()
    model.train()

    model_train = mnist_net().cuda()
    model_train.train()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr_max)
    opt_train = torch.optim.Adam(model_train.parameters(), lr=args.lr_max)
    if args.lr_type == 'cyclic':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2//5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_type == 'flat':
        lr_schedule = lambda t: args.lr_max
    else:
        raise ValueError('Unknown lr_type')

    # criterion = nn.CrossEntropyLoss()
    loss_fn = nn.CrossEntropyLoss()
    criterion = utils.LabelSmoothingLoss(classes=20, classes_target=10, smoothing=0.4)
    prev_robust_acc = 0.
    logger.info('Epoch \t Time \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0

        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            if i == 0:
                first_batch = (X, y)

            lr = lr_schedule(epoch + (i+1)/len(train_loader))
            opt.param_groups[0].update(lr=lr)
            opt_train.param_groups[0].update(lr=lr)

            if args.attack == 'fgsm':
                if i%2!=0:
                    delta = torch.zeros_like(X).uniform_(-args.epsilon, args.epsilon).cuda()
                    delta.requires_grad = True
                    output = model(X + delta)
                    loss = F.cross_entropy(output, y)
                    loss.backward()
                    grad = delta.grad.detach()
                    # 主要是这步
                    delta.data = torch.clamp(delta + args.alpha * torch.sign(grad), -args.epsilon, args.epsilon)

                    delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
                    delta = delta.detach()
                else:
                    delta = torch.zeros_like(X).uniform_(-args.epsilon, args.epsilon).cuda()
                    delta.requires_grad = True
                    output_train = model_train(X + delta)
                    loss = F.cross_entropy(output_train, y)
                    loss.backward()
                    grad = delta.grad.detach()
                    # 主要是这步
                    delta.data = torch.clamp(delta + args.alpha * torch.sign(grad), -args.epsilon, args.epsilon)

                    delta.data = torch.max(torch.min(1 - X, delta.data), 0 - X)
                    delta = delta.detach()
            elif args.attack == 'none':
                delta = torch.zeros_like(X)
            elif args.attack == 'pgd':
                delta = torch.zeros_like(X).uniform_(-args.epsilon, args.epsilon)
                delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
                for _ in range(args.attack_iters):
                    delta.requires_grad = True
                    output = model(X + delta)
                    loss = loss_fn(output, y)
                    opt.zero_grad()
                    loss.backward()
                    grad = delta.grad.detach()
                    I = output.max(1)[1] == y
                    #传统PGD对抗训练单步扰动生成，eps为小扰动。这里为大的
                    delta.data[I] = torch.clamp(delta + args.alpha * torch.sign(grad), -args.epsilon, args.epsilon)[I]
                    delta.data[I] = torch.max(torch.min(1-X, delta.data), 0-X)[I]
                delta = delta.detach()
            output_train = model_train(torch.clamp(X + delta, 0, 1))

            loss = loss_fn(output_train, y)
            opt_train.zero_grad()
            loss.backward()
            opt_train.step()

            train_loss += loss.item() * y.size(0)
            train_acc += (output_train.max(1)[1] == y).sum().item()
            train_n += y.size(0)

        X, y = first_batch

        # def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts):
        pgd_delta = delta = attack_pgd(model_train, X, y, 0.3, 1e-2, 20, 5)
        with torch.no_grad():
            output = model_train(X + pgd_delta)
        robust_acc = (output.max(1)[1] == y).sum().item() / y.size(0)
        if robust_acc - prev_robust_acc < -0.1:
            break
        prev_robust_acc = robust_acc
        best_state_dict = copy.deepcopy(model_train.state_dict())


        train_time = time.time()
        logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f',
            epoch, train_time - start_time, lr, train_loss/train_n, train_acc/train_n)


        torch.save(best_state_dict, args.fname)


if __name__ == "__main__":
    main()
