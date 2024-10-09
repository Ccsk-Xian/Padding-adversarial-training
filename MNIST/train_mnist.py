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
from utils import makeRandom
# from utils import LabelSmoothingLoss
from utils import LabelSmoothing,LabelSmoothingLoss
from utils import attack_pgd_fgsm,clamp

from mnist_net import mnist_net

logger = logging.getLogger(__name__)


logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
lower = torch.tensor([0]).cuda()
upper =torch.tensor([1]).cuda()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--data-dir', default='../mnist-data', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--attack', default='fgsm', type=str, choices=['none', 'pgd', 'fgsm'])
    parser.add_argument('--out-dir', default='exam_1_output', type=str, help='Output directory')
    parser.add_argument('--epsilon', default=0.4, type=float)
    parser.add_argument('--interpret', default=[0.5,0.7], type=float)
    parser.add_argument('--smoothing', default=0.35, type=float)
    parser.add_argument('--alpha', default=0.375, type=float)
    parser.add_argument('--attack-iters', default=40, type=int)
    parser.add_argument('--lr-max', default=5e-3, type=float)
    parser.add_argument('--lr-type', default='cyclic')
    parser.add_argument('--fname', default='./eval/118.pth', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--max', default=1, type=int)
    parser.add_argument('--min', default=0, type=int)
    parser.add_argument('--channel', default=1, type=int)
    parser.add_argument('--mean', default=[0.1307], type=float)
    parser.add_argument('--std', default=[0.3081], type=float)
    parser.add_argument('--early-stop', action='store_true', default=True, help='Early stop if overfitting occurs')

    return parser.parse_args()


def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    # 建立相关文件夹
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    # 日志文件路径
    logfile = os.path.join(args.out_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    # 创建 StreamHandler
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(args.out_dir, 'output.log'))

    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    mnist_train = datasets.MNIST("../mnist-data", train=True, download=True, transform=transforms.ToTensor())
    # mnist_train = datasets.CIFAR10("../cifar10_data", train=True, download=True, transform=transforms.ToTensor())
    # 这里的mnist_train可以替换为上传的数据。但这就要求上传的数据是处理好的。这部分我后面写吧。我把前端需求写清楚
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True)

    # 这里需要更换为上传的模型结构，方法就是上传时约束模型的方法名就为mnist_net。直接使用上传的文件覆盖mnist_net这个py文件。
    # model = mnist_net().cuda()
    model = torchvision.models.resnet18(num_classes = 11).cuda()
    #fine-tunning
    # model.fc = nn.Linear(in_features=512,out_features=11).cuda()
    print(model)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr_max)
    if args.lr_type == 'cyclic':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2//5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_type == 'flat':
        lr_schedule = lambda t: args.lr_max
    else:
        raise ValueError('Unknown lr_type')

    criterion = nn.CrossEntropyLoss()

    loss_func = LabelSmoothingLoss(classes_target=10, classes=11, smoothing=args.smoothing)
    # loss_func = LabelSmoothing(smoothing=0.55)
    logger.info('Epoch \t Time \t LR \t \t Train Loss \t Train Acc')
    prev_robust_acc = 0.
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        train_acc_true = 0
        train_loss_true = 0
        train_acc_defense = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            if i == 0:
                first_batch = (X, y)
            lr = lr_schedule(epoch + (i+1)/len(train_loader))
            opt.param_groups[0].update(lr=lr)

            if args.attack == 'fgsm':
                #随机初始化
                delta = torch.zeros_like(X).uniform_(-args.epsilon, args.epsilon).cuda()
                delta.requires_grad = True
                output = model(X + delta)
                loss = loss_func(output, y)
                # loss = F.cross_entropy(output, y)
                loss.backward()
                grad = delta.grad.detach()
                # 主要是这步
                delta.data = torch.clamp(delta + args.alpha * torch.sign(grad), -args.epsilon, args.epsilon)
                # 这里应该是控制扰动大小。
                delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
                delta = delta.detach()
            elif args.attack == 'none':
                delta = torch.zeros_like(X)
            elif args.attack == 'pgd':
                delta = torch.zeros_like(X).uniform_(-args.epsilon, args.epsilon)
                delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
                for _ in range(args.attack_iters):
                    delta.requires_grad = True
                    output = model(X + delta)
                    loss = loss_func(output, y)
                    # loss = criterion(output, y)
                    opt.zero_grad()
                    loss.backward()
                    grad = delta.grad.detach()
                    I = output.max(1)[1] == y
                    #传统PGD对抗训练单步扰动生成，eps为小扰动。这里为大的
                    delta.data[I] = torch.clamp(delta + args.alpha * torch.sign(grad), -args.epsilon, args.epsilon)[I]
                    delta.data[I] = torch.max(torch.min(1-X, delta.data), 0-X)[I]
                delta = delta.detach()

            X = torch.clamp(X + delta, 0, 1)
            # padding装配
            P_X = X[:10, :, :, :]
            data_padding = makeRandom(channel=args.channel, data=P_X, max=args.max, min=args.min, mean=args.mean,
                                      std=args.std, Epoch=args.epochs, epoch=epoch, delta=delta)



            data_padding = data_padding.cuda()
            y_padding = torch.ones_like(y[:10]) * 10
            y_padding = y_padding.cuda()





            # X = torch.cat((X, data_padding[:10, :, :, :]), 0)
            # y = torch.cat((y, y_padding[:10]), 0)

            X = torch.cat((X, data_padding), 0)
            X = torch.clamp(X, 0, 1)
            y = torch.cat((y, y_padding), 0)
            X, y = X.cuda(), y.cuda()

            output = model(X)





            # 需要测试一下，到底全部是陷阱损失还是这里要变回去
            loss = loss_func(output, y)
            # loss  = criterion(output,y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)



        if args.early_stop :
            # Check current PGD robustness of model using random minibatch
            X_t, y_t = first_batch
            import torchattacks
            # epsilon = torch.tensor([args.epsilon])
            # pgd_delta = attack_pgd_fgsm(model, X_t, y_t, 0.3, 0.02, 5, 1)
            atk = torchattacks.PGD(model,eps=0.3)
            X_adv = atk(X_t,y_t)
            with torch.no_grad():
                output = model(X_adv)
            pred_ = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            robust_acc_be = pred_.eq(y_t.view_as(pred_)).sum().item()/ y_t.size(0)
            clean = [i for i, x in enumerate(pred_) if x == 10]
            for j in range(len(clean)):
                pred_[clean[j]] = y[clean[j]]
            
            robust_acc_aft = pred_.eq(y_t.view_as(pred_)).sum().item()/ y_t.size(0)
            # if robust_acc_aft - robust_acc_be > 0.1:
            #     break
            # prev_robust_acc = robust_acc
            best_state_dict = model.state_dict()
        epoch_time = time.time()
        logger.info(' %.4f \t %.4f',
                    robust_acc_be,robust_acc_aft)
        logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f',
                    epoch, epoch_time - start_time, lr, train_loss / train_n, train_acc / train_n)
        if not args.early_stop:
            best_state_dict = model.state_dict()


    train_time = time.time()
    logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f',
        epoch, train_time - start_time, lr, train_loss/train_n, train_acc/train_n)
    torch.save(best_state_dict, args.fname)

    test_output = model(X[:10])
    pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
    print(pred_y, 'prediction number')
    print(y[:10].cpu().numpy(), 'real number')

    test_output = model(X[-10:])
    pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
    print(pred_y, 'prediction number')
    print(y[-10:].cpu().numpy(), 'real number')


if __name__ == "__main__":
    main()
