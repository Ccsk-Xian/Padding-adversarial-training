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
from utils import LabelSmoothingLoss
from utils import attack_pgd_fgsm,clamp

from mnist_net import mnist_net

logger = logging.getLogger(__name__)


logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
lower = torch.tensor([0]).cuda()
upper =torch.tensor([1]).cuda()
def main():


    mnist_train = datasets.MNIST("../mnist-data", train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)

    model = mnist_net().cuda()
    # model = torchvision.models.resnet18(num_classes = 11).cuda()
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    lr_schedule = lambda t: np.interp([t], [0, 10 * 2//5, 10], [0, 0.01, 0])[0]

    criterion = nn.CrossEntropyLoss()
    # loss_func = LabelSmoothingLoss(classes_target=10, classes=11, smoothing=0.1)
    # logger.info('Epoch \t Time \t LR \t \t Train Loss \t Train Acc')
    prev_robust_acc = 0.
    for epoch in range(10):
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




            output = model(X)

            # 需要测试一下，到底全部是陷阱损失还是这里要变回去
            loss = criterion(output, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)



        if True:
            # Check current PGD robustness of model using random minibatch
            X_t, y_t = first_batch
            epsilon = torch.tensor([0.3])
            pgd_delta = attack_pgd_fgsm(model, X_t, y_t, 0.3, 0.02, 5, 1)
            with torch.no_grad():
                output = model(clamp(X_t + pgd_delta[:X.size(0)],lower,upper))
            robust_acc = (output.max(1)[1] == y_t).sum().item() / y.size(0)
            if robust_acc - prev_robust_acc < -0.1:
                break
            prev_robust_acc = robust_acc
            best_state_dict = copy.deepcopy(model.state_dict())
        epoch_time = time.time()
        logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f',
                    epoch, epoch_time - start_time, lr, train_loss / train_n, train_acc / train_n)



    train_time = time.time()
    logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f',
        epoch, train_time - start_time, lr, train_loss/train_n, train_acc/train_n)
    torch.save(best_state_dict, './exam_1/mnist_inil.pth')

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