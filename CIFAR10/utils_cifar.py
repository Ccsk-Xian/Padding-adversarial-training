import os
import sys

import numpy as np
# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchsnooper
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path
import random

import apex.amp as amp
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from advertorch.defenses import MedianSmoothing2D
from advertorch.defenses import BitSqueezing
from advertorch.defenses import JPEGFilter
import torch.nn as nn



def getCifarTrain(rollNumber:[],):

    cifar_data=torch.load(Path('../Data/CIFAR10Train.pt'))
    X_0,X_1,X_2,X_3,X_4,X_5,X_6,X_7,X_8,X_9 = cifar_data
    # X = torch.cat((X_0,X_1,X_2,X_3,X_4,X_5,X_6,X_7,X_8,X_9),0)  #chage sample_number
    choseList = [X_0,X_1,X_2,X_3,X_4,X_5,X_6,X_7,X_8,X_9]
    Final_X = []
    Final_Y = []
    for step,i in enumerate(rollNumber):
        Final_X.append(choseList[i])
        Y_temp = torch.ones(choseList[i].shape[0],dtype=int)*(step+10)
        Final_Y.append(Y_temp)
    X = torch.cat(tuple(Final_X),0)
    Y = torch.cat(tuple(Final_Y),0)
    # X = torch.cat((X,X1))
    # Y =torch.cat((Y,Y1)))
    Cifar_10tr=Data.TensorDataset(X,Y)
    return Cifar_10tr

def getCifarTest(rollNumber:[],):
    cifar_data=torch.load(Path('../Data/CIFAR10Test.pt'))
    X_0,X_1,X_2,X_3,X_4,X_5,X_6,X_7,X_8,X_9 = cifar_data
    # X = torch.cat((X_0,X_1,X_2,X_3,X_4,X_5,X_6,X_7,X_8,X_9),0)  #chage sample_number
    choseList = [X_0,X_1,X_2,X_3,X_4,X_5,X_6,X_7,X_8,X_9]
    Final_X = []
    Final_Y = []
    for step,i in enumerate(rollNumber):
        Final_X.append(choseList[i])
        Y_temp = torch.ones(choseList[i].shape[0],dtype=int)*(step+10)
        Final_Y.append(Y_temp)
    X = torch.cat(tuple(Final_X),0)
    Y = torch.cat(tuple(Final_Y),0)
    # X = torch.cat((X,X1))
    # Y =torch.cat((Y,Y1)))
    Cifar_10tr=Data.TensorDataset(X,Y)
    return Cifar_10tr

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()
# [
#   [[]],
#   [[]],
#   [[]]
# ]
upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def get_loaders(dir_, batch_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
        Cutout(n_holes=1, length=16),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    num_workers = 2
    train_dataset = datasets.CIFAR10(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(
        dir_, train=False, transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    return train_loader, test_loader


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            if opt is not None:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta
lower = torch.tensor([0]).cuda()
upper =torch.tensor([1]).cuda()
def attack_pgd_fgsm(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        delta.uniform_(-epsilon, epsilon)
        delta.data = clamp(delta, 0 - X, 1 - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            if opt is not None:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            t_e = torch.tensor([epsilon]).cuda()
            d = clamp(d + alpha * torch.sign(g), -t_e, t_e)
            d = clamp(d, lower - X[index[0], :, :, :], upper - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd_change(test_loader, model, attack_iters, restarts,squeezing=False):
    epsilon = (80 / 255.) / std
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    test_acc_defense = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
            if squeezing:
                # 特征压缩
                bits_squeezing = BitSqueezing(bit_depth=3)
                median_filter = MedianSmoothing2D(kernel_size=3)


                defense = nn.Sequential(
                    # median_filter,
                    bits_squeezing,
                )
                X_defense = defense(X + pgd_delta).cuda()

                output_defense = model(X_defense)
                test_acc_defense += (output_defense.max(1)[1] == y).sum().item()

    return pgd_loss/n, pgd_acc/n, test_acc_defense/n

def transform_img(img):

    # X_out = img.transpose(1, 2,0)
    img = img.cpu().numpy()
    X_out  = img.swapaxes(0, 1)
    X_out = X_out.swapaxes(1,2)
    return X_out

def evaluate_pgd(test_loader, model, attack_iters, restarts,Trap=False):
    epsilon = (64/ 255.) / std
    alpha = (1 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    total_acc = 0
    total_acc_before = 0
    model.eval()

    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        # print(X.shape)
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():

            # plt.figure()
            # plt.subplot(3,3,1)
            # plt.imshow(transform_img(X[1]))
            # plt.xticks([])
            # plt.yticks([])
            # plt.axis('off')
            # # plt.title(y[1])
            # plt.subplot(3, 3, 4)
            # plt.imshow(transform_img(X[2]))
            # plt.xticks([])
            # plt.yticks([])
            # plt.axis('off')
            # # plt.title(y[2])
            # plt.subplot(3, 3, 7)
            # plt.imshow(transform_img(X[3]))
            # # plt.title(y[3])
            # plt.xticks([])
            # plt.yticks([])
            # plt.axis('off')
            #
            #
            # plt.subplot(3, 3, 2)
            # plt.imshow(transform_img(pgd_delta[1]))
            # plt.xticks([])
            # plt.yticks([])
            # plt.axis('off')
            #
            # plt.subplot(3, 3, 5)
            # plt.imshow(transform_img(pgd_delta[2]))
            # plt.xticks([])
            # plt.yticks([])
            # plt.axis('off')
            #
            # plt.subplot(3, 3, 8)
            # plt.imshow(transform_img(pgd_delta[3]))
            # plt.xticks([])
            # plt.yticks([])
            # plt.axis('off')

            output = model(X + pgd_delta)


            # plt.subplot(3, 3, 3)
            # plt.imshow(transform_img((X+pgd_delta)[1]))
            # plt.xticks([])
            # plt.yticks([])
            # plt.axis('off')
            # # plt.title(output[1])
            # plt.subplot(3, 3, 6)
            # plt.imshow(transform_img((X+pgd_delta)[2]))
            # plt.xticks([])
            # plt.yticks([])
            # plt.axis('off')
            # # plt.title(output[2])
            # plt.subplot(3, 3, 9)
            # plt.imshow((transform_img((X+pgd_delta)[3])))
            # # plt.title(output[3])
            # plt.xticks([])
            # plt.yticks([])
            # plt.axis('off')
            # plt.tight_layout(pad=0.5)
            # plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
            #                     wspace=None, hspace=None)
            # plt.savefig('1.png')
            # plt.clf()
            # plt.close()
            # print("**")


            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
            if Trap == True:
                pred_ = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

                total_acc_before += pred_.eq(y.view_as(pred_)).sum().item()

                clean = [i for i, x in enumerate(pred_) if x == 10]
                for j in range(len(clean)):
                    pred_[clean[j]] = y[clean[j]]

                total_acc += pred_.eq(y.view_as(pred_)).sum().item()
                # print(pgd_acc / n)
                # print(total_acc / n)
                # print('*')
    if Trap == True:
        return pgd_loss/n, pgd_acc/n,total_acc/n
    return pgd_loss/n, pgd_acc/n


def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n

def evaluate_standard_change(test_loader, model,squeezing=False):
    test_loss = 0
    test_acc = 0
    n = 0
    test_acc_defense = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
            if squeezing:
                # 特征压缩
                bits_squeezing = BitSqueezing(bit_depth=3)
                median_filter = MedianSmoothing2D(kernel_size=3)

                defense = nn.Sequential(
                    # median_filter,
                    bits_squeezing,
                )
                X_defense = defense(X).cuda()

                output_defense = model(X_defense)
                test_acc_defense += (output_defense.max(1)[1] == y).sum().item()
    return test_loss/n, test_acc/n,test_acc_defense/n

def makeRandom(channel,data,max,min,mean,std,Epoch,epoch):
    # schedule = lambda t: np.interp([t], [0, Epoch * 0.7, Epoch], [(max-min)/2, 2*max, 5*max])[0]
    # schedule = lambda t: np.interp([t], [0, epoch * 0.5, epoch], [0.25, 1, 10])[0]
    schedule = lambda t: np.interp([t], [0, epoch * 0.5, epoch], [0.35, 0.8, 1])[0]

    a = torch.sign(torch.randn_like(data)) * schedule(epoch)
    a = a.cuda()
    data_padding = data + a
    return data_padding

    # size = data.size()
    # # seed = random.randint(1)
    # seed =1
    # try:
    #     if seed ==3:
    #         r1 = torch.randint(-sys.maxsize, -10, size).cuda()
    #         r2 = torch.sign(torch.randn_like(data)).cuda()
    #         data = data + r1*r2
    #     for i in range(channel):
    #         if seed == 1:
    #             r1 = torch.normal(mean[i],std[i],size).cuda()
    #             data[:, i, :, :] = data[:,i,:,:]+r1[:,i,:,:]
    #             data = torch.clamp(data, min, max)
    #         elif seed ==2:
    #             r1 = torch.normal(schedule(epoch),std[i],size).cuda()
    #             r2 = torch.sign(torch.randn_like(data[:,i,:,:])).cuda()
    #             data[:, i, :, :] = data[:,i,:,:] + r1[:,i,:,:]*r2
    #     return data
    # except Exception as err:
    #     print("your code has problem:"+str(err))

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes_target,classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing/2
        self.cls_target = classes_target
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        index_target = torch.where(target < 10)
        index_trap = torch.where(target >9)
        pred = pred.log_softmax(dim=self.dim)
        distribution =  self.smoothing/9

        with torch.no_grad():
            # true_dist = pred.data.clone()

            # #不为陷阱类分配分布，这两种尝试是因为我们并不太清楚哪种方法可以增大检测效率。可能因为数据分布本身不处于一个流形，我们需要先对其进行自编码处理
            # true_dist = torch.zeros((pred.size()[0], self.cls_target))
            # add_dist = torch.zeros((true_dist.size()[0], self.cls - 10))
            # true_dist.fill_(self.smoothing / (self.cls_target - 1))
            # true_dist = torch.cat((true_dist, add_dist), 1).to(device)


            # 框架建立
            true_dist = torch.zeros((pred.size()[0],self.cls_target))
            add_dist = torch.zeros((true_dist.size()[0],self.cls-10))
            # 目标填充
            true_dist[index_target[0], :]=distribution
            Y = target.data[index_target]
            true_dist[index_target[0], Y]= self.confidence
            # true_dist.scatter_(1, target.cpu().detach().data.unsqueeze(1), self.confidence)

            add_dist[index_target[0],:]=(self.smoothing/(self.cls - 10))
            add_dist[index_trap[0],:]=1

            true_dist = torch.cat((true_dist,add_dist),1).cuda()

            #原方法
            # true_dist.fill_(self.smoothing / (self.cls - 1))
            # print(true_dist,'**')
            #我把这里的填补放在陷阱类里


        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img