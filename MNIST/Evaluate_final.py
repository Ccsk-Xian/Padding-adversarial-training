# 测试方法的黑盒防御性，和其它传统防御方式进行效力对比。
# 模型应有原始模型，FGSM-对抗训练模型，PGD-对抗训练模型，Trap-对抗训练模型。
# *2 应该有个toy模型。所以就是8*8=64组对比


# 3.单对抗样本训练和对抗样本+干净样本

# eps=0.3
# F:\Python3.7.3\python.exe D:/fast_adversarial-master/MNIST/Evaluate_final.py
# *0
# **0
# source-target:0-0
#
# Test set: Clean loss: 0.039, Clean Accuracy: 9904/10000 (99l%)
#
#
# ADV set_fgsm:  loss: 4.388,  Accuracy_before: 0/10000 (0l%), Accuracy_after: (0l%)
#
# **1
# source-target:0-1
#
# Test set: Clean loss: 0.051, Clean Accuracy: 9824/10000 (98l%)
#
#
# ADV set_fgsm:  loss: 0.061,  Accuracy_before: 9793/10000 (98l%), Accuracy_after: (98l%)
#
# **2
# source-target:0-2
#
# Test set: Clean loss: 0.264, Clean Accuracy: 9408/10000 (94l%)
#
#
# ADV set_fgsm:  loss: 0.315,  Accuracy_before: 9245/10000 (92l%), Accuracy_after: (92l%)
#
# **3
# source-target:0-3
#
# Test set: Clean loss: 0.632, Clean Accuracy: 9661/10000 (97l%)
#
#
# ADV set_fgsm:  loss: 0.668,  Accuracy_before: 9610/10000 (96l%), Accuracy_after: (99l%)
#
# *1
# **0
# source-target:1-0
#
# Test set: Clean loss: 0.039, Clean Accuracy: 9904/10000 (99l%)
#
#
# ADV set_fgsm:  loss: 1.483,  Accuracy_before: 6280/10000 (63l%), Accuracy_after: (63l%)
#
# **1
# source-target:1-1
#
# Test set: Clean loss: 0.051, Clean Accuracy: 9824/10000 (98l%)
#
#
# ADV set_fgsm:  loss: 0.402,  Accuracy_before: 7278/10000 (73l%), Accuracy_after: (73l%)
#
# **2
# source-target:1-2
#
# Test set: Clean loss: 0.264, Clean Accuracy: 9408/10000 (94l%)
#
#
# ADV set_fgsm:  loss: 0.517,  Accuracy_before: 8425/10000 (84l%), Accuracy_after: (84l%)
#
# **3
# source-target:1-3
#
# Test set: Clean loss: 0.632, Clean Accuracy: 9661/10000 (97l%)
#
#
# ADV set_fgsm:  loss: 0.789,  Accuracy_before: 9336/10000 (93l%), Accuracy_after: (98l%)
#
# *2
# **0
# source-target:2-0
#
# Test set: Clean loss: 0.039, Clean Accuracy: 9904/10000 (99l%)
#
#
# ADV set_fgsm:  loss: 1.333,  Accuracy_before: 6592/10000 (66l%), Accuracy_after: (66l%)
#
# **1
# source-target:2-1
#
# Test set: Clean loss: 0.051, Clean Accuracy: 9824/10000 (98l%)
#
#
# ADV set_fgsm:  loss: 0.086,  Accuracy_before: 9734/10000 (97l%), Accuracy_after: (97l%)
#
# **2
# source-target:2-2
#
# Test set: Clean loss: 0.264, Clean Accuracy: 9408/10000 (94l%)
#
#
# ADV set_fgsm:  loss: 0.901,  Accuracy_before: 4156/10000 (42l%), Accuracy_after: (42l%)
#
# **3
# source-target:2-3
#
# Test set: Clean loss: 0.632, Clean Accuracy: 9661/10000 (97l%)
#
#
# ADV set_fgsm:  loss: 0.750,  Accuracy_before: 9497/10000 (95l%), Accuracy_after: (99l%)
#
# *3
# **0
# source-target:3-0
#
# Test set: Clean loss: 0.039, Clean Accuracy: 9904/10000 (99l%)
#
#
# ADV set_fgsm:  loss: 0.320,  Accuracy_before: 9044/10000 (90l%), Accuracy_after: (90l%)
#
# **1
# source-target:3-1
#
# Test set: Clean loss: 0.051, Clean Accuracy: 9824/10000 (98l%)
#
#
# ADV set_fgsm:  loss: 0.088,  Accuracy_before: 9726/10000 (97l%), Accuracy_after: (97l%)
#
# **2
# source-target:3-2
#
# Test set: Clean loss: 0.264, Clean Accuracy: 9408/10000 (94l%)
#
#
# ADV set_fgsm:  loss: 0.395,  Accuracy_before: 8923/10000 (89l%), Accuracy_after: (89l%)
#
# **3
# source-target:3-3
#
# Test set: Clean loss: 0.632, Clean Accuracy: 9661/10000 (97l%)
#
#
# ADV set_fgsm:  loss: 1.104,  Accuracy_before: 7547/10000 (75l%), Accuracy_after: (97l%)
#
#
# Process finished with exit code 0


#eps = 0.6
#
# F:\Python3.7.3\python.exe D:/fast_adversarial-master/MNIST/Evaluate_final.py
# *0
# **0
# source-target:0-0
#
# Test set: Clean loss: 0.039, Clean Accuracy: 9904/10000 (99l%)
#
#
# ADV set_fgsm:  loss: 4.979,  Accuracy_before: 0/10000 (0l%), Accuracy_after: (0l%)
#
# **1
# source-target:0-1
#
# Test set: Clean loss: 0.051, Clean Accuracy: 9824/10000 (98l%)
#
#
# ADV set_fgsm:  loss: 0.697,  Accuracy_before: 7548/10000 (75l%), Accuracy_after: (75l%)
#
# **2
# source-target:0-2
#
# Test set: Clean loss: 0.264, Clean Accuracy: 9408/10000 (94l%)
#
#
# ADV set_fgsm:  loss: 0.404,  Accuracy_before: 9159/10000 (92l%), Accuracy_after: (92l%)
#
# **3
# source-target:0-3
#
# Test set: Clean loss: 0.632, Clean Accuracy: 9661/10000 (97l%)
#
#
# ADV set_fgsm:  loss: 2.672,  Accuracy_before: 106/10000 (1l%), Accuracy_after: (100l%)
#
# *1
# **0
# source-target:1-0
#
# Test set: Clean loss: 0.039, Clean Accuracy: 9904/10000 (99l%)
#
#
# ADV set_fgsm:  loss: 0.424,  Accuracy_before: 8711/10000 (87l%), Accuracy_after: (87l%)
#
# **1
# source-target:1-1
#
# Test set: Clean loss: 0.051, Clean Accuracy: 9824/10000 (98l%)
#
#
# ADV set_fgsm:  loss: 2.177,  Accuracy_before: 0/10000 (0l%), Accuracy_after: (0l%)
#
# **2
# source-target:1-2
#
# Test set: Clean loss: 0.264, Clean Accuracy: 9408/10000 (94l%)
#
#
# ADV set_fgsm:  loss: 0.416,  Accuracy_before: 9131/10000 (91l%), Accuracy_after: (91l%)
#
# **3
# source-target:1-3
#
# Test set: Clean loss: 0.632, Clean Accuracy: 9661/10000 (97l%)
#
#
# ADV set_fgsm:  loss: 2.802,  Accuracy_before: 58/10000 (1l%), Accuracy_after: (100l%)
#
# *2
# **0
# source-target:2-0
#
# Test set: Clean loss: 0.039, Clean Accuracy: 9904/10000 (99l%)
#
#
# ADV set_fgsm:  loss: 3.342,  Accuracy_before: 3417/10000 (34l%), Accuracy_after: (34l%)
#
# **1
# source-target:2-1
#
# Test set: Clean loss: 0.051, Clean Accuracy: 9824/10000 (98l%)
#
#
# ADV set_fgsm:  loss: 2.609,  Accuracy_before: 2211/10000 (22l%), Accuracy_after: (22l%)
#
# **2
# source-target:2-2
#
# Test set: Clean loss: 0.264, Clean Accuracy: 9408/10000 (94l%)
#
#
# ADV set_fgsm:  loss: 1.632,  Accuracy_before: 0/10000 (0l%), Accuracy_after: (0l%)
#
# **3
# source-target:2-3
#
# Test set: Clean loss: 0.632, Clean Accuracy: 9661/10000 (97l%)
#
#
# ADV set_fgsm:  loss: 3.354,  Accuracy_before: 19/10000 (0l%), Accuracy_after: (100l%)
#
# *3
# **0
# source-target:3-0
#
# Test set: Clean loss: 0.039, Clean Accuracy: 9904/10000 (99l%)
#
#
# ADV set_fgsm:  loss: 0.260,  Accuracy_before: 9190/10000 (92l%), Accuracy_after: (92l%)
#
# **1
# source-target:3-1
#
# Test set: Clean loss: 0.051, Clean Accuracy: 9824/10000 (98l%)
#
#
# ADV set_fgsm:  loss: 0.485,  Accuracy_before: 8411/10000 (84l%), Accuracy_after: (84l%)
#
# **2
# source-target:3-2
#
# Test set: Clean loss: 0.264, Clean Accuracy: 9408/10000 (94l%)
#
#
# ADV set_fgsm:  loss: 0.361,  Accuracy_before: 9279/10000 (93l%), Accuracy_after: (93l%)
#
# **3
# source-target:3-3
#
# Test set: Clean loss: 0.632, Clean Accuracy: 9661/10000 (97l%)
#
#
# ADV set_fgsm:  loss: 3.374,  Accuracy_before: 0/10000 (0l%), Accuracy_after: (100l%)
#
#
# Process finished with exit code 0

import logging

logger = logging.getLogger()
logging.basicConfig(
    filemode='a',
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filename='10.log',
    )


logger.error("222")
logging.info("eee")
from deeprobust.image.defense import fgsmtraining
from deeprobust.image.defense import pgdtraining
from deeprobust.image.defense import AWP
import os
import numpy as np
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
import argparse
from sklearn.neighbors import KernelDensity
import torch
import torchvision
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch.utils.data as Data
from pathlib import Path
from torchattacks import DeepFool
from scipy.interpolate import interpn
import torch.nn.functional as F
import torchvision.transforms as transforms
from mnist_net import mnist_net
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# logger = logging.getLogger()




class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output  # return x for visualization


class CNN_2(nn.Module):
    def __init__(self):
        super(CNN_2, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=64,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (64, 28, 28)
            nn.ReLU(),  # activation

        )
        self.conv2 = nn.Sequential(  # input shape (64, 28, 28)
            nn.Conv2d(
                in_channels=64,  # input height
                out_channels=64,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (64, 28, 28)
            nn.ReLU(),  # activation

        )
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.dens = nn.Sequential(nn.Linear(64 * 28 * 28, 128))
        self.out = nn.Sequential(nn.Linear(128, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = self.dens(x)
        x = self.dropout2(x)
        output = self.out(x)
        return output  # return x for visualization

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

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
train_data_mn = torchvision.datasets.MNIST(

    root='../mnist-data',
    train=True,  # this is training data
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ]),  # Converts a PIL.Image or numpy.ndarray to
    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=False,
)
test_data_mn = torchvision.datasets.MNIST(
    root='../mnist-data',
    train=False,  # this is training data
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ]),  # Converts a PIL.Image or numpy.ndarray to
    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=False,
)
dl_train_mn = Data.DataLoader(dataset=train_data_mn, batch_size=256, shuffle=False)
dl_test_mn = Data.DataLoader(dataset=test_data_mn, batch_size=100, shuffle=False)

# logging.basicConfig(
#     filemode='a',
#     format='[%(asctime)s] - %(message)s',
#     datefmt='%Y/%m/%d %H:%M:%S',
#     level=logging.DEBUG,
#     filename='1.log',
#     )
# logger = logging.getLogger()


for number in range(6):
    # net_source = torchvision.models.resnet18(num_classes=10).to(device)

    if number == 0:
        net_source = mnist_net().to(device)
        net_source.load_state_dict(torch.load(Path("./eval/1.pth")))
        logger.info('0')
        logging.info('*0')
    elif number == 1:
        net_source = mnist_net().to(device)
        net_source.load_state_dict(
            torch.load(Path("./eval/2.pth")))
        logging.info('*1')
    elif number == 2:
        net_source = mnist_net().to(device)
        net_source.load_state_dict(
            torch.load(Path("./eval/3.pth")))
        logging.info('*2')
    elif number == 3:
        net_source = mnist_net().to(device)
        net_source.load_state_dict(
            torch.load(Path("./eval/4.pth")))
        logging.info('*3')
    elif number == 4:
        net_source = torchvision.models.resnet18(num_classes=11).to(device)
        net_source.load_state_dict(
            torch.load(Path("./eval/6.pth")))
        logging.info('*4')
    elif number == 5:
        net_source = torchvision.models.resnet18(num_classes=11).to(device)
        net_source.load_state_dict(
            torch.load(Path("./eval/8.pth")))
        logging.info('*5')
    elif number == 6:
        net_source = torchvision.models.resnet18(num_classes=11).to(device)
        net_source.load_state_dict(
            torch.load(Path("./eval/8.pth")))
        logging.info('*6')
    elif number == 7:
        net_source = torchvision.models.resnet18(num_classes=11).to(device)
        net_source.load_state_dict(
            torch.load(Path("./eval/7.pth")))
        logging.info('*7')
    elif number == 8:
        net_source = torchvision.models.resnet18(num_classes=11).to(device)
        net_source.load_state_dict(
            torch.load(Path("./eval/8.pth")))
        logging.info('*8')






    for number_target in range(6):
        # net_target = torchvision.models.resnet18(num_classes=10).to(device)

        if number_target == 0:
            net_target = mnist_net().to(device)
            net_target.load_state_dict(torch.load(Path("./eval/1.pth")))
            logging.info('**0')
        elif number_target == 1:
            net_target = mnist_net().to(device)
            net_target.load_state_dict(
                torch.load(Path("./eval/2.pth")))
            logging.info('**1')
        elif number_target == 2:
            net_target = mnist_net().to(device)
            net_target.load_state_dict(
                torch.load(Path("./eval/3.pth")))
            logging.info('**2')
        elif number_target == 3:
            net_target = mnist_net().to(device)
            net_target.load_state_dict(
                torch.load(Path("./eval/4.pth")))
            logging.info('**3')
        elif number_target == 4:
            net_target = torchvision.models.resnet18(num_classes=11).to(device)
            net_target.load_state_dict(
                torch.load(Path("./eval/6.pth")))
            logging.info('**4-p')
        elif number_target == 5:
            net_target = torchvision.models.resnet18(num_classes=11).to(device)
            net_target.load_state_dict(
                torch.load(Path("./eval/8.pth")))
            logging.info('**5')
        elif number_target == 6:
            net_target = torchvision.models.resnet18(num_classes=11).to(device)
            net_target.load_state_dict(
                torch.load(Path("./eval/6.pth")))
            logging.info('**6')
        elif number_target == 7:
            net_target = torchvision.models.resnet18(num_classes=11).to(device)
            net_target.load_state_dict(
                torch.load(Path("./eval/7.pth")))
            logging.info('**7')
        elif number_target == 8:
            net_target = torchvision.models.resnet18(num_classes=11).to(device)
            net_target.load_state_dict(
                torch.load(Path("./eval/8.pth")))
            logging.info('**8')




        test_loss = 0
        test_loss_fgsm = 0
        correct = 0
        correct_fgsm = 0
        test_loss_adv = 0
        correct_adv = 0
        total_acc = 0
        EPS = 0.6
        # net_source生成对抗样本，net_target用于检验。
        net_target.eval()
        net_source.eval()

        for data, target in dl_test_mn:
            data, target = data.to(device), target.to(device)
            # logging.info clean accuracy
            output = net_target(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            # fgsm
            # x_fgm_ts4 = projected_gradient_descent(net_eval, data, EPS, 0.02, 40, np.inf, clip_min=-1, clip_max=1,targeted=True, y=(torch.ones_like(target) * 2).to(device))
            x_adv = projected_gradient_descent(net_source, data, EPS, 0.01, 40, np.inf, clip_min=0, clip_max=1
                                               )
            # delta = attack_pgd(net_source, data, target, EPS,0.01, 40, 2)

            # x_adv = fast_gradient_method(net_source,data,EPS,norm=np.inf,clip_min=-1,clip_max=1)
            # x_fgm_ts4 = DeepFool(net_eval, 10, 0.02)
            # x_fgm_ts4 = x_fgm_ts4(data, target)
            with torch.no_grad():
                output_adv = net_target(x_adv)
                test_loss_adv += F.cross_entropy(output_adv, target, reduction='sum').item()  # sum up batch loss
                pred_fgsm = output_adv.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct_fgsm += pred_fgsm.eq(target.view_as(pred_fgsm)).sum().item()

                # 检测前后

                clean = [i for i, x in enumerate(pred_fgsm) if x == 10]
                for j in range(len(clean)):
                    pred_fgsm[clean[j]] = target[clean[j]]

                total_acc += pred_fgsm.eq(target.view_as(pred_fgsm)).sum().item()


        logging.info("source-target:{}-{}".format(number, number_target))
        logging.info('\nTest set: Clean loss: {:.3f}, Clean Accuracy: {}/{} ({:.0f}l%)\n'.format(
            test_loss / len(dl_test_mn.dataset), correct, len(dl_test_mn.dataset),
            100. * correct / len(dl_test_mn.dataset)))
        logging.info('\nADV set_fgsm:  loss: {:.3f},  Accuracy_before: {}/{} ({:.0f}l%), Accuracy_after: ({:.0f}l%)\n'.format(
            test_loss_adv / len(dl_test_mn.dataset), correct_fgsm, len(dl_test_mn.dataset),
            100. * correct_fgsm / len(dl_test_mn.dataset),100. * total_acc / len(dl_test_mn.dataset)))




