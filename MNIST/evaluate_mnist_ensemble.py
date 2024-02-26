import argparse
import logging
import sys
import time
import torchattacks
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from mnist_net import mnist_net
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

# Acc: 0.9518--0.25
# Acc: 0.7994--0.35

# Acc: 0.0729--0.45
# Acc: 0.0146--0.50
# Acc: 0.0032-0.55
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s %(filename)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_fgsm(model, X, y, epsilon):
    delta = torch.zeros_like(X, requires_grad=True)
    output = model(X + delta)
    loss = F.cross_entropy(output, y)
    loss.backward()
    grad = delta.grad.detach()
    delta.data = epsilon * torch.sign(grad)
    return delta.detach()


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).cuda()
        # 确保添加扰动后图片仍在0，1之间
        delta.data = clamp(delta, 0 - X, 1 - X)
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
            d = clamp(d, 0 - X, 1 - X)
            delta.data[index] = d[index]
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X + delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--data-dir', default='../mnist-data', type=str)
    # parser.add_argument('--fname', default='./mnist_model/test_1.pth',type=str)
    # parser.add_argument('--fname', default='./mnist_model/test_1.pth', type=str) # 84.5
    # parser.add_argument('--fname', default='./eval/4.pth', type=str)  # fgsm对抗训练
    # parser.add_argument('--fname', default='./eval/4.pth', type=str) #0.45smoothing的最终模型
    # parser.add_argument('--fname', default='./exam_1/mnist.pth', type=str) #82.8 无改 loss  0.2685
    # parser.add_argument('--fname', default='./exam_1/mnist_l1.pth', type=str) #fgsm试trap 71.8
    # parser.add_argument('--fname', default='./exam_1/mnist_l2.pth', type=str) #更新时trap 88.29
    parser.add_argument('--fname', default='./exam_1/mnist_l9.pth',
                        type=str)  # 0.3-98.41 90.02/92.33(0.25-92.71/94.32 ρ=0.35-29/69 0.4-55  0.5-42.93  0.8 21.26) 0.4-97.79,93.74  0.5-96.59,98.50(不受扰动大小影响) 0.6-92.53/99.79  0.7 6.48 99.99
    # 0.3 97.84-88.61/94.03  0.35 97.36-89.06/94.50 0.4 97.45-89.85/95.05
    # loss 1.4589 和原始对抗训练相比他的loss提升很多，如果只从loss的观点来思考，是消除部分梯度遮蔽的。
    # 以对抗训练得扰动为界，以下是保持对抗训练，以上是检测。
    parser.add_argument('--fname1', default='./exam_1/mnist_l6.pth',
                        type=str)
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'none'])
    parser.add_argument('--epsilon', default=0.3, type=float)
    parser.add_argument('--attack-iters', default=50, type=int)
    parser.add_argument('--alpha', default=1e-2, type=float)
    parser.add_argument('--restarts', default=3, type=int)
    parser.add_argument('--seed', default=0, type=int)
    return parser.parse_args()

def model():
    args = get_args()
    model = mnist_net().cuda()
    checkpoint = torch.load(args.fname)
    model.load_state_dict(checkpoint)
    model.eval()
    return model
def model1():
    args = get_args()
    model1 = mnist_net().cuda()
    checkpoint1 = torch.load(args.fname1)
    model1.load_state_dict(checkpoint1)
    model1.eval()
    return model1

class OursNet(nn.Module):
    def __init__(self):
        super(OursNet, self).__init__()
        self.net1 = model()
        self.net2 = model1()



    def forward(self, x):
        net2_out = self.net2(x)
        _, pred2 = net2_out.max(
            1
        )
        # 根据最大输出类补充logits
        net1_out = self.net1(x)
        _, pred = net1_out.max(
            1
        )
        result = torch.zeros(net2_out.size()).cuda()
        for i in range(net2_out.size()[0]):
            if pred2[i] > 9 or pred[i] > 9:
                if pred2[i] > 9:
                    result[i] = net2_out[i].cuda()
                    # print('>9')
                elif pred[i] > 9:
                    result[i] = net1_out[i].cuda()
                    # print('>9')
            else:
                result[i] = ((net1_out[i] + net2_out[i] ) / 2).cuda()

        # result = (net1_out+net2_out+net3_out)/3
        return result

def main():
    args = get_args()
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    mnist_test = datasets.MNIST("../mnist-data", train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=args.batch_size, shuffle=False)


    model = OursNet().cuda()
    model.eval()




    total_loss = 0
    total_loss_clean = 0
    total_acc = 0
    total_acc_clean = 0
    total_acc_before = 0
    n = 0

    if args.attack == 'none':
        with torch.no_grad():
            for i, (X, y) in enumerate(test_loader):
                X, y = X.cuda(), y.cuda()
                output = model(X)
                loss = F.cross_entropy(output, y)
                total_loss += loss.item() * y.size(0)
                total_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)
    else:
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()

            output_clean = model(X)
            loss_clean = F.cross_entropy(output_clean, y)
            total_loss_clean += loss_clean.item() * y.size(0)
            total_acc_clean += (output_clean.max(1)[1] == y).sum().item()

            if args.attack == 'pgd':
                X_adv = projected_gradient_descent(model, X, args.epsilon, 0.02, 40, np.inf, clip_min=0, clip_max=1
                                                   ,targeted=True,y=(torch.ones_like(y) * 2).cuda())
                # delta = attack_pgd(model, X, y, args.epsilon, args.alpha, args.attack_iters, args.restarts)
            elif args.attack == 'fgsm':
                delta = attack_fgsm(model, X, y, args.epsilon)
            elif args.attack == 'cw':
                X_adv = carlini_wagner_l2(model, X, n_classes=11, initial_const=0.5, confidence=0.01, clip_min=0,
                                          clip_max=1, binary_search_steps=5, targeted=True,y=(torch.ones_like(y) * 2).cuda())
                # cw_attack = torchattacks.CW(model,c=1,kappa=0.1)
                # X_adv = cw_attack(X,y)
            with torch.no_grad():
                # if args.attack=="cw":
                output = model(X_adv)  # cw
                # else:
                # output = model(X + delta)
                #
                loss = F.cross_entropy(output, y)
                total_loss += loss.item() * y.size(0)

                pred_ = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

                total_acc_before += (output.max(1)[1] == y).sum().item()

                clean = [i for i, x in enumerate(pred_) if x == 10]
                for j in range(len(clean)):
                    pred_[clean[j]] = y[clean[j]]

                total_acc += pred_.eq(y.view_as(pred_)).sum().item()

                n += y.size(0)
                logger.info('i:%d', i)
                logger.info('Clean Loss: %.4f, Clean_Acc: %.4f -- Test Loss: %.4f, Before_Acc: %.4f,Acc: %.4f',
                            total_loss_clean / n, total_acc_clean / n, total_loss / n, total_acc_before / n,
                            total_acc / n)

        test_output = model(X[-10:])
        pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
        print(pred_y, 'prediction number')
        print(y[-10:].cpu().numpy(), 'real number')

        test_output = model((X + delta)[-10:])
        pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
        print(pred_y, 'prediction number')
        print(y[-10:].cpu().numpy(), 'real number')

    logger.info('Clean Loss: %.4f, Clean_Acc: %.4f -- Test Loss: %.4f, Before_Acc: %.4f,Acc: %.4f',
                total_loss_clean / n, total_acc_clean / n, total_loss / n, total_acc_before / n, total_acc / n)


if __name__ == "__main__":
    main()
