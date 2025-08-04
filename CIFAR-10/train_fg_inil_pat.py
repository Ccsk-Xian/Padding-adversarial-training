import argparse
import copy
import logging
import os
import time
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from apex import amp
from vgg import vgg16_bn,vgg19_bn
import torchattacks
from preact_resnet import PreActResNet18
from GoogleNet import GoogLeNet
from rec import ResNet18
# from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
#     attack_pgd, evaluate_pgd, evaluate_standard)
from device_config import device
from utils_cifar_out import (upper_limit, lower_limit, std, clamp, get_loaders,
    attack_pgd, evaluate_pgd, evaluate_standard, makeRandom, LabelSmoothingLoss)

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='cifar10_data', type=str)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=10, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous'],
        help='Perturbation initialization method')
    parser.add_argument('--out-dir', default='save', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--early-stop', action='store_true', help='Early stop if overfitting occurs')
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    return parser.parse_args()


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = get_args()
    
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    logfile = os.path.join(args.out_dir, 'ZZ1.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(args.out_dir, 'ZZ1.log'))
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    pgd_alpha = (2 / 255.) / std

    model = PreActResNet18().to(device)
    # model = ResNet18().to(device)
    # model = torchvision.models.resnet50(num_classes=11).to(device)

    # model = vgg16_bn(num_classes=10).to(device)
    # model = vgg19_bn(num_classes=11).to(device)
    # model = GoogLeNet(num_classes=11).to(device)
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    # amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    # if args.opt_level == 'O2':
    #     amp_args['master_weights'] = args.master_weights
    # model, opt = amp.initialize(model, opt, **amp_args)
    criterion = nn.CrossEntropyLoss()
    loss_func = LabelSmoothingLoss(classes_target=10, classes=11, smoothing=0.3)

    if args.delta_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).to(device)

    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # Training
    prev_robust_acc = 0.
    start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            if i == 0:
                first_batch = (X, y)
            # if args.delta_init != 'previous':
            #     delta = torch.zeros_like(X).to(device)
            # if args.delta_init == 'random':
            #     for j in range(len(epsilon)):
            #         delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
            #     # print(delta.device)
            #     # print(X.device)
            #     # print(lower_limit.device)
            #     delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            # delta.requires_grad = True
            # output = model(X + delta[:X.size(0)])
            # # loss = F.cross_entropy(output, y)
            # loss = loss_func(output,y)
            # # with amp.scale_loss(loss, opt) as scaled_loss:
            #     # scaled_loss.backward()
            # loss.backward()
            # grad = delta.grad.detach()
            # delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            # delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            # delta = delta.detach()
            # X_adv = X + delta[:X.size(0)]
            # # atk = torchattacks.PGD(model,eps=8/255,alpha=2/255,steps=10,loss="")
            # # atk = torchattacks.PGD(model,eps=8/255,alpha=2/255,steps=10,loss=loss_func)
            # # atk.set_normalization_used(mean=[0.4914, 0.4822, 0.4465],std=[0.2471, 0.2435, 0.2616])
            # # X = atk(X,y)

            # y_padding = torch.ones_like(y) * 10
            # X = torch.cat((X, X_adv), 0)
            # y = torch.cat((y, y_padding), 0)
            # X = clamp(X, lower_limit, upper_limit)
            # X, y = X.to(device), y.to(device)

            # num=10
            # # adv_X = atk(X,y)
            # # padding = torchattacks.PGD(model,eps=64/255,alpha=2/255,steps=10,loss="")
            # padding = torchattacks.PGD(model,eps=64/255,alpha=2/255,steps=10,loss=loss_func)
            # padding.set_normalization_used(mean=[0.4914, 0.4822, 0.4465],std=[0.2471, 0.2435, 0.2616])
            # data_padding = padding(X,y)
            # # data_padding = atk(X[:num,:,:,:],y[:num])
            # y_padding = torch.ones_like(y) * 10
            # # X = torch.cat((X, data_padding[:num,:,:,:]), 0)
            # # y = torch.cat((y, y_padding[:num]), 0)
            # X = torch.cat((X, data_padding[:num,:,:,:]), 0)
            # y = torch.cat((y, y_padding[:num]), 0)
            # X = clamp(X, lower_limit, upper_limit)
            # X, y = X.to(device), y.to(device)

            # data_padding = makeRandom(channel=3, data=X, Epoch=args.epochs, epoch=epoch)
            # data_padding = data_padding.to(device)
            # y_padding = torch.ones_like(y) * 10
            # y_padding = y_padding.to(device)
            # # num = int(round((epoch+1)/args.epochs * args.batch_size)) 
            # num=10
            # # num = int(round((epoch+1)/args.epochs * args.batch_size*0.5)) 
            # X = torch.cat((X, data_padding[:num,:,:,:]), 0)
            # y = torch.cat((y, y_padding[:num]), 0)
            # X = clamp(X, lower_limit, upper_limit)
            # X, y = X.to(device), y.to(device)

            output = model(X)
            # loss = criterion(output, y)
            loss = loss_func(output,y)
            opt.zero_grad()
            # with amp.scale_loss(loss, opt) as scaled_loss:
            #     scaled_loss.backward()
            loss.backward()
            opt.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()
        if args.early_stop:
            # Check current PGD robustness of model using random minibatch
            X, y = first_batch
            pgd_delta = attack_pgd(model, X, y, epsilon, pgd_alpha, 5, 1, opt)
            with torch.no_grad():
                output = model(clamp(X + pgd_delta[:X.size(0)], lower_limit, upper_limit))
            robust_acc = (output.max(1)[1] == y).sum().item() / y.size(0)
            if robust_acc - prev_robust_acc < -0.2:
                break
            prev_robust_acc = robust_acc
            best_state_dict = copy.deepcopy(model.state_dict())
        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
                    epoch, epoch_time - start_epoch_time, lr, train_loss / train_n, train_acc / train_n)
        print(epoch, train_acc/train_n)
    train_time = time.time()
    if not args.early_stop:
        best_state_dict = model.state_dict()
    torch.save(best_state_dict, os.path.join(args.out_dir, 'ZZ1.pth'))
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    # Evaluation
    # model_test = PreActResNet18().to(device)
    # model_test.load_state_dict(best_state_dict)
    # model_test.float()
    # model_test.eval()

    pgd_loss, pgd_acc, total_acc = evaluate_pgd(test_loader, model, 10, 5,epsilon=8,device=device)
    test_loss, test_acc = evaluate_standard(test_loader, model,device=device)
    # pgd_loss16, pgd_acc16,total_acc16 = evaluate_pgd(test_loader, model_test, 50, 10,epsilon=16,device=device)
    pgd_loss32, pgd_acc32,total_acc32 = evaluate_pgd(test_loader, model, 10, 5,epsilon=32,device=device)

    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc \t PGD After ACC')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc, total_acc)

    # logger.info('16Test Loss \t Test Acc \t PGD Loss \t PGD Acc \t PGD After ACC')
    # logger.info('%.4f \t \t %.4f \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss16, pgd_acc16, total_acc16)

    logger.info('32Test Loss \t Test Acc \t PGD Loss \t PGD Acc \t PGD After ACC')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss32, pgd_acc32, total_acc32)


if __name__ == "__main__":
    main()