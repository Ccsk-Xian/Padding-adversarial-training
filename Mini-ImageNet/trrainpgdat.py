# # 普通模型训练
# from __future__ import print_function

# import os
# import argparse
# import socket
# import time
# import itertools
# import tensorboard_logger as tb_logger
# import torch
# import torch.optim as optim
# import torch.nn as nn
# import torch.backends.cudnn as cudnn
# from timm.scheduler.cosine_lr import CosineLRScheduler
# from models import model_dict
# import sys
# import numpy as np
# from dataset.cifar100 import get_cifar100_dataloaders
# from dataset.miniImage import get_miniImagenet_dataloader
# import logging
# from fvcore.nn import FlopCountAnalysis, parameter_count_table
# from helper.util import  accuracy, AverageMeter,adjust_learning_rate
# # from helper.loops import train_vanilla as train, validate
# from termcolor import colored
# import timm
# import torch.nn.functional as F

# def validate(val_loader, model, criterion, opt,device):
#     """validation"""
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()

#     # switch to evaluate mode
#     model.eval()

#     with torch.no_grad():
#         end = time.time()
#         for idx, (input, target) in enumerate(val_loader):
            
#             input = input.float()
#             if torch.cuda.is_available():
#                 input = input.to(device)
#                 target = target.to(device)

#             # compute output
#             out_1,out_2 = model(input)
#             output = torch.matmul(out_1.view(out_1.size(0),-1, 1), out_2.view(out_2.size(0),1, -1))
#             output = output.view(output.size(0), -1)
#             loss = criterion(output, target)

#             # measure accuracy and record loss
#             acc1, acc5 = accuracy(output, target, topk=(1, 5))
#             losses.update(loss.item(), input.size(0))
#             top1.update(acc1[0], input.size(0))
#             top5.update(acc5[0], input.size(0))

#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()

#             if idx % opt.print_freq == 0:
#                 print('Test: [{0}/{1}]\t'
#                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                       'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                        idx, len(val_loader), batch_time=batch_time, loss=losses,
#                        top1=top1, top5=top5))

#         print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
#               .format(top1=top1, top5=top5))

#     return top1.avg, top5.avg, losses.avg



# def create_logger(output_dir, dist_rank=0, name=''):
#     # create logger
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.DEBUG)
#     logger.propagate = False

#     # create formatter
#     fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
#     color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
#                 colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

#     # create console handlers for master process
#     if dist_rank == 0:
#         console_handler = logging.StreamHandler(sys.stdout)
#         console_handler.setLevel(logging.DEBUG)
#         console_handler.setFormatter(
#             logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
#         logger.addHandler(console_handler)

#     # create file handlers
#     path_file = os.path.join(output_dir, f'{name}log_rank{dist_rank}.txt')
#     if os.path.isfile(path_file):
#         os.remove(path_file)
#     file_handler = logging.FileHandler(path_file,mode='a')
#     file_handler.setLevel(logging.DEBUG)
#     file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
#     logger.addHandler(file_handler)

#     return logger

# def makeRandom(channel,data,max,min,mean,std,Epoch,epoch):
#     # schedule = lambda t: np.interp([t], [0, Epoch * 0.7, Epoch], [(max-min)/2, 2*max, 5*max])[0]
#     # schedule = lambda t: np.interp([t], [0, epoch * 0.5, epoch], [0.25, 1, 10])[0]
#     schedule = lambda t: np.interp([t], [0, epoch * 0.5, epoch], [0.35, 0.8, 1])[0]

#     a = torch.sign(torch.randn_like(data)) * schedule(epoch)
#     a = a.cuda()
#     data_padding = data + a
#     return data_padding

# def train(epoch, train_loader, model, criterion, optimizer, opt,loggering,device,trap_smoothing):
#     """vanilla training"""
#     model.train()

#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#     std = [0.2471, 0.2435, 0.2616]
#     mean = [0.48025, 0.4481, 0.3975]
#     std = torch.tensor(std).view(3,1,1).cuda()
#     #设置超参数：/255.是归一化到[0,1]。/std是消除量级
#     epsilon = (16 / 255.) / std
#     alpha = (2 / 255.) / std
#     pgd_alpha = (2 / 255.) / std

#     end = time.time()
#     for idx, (input, target) in enumerate(train_loader):
        
#         data_time.update(time.time() - end)

#         delta = torch.zeros_like(X).uniform_(epsilon, epsilon).cuda()
#         delta.requires_grad = True
#         output = model(input + delta)
#                 # loss = loss_func(output, y)
#         loss = F.cross_entropy(output, target)
#         loss.backward()
#         grad = delta.grad.detach()
#                 # 主要是这步
#         delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
#                 # 这里应该是控制扰动大小。
#         delta.data = torch.max(torch.min(1-input, delta.data), 0-input)
#         delta = delta.detach()
#         input = torch.clamp(input + delta, -1, 1)
#         # data_padding = makeRandom(channel=3, data=input, max=1, min=-1, mean=mean,
#         #                               std=std, Epoch=80, epoch=epoch, delta=delta)



#         # data_padding = data_padding.cuda()
#         # y_padding = torch.ones_like(target) * 100
#         # y_padding = y_padding.cuda()
#         # input = torch.cat((input, data_padding[:10,:,:,:]), 0)
#         # target = torch.cat((target, y_padding[:10]), 0)
#         input, target = input.cuda(), target.cuda()

#         input = input.float()
#         if torch.cuda.is_available():
#             input = input.to(device)
#             target = target.to(device)
#         # print(input.shape)
        
#         # ===================forward=====================
#         out_1,out_2 = model(input)
#         # output = torch.matmul(out_1.view(out_1.size(0),-1, 1), out_2.view(out_2.size(0),1, -1))
#         output = torch.matmul(out_2.view(out_2.size(0),-1, 1), out_1.view(out_1.size(0),1, -1))
#         # print(output.shape)
#         output = output.view(output.size(0), -1)
#         # print(output.shape)
#         loss = criterion(output, target)
#         # loss = trap_smoothing(output,target)

#         acc1, acc5 = accuracy(output, target, topk=(1, 5))
#         losses.update(loss.item(), input.size(0))
#         top1.update(acc1[0], input.size(0))
#         top5.update(acc5[0], input.size(0))

#         # ===================backward=====================
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # lr_scheduler.step_update(epoch * len(train_loader) + idx)
#         # ===================meters=====================
#         batch_time.update(time.time() - end)
#         end = time.time()

#         # tensorboard logger
#         pass

#         # print info
#         if idx % opt.print_freq == 0:
#             print('Epoch: [{0}][{1}/{2}]\t'
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                   'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                   'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                    epoch, idx, len(train_loader), batch_time=batch_time,
#                    data_time=data_time, loss=losses, top1=top1, top5=top5))
#             sys.stdout.flush()

#     print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
#           .format(top1=top1, top5=top5))

#     return top1.avg, losses.avg

# class LabelSmoothingLoss(nn.Module):
#     def __init__(self, classes_target,classes, smoothing=0.0, dim=-1):
#         super(LabelSmoothingLoss, self).__init__()
#         self.confidence = 1.0 - smoothing
#         self.smoothing = smoothing/2
#         self.cls_target = classes_target
#         self.cls = classes
#         self.dim = dim

#     def forward(self, pred, target):
#         index_target = torch.where(target < 100)
#         index_trap = torch.where(target >99)
#         pred = pred.log_softmax(dim=self.dim)
#         distribution =  self.smoothing/99
#         # print(pred)

#         with torch.no_grad():
#             # true_dist = pred.data.clone()

#             # #不为陷阱类分配分布，这两种尝试是因为我们并不太清楚哪种方法可以增大检测效率。可能因为数据分布本身不处于一个流形，我们需要先对其进行自编码处理
#             # true_dist = torch.zeros((pred.size()[0], self.cls_target))
#             # add_dist = torch.zeros((true_dist.size()[0], self.cls - 10))
#             # true_dist.fill_(self.smoothing / (self.cls_target - 1))
#             # true_dist = torch.cat((true_dist, add_dist), 1).to(device)


#             # 框架建立
#             true_dist = torch.zeros((pred.size()[0],self.cls_target))
#             add_dist = torch.zeros((true_dist.size()[0],self.cls-100))
#             # 目标填充
#             true_dist[index_target[0], :]=distribution
#             Y = target.data[index_target]
#             true_dist[index_target[0], Y]= self.confidence
#             # true_dist.scatter_(1, target.cpu().detach().data.unsqueeze(1), self.confidence)

#             add_dist[index_target[0],:]=(self.smoothing/(self.cls - 100))
#             add_dist[index_trap[0],:]=1

#             true_dist = torch.cat((true_dist,add_dist),1).cuda()

#             #原方法
#             # true_dist.fill_(self.smoothing / (self.cls - 1))
#             # print(true_dist,'**')
#             #我把这里的填补放在陷阱类里


#         return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# def parse_option():

#     hostname = socket.gethostname()

#     parser = argparse.ArgumentParser('argument for training')

#     parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
#     parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
#     parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
#     parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
#     parser.add_argument('--num_workers', type=int, default=10, help='num of workers to use')
#     parser.add_argument('--epochs', type=int, default=80, help='number of training epochs')
#     parser.add_argument('--freeze-layers', type=bool, default=False)

#     # optimization
#     parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
#     parser.add_argument('--lr_decay_epochs', type=str, default='50,60,70', help='where to decay lr, can be a list')
#     parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
#     parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
#     parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
#     parser.add_argument('--weights', type=str, default='',
#                         help='initial weights path')

#     # dataset
#     parser.add_argument('--model', type=str, default='mobile_vit_tiny_likevitpp',
#                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
#                                  'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
#                                  'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
#                                  'MobileNetV2','MobileNetV3', 'ShuffleV1', 'ShuffleV2','mobile_vit_tiny','mobile_vit_xx_small_init','mobile_vit_x_small_init',
#                                   'mobile_vit_small_init','mobile_vit_tiny_init','mobile_vit_xx_small_in7','mobile_vit_x_small_in7',
#                                    'mobile_vit_small_in7','mobile_vit_tiny_in7','mobile_vit_tiny_likevit','mobile_vit_tiny_novit','mobile_vit_xx_small_best','mobile_vit_tiny_best','mobile_vit_tiny_likevitpp',
#                                    'mobile_vit_tiny_novit_test1','mobile_vit_tiny_novit_test2','mobile_vit_tiny_novit_test3','mobile_vit_tiny_novit_test4',
#                                    'mobile_vit_tiny_novit_test5','mobile_vit_tiny_novit_test6','mobile_vit_tiny_novit_test0','mobile_vit_tiny_dilatedblock_5','mobile_vit_tiny_dilatedblock_7',
#                                    'mobile_vit_tiny_dilatedblock_9','mobile_vit_tiny_dilatedblock_5_1','mobile_vit_tiny_dilatedblock_5_2','mobile_vit_tiny_dilatedblock_5_3','mobile_vit_tiny_dilatedblock_5_4','mobile_vit_tiny_dilatedblock_5_5',
#                                    'mobile_vit_tiny_dilatedblock_5_noweight','mcunet','SwiftFormer_XXS','edgenext_xxx_small','mobilecontainer','repvit_m0_6','repvit_m0_6_infiuni','RepInfiniteUniVit_initial','repvit_m0_6_uni_ours',
#                                    'mobile_half_1_1_1','mobile_half_1_1_2','mobile_half_1_1_3','mobile_half_1_1_4','mobile_half_1_2_1','mobile_half_1_2_2','mobile_half_1_2_3','mobile_half_1_2_4','mobile_half_1_3_1','mobile_half_1_1_5',
#                                    'mobile_half_2_1_1','mobile_half_2_1_1_1','mobile_half_2_1_2','mobile_half_2_1_3','mobile_half_2_1_4','mobile_half_2_1_5','mobile_half_2_1_6','mobile_half_3_1_1',
#                                    'mobile_half_4_1_1','mobile_half_4_1_2','mobile_half_4_1_3','mobile_half_4_1_4','mobile_half_4_1_5','mobile_half_4_1_6','mobile_half_4_1_7','mobile_half_5_1_1','mobile_half_percent',
#                                    'mobile_half_1_1_7','mobile_half_1_1_8','mobile_half_1_1_9','mobile_half_1_1_1_1','mobile_half_1_1_10','mobile_half_1_1_11','mobile_half_1_1_12','mobile_half_4_1_2_1','mobile_half_4_1_2_2','mobile_half_4_1_3_1','mobile_half_4_1_3_3','mobile_half_4_1_3_2',
#                                    'mobile_half_1_1_8_1','mobile_half_1_1_8_2','mobile_half_6_1_1_1','mobile_half_6_1_1_2','mobile_half_6_1_2_1','mobile_half_1_1_12_1','mobile_half_1_1_8_3','mobile_half_6_1_1_2_1','mobile_half_1_1_5_1','mobile_half_1_1_5_2','mobile_half_1_1_5_3','mobile_half_1_1_5_4','mobile_half_1_1_5_3_1',
#                                    'mobile_half_1_2_2_1','mobile_half_1_2_1_1','mobile_half_7_1_1','mobile_half_7_1_2','mobile_half_7_1_1_1','mobile_half_7_1_2_1','MobileNetV2_9','ContrastNet1'])
#     parser.add_argument('--dataset', type=str, default='miniImage', choices=['cifar100','miniImage'], help='dataset')

#     parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')
#     parser.add_argument('-cuda', '--cuda', type=str, default='0', help='the cuda number')
#     parser.add_argument('--arch_name', type=str, default='mobile_vit_tiny_likevitpp',
#                         help='log name')
#     parser.add_argument('--OUTPUT', type=str, default='./log',
#                         help='log output path')

#     opt = parser.parse_args()
#     print('111'+opt.model)
    
#     # set different learning rate from these 4 models
#     if opt.model in ['MobileNetV2','MobileNetV3' 'ShuffleV1', 'ShuffleV2','mobile_vit_tiny','mobile_vit_xx_small_init','mobile_vit_x_small_init',
#                                   'mobile_vit_small_init','mobile_vit_tiny_init','mobile_vit_xx_small_in7','mobile_vit_x_small_in7',
#                                    'mobile_vit_small_in7','mobile_vit_tiny_in7','mobile_vit_tiny_likevit','mobile_vit_tiny_novit','mobile_vit_xx_small_best','mobile_vit_tiny_best',
#                                    'mobile_vit_tiny_likevitpp','mobile_vit_tiny_novit_test1','mobile_vit_tiny_novit_test2','mobile_vit_tiny_novit_test3','mobile_vit_tiny_novit_test4',
#                                    'mobile_vit_tiny_novit_test5','mobile_vit_tiny_novit_test6','mobile_vit_tiny_novit_test0','mobile_vit_tiny_dilatedblock_5','mobile_vit_tiny_dilatedblock_7',
#                                    'mobile_vit_tiny_dilatedblock_9','mobile_vit_tiny_dilatedblock_5_1','mobile_vit_tiny_dilatedblock_5_2','mobile_vit_tiny_dilatedblock_5_3','mobile_vit_tiny_dilatedblock_5_4','mobile_vit_tiny_dilatedblock_5_5',
#                                    'mobile_vit_tiny_dilatedblock_5_noweight','mobilecontainer','repvit_m0_6','repvit_m0_6_infiuni','RepInfiniteUniVit_initial','repvit_m0_6_uni_ours',
#                                    'mobile_half_1_1_1','mobile_half_1_1_2','mobile_half_1_1_3','mobile_half_1_1_4','mobile_half_1_2_1','mobile_half_1_2_2','mobile_half_1_2_3','mobile_half_1_2_4','mobile_half_1_3_1','mobile_half_1_1_5',
#                                    'mobile_half_2_1_1','mobile_half_2_1_1_1','mobile_half_2_1_2','mobile_half_2_1_3','mobile_half_2_1_4','mobile_half_2_1_5','mobile_half_2_1_6','mobile_half_3_1_1',
#                                    'mobile_half_4_1_1','mobile_half_4_1_2','mobile_half_4_1_3','mobile_half_4_1_4','mobile_half_4_1_5','mobile_half_4_1_6','mobile_half_4_1_7','mobile_half_5_1_1','mobile_half_percent',
#                                    'mobile_half_1_1_7','mobile_half_1_1_8','mobile_half_1_1_9','mobile_half_1_1_1_1','mobile_half_1_1_10','mobile_half_1_1_11','mobile_half_1_1_12','mobile_half_4_1_2_1','mobile_half_4_1_2_2','mobile_half_4_1_3_1','mobile_half_4_1_3_3','mobile_half_4_1_3_2','mobile_half_1_1_12_1',
#                                    'mobile_half_1_1_8_1','mobile_half_1_1_8_2','mobile_half_6_1_1_1','mobile_half_6_1_1_2','mobile_half_6_1_2_1','mobile_half_1_1_8_3','mobile_half_6_1_1_2_1','mobile_half_1_1_5_1','mobile_half_1_1_5_2','mobile_half_1_1_5_3','mobile_half_1_1_5_4','mobile_half_1_1_5_3_1',
#                                    'mobile_half_1_2_2_1','mobile_half_1_2_1_1','mobile_half_7_1_1','mobile_half_7_1_2','mobile_half_7_1_1_1','mobile_half_7_1_2_1','MobileNetV2_9','ContrastNet1']:
#         opt.learning_rate = 0.01

#     # set the path according to the environment
#     if hostname.startswith('wsl'):
#         opt.model_path = './path/teacher_model'
#         opt.tb_path = './path/teacher_tensorboards'
#     else:
#         opt.model_path = './save/teacher_model'
#         opt.tb_path = './save/teacher_tensorboards'

#     iterations = opt.lr_decay_epochs.split(',')
#     opt.lr_decay_epochs = list([])
#     for it in iterations:
#         opt.lr_decay_epochs.append(int(it))

#     opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate,
#                                                             opt.weight_decay, opt.trial)

#     opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
#     if not os.path.isdir(opt.tb_folder):
#         os.makedirs(opt.tb_folder)

#     opt.save_folder = os.path.join(opt.model_path, opt.model_name)
#     if not os.path.isdir(opt.save_folder):
#         os.makedirs(opt.save_folder)

#     return opt


# def main():
#     np.random.seed(970203)
#     torch.manual_seed(970203)
#     torch.cuda.manual_seed_all(970203) #所有GPU
#     torch.cuda.manual_seed(970203)     # 当前GPU
#     best_acc = 0
#     opt = parse_option()
#     device = torch.device("cuda:"+opt.cuda if torch.cuda.is_available() else "cpu")
    
    

#     if not os.path.exists(opt.OUTPUT):
#         os.mkdir(opt.OUTPUT)
#     loggering = create_logger(output_dir=opt.OUTPUT, dist_rank=0, name=f"{opt.arch_name}")

#     # dataloader
#     if opt.dataset == 'cifar100':
#         train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
#         n_cls = 100
#     elif opt.dataset == 'miniImage':
#         train_loader, val_loader = get_miniImagenet_dataloader(batch_size=opt.batch_size, num_workers=opt.num_workers)
#         n_cls = 100
#     else:
#         raise NotImplementedError(opt.dataset)

#     # model
#     # from mcunet.mcunet.model_zoo import net_id_list, build_model, download_tflite
#     # model ,_,_ = build_model(net_id="mcunet-in3",pretrained=False)
#     # model.classifier.out_features = n_cls
#     model = model_dict[opt.model](num_classes=n_cls)

#     if opt.weights != '':
#         print('finetune')
#         if os.path.exists(opt.weights):
#             assert os.path.exists(opt.weights), "file {} does not exist.".format(opt.weights)
#             # net.load_state_dict(torch.load(args.weights, map_location='cpu'))

#             weights_dict = torch.load(opt.weights)

#             cycleer = itertools.cycle(list(weights_dict.keys()))
#             cycleer_asis = itertools.cycle(list(weights_dict.keys())) 
#             load_weights_dict = {k: weights_dict[next(cycleer)]  if 'repadd' not in k  and weights_dict[next(cycleer_asis)].numel() == model.state_dict()[k].numel() else v for k, v, in model.state_dict().items()}                                
#             model.load_state_dict(load_weights_dict, strict=False)
#         else:
#             raise FileNotFoundError("not found weights file: {}".format(opt.weights))
#         if opt.freeze_layers:
#             for name, para in model.named_parameters():
#                 # 除最后的全连接层外，其他权重全部冻结
#                 if "fc" not in name:
#                     if "repadd" not in name:
#                         print(name)
#                         para.requires_grad_(False)
#                     else:
#                         print('*'+name)
#                 else:
#                     print('*'+name)

#     # optimizer
#     optimizer = optim.SGD(model.parameters(),
#                           lr=opt.learning_rate,
#                           momentum=opt.momentum,
#                           weight_decay=opt.weight_decay)

#     criterion = nn.CrossEntropyLoss()

#     if torch.cuda.is_available():
#         model = model.to(device)
#         criterion = criterion.to(device)
#         cudnn.benchmark = True

#     # tensorboard
#     logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

#     # total_steps = opt.epochs*len(train_loader)
#     # warmup_steps  = int(len(train_loader))
#     # lr_scheduler = CosineLRScheduler(
#     #         optimizer,
#     #         t_initial=total_steps,
#     #         lr_min=0.0001,
#     #         warmup_lr_init=0.0,
#     #         warmup_t=warmup_steps,
#     #         cycle_limit=1,
#     #         t_in_epochs=False,
#     #     )

#     # routine
#     acc_arr = []
#     acc5_arr = []
#     loss_arr = []
#     for epoch in range(1, opt.epochs + 1):

#         adjust_learning_rate(epoch, opt, optimizer)
#         print("==> training...")
#         if epoch==1:
#             for input,_ in (train_loader):
#                 model.eval()
#                 img = input[0]
#                 img = torch.unsqueeze(img, dim=0)
#                 # predict class
#                 flops = FlopCountAnalysis(model,img.to(device))
#                 params = parameter_count_table(model)
#                 print(flops.total())
#                 print(params)
#                 loggering.info(flops.total())
#                 loggering.info(params)
#                 model.train()
#                 break
#         time1 = time.time()
#         loss_func = LabelSmoothingLoss(classes_target=200, classes=201, smoothing=0.02)
#         train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt,loggering,device,loss_func)
#         print(optimizer.param_groups[0]['lr'])
#         time2 = time.time()
#         print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

#         logger.log_value('train_acc', train_acc, epoch)
#         logger.log_value('train_loss', train_loss, epoch)

#         test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt,device)
#         acc_arr.append(float(test_acc))
#         acc5_arr.append(float(test_acc_top5))
#         loss_arr.append(float(test_loss))
#         logger.log_value('test_acc', test_acc, epoch)
#         logger.log_value('test_acc_top5', test_acc_top5, epoch)
#         logger.log_value('test_loss', test_loss, epoch)
#         loggering.info(f"test_acc{test_acc} epoch{epoch} top5 {test_acc_top5} loss{test_loss}")
#         loggering.info(f"acc1{acc_arr}")
#         loggering.info(f"acc5{acc5_arr}")
#         loggering.info(f"loss{loss_arr}")
        
        

#         # save the best model
#         if test_acc > best_acc:
#             best_acc = test_acc
#             state = {
#                 'epoch': epoch,
#                 'model': model.state_dict(),
#                 'best_acc': best_acc,
#                 'optimizer': optimizer.state_dict(),
#             }
#             save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
#             print('saving the best model!')
#             torch.save(state, save_file)

#         # regular saving
#         if epoch % opt.save_freq == 0:
#             print('==> Saving...')
#             state = {
#                 'epoch': epoch,
#                 'model': model.state_dict(),
#                 'accuracy': test_acc,
#                 'optimizer': optimizer.state_dict(),
#             }
#             save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
#             torch.save(state, save_file)

#     # This best accuracy is only for printing purpose.
#     # The results reported in the paper/README is from the last epoch.
#     print('best accuracy:', best_acc)
#     loggering.info(f"best_accuracy{best_acc}")

#     # save model
#     state = {
#         'opt': opt,
#         'model': model.state_dict(),
#         'optimizer': optimizer.state_dict(),
#     }
#     save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
#     torch.save(state, save_file)


# if __name__ == '__main__':
#     main()

# 普通模型训练
from __future__ import print_function

import os
import argparse
import socket
import time
import itertools
# import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from timm.scheduler.cosine_lr import CosineLRScheduler
from models import model_dict
import sys
import numpy as np
from dataset.cifar100 import get_cifar100_dataloaders
from dataset.miniImage import get_miniImagenet_dataloader
import logging
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from helper.util import  accuracy, AverageMeter,adjust_learning_rate
# from helper.loops import train_vanilla as train, validate
from termcolor import colored
import timm
import torch.nn.functional as F
import torchattacks
import time
def validate(val_loader, model, criterion, opt,device):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.to(device)
                target = target.to(device)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg



def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    path_file = os.path.join(output_dir, f'{name}log_rank{dist_rank}.txt')
    if os.path.isfile(path_file):
        os.remove(path_file)
    file_handler = logging.FileHandler(path_file,mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


def makeRandom(channel,data,max,min,mean,std,Epoch,epoch,device):
    # schedule = lambda t: np.interp([t], [0, Epoch * 0.7, Epoch], [(max-min)/2, 2*max, 5*max])[0]
    schedule = lambda t: np.interp([t], [0, epoch * 0.5, epoch], [0.25, 1, 10])[0]
    # schedule = lambda t: np.interp([t], [0, epoch * 0.5, epoch], [0.35, 0.8, 1])[0]

    a = torch.sign(torch.randn_like(data)) * schedule(epoch)
    a = a.to(device)
    data_padding = data + a
    return data_padding


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes_target,classes, device,smoothing=0.0,  dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing/2
        self.cls_target = classes_target
        self.cls = classes
        self.dim = dim
        self.device = device

    def forward(self, pred, target):
        index_target = torch.where(target < 100)
        index_trap = torch.where(target >99)
        pred = pred.log_softmax(dim=self.dim)
        distribution =  self.smoothing/99
        # print(pred)

        with torch.no_grad():
            # true_dist = pred.data.clone()

            # #不为陷阱类分配分布，这两种尝试是因为我们并不太清楚哪种方法可以增大检测效率。可能因为数据分布本身不处于一个流形，我们需要先对其进行自编码处理
            # true_dist = torch.zeros((pred.size()[0], self.cls_target))
            # add_dist = torch.zeros((true_dist.size()[0], self.cls - 10))
            # true_dist.fill_(self.smoothing / (self.cls_target - 1))
            # true_dist = torch.cat((true_dist, add_dist), 1).to(device)


            # 框架建立
            true_dist = torch.zeros((pred.size()[0],self.cls_target))
            add_dist = torch.zeros((true_dist.size()[0],self.cls-100))
            # 目标填充
            true_dist[index_target[0], :]=distribution
            Y = target.data[index_target]
            true_dist[index_target[0], Y]= self.confidence
            # true_dist.scatter_(1, target.cpu().detach().data.unsqueeze(1), self.confidence)

            add_dist[index_target[0],:]=(self.smoothing/(self.cls - 100))
            add_dist[index_trap[0],:]=1

            true_dist = torch.cat((true_dist,add_dist),1).to(self.device)

            #原方法
            # true_dist.fill_(self.smoothing / (self.cls - 1))
            # print(true_dist,'**')
            #我把这里的填补放在陷阱类里


        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def train(epoch, train_loader, model, criterion, optimizer, opt,loggering,device,trap_smoothing):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    std = [0.2471, 0.2435, 0.2616]
    mean = [0.48025, 0.4481, 0.3975]
    std = torch.tensor(std).view(3,1,1).to(device)
    #设置超参数：/255.是归一化到[0,1]。/std是消除量级
    epsilon = 16 / 255
    alpha =2 / 255
    pgd_alpha = (2 / 255.) / std
    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        
        input,target = input.to(device),target.to(device)
        data_time.update(time.time() - end)
        # delta = torch.zeros_like(input).to(device)
        # for j in range(len(epsilon)):
        #     delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
        # delta.requires_grad = True
        # output1 = model(input + delta)
        # loss = trap_smoothing(output1, target)
        # # loss = F.cross_entropy(output1, target)
        # loss.backward()
        # grad = delta.grad.detach()
        # #         # 主要是这步
        # delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        # #         # 这里应该是控制扰动大小。
        # # delta.data = torch.max(torch.min(1-input, delta.data), 0-input)
        # # delta = delta.detach()
        # X_adv = (input+delta).to(device)
        # input = torch.clamp(input + delta, -1, 1)\
        # atk = torchattacks.PGD(model,eps=epsilon,alpha=alpha,steps=10,loss=trap_smoothing)
        atk = torchattacks.PGD(model,eps=epsilon,alpha=alpha,steps=10,loss=trap_smoothing)
        # atk = torchattacks.FGSM(model,eps=epsilon,loss="")
        atk.set_normalization_used(mean=[0.48025, 0.4481, 0.3975], std=[0.2471, 0.2435, 0.2616])
        X_adv = atk(input,target).to(device)
        P_X = X_adv[:10, :, :, :] 
        data_padding = makeRandom(channel=3, data=P_X, max=1, min=-1, mean=mean,
                                      std=std, Epoch=40, epoch=epoch,device=device)


        

        data_padding = data_padding.to(device)
        y_padding = torch.ones_like(target[:50]) * 100
        y_padding = y_padding.to(device)
        input = torch.cat((X_adv, data_padding), 0)
        target = torch.cat((target, y_padding), 0)
        input, target = input.to(device), target.to(device)

        # input = X_adv.to(device)
        # input = input.float()
        # if torch.cuda.is_available():
        #     input = input.to(device)
        #     target = target.to(device)
        # print(input.shape)
        
        # ===================forward=====================
        output = model(input)
  
        # loss = criterion(output, target)
        loss = trap_smoothing(output,target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # lr_scheduler.step_update(epoch * len(train_loader) + idx)
        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()
            print(time.time())

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=10, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=40, help='number of training epochs')
    parser.add_argument('--start_epochs', type=int, default=1, help='restrart epochs')
    parser.add_argument('--restart', type=bool, default=False, help='restrart epochs')
    parser.add_argument('--freeze-layers', type=bool, default=False)

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='20, 30,35', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')

    # dataset
    parser.add_argument('--model', type=str, default='mobile_vit_tiny_likevitpp',
                       choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2','MobileNetV3', 'ShuffleV1', 'ShuffleV2','mobile_vit_tiny','mobile_vit_xx_small_init','mobile_vit_x_small_init',
                                  'mobile_vit_small_init','mobile_vit_tiny_init','mobile_vit_xx_small_in7','mobile_vit_x_small_in7',
                                   'mobile_vit_small_in7','mobile_vit_tiny_in7','mobile_vit_tiny_likevit','mobile_vit_tiny_novit','mobile_vit_xx_small_best','mobile_vit_tiny_best','mobile_vit_tiny_likevitpp',
                                   'mobile_vit_tiny_novit_test1','mobile_vit_tiny_novit_test2','mobile_vit_tiny_novit_test3','mobile_vit_tiny_novit_test4',
                                   'mobile_vit_tiny_novit_test5','mobile_vit_tiny_novit_test6','mobile_vit_tiny_novit_test0','mobile_vit_tiny_dilatedblock_5','mobile_vit_tiny_dilatedblock_7',
                                   'mobile_vit_tiny_dilatedblock_9','mobile_vit_tiny_dilatedblock_5_1','mobile_vit_tiny_dilatedblock_5_2','mobile_vit_tiny_dilatedblock_5_3','mobile_vit_tiny_dilatedblock_5_4','mobile_vit_tiny_dilatedblock_5_5',
                                   'mobile_vit_tiny_dilatedblock_5_noweight','mcunet','SwiftFormer_XXS','edgenext_xxx_small','mobilecontainer','repvit_m0_6','repvit_m0_6_infiuni','RepInfiniteUniVit_initial','repvit_m0_6_uni_ours',
                                   'mobile_half_1_1_1','mobile_half_1_1_2','mobile_half_1_1_3','mobile_half_1_1_4','mobile_half_1_2_1','mobile_half_1_2_2','mobile_half_1_2_3','mobile_half_1_2_4','mobile_half_1_3_1','mobile_half_1_1_5',
                                   'mobile_half_2_1_1','mobile_half_2_1_1_1','mobile_half_2_1_2','mobile_half_2_1_3','mobile_half_2_1_4','mobile_half_2_1_5','mobile_half_2_1_6','mobile_half_3_1_1',
                                   'mobile_half_4_1_1','mobile_half_4_1_2','mobile_half_4_1_3','mobile_half_4_1_4','mobile_half_4_1_5','mobile_half_4_1_6','mobile_half_4_1_7','mobile_half_5_1_1','mobile_half_percent',
                                   'mobile_half_1_1_7','mobile_half_1_1_8','mobile_half_1_1_9','mobile_half_1_1_1_1','mobile_half_1_1_10','mobile_half_1_1_11','mobile_half_1_1_12','mobile_half_4_1_2_1','mobile_half_4_1_2_2','mobile_half_4_1_3_1','mobile_half_4_1_3_3','mobile_half_4_1_3_2',
                                   'mobile_half_1_1_8_1','mobile_half_1_1_8_2','mobile_half_6_1_1_1','mobile_half_6_1_1_2','mobile_half_6_1_2_1','mobile_half_1_1_12_1','mobile_half_1_1_8_3','mobile_half_6_1_1_2_1','mobile_half_1_1_5_1','mobile_half_1_1_5_2','mobile_half_1_1_5_3','mobile_half_1_1_5_4','mobile_half_1_1_5_3_1',
                                   'mobile_half_1_2_2_1','mobile_half_1_2_1_1','mobile_half_7_1_1','mobile_half_7_1_2','mobile_half_7_1_1_1','mobile_half_7_1_2_1','mobile_half_1_1_1_2','mobile_half_5_1_1_1','mobile_half_5_1_1_2','mobile_half_1_1_8_4','mobile_half_1_1_8_5','mobile_half_1_1_8_6','mobile_half_1_1_8_7',
                                   'mobilemetanet','mobilemetanet_1','mobilemetanet_2','mobile_half_3_1_2','RepTinynet','RepTinynet1','RepTinynet2','RepTinynet3','RepTinynet4','RepTinynet5','RepTinynet6','RepTinynet7','RepTinynet8','RepTinynet9','RepTinynet10','RepTinynet11','RepTinynet12','RepTinynet13','mcunetlike','RepTinynet14','RepTinynet15','RepTinynet16','RepTinynet17',
                                   'mobile_half_5_1_1_3','mobile_half_class','mobile_half_class_2','mobile_half_class_3','RepTinynet18','RepTinynet19','RepTinynet20','RepTinynet21','RepTinynet22','RepTinynet23','mobile_half_5_1_1_4','RepTinynet24','RepTinynet25','repvggnet','McuNetv1','ContrastNet1','ContrastNet2','ContrastNet3','ContrastNet12','ContrastNet11','ContrastNet10','ContrastNet9','ContrastNet8','finalNet','ResNet50','ResNet18','squeezenet1'])
    parser.add_argument('--dataset', type=str, default='miniImage', choices=['cifar100','miniImage'], help='dataset')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')
    parser.add_argument('-cuda', '--cuda', type=str, default='0', help='the cuda number')
    parser.add_argument('--arch_name', type=str, default='mobile_vit_tiny_likevitpp',
                        help='log name')
    parser.add_argument('--OUTPUT', type=str, default='./log',
                        help='log output path')

    opt = parser.parse_args()
    print('111'+opt.model)
    
    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2','MobileNetV3' 'ShuffleV1', 'ShuffleV2','mobile_vit_tiny','mobile_vit_xx_small_init','mobile_vit_x_small_init',
                                  'mobile_vit_small_init','mobile_vit_tiny_init','mobile_vit_xx_small_in7','mobile_vit_x_small_in7',
                                   'mobile_vit_small_in7','mobile_vit_tiny_in7','mobile_vit_tiny_likevit','mobile_vit_tiny_novit','mobile_vit_xx_small_best','mobile_vit_tiny_best',
                                   'mobile_vit_tiny_likevitpp','mobile_vit_tiny_novit_test1','mobile_vit_tiny_novit_test2','mobile_vit_tiny_novit_test3','mobile_vit_tiny_novit_test4',
                                   'mobile_vit_tiny_novit_test5','mobile_vit_tiny_novit_test6','mobile_vit_tiny_novit_test0','mobile_vit_tiny_dilatedblock_5','mobile_vit_tiny_dilatedblock_7',
                                   'mobile_vit_tiny_dilatedblock_9','mobile_vit_tiny_dilatedblock_5_1','mobile_vit_tiny_dilatedblock_5_2','mobile_vit_tiny_dilatedblock_5_3','mobile_vit_tiny_dilatedblock_5_4','mobile_vit_tiny_dilatedblock_5_5',
                                   'mobile_vit_tiny_dilatedblock_5_noweight','mobilecontainer','repvit_m0_6','repvit_m0_6_infiuni','RepInfiniteUniVit_initial','repvit_m0_6_uni_ours',
                                   'mobile_half_1_1_1','mobile_half_1_1_2','mobile_half_1_1_3','mobile_half_1_1_4','mobile_half_1_2_1','mobile_half_1_2_2','mobile_half_1_2_3','mobile_half_1_2_4','mobile_half_1_3_1','mobile_half_1_1_5',
                                   'mobile_half_2_1_1','mobile_half_2_1_1_1','mobile_half_2_1_2','mobile_half_2_1_3','mobile_half_2_1_4','mobile_half_2_1_5','mobile_half_2_1_6','mobile_half_3_1_1',
                                   'mobile_half_4_1_1','mobile_half_4_1_2','mobile_half_4_1_3','mobile_half_4_1_4','mobile_half_4_1_5','mobile_half_4_1_6','mobile_half_4_1_7','mobile_half_5_1_1','mobile_half_percent',
                                   'mobile_half_1_1_7','mobile_half_1_1_8','mobile_half_1_1_9','mobile_half_1_1_1_1','mobile_half_1_1_10','mobile_half_1_1_11','mobile_half_1_1_12','mobile_half_4_1_2_1','mobile_half_4_1_2_2','mobile_half_4_1_3_1','mobile_half_4_1_3_3','mobile_half_4_1_3_2','mobile_half_1_1_12_1',
                                   'mobile_half_1_1_8_1','mobile_half_1_1_8_2','mobile_half_6_1_1_1','mobile_half_6_1_1_2','mobile_half_6_1_2_1','mobile_half_1_1_8_3','mobile_half_6_1_1_2_1','mobile_half_1_1_5_1','mobile_half_1_1_5_2','mobile_half_1_1_5_3','mobile_half_1_1_5_4','mobile_half_1_1_5_3_1',
                                   'mobile_half_1_2_2_1','mobile_half_1_2_1_1','mobile_half_7_1_1','mobile_half_7_1_2','mobile_half_7_1_1_1','mobile_half_7_1_2_1','mobile_half_1_1_1_2','mobile_half_5_1_1_1','mobile_half_5_1_1_2','mobile_half_1_1_8_4','mobile_half_1_1_8_5','mobile_half_1_1_8_6','mobile_half_1_1_8_7',
                                   'mobilemetanet','mobilemetanet_1','mobilemetanet_2','mobile_half_3_1_2','RepTinynet','RepTinynet1','RepTinynet2','RepTinynet3','RepTinynet4','RepTinynet5','RepTinynet6','RepTinynet7','RepTinynet8','RepTinynet9','RepTinynet10','RepTinynet11','RepTinynet12','RepTinynet13','RepTinynet14','RepTinynet15','mcunetlike','RepTinynet16','RepTinynet17',
                                   'mobile_half_5_1_1_3','mobile_half_class','mobile_half_class_2','mobile_half_class_3','RepTinynet18','RepTinynet19','RepTinynet20','RepTinynet21','RepTinynet22','RepTinynet23','mobile_half_5_1_1_4','RepTinynet24','RepTinynet25','repvggnet','McuNetv1','ContrastNet1','ContrastNet2','ContrastNet3','ContrastNet12','ContrastNet11','ContrastNet10','ContrastNet9','ContrastNet8','finalNet','squeezenet1']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('wsl'):
        opt.model_path = './path/teacher_model'
        opt.tb_path = './path/teacher_tensorboards'
    else:
        opt.model_path = './save/teacher_model'
        opt.tb_path = './save/teacher_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

import torchvision
def main():
    np.random.seed(970203)
    torch.manual_seed(970203)
    torch.cuda.manual_seed_all(970203) #所有GPU
    torch.cuda.manual_seed(970203)     # 当前GPU
    best_acc = 0
    start_epochs = 1
    opt = parse_option()
    device = torch.device("cuda:"+opt.cuda if torch.cuda.is_available() else "cpu")
    
    

    if not os.path.exists(opt.OUTPUT):
        os.mkdir(opt.OUTPUT)
    loggering = create_logger(output_dir=opt.OUTPUT, dist_rank=0, name=f"{opt.arch_name}")

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 100
    elif opt.dataset == 'miniImage':
        train_loader, val_loader = get_miniImagenet_dataloader(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 101
    else:
        raise NotImplementedError(opt.dataset)

    # model
    # from mcunet.mcunet.model_zoo import net_id_list, build_model, download_tflite
    # model ,_,_ = build_model(net_id="mcunet-in3",pretrained=False)
    # model.classifier.out_features = n_cls
    # model = model_dict[opt.model](num_classes=n_cls)
    # model = torchvision.models.resnet50(num_classes=1000, pretrained=True).to(device)
    model = torchvision.models.resnet18(num_classes=1000, pretrained=True).to(device)

    # for param in model.parameters():
    #     param.requires_grad = False
    # model.fc = nn.Linear(in_features=2048, out_features=101).cuda()
    model.fc = nn.Linear(in_features=512, out_features=101).cuda()

    

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.to(device)
        criterion = criterion.to(device)
        cudnn.benchmark = True
    if opt.restart:
        opt.weights = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=opt.start_epochs))
    if opt.weights != '':
        print('finetune')
        if os.path.exists(opt.weights):
            assert os.path.exists(opt.weights), "file {} does not exist.".format(opt.weights)
            # net.load_state_dict(torch.load(args.weights, map_location='cpu'))

            weights_dict = torch.load(opt.weights,map_location=device)

            # cycleer = itertools.cycle(list(weights_dict['model'].keys()))
            # cycleer_asis = itertools.cycle(list(weights_dict['model'].keys())) 
            # load_weights_dict = {k: weights_dict[next(cycleer)]  if 'repadd' not in k  and weights_dict[next(cycleer_asis)].numel() == model.state_dict()[k].numel() else v for k, v, in model.state_dict().items()}                                
            model.load_state_dict(weights_dict['model'], strict=True)
            print('model successful load')
            if  'optimizer' in weights_dict.keys():
                optimizer.load_state_dict(weights_dict['optimizer'])
                print('optimizer successful load')
            if  'epoch' in weights_dict.keys():
                start_epochs = weights_dict['epoch']+1
                print('epoch successful load')
            if  'best_acc' in weights_dict.keys():
                best_acc = weights_dict['best_acc']
                print('best_acc successful load')
            if  'acc1' in weights_dict.keys():
                acc_arr = weights_dict['acc1']
                print('acc1 successful load')
            if  'acc5' in weights_dict.keys():
                acc5_arr = weights_dict['acc5']
                print('acc5 successful load')
            if  'loss' in weights_dict.keys():
                loss_arr = weights_dict['loss']
                print('loss successful load')
        else:
            raise FileNotFoundError("not found weights file: {}".format(opt.weights))
        if opt.freeze_layers:
            for name, para in model.named_parameters():
                # 除最后的全连接层外，其他权重全部冻结
                if "fc" not in name:
                    if "repadd" not in name:
                        print(name)
                        para.requires_grad_(False)
                    else:
                        print('*'+name)
                else:
                    print('*'+name)
    # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # total_steps = opt.epochs*len(train_loader)
    # warmup_steps  = int(len(train_loader))
    # lr_scheduler = CosineLRScheduler(
    #         optimizer,
    #         t_initial=total_steps,
    #         lr_min=0.0001,
    #         warmup_lr_init=0.0,
    #         warmup_t=warmup_steps,
    #         cycle_limit=1,
    #         t_in_epochs=False,
    #     )

    # routine
    acc_arr = []
    acc5_arr = []
    loss_arr = []
    for epoch in range(start_epochs, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")
        if epoch==1:
            for input,_ in (train_loader):
                model.eval()
                img = input[0]
                print(img.shape)
                img = torch.unsqueeze(img, dim=0)
                # predict class
                flops = FlopCountAnalysis(model,img.to(device))
                params = parameter_count_table(model)
                print(flops.total())
                print(params)
                loggering.info(flops.total())
                loggering.info(params)
                model.train()
                break
        time1 = time.time()
        loss_func = LabelSmoothingLoss(classes_target=100, classes=101, smoothing=0.1,device = device)
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt,loggering,device,loss_func)
        print(optimizer.param_groups[0]['lr'])
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # logger.log_value('train_acc', train_acc, epoch)
        # logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt,device)
        acc_arr.append(float(test_acc))
        acc5_arr.append(float(test_acc_top5))
        loss_arr.append(float(test_loss))
        # logger.log_value('test_acc', test_acc, epoch)
        # logger.log_value('test_acc_top5', test_acc_top5, epoch)
        # logger.log_value('test_loss', test_loss, epoch)
        loggering.info(f"test_acc{test_acc} epoch{epoch} top5 {test_acc_top5} loss{test_loss}")
        loggering.info(f"acc1{acc_arr}")
        loggering.info(f"acc5{acc5_arr}")
        loggering.info(f"loss{loss_arr}")
        
        

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)
    loggering.info(f"best_accuracy{best_acc}")

    # save model
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
