import torch.nn as nn
import torch
import torchvision
from pathlib import Path
import torch.nn.functional as F
from Mnist_trip import mnist_net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def trip13_0():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = mnist_net(13).to(device)
    net_eval.load_state_dict(torch.load(Path("./mnist_model/trip_13_ls0.4_rd0.0.pth")))
    return net_eval

def trip13_0_noT():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = mnist_net(13).to(device)
    # net_eval.load_state_dict(torch.load(Path("./mnist_model/trip_13_ls0.4_rd0.0.pth")))
    return net_eval

def trip15_0():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = mnist_net(15).to(device)
    net_eval.load_state_dict(torch.load(Path("./mnist_model/trip_15_ls0.4_rd0.0.pth")))
    return net_eval

def trip17_0():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = mnist_net(17).to(device)
    net_eval.load_state_dict(torch.load(Path("./mnist_model/trip_17_ls0.4_rd0.0.pth")))
    return net_eval

def trip20_0():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = mnist_net(20).to(device)
    net_eval.load_state_dict(torch.load(Path("./mnist_model/trip_20_ls0.4_rd0.0.pth")))
    return net_eval

def trip13_15():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = mnist_net(13).to(device)
    net_eval.load_state_dict(torch.load(Path("./mnist_model/trip_13_ls0.4_rd0.15.pth")))
    return net_eval

def trip15_15():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = mnist_net(15).to(device)
    net_eval.load_state_dict(torch.load(Path("./mnist_model/trip_15_ls0.4_rd0.15.pth")))
    return net_eval

def trip15_15_noT():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = mnist_net(15).to(device)
    # net_eval.load_state_dict(torch.load(Path("./mnist_model/trip_15_ls0.4_rd0.15.pth")))
    return net_eval

def trip17_15():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = mnist_net(17).to(device)
    net_eval.load_state_dict(torch.load(Path("./mnist_model/trip_17_ls0.4_rd0.15.pth")))
    return net_eval

def trip20_15():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = mnist_net(20).to(device)
    net_eval.load_state_dict(torch.load(Path("./mnist_model/trip_20_ls0.4_rd0.15.pth")))
    return net_eval

def trip13_25():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = mnist_net(13).to(device)
    net_eval.load_state_dict(torch.load(Path("./mnist_model/trip_13_ls0.4_rd0.25.pth")))
    return net_eval

def trip15_25():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = mnist_net(15).to(device)
    net_eval.load_state_dict(torch.load(Path("./mnist_model/trip_15_ls0.4_rd0.25.pth")))
    return net_eval

def trip17_25():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = mnist_net(17).to(device)
    net_eval.load_state_dict(torch.load(Path("./mnist_model/trip_17_ls0.4_rd0.25.pth")))
    return net_eval

def trip20_25():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = mnist_net(20).to(device)
    net_eval.load_state_dict(torch.load(Path("./mnist_model/trip_20_ls0.4_rd0.25.pth")))
    return net_eval

def trip20_25_noT():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = mnist_net(20).to(device)
    # net_eval.load_state_dict(torch.load(Path("./mnist_model/trip_20_ls0.4_rd0.25.pth")))
    return net_eval

class trip_3_5_20(nn.Module):
    def __init__(self):
        super(trip_3_5_20, self).__init__()
        self.net1 = trip13_0()
        self.net2 = trip15_15()
        self.net3 = trip20_25()

#如果是我们自己的方法，那就需要多一列，如果判别是>正确类别的话，就要将后位全移到11
    def forward(self,x):
        net3_out = self.net3(x)
        _, pred3 = net3_out.max(
            1
        )
        net2_out = self.net2(x)
        net2_add = torch.zeros((net2_out.size()[0], net3_out.size()[1] - net2_out.size()[1])).to(device)
        net2_out = torch.cat((net2_out, net2_add), 1).to(device)
        _, pred2 = net2_out.max(
            1
        )
        #根据最大输出类补充logits
        net1_out = self.net1(x)
        net1_add = torch.zeros((net1_out.size()[0],net3_out.size()[1]-net1_out.size()[1])).to(device)
        net1_out = torch.cat((net1_out,net1_add),1).to(device)
        _, pred = net1_out.max(
            1
        )
        result = torch.zeros(net3_out.size()).to(device)
        for i in range (net3_out.size()[0]):
            if pred2[i] >9 or pred3[i]>9 or pred[i]>9:
                if pred3[i]>9:
                    result[i] = net3_out[i].to(device)
                    # print('>9')
                elif pred2[i]>9:
                    result[i] = net2_out[i].to(device)
                    # print('>9')
                elif pred[i]>9:
                    result[i] = net1_out[i].to(device)
                    # print('>9')
            else:
                result[i] = ((net1_out[i] + net2_out[i] + net3_out[i]) / 3).to(device)

        # result = (net1_out+net2_out+net3_out)/3
        return result


class trip_3_5_20_noT(nn.Module):
    def __init__(self):
        super(trip_3_5_20_noT, self).__init__()
        self.net1 = trip13_0_noT()
        self.net2 = trip15_15_noT()
        self.net3 = trip20_25_noT()

#如果是我们自己的方法，那就需要多一列，如果判别是>正确类别的话，就要将后位全移到11
    def forward(self,x):
        net3_out = self.net3(x)
        _, pred3 = net3_out.max(
            1
        )
        net2_out = self.net2(x)
        net2_add = torch.zeros((net2_out.size()[0], net3_out.size()[1] - net2_out.size()[1])).to(device)
        net2_out = torch.cat((net2_out, net2_add), 1).to(device)
        _, pred2 = net2_out.max(
            1
        )
        #根据最大输出类补充logits
        net1_out = self.net1(x)
        net1_add = torch.zeros((net1_out.size()[0],net3_out.size()[1]-net1_out.size()[1])).to(device)
        net1_out = torch.cat((net1_out,net1_add),1).to(device)
        _, pred = net1_out.max(
            1
        )
        result = torch.zeros(net3_out.size()).to(device)
        for i in range (net3_out.size()[0]):
            if pred2[i] >9 or pred3[i]>9 or pred[i]>9:
                if pred3[i]>9:
                    result[i] = net3_out[i].to(device)
                    # print('>9')
                elif pred2[i]>9:
                    result[i] = net2_out[i].to(device)
                    # print('>9')
                elif pred[i]>9:
                    result[i] = net1_out[i].to(device)
                    # print('>9')
            else:
                result[i] = ((net1_out[i] + net2_out[i] + net3_out[i]) / 3).to(device)

        # result = (net1_out+net2_out+net3_out)/3
        return result