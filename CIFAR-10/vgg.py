'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei
'''
import torch.nn as nn
import torch.nn.functional as F
import math
import torch

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

# 这里实现的是多参数的方案， 提升不大
# def create_gaussian_mask(KH,KW, noise = True, level=0.1, sigma=1.0,mask_type=1,normalization=False,weight=[],circle=True,offset=[]):
#     # Create a grid of indices
    
#     # grid_x = torch.arange(start=(-(KW - 1) / 2.), end=((KW - 1) / 2.))
#     grid_x = torch.linspace(-(KW - 1) / 2., (KW - 1) / 2., KW).to(sigma.device)
#     # print(grid_x)
#     grid_y = torch.linspace(-(KH - 1) / 2., (KH - 1) / 2., KH).to(sigma.device)
#     # grid_y = torch.arange(-(KH - 1) / 2., (KH - 1) / 2., KH)
#     # print(grid_x)
#     grid_x, grid_y = torch.meshgrid(grid_x, grid_y,indexing='xy')
#     if noise:
#         noise_level = level
#         grid_x = grid_x + noise_level * torch.randn(*grid_x.shape).to(sigma.device)
#         grid_y = grid_y + noise_level * torch.randn(*grid_y.shape).to(sigma.device)
    
#     if circle:
#         if mask_type==2:
#             sigma = sigma.view(sigma.shape[0],sigma.shape[1],1,1)
#             grid_x=grid_x.expand(sigma.shape[1],KH,KW)
#             grid_y=grid_y.expand(sigma.shape[1],KH,KW)
#         else:
#             sigma = sigma.view(sigma.shape[0],1,1)
        
#             grid_x=grid_x.expand(sigma.shape[0],KH,KW)
#             grid_y=grid_y.expand(sigma.shape[0],KH,KW)
#     else:
#         if mask_type==2:
#             sigma = sigma.view(2,sigma.shape[1],sigma.shape[2],1,1)
#             grid_x=grid_x.expand(sigma.shape[2],KH,KW)
#             grid_y=grid_y.expand(sigma.shape[2],KH,KW)
#         else:
#             sigma = sigma.view(2,sigma.shape[1],1,1)
#             grid_x=grid_x.expand(sigma.shape[1],KH,KW)
#             grid_y=grid_y.expand(sigma.shape[1],KH,KW)
#     if mask_type ==1:
        
#         # print(grid_x)
#         # 添加随机噪声到网格坐标
#         # if noise:
#         #     noise_level = level  # 噪声水平，可以根据需要调整
#         #     grid_x = grid_x + noise_level * torch.randn(*grid_x.shape).to(sigma.device)
#         #     grid_y = grid_y + noise_level * torch.randn(*grid_y.shape).to(sigma.device)
#         # print(grid)
#     # grid_z = np.zeros_like(grid_x)
    
#     # Calculate the Gaussian function
#         if circle:

#             if len(offset)>1:
#                 # gaussian = torch.exp(-0.5 * ((grid_x-offset[0].reshape(grid_x.shape))**2 + (grid_y-offset[1].reshape(grid_x.shape))**2) / sigma**2)
#                 gaussian = torch.exp(-0.5 * ((grid_x-offset[0])**2 + (grid_y-offset[1])**2) / sigma**2)
#             else:
#                 gaussian = torch.exp(-0.5 * (grid_x**2 + grid_y**2) / sigma**2)
#         else:
            
#             if len(offset)>1:
#                 # gaussian = torch.exp(-0.5 * ((grid_x-offset[0].reshape(grid_x.shape))**2/sigma[0]**2 + (grid_y-offset[1].reshape(grid_x.shape))**2/sigma[1]**2))

#                 gaussian = torch.exp(-0.5 * ((grid_x-offset[0])**2/sigma[0]**2 + (grid_y-offset[1])**2/sigma[1]**2))
#             else:
#                 gaussian = torch.exp(-0.5 * (grid_x**2/sigma[0]**2 + grid_y**2/sigma[1]**2))
        
#     # print(gaussian.shape)
#     # Normalize the Gaussian mask so that the maximum value is 1
    
#     if mask_type == 2:
#         b = torch.softmax(weight, dim=0)
#         # for i,x in enumerate([-1,0,1]):
#             # for o,y in enumerate([-1,0,1]):
#         for i in [-1,0,1]:
#             for o in [-1,0,1]:
#                 if i==-1 and o==-1:
#         #             if circle==True:
#         #                 gaussian = torch.exp(-0.5 * ((grid_x-i)**2 + (grid_y-o)**2) / sigma[i*KH+o]**2)
#         #             else:
#         #                 gaussian = torch.exp(-0.5 * (((grid_x-i)**2)/sigma[0][i*KH+o]**2 + ((grid_y-o)**2)/sigma[1][i*KH+o]**2))
#         #         else:
#         #             if circle==True:
#         #                 gaussian = gaussian + torch.exp(-0.5 * ((grid_x-i)**2 + (grid_y-o)**2) / sigma[i*KH+o]**2)
#         #             else:
#         #                 gaussian = gaussian+ torch.exp(-0.5 * (((grid_x-i)**2)/sigma[0][i*KH+o]**2 + ((grid_y-o)**2)/sigma[1][i*KH+o]**2))
#         # gaussian = gaussian/(KH*KW)
#                     if circle==True:
#                         if len(offset)>1:
#                             # gaussian = torch.exp(-0.5 * ((grid_x-i-offset[0][i*KH+o].reshape(grid_x.shape))**2 + (grid_y-o-offset[1][i*KH+o].reshape(grid_x.shape))**2) / sigma[i*KH+o]**2)*b[i*KH+o]
#                             gaussian = torch.exp(-0.5 * ((grid_x-i-offset[0][i*KH+o])**2 + (grid_y-o-offset[1][i*KH+o])**2) / sigma[i*KH+o]**2)*b[i*KH+o]
#                         else:
#                             # print(grid_x.shape)
#                             # print(sigma[i*KH+o].shape)
#                             gaussian = torch.exp(-0.5 * ((grid_x-i)**2 + (grid_y-o)**2) / sigma[i*KH+o]**2)*b[i*KH+o]
#                     else:
#                         if len(offset)>1:
#                             # gaussian = torch.exp(-0.5 * (((grid_x-i-offset[0][i*KH+o].reshape(grid_x.shape))**2)/sigma[0][i*KH+o]**2 + ((grid_y-o-offset[1][i*KH+o].reshape(grid_x.shape))**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]
#                             gaussian = torch.exp(-0.5 * (((grid_x-i-offset[0][i*KH+o])**2)/sigma[0][i*KH+o]**2 + ((grid_y-o-offset[1][i*KH+o])**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]
#                         else:
#                             gaussian = torch.exp(-0.5 * (((grid_x-i)**2)/sigma[0][i*KH+o]**2 + ((grid_y-o)**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]
#                 else:
#                     if circle==True:
#                         if len(offset)>1:
#                             # gaussian = gaussian+torch.exp(-0.5 * ((grid_x-i-offset[0][i*KH+o].reshape(grid_x.shape))**2 + (grid_y-o-offset[1][i*KH+o].reshape(grid_x.shape))**2) / sigma[i*KH+o]**2)*b[i*KH+o]
#                             gaussian = gaussian+torch.exp(-0.5 * ((grid_x-i-offset[0][i*KH+o])**2 + (grid_y-o-offset[1][i*KH+o])**2) / sigma[i*KH+o]**2)*b[i*KH+o]
#                         else:
#                             gaussian = gaussian+torch.exp(-0.5 * ((grid_x-i)**2 + (grid_y-o)**2) / sigma[i*KH+o]**2)*b[i*KH+o]
#                     else:
#                         if len(offset)>1:
#                             # gaussian = gaussian+torch.exp(-0.5 * (((grid_x-i-offset[0][i*KH+o].reshape(grid_x.shape))**2)/sigma[0][i*KH+o]**2 + ((grid_y-o-offset[1][i*KH+o].reshape(grid_x.shape))**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]
#                             gaussian = gaussian+torch.exp(-0.5 * (((grid_x-i-offset[0][i*KH+o])**2)/sigma[0][i*KH+o]**2 + ((grid_y-o-offset[1][i*KH+o])**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]
#                         else:
#                             gaussian = gaussian+torch.exp(-0.5 * (((grid_x-i)**2)/sigma[0][i*KH+o]**2 + ((grid_y-o)**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]

#                 # if i==-1 and o==-1:
#                 #     if circle==True:
#                 #         gaussian = torch.exp(-0.5 * ((grid_x-i)**2 + (grid_y-o)**2) / sigma[i*KH+o]**2)*b[i*KH+o]
#                 #     else:
#                 #         gaussian = torch.exp(-0.5 * (((grid_x-i)**2)/sigma[0][i*KH+o]**2 + ((grid_y-o)**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]
#                 # else:
#                 #     if circle==True:
#                 #         gaussian = gaussian + torch.exp(-0.5 * ((grid_x-i)**2 + (grid_y-o)**2) / sigma[i*KH+o]**2)*b[i*KH+o]
#                 #     else:
#                 #         gaussian = gaussian+ torch.exp(-0.5 * (((grid_x-i)**2)/sigma[0][i*KH+o]**2 + ((grid_y-o)**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]
#         # gaussian = gaussian/9.0
            
#         # all_x = torch.zeros((KH*KW,KH,KW),dtype=torch.float32).to(weight.device)
#         # all_y = torch.zeros((KH*KW,KH,KW),dtype=torch.float32).to(weight.device)
#         # for i in range(KH):
#         #     # grid_x = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
#         #     # grid_y_pre = np.linspace(-(KH-1-i),i,KH)
#         #     grid_y_pre = torch.linspace(-(KH-1-i),i,KH)
#         #     # print(grid_x)
#         #     for o in range(KW):
#         #         grid_x = torch.linspace(-(KW-1-o),o,KW)
#         #         grid_x, grid_y = torch.meshgrid(grid_x, grid_y_pre,indexing='xy')
#         #         # print(grid_x)
#         #         all_x[i*KH+o]=grid_x
#         #         all_y[i*KH+o]=grid_y
#         # # weight = weight
#         # # gaussian = torch.zeros((KH,KW),requires_grad=True).to(weight.device)
#         # # grid_x = grid_x.to(weight.device)
#         # # grid_y = grid_y.to(weight.device)
#         # b = torch.softmax(weight, dim=0)
#         # for i in range(KH*KW):
#         #     if noise:
#         #         noise_level = level
#         #         grid_x = all_x[i] + noise_level * torch.randn(*grid_x.shape).to(b.device)
#         #         grid_y = all_y[i] + noise_level * torch.randn(*grid_y.shape).to(b.device)
#         #     # print(i)
#         #     # print(type(grid_x))
#         #     # print(grid_y.device)
#         #     # print(weight.device)
#         #         if i==0:
#         #             gaussian = torch.exp(-0.5 * (grid_x**2 + grid_y**2) / sigma[i]**2)*b[i]
#         #         else:
#         #             gaussian=gaussian+torch.exp(-0.5 * (grid_x**2 + grid_y**2) / sigma[i]**2)*b[i]
#         #     else:
#         #         if i==0:
#         #             gaussian = torch.exp(-0.5 * ((all_x[i]**2)/sigma[0][i]**2 + (all_y[i]**2)/sigma[1][i]**2) )*b[i]
#         #         else:
#         #             gaussian=gaussian+torch.exp(-0.5 * (all_x[i]**2 + all_y[i]**2) / sigma[i]**2)*b[i]

#     if normalization:
#         gaussian = gaussian / gaussian.max()
    
#     # Plot the Gaussian mask
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111, projection='3d')
#     # # X, Y, Z = np.meshgrid(grid_x, grid_y, grid_z)
#     # ax.view_init(elev=20., azim=35)
#     # print(gaussian.shape)
#     # print(gaussian)
#     # print(sigma)
#     # ax.plot_surface(grid_x, grid_y, gaussian, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, alpha=0.8)
#     # plt.title('3D Gaussian Mask')
#     # plt.xlabel('X axis')
#     # plt.ylabel('Y axis')
#     # ax.set_zlim(-1,1)  # Limit the Z axis for better visualization
#     # plt.savefig("1.png")
#     # plt.show()
#     return gaussian
#     # print(gaussian.shape)
#     # return torch.tensor(gaussian, dtype=torch.float32,requires_grad=True).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

def create_gaussian_mask(KH,KW, noise = True, level=0.1, sigma=1.0,mask_type=1,normalization=False,weight=[],circle=True,offset=[]):
    # Create a grid of indices
    
    # grid_x = torch.arange(start=(-(KW - 1) / 2.), end=((KW - 1) / 2.))
    grid_x = torch.linspace(-(KW - 1) / 2., (KW - 1) / 2., KW).to(sigma.device)
    # print(grid_x)
    grid_y = torch.linspace(-(KH - 1) / 2., (KH - 1) / 2., KH).to(sigma.device)
    # grid_y = torch.arange(-(KH - 1) / 2., (KH - 1) / 2., KH)
    # print(grid_x)
    grid_x, grid_y = torch.meshgrid(grid_x, grid_y,indexing='xy')
    if noise:
        noise_level = level
        grid_x = grid_x + noise_level * torch.randn(*grid_x.shape).to(sigma.device)
        grid_y = grid_y + noise_level * torch.randn(*grid_y.shape).to(sigma.device)
    
    # if circle:
    #     if mask_type==2:
    #         sigma = sigma.view(sigma.shape[0],sigma.shape[1],1,1)
    #         grid_x=grid_x.expand(sigma.shape[1],KH,KW)
    #         grid_y=grid_y.expand(sigma.shape[1],KH,KW)
    #     else:
    #         sigma = sigma.view(sigma.shape[0],1,1)
        
    #         grid_x=grid_x.expand(sigma.shape[0],KH,KW)
    #         grid_y=grid_y.expand(sigma.shape[0],KH,KW)
    # else:
    #     if mask_type==2:
    #         sigma = sigma.view(2,sigma.shape[1],sigma.shape[2],1,1)
    #         grid_x=grid_x.expand(sigma.shape[2],KH,KW)
    #         grid_y=grid_y.expand(sigma.shape[2],KH,KW)
    #     else:
    #         sigma = sigma.view(2,sigma.shape[1],1,1)
    #         grid_x=grid_x.expand(sigma.shape[1],KH,KW)
    #         grid_y=grid_y.expand(sigma.shape[1],KH,KW)

    if mask_type ==1:
        
        # print(grid_x)
        # 添加随机噪声到网格坐标
        # if noise:
        #     noise_level = level  # 噪声水平，可以根据需要调整
        #     grid_x = grid_x + noise_level * torch.randn(*grid_x.shape).to(sigma.device)
        #     grid_y = grid_y + noise_level * torch.randn(*grid_y.shape).to(sigma.device)
        # print(grid)
    # grid_z = np.zeros_like(grid_x)
    
    # Calculate the Gaussian function
        if circle:
            
            if len(offset)>1:
                # gaussian = torch.exp(-0.5 * ((grid_x-offset[0].reshape(grid_x.shape))**2 + (grid_y-offset[1].reshape(grid_x.shape))**2) / sigma**2)
                gaussian = torch.exp(-0.5 * ((grid_x-offset[0])**2 + (grid_y-offset[1])**2) / sigma**2)
            else:
                gaussian = torch.exp(-0.5 * (grid_x**2 + grid_y**2) / sigma**2)
                # gaussian = torch.exp(-0.5 * (grid_x**2) / sigma**2)
                # gaussian = gaussian.view(-1,1)*gaussian.view(1,-1)
        else:
            
            if len(offset)>1:
                # gaussian = torch.exp(-0.5 * ((grid_x-offset[0].reshape(grid_x.shape))**2/sigma[0]**2 + (grid_y-offset[1].reshape(grid_x.shape))**2/sigma[1]**2))

                gaussian = torch.exp(-0.5 * ((grid_x-offset[0])**2/sigma[0]**2 + (grid_y-offset[1])**2/sigma[1]**2))
            else:
                gaussian = torch.exp(-0.5 * (grid_x**2/sigma[0]**2 + grid_y**2/sigma[1]**2))
        
    # print(gaussian.shape)
    # Normalize the Gaussian mask so that the maximum value is 1
    
    if mask_type == 2:
        b = torch.softmax(weight, dim=0)
        i=torch.tensor([-1,-1,-1,0,0,0,1,1,1]).to(sigma.device)
        o=torch.tensor([-1,0,1,-1,0,1,-1,0,1]).to(sigma.device)
        i = i.view(KH*KW,1,1)
        o = o.view(KH*KW,1,1)
        grid_x=grid_x.expand(KH*KW,-1,-1)
        grid_y=grid_y.expand(KH*KW,-1,-1)
        b=b.view(KH*KW,1,1)
        if circle==True:
            sigma = sigma.view(KH*KW,1,1)

        #     if len(offset)>1:
        #         offset = offset.view(2,KH*KW,1,1)
        #         # gaussian = torch.exp(-0.5 * ((grid_x-i-offset[0][i*KH+o].reshape(grid_x.shape))**2 + (grid_y-o-offset[1][i*KH+o].reshape(grid_x.shape))**2) / sigma[i*KH+o]**2)*b[i*KH+o]
        #         gaussian = torch.exp(-0.5 * ((grid_x-i-offset[0])**2 + (grid_y-o-offset[1])**2) / sigma**2)
        #     else:
        #         # print(grid_x.shape)
        #         # print(sigma[i*KH+o].shape)
        #         gaussian = torch.exp(-0.5 * ((grid_x-i)**2 + (grid_y-o)**2) / sigma**2)
        # else:
        #     sigma = sigma.view(2,KH*KW,1,1)
        #     if len(offset)>1:
        #         offset = offset.view(2,KH*KW,1,1)
        #         # gaussian = torch.exp(-0.5 * (((grid_x-i-offset[0][i*KH+o].reshape(grid_x.shape))**2)/sigma[0][i*KH+o]**2 + ((grid_y-o-offset[1][i*KH+o].reshape(grid_x.shape))**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]
        #         gaussian = torch.exp(-0.5 * (((grid_x-i-offset[0])**2)/sigma[0]**2 + ((grid_y-o-offset[1])**2)/sigma[1]**2))
        #     else:
        #         gaussian = torch.exp(-0.5 * (((grid_x-i)**2)/sigma[0]**2 + ((grid_y-o)**2)/sigma[1]**2))
        # gaussian = gaussian.sum(dim=0)
        # gaussian = gaussian/(KH*KW)
            

            if len(offset)>1:
                offset = offset.view(2,KH*KW,1,1)
                # gaussian = torch.exp(-0.5 * ((grid_x-i-offset[0][i*KH+o].reshape(grid_x.shape))**2 + (grid_y-o-offset[1][i*KH+o].reshape(grid_x.shape))**2) / sigma[i*KH+o]**2)*b[i*KH+o]
                gaussian = torch.exp(-0.5 * ((grid_x-i-offset[0])**2 + (grid_y-o-offset[1])**2) / sigma**2)*b
            else:
                # print(grid_x.shape)
                # print(sigma[i*KH+o].shape)
                gaussian = torch.exp(-0.5 * ((grid_x-i)**2 + (grid_y-o)**2) / sigma**2)*b
        else:
            sigma = sigma.view(2,KH*KW,1,1)
            if len(offset)>1:
                offset = offset.view(2,KH*KW,1,1)
                # gaussian = torch.exp(-0.5 * (((grid_x-i-offset[0][i*KH+o].reshape(grid_x.shape))**2)/sigma[0][i*KH+o]**2 + ((grid_y-o-offset[1][i*KH+o].reshape(grid_x.shape))**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]
                gaussian = torch.exp(-0.5 * (((grid_x-i-offset[0])**2)/sigma[0]**2 + ((grid_y-o-offset[1])**2)/sigma[1]**2))*b
            else:
                gaussian = torch.exp(-0.5 * (((grid_x-i)**2)/sigma[0]**2 + ((grid_y-o)**2)/sigma[1]**2))*b        
        gaussian = gaussian.sum(dim=0)

        # for i,x in enumerate([-1,0,1]):
            # for o,y in enumerate([-1,0,1]):
        # for i in [-1,0,1]:
        #     for o in [-1,0,1]:
        #         if i==-1 and o==-1:
        # #             if circle==True:
        # #                 gaussian = torch.exp(-0.5 * ((grid_x-i)**2 + (grid_y-o)**2) / sigma[i*KH+o]**2)
        # #             else:
        # #                 gaussian = torch.exp(-0.5 * (((grid_x-i)**2)/sigma[0][i*KH+o]**2 + ((grid_y-o)**2)/sigma[1][i*KH+o]**2))
        # #         else:
        # #             if circle==True:
        # #                 gaussian = gaussian + torch.exp(-0.5 * ((grid_x-i)**2 + (grid_y-o)**2) / sigma[i*KH+o]**2)
        # #             else:
        # #                 gaussian = gaussian+ torch.exp(-0.5 * (((grid_x-i)**2)/sigma[0][i*KH+o]**2 + ((grid_y-o)**2)/sigma[1][i*KH+o]**2))
        # # gaussian = gaussian/(KH*KW)
        #             if circle==True:
        #                 if len(offset)>1:
        #                     # gaussian = torch.exp(-0.5 * ((grid_x-i-offset[0][i*KH+o].reshape(grid_x.shape))**2 + (grid_y-o-offset[1][i*KH+o].reshape(grid_x.shape))**2) / sigma[i*KH+o]**2)*b[i*KH+o]
        #                     gaussian = torch.exp(-0.5 * ((grid_x-i-offset[0][i*KH+o])**2 + (grid_y-o-offset[1][i*KH+o])**2) / sigma[i*KH+o]**2)*b[i*KH+o]
        #                 else:
        #                     # print(grid_x.shape)
        #                     # print(sigma[i*KH+o].shape)
        #                     gaussian = torch.exp(-0.5 * ((grid_x-i)**2 + (grid_y-o)**2) / sigma[i*KH+o]**2)*b[i*KH+o]
        #             else:
        #                 if len(offset)>1:
        #                     # gaussian = torch.exp(-0.5 * (((grid_x-i-offset[0][i*KH+o].reshape(grid_x.shape))**2)/sigma[0][i*KH+o]**2 + ((grid_y-o-offset[1][i*KH+o].reshape(grid_x.shape))**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]
        #                     gaussian = torch.exp(-0.5 * (((grid_x-i-offset[0][i*KH+o])**2)/sigma[0][i*KH+o]**2 + ((grid_y-o-offset[1][i*KH+o])**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]
        #                 else:
        #                     gaussian = torch.exp(-0.5 * (((grid_x-i)**2)/sigma[0][i*KH+o]**2 + ((grid_y-o)**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]
        #         else:
        #             if circle==True:
        #                 if len(offset)>1:
        #                     # gaussian = gaussian+torch.exp(-0.5 * ((grid_x-i-offset[0][i*KH+o].reshape(grid_x.shape))**2 + (grid_y-o-offset[1][i*KH+o].reshape(grid_x.shape))**2) / sigma[i*KH+o]**2)*b[i*KH+o]
        #                     gaussian = gaussian+torch.exp(-0.5 * ((grid_x-i-offset[0][i*KH+o])**2 + (grid_y-o-offset[1][i*KH+o])**2) / sigma[i*KH+o]**2)*b[i*KH+o]
        #                 else:
        #                     gaussian = gaussian+torch.exp(-0.5 * ((grid_x-i)**2 + (grid_y-o)**2) / sigma[i*KH+o]**2)*b[i*KH+o]
        #             else:
        #                 if len(offset)>1:
        #                     # gaussian = gaussian+torch.exp(-0.5 * (((grid_x-i-offset[0][i*KH+o].reshape(grid_x.shape))**2)/sigma[0][i*KH+o]**2 + ((grid_y-o-offset[1][i*KH+o].reshape(grid_x.shape))**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]
        #                     gaussian = gaussian+torch.exp(-0.5 * (((grid_x-i-offset[0][i*KH+o])**2)/sigma[0][i*KH+o]**2 + ((grid_y-o-offset[1][i*KH+o])**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]
        #                 else:
        #                     gaussian = gaussian+torch.exp(-0.5 * (((grid_x-i)**2)/sigma[0][i*KH+o]**2 + ((grid_y-o)**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]

                # if i==-1 and o==-1:
                #     if circle==True:
                #         gaussian = torch.exp(-0.5 * ((grid_x-i)**2 + (grid_y-o)**2) / sigma[i*KH+o]**2)*b[i*KH+o]
                #     else:
                #         gaussian = torch.exp(-0.5 * (((grid_x-i)**2)/sigma[0][i*KH+o]**2 + ((grid_y-o)**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]
                # else:
                #     if circle==True:
                #         gaussian = gaussian + torch.exp(-0.5 * ((grid_x-i)**2 + (grid_y-o)**2) / sigma[i*KH+o]**2)*b[i*KH+o]
                #     else:
                #         gaussian = gaussian+ torch.exp(-0.5 * (((grid_x-i)**2)/sigma[0][i*KH+o]**2 + ((grid_y-o)**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]
        # gaussian = gaussian/9.0
            
        # all_x = torch.zeros((KH*KW,KH,KW),dtype=torch.float32).to(weight.device)
        # all_y = torch.zeros((KH*KW,KH,KW),dtype=torch.float32).to(weight.device)
        # for i in range(KH):
        #     # grid_x = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
        #     # grid_y_pre = np.linspace(-(KH-1-i),i,KH)
        #     grid_y_pre = torch.linspace(-(KH-1-i),i,KH)
        #     # print(grid_x)
        #     for o in range(KW):
        #         grid_x = torch.linspace(-(KW-1-o),o,KW)
        #         grid_x, grid_y = torch.meshgrid(grid_x, grid_y_pre,indexing='xy')
        #         # print(grid_x)
        #         all_x[i*KH+o]=grid_x
        #         all_y[i*KH+o]=grid_y
        # # weight = weight
        # # gaussian = torch.zeros((KH,KW),requires_grad=True).to(weight.device)
        # # grid_x = grid_x.to(weight.device)
        # # grid_y = grid_y.to(weight.device)
        # b = torch.softmax(weight, dim=0)
        # for i in range(KH*KW):
        #     if noise:
        #         noise_level = level
        #         grid_x = all_x[i] + noise_level * torch.randn(*grid_x.shape).to(b.device)
        #         grid_y = all_y[i] + noise_level * torch.randn(*grid_y.shape).to(b.device)
        #     # print(i)
        #     # print(type(grid_x))
        #     # print(grid_y.device)
        #     # print(weight.device)
        #         if i==0:
        #             gaussian = torch.exp(-0.5 * (grid_x**2 + grid_y**2) / sigma[i]**2)*b[i]
        #         else:
        #             gaussian=gaussian+torch.exp(-0.5 * (grid_x**2 + grid_y**2) / sigma[i]**2)*b[i]
        #     else:
        #         if i==0:
        #             gaussian = torch.exp(-0.5 * ((all_x[i]**2)/sigma[0][i]**2 + (all_y[i]**2)/sigma[1][i]**2) )*b[i]
        #         else:
        #             gaussian=gaussian+torch.exp(-0.5 * (all_x[i]**2 + all_y[i]**2) / sigma[i]**2)*b[i]

    if normalization:
        gaussian = gaussian / gaussian.max()
    
    # Plot the Gaussian mask
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # X, Y, Z = np.meshgrid(grid_x, grid_y, grid_z)
    # ax.view_init(elev=20., azim=35)
    # print(gaussian.shape)
    # print(gaussian)
    # print(sigma)
    # ax.plot_surface(grid_x, grid_y, gaussian, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, alpha=0.8)
    # plt.title('3D Gaussian Mask')
    # plt.xlabel('X axis')
    # plt.ylabel('Y axis')
    # ax.set_zlim(-1,1)  # Limit the Z axis for better visualization
    # plt.savefig("1.png")
    # plt.show()
    return gaussian


class RepVGGBlock(nn.Module):

    def __init__(self, conv_type=1, sigma = 1.0, noise=False, normalization=True, offset= False, circle=True,*args, mask=False, **kwargs):
        super(RepVGGBlock, self).__init__()
        self.conv = nn.Conv2d(*args,**kwargs)
        self.conv_1 = nn.Conv2d(*args,**kwargs)
        # self.conv_2 = nn.Conv2d(*args,**kwargs)

        self.bn = nn.BatchNorm2d(self.conv.out_channels)
        self.bn_1 = nn.BatchNorm2d(self.conv_1.out_channels)
        # self.bn_2 = nn.BatchNorm2d(self.conv_2.out_channels)

        self.circle = circle
        self.mask = mask
        # print(mask)
        if self.mask:
            assert conv_type in (1,2)
            self.conv_type = conv_type
            self.noise = noise 
            self.normalization = normalization



            if circle:
                self.mask_sigma = nn.Parameter(torch.ones((self.conv.out_channels),dtype=torch.float32)*sigma)
                self.mask_sigma_1 = nn.Parameter(torch.ones((self.conv.out_channels),dtype=torch.float32)*sigma)

                # self.mask_sigma = nn.Parameter(torch.tensor(sigma,dtype=torch.float32),requires_grad=True)
                # self.mask_sigma_1 = nn.Parameter(torch.tensor(sigma,dtype=torch.float32),requires_grad=True)
                # self.mask_sigma_2 = nn.Parameter(torch.tensor(sigma,dtype=torch.float32),requires_grad=True)
            else:
                self.mask_sigma = nn.Parameter(torch.ones((2,self.conv.out_channels),dtype=torch.float32)*sigma)
                self.mask_sigma_1 = nn.Parameter(torch.ones((2,self.conv.out_channels),dtype=torch.float32)*sigma)

                # self.mask_sigma = nn.Parameter(torch.ones((2),dtype=torch.float32)*sigma,requires_grad=True)
                # self.mask_sigma_1 = nn.Parameter(torch.ones((2),dtype=torch.float32)*sigma,requires_grad=True)
                # self.mask_sigma_2 = nn.Parameter(torch.ones((2),dtype=torch.float32)*sigma,requires_grad=True)
                
            
            _, _, KH, KW = self.conv.weight.data.size()
            if offset:
                self.offset = nn.Parameter(torch.tensor([0.0,0.0],dtype=torch.float32),requires_grad=True)
                self.offset_1 = nn.Parameter(torch.tensor([0.0,0.0],dtype=torch.float32),requires_grad=True)
                # self.offset_2 = nn.Parameter(torch.tensor([0.0,0.0],dtype=torch.float32),requires_grad=True)
            else:
                self.offset = []
                self.offset_1 = []
                # self.offset_2 = []

            self.mask_weight=[]
            self.mask_weight_1=[]
            if conv_type==2:
                if KH==KW:
                    self.mask_weight = nn.Parameter(torch.tensor([0.5/(KH*KW) for i in range((KH*KW))],dtype=torch.float32),requires_grad=False)
                    self.mask_weight[int((KH*KW)/2)] +=torch.tensor(0.5) 
                    self.mask_weight_1 = nn.Parameter(torch.tensor([0.5/(KH*KW) for i in range((KH*KW))],dtype=torch.float32),requires_grad=False)
                    self.mask_weight_1[int((KH*KW)/2)] +=torch.tensor(0.5) 
                    # self.mask_weight_2 = nn.Parameter(torch.tensor([0.5/(KH*KW) for i in range((KH*KW))],dtype=torch.float32),requires_grad=False)
                    # self.mask_weight_2[int((KH*KW)/2)] +=torch.tensor(0.5) 
                else:
                    self.mask_weight = nn.Parameter(torch.tensor([1.0/(KH*KW) for i in range((KH*KW))],dtype=torch.float32),requires_grad=True)
                    self.mask_weight_1 = nn.Parameter(torch.tensor([1.0/(KH*KW) for i in range((KH*KW))],dtype=torch.float32),requires_grad=True)
                    # self.mask_weight_2 = nn.Parameter(torch.tensor([1.0/(KH*KW) for i in range((KH*KW))],dtype=torch.float32),requires_grad=True)
                self.mask_weight.requires_grad=True
                self.mask_weight_1.requires_grad=True
                # self.mask_weight_2.requires_grad=True



                if circle==True:
                    self.mask_sigma = nn.Parameter(torch.ones((KH*KW,self.conv.out_channels),dtype=torch.float32)*sigma,requires_grad=True)
                    self.mask_sigma_1 = nn.Parameter(torch.ones((KH*KW,self.conv.out_channels),dtype=torch.float32)*sigma,requires_grad=True)

                    # self.mask_sigma = nn.Parameter(torch.ones((KH*KW),dtype=torch.float32)*sigma,requires_grad=True)
                    # self.mask_sigma_1 = nn.Parameter(torch.ones((KH*KW),dtype=torch.float32)*sigma,requires_grad=True)
                    # self.mask_sigma_2 = nn.Parameter(torch.ones((KH*KW),dtype=torch.float32)*sigma,requires_grad=True)
                else:
                    self.mask_sigma = nn.Parameter(torch.ones((2,KH*KW,self.conv.out_channels),dtype=torch.float32)*sigma,requires_grad=True)
                    self.mask_sigma_1 = nn.Parameter(torch.ones((2,KH*KW,self.conv.out_channels),dtype=torch.float32)*sigma,requires_grad=True)
                    
                    # self.mask_sigma = nn.Parameter(torch.ones((2,KH*KW),dtype=torch.float32)*sigma,requires_grad=True)
                    # self.mask_sigma_1 = nn.Parameter(torch.ones((2,KH*KW),dtype=torch.float32)*sigma,requires_grad=True)
                    # self.mask_sigma_2 = nn.Parameter(torch.ones((2,KH*KW),dtype=torch.float32)*sigma,requires_grad=True)

                if offset:
                    self.offset = nn.Parameter(torch.ones((2,KH*KW),dtype=torch.float32)*sigma,requires_grad=True)
                    self.offset_1 = nn.Parameter(torch.ones((2,KH*KW),dtype=torch.float32)*sigma,requires_grad=True)
                    # self.offset_2 = nn.Parameter(torch.ones((2,KH*KW),dtype=torch.float32)*sigma,requires_grad=True)
                else:
                    self.offset = []
                    self.offset_1 = []
                    # self.offset_2 = []
            print(self.mask_weight)
            print(self.mask_sigma)
    def forward(self,x):
        # print(self.conv.weight.data[0][0])
        if self.mask:
            out_ch, _, kH, kW = self.conv.weight.data.size()
            self.mask_sigma.data = torch.clamp(self.mask_sigma,min=1)
            self.mask_sigma_1.data = torch.clamp(self.mask_sigma_1,min=1)
            # self.mask_sigma_2.data = torch.clamp(self.mask_sigma_2,min=1)
            mask = create_gaussian_mask(kH,kW, level=0.1, sigma=self.mask_sigma,mask_type=self.conv_type,weight= self.mask_weight,noise=self.noise,normalization=self.normalization,offset=self.offset,circle=self.circle).reshape(out_ch,1, kH, kW)
            mask_1 = create_gaussian_mask(kH,kW, level=0.1, sigma=self.mask_sigma_1,mask_type=self.conv_type,weight= self.mask_weight_1,noise=self.noise,normalization=self.normalization,offset=self.offset_1,circle=self.circle).reshape(out_ch,1, kH, kW)
            # mask = create_gaussian_mask(kH,kW, level=0.1, sigma=self.mask_sigma,mask_type=self.conv_type,weight= self.mask_weight,noise=self.noise,normalization=self.normalization,offset=self.offset,circle=self.circle).reshape(1,1, kH, kW)
            # mask_1 = create_gaussian_mask(kH,kW, level=0.1, sigma=self.mask_sigma_1,mask_type=self.conv_type,weight= self.mask_weight_1,noise=self.noise,normalization=self.normalization,offset=self.offset_1,circle=self.circle).reshape(1,1, kH, kW)
            # mask_2 = create_gaussian_mask(kH,kW, level=0.1, sigma=self.mask_sigma_2,mask_type=self.conv_type,weight= self.mask_weight_2,noise=self.noise,normalization=self.normalization,offset=self.offset_2,circle=self.circle).reshape(1,1, kH, kW)
            masked_weights = self.conv.weight * mask
            masked_weights_1 = self.conv_1.weight * mask_1
            # masked_weights_2 = self.conv_2.weight * mask_2
            x1 = nn.functional.conv2d(x, masked_weights, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
            x2 = nn.functional.conv2d(x, masked_weights_1, self.conv_1.bias, self.conv_1.stride, self.conv_1.padding, self.conv_1.dilation, self.conv_1.groups)
            # x3 = nn.functional.conv2d(x, masked_weights_2, self.conv_2.bias, self.conv_2.stride, self.conv_2.padding, self.conv_2.dilation, self.conv_2.groups)
        #   
        # masked_weights = self.conv.weight * self.mask
        # print(self.conv.weight.data[0][0])
        # print(x.shape)
        else:
            x1 = self.conv(x)
            x2 = self.conv_1(x)  
            # x3 = self.conv_2(x)
        # output = (self.bn(x1)+self.bn_1(x2))/2.0
        output = self.bn(x1)+self.bn_1(x2)
        # output = self.bn(x1)+self.bn_1(x2)+self.bn_2(x3)
        # output = (self.bn(x1)+self.bn_1(x2))/2
        # output = self.relu((self.bn(self.conv(x))+self.bn_1(self.conv_1(x))))
        # output = self.conv(x)
        
        return output
    
class MaskConv2d(nn.Module):
    def __init__(self, conv_type=1, sigma = 1.0, noise=True, normalization=True, offset=True, circle=True,*args, **kwargs):
        super().__init__()
        
        assert conv_type in (1,2)
        self.conv_type = conv_type
        self.noise = noise 
        self.normalization = normalization
        self.conv = nn.Conv2d(*args,**kwargs)
        _, _, KH, KW = self.conv.weight.data.size()
        self.circle = circle
        if offset:
            self.offset = nn.Parameter(torch.tensor([0.0,0.0],dtype=torch.float32),requires_grad=True)
            # self.offset = nn.Parameter(torch.zeros((2,KH*KW),dtype=torch.float32),requires_grad=True)
        else:
            self.offset = []
        # out_channels, in_channels, kH, kW = self.conv.weight.data.size()
        if circle:
            # self.mask_sigma = nn.Parameter(torch.ones((self.conv.out_channels),dtype=torch.float32)*sigma)
            self.mask_sigma = nn.Parameter(torch.tensor(sigma,dtype=torch.float32),requires_grad=True)
        else:
            # self.mask_sigma = nn.Parameter(torch.ones((2,self.conv.out_channels),dtype=torch.float32)*sigma)
            self.mask_sigma = nn.Parameter(torch.tensor([1.0,1.0],dtype=torch.float32)*sigma,requires_grad=True)
        # self.mask = nn.Parameter(torch.ones((KH,KW),dtype=torch.float32).reshape(1,1,KH,KW),requires_grad=True)
        

        
        

        # 自由的mask
        # self.mask = nn.Parameter((torch.ones((KH,KW),dtype=torch.float32).reshape(1,1,KH,KW)),requires_grad=True)
        # self.mask = nn.Parameter((torch.randn((KH,KW),dtype=torch.float32).reshape(1,1,KH,KW)),requires_grad=True)
        self.mask_weight=[]
        if conv_type==2:
            if KH==KW:
                self.mask_weight = nn.Parameter(torch.tensor([0.5/(KH*KW) for i in range((KH*KW))],dtype=torch.float32),requires_grad=False)
                self.mask_weight[int((KH*KW)/2)] +=torch.tensor(0.5) 
            else:
                self.mask_weight = nn.Parameter(torch.tensor([1.0/(KH*KW) for i in range((KH*KW))],dtype=torch.float32),requires_grad=True)
            self.mask_weight.requires_grad=True
            if circle==True:
                self.mask_sigma = nn.Parameter(torch.ones((KH*KW),dtype=torch.float32)*sigma,requires_grad=True)
                # self.mask_sigma = nn.Parameter(torch.ones((KH*KW,self.conv.out_channels),dtype=torch.float32)*sigma,requires_grad=True)
            else:
                # self.mask_sigma = nn.Parameter(torch.ones((2,KH*KW,self.conv.out_channels),dtype=torch.float32)*sigma,requires_grad=True)
                self.mask_sigma = nn.Parameter(torch.ones((2,KH*KW),dtype=torch.float32)*sigma,requires_grad=True)
            if offset:
                self.offset = nn.Parameter(torch.zeros((2,KH*KW),dtype=torch.float32),requires_grad=True)
                # self.offset = nn.Parameter(torch.zeros((2,KH*KW,KH*KW),dtype=torch.float32),requires_grad=True)
            else:
                self.offset = []
        # print(self.mask_weight)
        # print(self.mask_sigma)
        # self.mask = create_gaussian_mask(kH,kW, level=0.1, sigma=sigma,mask_type=conv_type).reshape(1,1, kH, kW)
        # print(mask)
        # self.register_buffer('mask',mask,False)

    def forward(self,x):
        # print(self.conv.weight.data[0][0])
        output_ch, _, kH, kW = self.conv.weight.data.size()
        if type(kH) is not int:
            kH=kH.to(self.mask_sigma.device)
            kW=kW.to(self.mask_sigma.device)
        self.mask_sigma.data = torch.clamp(self.mask_sigma,min=1)
        # if len(self.offset)>1:
        #     self.offset.requires_grad=False
        #     self.offset[0] = torch.clamp(self.offset[0],min=-0.5,max=0.5)
        #     self.offset[1] = torch.clamp(self.offset[1],min=-0.5,max=0.5)
        #     self.offset.requires_grad=True
        mask = create_gaussian_mask(kH,kW, level=0.1, sigma=self.mask_sigma,mask_type=self.conv_type,weight= self.mask_weight,noise=self.noise,normalization=self.normalization,offset=self.offset,circle=self.circle).reshape(1,1, kH, kW)
        masked_weights = self.conv.weight * mask
        # # 
        # masked_weights = self.conv.weight * self.mask
        # print(self.conv.weight.data[0][0])
        output = nn.functional.conv2d(x, masked_weights, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        # output = self.conv(x)
        return output
    # def post_op_mask(self,optimizer):
        

class VGG(nn.Module):

    def __init__(self, cfg, batch_norm=False,rate=1, mask=False, num_classes=1000):
        super(VGG, self).__init__()
        # print(mask)
        self.block0 = self._make_layers(cfg[0], batch_norm, 3,mask,mask_number=2,rate=rate)
        self.block1 = self._make_layers(cfg[1], batch_norm, int(cfg[0][-1]*rate),mask,mask_number=2,rate=rate)
        self.block2 = self._make_layers(cfg[2], batch_norm, int(cfg[1][-1]*rate),mask,mask_number=2,rate=rate)
        self.block3 = self._make_layers(cfg[3], batch_norm, int(cfg[2][-1]*rate),mask,mask_number=2,rate=rate)
        self.block4 = self._make_layers(cfg[4], batch_norm, int(cfg[3][-1]*rate),mask,mask_number=2,rate=rate)
        # self.block3 = self._make_layers(cfg[3], batch_norm, int(cfg[2][-1]*rate),rate=rate)
        # self.block4 = self._make_layers(cfg[4], batch_norm, int(cfg[3][-1]*rate),rate=rate)

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Linear(int(512*rate), num_classes)
        self._initialize_weights()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.block0)
        feat_m.append(self.pool0)
        feat_m.append(self.block1)
        feat_m.append(self.pool1)
        feat_m.append(self.block2)
        feat_m.append(self.pool2)
        feat_m.append(self.block3)
        feat_m.append(self.pool3)
        feat_m.append(self.block4)
        feat_m.append(self.pool4)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block1[-1]
        bn2 = self.block2[-1]
        bn3 = self.block3[-1]
        bn4 = self.block4[-1]
        return [bn1, bn2, bn3, bn4]

    def forward(self, x, is_feat=False, preact=False,is_mask=False):
        h = x.shape[2]
        # print(h)
        x = F.relu(self.block0(x))
        f0 = x
        if h >= 128:
            x = self.pool0(x)
        x = self.block1(x)
        f1_pre = x
        x = F.relu(x)
        f1 = x
        if h >= 64:
            x = self.pool1(x)
        x = self.block2(x)
        f2_pre = x
        x = F.relu(x)
        f2 = x
        x = self.pool2(x)
        x = self.block3(x)
        f3_pre = x
        x = F.relu(x)
        f3 = x
        
        
        x = self.pool3(x)
        x = self.block4(x)
        f4_pre = x
        x = F.relu(x)
        f4 = x
        
        x = self.pool4(x)
        f5 = x
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        
        x = self.classifier(x)

        if is_feat:
            if preact:
                return [f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], x
            else:
                return [f0, f1, f2, f3, f4, f5], x
        if is_mask:
            sigma_list = []
            for i in range(len(self.blocks)):
                block_i = self.blocks[i]
                # 这里根据实际情况写:
                #   如果 block_i 本身就是 GMConvLayer，则直接 block_i.sigma
                #   如果 block_i 内部还包含 sub_layer.sigma，需进一步获取
                if hasattr(block_i, 'sigma'):
                    sigma_list.append(block_i.sigma)
        else:
            sigma_list = None
        # else:
        return x

    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3,mask=False,mask_number=1,rate=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # print(rate)
                v = int(v*rate)
                if mask:
                    # pass
                    # print('**************mask*******************')
                    if mask_number==1:
                        pass
                    # conv2d = MaskConv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1,conv_type=2,noise=False,normalization=False,sigma=2.0)
                        conv2d = MaskConv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1,conv_type=1,noise=False,normalization=True,offset=False,circle=True,sigma=5.0)
                        # conv2d = RepVGGBlock(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1,conv_type=1,noise=False,normalization=False,mask=mask,offset=True,circle=False,sigma=2.0)
                    else:
                        pass
                        conv2d = MaskConv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1,conv_type=2,noise=False,normalization=True,offset=False,circle=True,sigma=5.0)
                        # conv2d = RepVGGBlock(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1,conv_type=2,noise=False,normalization=False,mask=mask,offset=True,circle=False,sigma=2.0)
                else:
                    pass
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    # conv2d = MaskConv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1,conv_type=2,noise=False,normalization=True,offset=False,circle=True,sigma=5.0)
                    # conv2d = RepVGGBlock(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1)
                # conv2d = RepVGGBlock(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1)
                # layers += [conv2d, nn.ReLU(inplace=True)]
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                # print(v)
                in_channels = v
        layers = layers[:-1]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


cfg = {
    'A': [[64], [128], [256, 256], [512, 512], [512, 512]],
    'B': [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]],
    'D': [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    'E': [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]],
    'S': [[64], [128], [256], [512], [512]],
}


def vgg8(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['S'], **kwargs)
    return model


def vgg8_bn(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['S'], batch_norm=True, mask=True,rate=0.45,**kwargs)
    return model


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['A'], **kwargs)
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(cfg['A'], batch_norm=True, mask=True,**kwargs)
    return model


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['B'], **kwargs)
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(cfg['B'], batch_norm=True, **kwargs)
    return model


def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['D'], **kwargs)
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(cfg['D'], batch_norm=True, mask=False,**kwargs)
    return model


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['E'], **kwargs)
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(cfg['E'], batch_norm=True, mask=True,**kwargs)
    return model

from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count_table
if __name__ == '__main__':
    import torch

    x = torch.randn(2, 3, 224,224)
    net = vgg16_bn(num_classes=10)

    feats, logit = net(x, is_feat=True, preact=True)
    
    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)
    flops, params = profile(net, (x,))
    print('flops: ', flops, 'params: ', params)
    for m in net.get_bn_before_relu():
        if isinstance(m, nn.BatchNorm2d):
            print('pass')
        else:
            print('warning')
