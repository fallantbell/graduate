import torch
import torch.nn as nn
import torch.nn.functional as F


from activation import trunc_exp, biased_softplus
from .renderer import NeRFRenderer

from .mlp import VanillaNeRFRadianceField

import numpy as np
from encoding import get_encoder

from .utils import safe_normalize

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x

'''
    Nerfnetwork 主要定義了整體nerf 的主幹網路
    包括 hash encoder 和 MLP 網路
    在給定一個三為座標點 (x,y,z) 的情況下，經過網路可以得到相對應的 rgb 與 sigma

    #! Nerfnetwork 繼承了 NerfRender Class

''' 
class NeRFNetwork(NeRFRenderer):
    def __init__(self, 
                 opt,
                 num_layers=3,
                 hidden_dim=64,
                 num_layers_bg=2,
                 hidden_dim_bg=32,
                 ):

        #* 初始化 NerfRender    
        super().__init__(opt)

        #* 定義 MLP 網路層數與hidden dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        #* 定義 hash encoder
        #! self.encoder 是一個 nn.module 類別，所以也是要訓練的
        level = 2
        self.encoder, self.in_dim = get_encoder('hashgrid', input_dim=3,level_dim=level,log2_hashmap_size=19, desired_resolution=2048 * self.bound, interpolation='smoothstep')

        #* 定義MLP網路
        #* 輸入為encoder 出來的embedding, dim 預設為 32=16*2
        #* 輸出應該就是 rgb + sigma
        self.sigma_net = MLP(self.in_dim, 4, hidden_dim, num_layers, bias=True)

        self.density_activation = trunc_exp if self.opt.density_activation == 'exp' else biased_softplus

        #* background network
        if self.opt.bg_radius > 0:
            self.num_layers_bg = num_layers_bg   
            self.hidden_dim_bg = hidden_dim_bg
            
            #* use a very simple network to avoid it learning the prompt...
            #* 定義背景部分的encoder, 輸入是 xyz
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3, multires=6)
            #* 定義背景部分的MLP 網路
            #* 輸入為 xyz 的positional encoding, dim: 39 = 3(xyz)* 6 * 2(sin,cos) + 3(xyz) 
            #* 輸出是 rgb
            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
            
        else:
            self.bg_net = None

        # self.radiance_field = VanillaNeRFRadianceField()

    def common_forward(self, x):

        # sigma
        enc = self.encoder(x, bound=self.bound, max_level=self.max_level)

        h = self.sigma_net(enc)

        sigma = self.density_activation(h[..., 0] + self.density_blob(x))
        albedo = torch.sigmoid(h[..., 1:])

        return sigma, albedo
    
    # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        dx_pos, _ = self.common_forward((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dx_neg, _ = self.common_forward((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_pos, _ = self.common_forward((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_neg, _ = self.common_forward((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dz_pos, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        dz_neg, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        
        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon, 
            0.5 * (dy_pos - dy_neg) / epsilon, 
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return -normal

    def normal(self, x):
        normal = self.finite_difference_normal(x)
        normal = safe_normalize(normal)
        normal = torch.nan_to_num(normal)
        return normal
    
    def forward(self, x, d, l=None, ratio=1, shading='albedo'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)

        # rgbs, sigmas = self.radiance_field(x, d) #* rgbs (N,3) sigmas(N,1)
        # normal = self.normal(x)
        # return sigmas.squeeze(-1),rgbs,normal

        #* 給定每個採樣點 x 
        #* 輸出sigma: 密度 shape(N)
        #* albedo: 可以簡單把它當作顏色 shape(N,3)
        sigma, albedo = self.common_forward(x)

        if shading == 'albedo':
            normal = None
            color = albedo
        
        else: # lambertian shading

            #* 利用神奇的function 得到x的法向量
            normal = self.normal(x)

            lambertian = ratio + (1 - ratio) * (normal * l).sum(-1).clamp(min=0) # [N,]

            if shading == 'textureless': #* 訓練中後期隨機不加顏色來維持形狀
                color = lambertian.unsqueeze(-1).repeat(1, 3)
            elif shading == 'normal':   #* 訓練初期主要訓練形狀 (0,1)
                color = (normal + 1) / 2
            else: #* 'lambertian' 正常的顏色
                color = albedo * lambertian.unsqueeze(-1)
        
        #* 返回點的 密度, 顏色, 法向量
        return sigma, color, normal

      
    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        
        sigma, albedo = self.common_forward(x)
        
        return {
            'sigma': sigma,
            'albedo': albedo,
        }


    def background(self, d):

        h = self.encoder_bg(d) # [N, C]
        
        h = self.bg_net(h)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr * 10},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            # {'params': self.normal_net.parameters(), 'lr': lr},
        ]        

        if self.opt.bg_radius > 0:
            # params.append({'params': self.encoder_bg.parameters(), 'lr': lr * 10})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        if self.opt.dmtet:
            params.append({'params': self.sdf, 'lr': lr})
            params.append({'params': self.deform, 'lr': lr})

        return params