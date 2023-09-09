import os
import cv2
import glob
import json
import tqdm
import random
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

# import trimesh

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from PIL import Image

from terrain import Terrain
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras
)

from .utils import get_rays, safe_normalize

DIR_COLORS = np.array([
    [255, 0, 0, 255], # front
    [0, 255, 0, 255], # side
    [0, 0, 255, 255], # back
    [255, 255, 0, 255], # side
    [255, 0, 255, 255], # overhead
    [0, 255, 255, 255], # bottom
], dtype=np.uint8)

'''
def visualize_poses(poses, dirs, size=0.1):
    # poses: [B, 4, 4], dirs: [B]
    poses = poses.numpy()

    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes, sphere]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)

        # different color for different dirs
        # segs.colors = np.array([255, 0, 0, 255]).repeat(len(segs.entities), 0)

        objects.append(segs)

    trimesh.Scene(objects).show()
'''

def get_view_direction(thetas, phis, overhead, front):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (right) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (left) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    phis = phis % (2 * np.pi)
    res[(phis < front / 2) | (phis >= 2 * np.pi - front / 2)] = 0
    res[(phis >= front / 2) & (phis < np.pi - front / 2)] = 1
    res[(phis >= np.pi - front / 2) & (phis < np.pi + front / 2)] = 2
    res[(phis >= np.pi + front / 2) & (phis < 2 * np.pi - front / 2)] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res


def rand_poses(size, device, opt, radius_range=[1, 1.5], theta_range=[0, 120], phi_range=[0, 360], return_dirs=False, angle_overhead=30, angle_front=60, uniform_sphere_rate=0.5,target=torch.zeros(3)):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    #* 將角度(degree) 轉換成 弧度(radian)
    theta_range = np.array(theta_range) / 180 * np.pi
    phi_range = np.array(phi_range) / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    #* 隨機抽取相機與原點距離
    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    #* 隨機抽取相機角度
    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]
    phis[phis < 0] += 2 * np.pi

    #* 相機位置 xyz
    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    targets = target.to(device)

    # lookat
    #* 得到相機的xyz向量 (世界座標)
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    # right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    right_vector = safe_normalize(torch.cross(up_vector, forward_vector, dim=-1))

    up_noise = 0
    # up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)
    up_vector = safe_normalize(torch.cross(forward_vector, right_vector, dim=-1) + up_noise)

    #* 得到相機的旋轉矩陣
    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    #* 根據角度(theta phi)得出相機看向的是物體的哪個方位(前、後、兩側、上面)
    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None

    #* back to degree
    thetas = thetas / np.pi * 180
    phis = phis / np.pi * 180

    return poses, dirs, thetas, phis, radius,centers


def circle_poses(device, radius=torch.tensor([3.2]), theta=torch.tensor([60]), phi=torch.tensor([0]), return_dirs=False, angle_overhead=30, angle_front=60):

    #* 將角度(degree) 轉換成 弧度(radian)
    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    #* 代表 x,y,z 的點
    #* 給定環繞圓心角 phi(0,360) , 跟垂直圓心的角度 theta(0,120) , 還有圓的半徑
    #* 就能求出相對應的 xyz 座標
    centers = torch.stack([
        radius * torch.sin(theta) * torch.sin(phi),     #* x 左右
        radius * torch.cos(theta),                      #* y 上下
        radius * torch.sin(theta) * torch.cos(phi),     #* z 前後
    ], dim=-1) # [B, 3]

    # lookat
    #* safe_normalize 為歸一化操作
    forward_vector = safe_normalize(centers) #* 世界座標原點(0,0,0) 指向各個相機原點 (xyz) 的向量, 也代表相機座標的z 向量
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(len(centers), 1) #* 朝上的向量
    # right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    right_vector = safe_normalize(torch.cross(up_vector, forward_vector, dim=-1)) #* 做外積求相機右邊向量, 代表相機座標x向量
    # up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(forward_vector, right_vector, dim=-1)) #* 再做一次外積得到與 forward right 垂直的 up 向量, 代表相機座標y向量

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(len(centers), 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1) #* 得到相機的旋轉矩陣
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(theta, phi, angle_overhead, angle_front) #* 根據角度(theta phi)得出相機看向的是物體的哪個方位(前、後、兩側、上面)
    else:
        dirs = None

    return poses, dirs

#! 定義

class NeRFDataset:
    def __init__(self, opt, device, type='train', H=256, W=256, size=100):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type # train, val, test

        self.H = H #* 預設 64 
        self.W = W
        self.size = size #* 所有data的數量，也就是一個epoch要執行多少個iteration

        self.training = self.type in ['train', 'all']

        self.cx = self.H / 2 #* 32 相機成像中心點
        self.cy = self.W / 2

        self.near = self.opt.min_near #* 最近從距離相機多遠開始採樣
        self.far = 1000 #* infinite 最遠採樣到多遠

        if self.training:
            self.fov = opt.default_fovy
        else:
            self.fov = 60

        #!
        self.terrain = Terrain(opt)

        # [debug] visualize poses
        # poses, dirs, _, _, _ = rand_poses(100, self.device, opt, radius_range=self.opt.radius_range, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front, jitter=self.opt.jitter_pose, uniform_sphere_rate=1)
        # visualize_poses(poses.detach().cpu().numpy(), dirs.detach().cpu().numpy())

    def get_default_view_data(self):

        H = int(self.opt.known_view_scale * self.H) #* 1.5*64 = 96 不太確定為什麼要*1.5
        W = int(self.opt.known_view_scale * self.W)
        cx = H / 2 #* 48
        cy = W / 2

        radii = torch.FloatTensor(self.opt.ref_radii).to(self.device) #* tensor shape(0) , 半徑
        thetas = torch.FloatTensor(self.opt.ref_polars).to(self.device) #* tensor shape(0) , 定義其中一個方向的角度
        phis = torch.FloatTensor(self.opt.ref_azimuths).to(self.device) #* tensor shape(0) , 定義另一個方向的角度

        #* angle front 預設60 :定義了前、後、兩側的角度範圍, 0~angle 為物體的前方 , 180~180+angle 為物體後方, 剩餘為物體兩側 
        #* angle_overhead 預設30 :定義上方的角度
        #* 產生半圓形分布的相機 看向原點 (0,0,0)
        #* pose: 相機旋轉矩陣(cam2world) dirs: 代表相機看向物體哪個方位
        poses, dirs = circle_poses(self.device, radius=radii, theta=thetas, phi=phis, return_dirs=True, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front)
        
        #* fov預設20
        fov = self.opt.default_fovy
        focal = H / (2 * np.tan(np.deg2rad(fov) / 2)) #* 根據fov 計算相機焦距 focal
        intrinsics = np.array([focal, focal, cx, cy])

        #* mvp 轉換中的projection 矩陣
        #* 將 x,y,z 都映射到 [-1,1]
        #? 具體怎麼做不確定
        projection = torch.tensor([
            [2*focal/W, 0, 0, 0],
            [0, -2*focal/H, 0, 0],
            [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
            [0, 0, -1, 0]
        ], dtype=torch.float32, device=self.device).unsqueeze(0).repeat(len(radii), 1, 1)

        #* MVP變換,用來將世界座標的點轉換到相機平面上
        mvp = projection @ torch.inverse(poses) # [B, 4, 4]

        # sample a low-resolution but full image
        #* 輸入相機的座標、參數
        #* 返回 
        #* rays[rays_o] 相機世界座標的位置
        #* rays[rays_d] 這個相機原點射向每個pixel的向量 (世界座標)
        #* 兩個shape 都為 (B,N,3), batch=1 , N=H*W 每個pixel , 3 向量
        #* 一個batch 應該就代表一個相機
        rays = get_rays(poses, intrinsics, H, W, -1)

        #* 返回相機看向物體各種資訊
        data = {
            'H': H,
            'W': W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'dir': dirs,
            'mvp': mvp,
            'polar': self.opt.ref_polars,
            'azimuth': self.opt.ref_azimuths,
            'radius': self.opt.ref_radii,
        }

        return data
    
    def save_depth(self,depth):
        depth_np = depth.cpu().numpy()
        plt.imshow(depth_np, cmap='viridis', vmin=0, vmax=depth_np.max())
        plt.colorbar()
        plt.savefig(f"{self.opt.workspace}/depth_image_now.png")
        plt.close()
        # non_zero_mask = (depth_np != 0)
        # depth_np[non_zero_mask] = 1.0 / depth_np[non_zero_mask]
        # depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-6)
        # depth_np = (depth_np * 255).astype(np.uint8)
        # depth_image = Image.fromarray(depth_np, mode='L')
        # depth_image.save(f"{self.opt.workspace}/disparity_image_now.png")
    
    def get_depth(self,theta,phis,radius,target=torch.zeros(3)):

        '''
            這個function 是根據camera render 出height map 的深度圖
            output: depth map, numpy array ,沒有normalize
        '''

        #* 原本的top 角度是0 平視90 bottom 180
        #* pytorch3d top 90 平視0 bottom -90
        #* 所以theta 要做相對應調整
        tilt = 90-theta

        #* 0~360 => -180~180 一樣是對齊pytorch3d
        if phis>180:
            phis = -(360-phis)

        #* pytorch3D 好像座標軸不一樣，所以R T 要重算
        R2, T2 = look_at_view_transform(dist=radius, elev=tilt, azim=phis, at=((target[0],target[1],target[2]),))
        camera = FoVPerspectiveCameras(R=R2, T=T2, fov=self.fov, device='cuda')

        # 進行渲染
        image, depth, mask, _, _ = self.terrain.render_from_camera(camera, self.H, self.W)

        #* 儲存深度圖
        # self.save_depth(depth)

        depth_np = depth.cpu().numpy()
        image_np = image[:3,...].cpu().numpy()

        return depth_np,image_np

    def collate(self, index):

        B = len(index) #* 1

        if self.training:
            #* random pose on the fly
            #* 產生隨機的相機位置
            '''
                input:
                radius_range: 相機距離原點的距離 default [3,3.5]
                theta_range: 與圓心垂直的角度 default [45,105] , 0是top 180 是bottom
                phi_range: 圓心角 default [-180,180]
                angle_overhead: 定義超過多少角度為上方 default 30
                angle_front: 定義多少角度為物體前後方 default 60

                output:
                poses: 相機的旋轉矩陣 shape: (1,4,4)
                dirs: 相機看相物體的方位
                theta,phis: 相機的方位角
                radius: 相機距離原點的距離
            '''

            #! 固定100 個相機
            if self.opt.fixed_camera:
                r = 2 + (index[0]/self.size)
                self.opt.radius_range = [r,r]

                t = 45 + (((index[0]+45)%self.size)/self.size)*45
                p = (((index[0]+self.size/2)%self.size)/self.size)*360

                self.opt.theta_range = [t,t]
                self.opt.phi_range = [p,p]

            #! 固定 1 個相機
            if self.opt.one_camera:
                r = self.opt.default_radius
                t = self.opt.default_polar - 30
                p = 0
                self.opt.radius_range = [r,r]
                self.opt.theta_range = [t,t]
                self.opt.phi_range = [p,p]


            #! 不固定


            rand_target = torch.rand(3).to(self.device)
            rand_target = rand_target - 0.5
            rand_target = torch.zeros(3).to(self.device)

            poses, dirs, thetas, phis, radius,centers = rand_poses(
                B, self.device, self.opt, radius_range=self.opt.radius_range, 
                theta_range=self.opt.theta_range, phi_range=self.opt.phi_range, return_dirs=True, 
                angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front, 
                uniform_sphere_rate=self.opt.uniform_sphere_rate,
                target = rand_target
            )

            #! 取得實際深度圖
            depth_np,image_np = self.get_depth(thetas,phis,radius,target = rand_target)

            # fixed focal
            # fov = self.opt.default_fovy

        else:
            # circle pose
            thetas = torch.FloatTensor([self.opt.default_polar-30]).to(self.device)
            phis = torch.FloatTensor([(index[0] / self.size) * 360]).to(self.device)
            radius = torch.FloatTensor([self.opt.default_radius]).to(self.device)

            #! 固定 1 個相機
            if self.opt.one_camera:
                thetas = torch.FloatTensor([self.opt.default_polar-30]).to(self.device)
                phis = torch.FloatTensor([0]).to(self.device)
                radius = torch.FloatTensor([self.opt.default_radius]).to(self.device)

            poses, dirs = circle_poses(
                self.device, radius=radius, theta=thetas, phi=phis,
                return_dirs=True, angle_overhead=self.opt.angle_overhead, 
                angle_front=self.opt.angle_front
            )

            # fixed focal
            # fov = self.opt.default_fovy
                 
            #! 取得實際深度圖
            depth_np,image_np = self.get_depth(thetas,phis,radius)

        fov = self.fov

        #* 根據fov 計算相機焦距 focal
        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, self.cx, self.cy])

        #* mvp 轉換中的projection 矩陣
        #* 將 x,y,z 都映射到 [-1,1]
        #? 具體怎麼做不確定
        projection = torch.tensor([
            [2*focal/self.W, 0, 0, 0],
            [0, -2*focal/self.H, 0, 0],
            [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
            [0, 0, -1, 0]
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

        #* MVP變換,用來將世界座標的點轉換到相機平面上
        mvp = projection @ torch.inverse(poses) # [1, 4, 4]

        # sample a low-resolution but full image
        '''
            input:
                相機的座標、內參
            output: 
                rays[rays_o] 相機世界座標的位置
                rays[rays_d] 這個相機原點射向每個pixel的向量 (世界座標)
                兩個shape 都為 (B,N,3), batch=1 , N=H*W 每個pixel , 3 向量
                一個batch 應該就代表一個相機
        '''
        rays = get_rays(poses, intrinsics, self.H, self.W, -1)

        #* delta polar/azimuth/radius to default view
        #* 與default 的差距
        delta_polar = thetas - self.opt.default_polar #* default 90
        delta_azimuth = phis - self.opt.default_azimuth #* default 0
        delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
        delta_radius = radius - self.opt.default_radius #* default 3.2

        data = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'dir': dirs,
            'mvp': mvp,
            'polar': delta_polar,
            'azimuth': delta_azimuth,
            'radius': delta_radius,
            "depth_gt":depth_np,
            "image_gt":image_np,
        }

        return data

    def dataloader(self, batch_size=None):
        batch_size = batch_size or self.opt.batch_size
        #* 創建dataloader
        loader = DataLoader(list(range(self.size)), batch_size=batch_size, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        #* 將nerfdataset 這個class指定給 dataloader._data
        loader._data = self
        return loader