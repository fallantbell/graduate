import os
import glob
import tqdm
import math
import imageio
import psutil
from pathlib import Path
import random
import shutil
import warnings
import tensorboardX

import numpy as np

import time

import cv2
import matplotlib.pyplot as plt
from PIL import Image
from einops import rearrange

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms.functional as TF
from torchmetrics import PearsonCorrCoef
from torchvision import transforms

from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def safe_normalize(x, eps=1e-20):
    #* 歸一化
    #* 為了防止除以0, 使用clamp 限制最小為eps
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]i = i*0+
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics


    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device))
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    zs = - torch.ones_like(i)
    xs = - (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1) #* 相機座標下原點到每個pixel 的向量

    # directions = safe_normalize(directions)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) #* (B, N, 3) 轉換到世界座標 

    rays_o = poses[..., :3, 3] #* [B, 3] 每個相機的原點世界座標
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3] , N = H*W 

    results['rays_o'] = rays_o #* 各個相機的原點
    results['rays_d'] = rays_d #* 相機原點射向平面上各個pixel的向量

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


class Trainer(object):
    def __init__(self,
		         argv, # command line args
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network
                 guidance, # guidance network
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):

        self.argv = argv
        self.name = name
        self.opt = opt                  #* 輸入參數
        self.mute = mute                
        self.metrics = metrics          
        self.local_rank = local_rank    #* 當前是第幾個gpu
        self.world_size = world_size    #* 總共的gpu
        self.workspace = workspace      #* 儲存的資料夾
        self.ema_decay = ema_decay      #* 移動平均的decay 預設0.95
        self.fp16 = fp16                #* 使用fp16 精度
        self.best_mode = best_mode      
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model  #* nerf 網路

        # guide model
        self.guidance = guidance   #* stable diffusion guidance 網路
        self.embeddings = {}

        # text prompt / images
        if self.guidance is not None:
            for key in self.guidance: #* key = SD
                for p in self.guidance[key].parameters(): #* stable diffusion 所有參數
                    p.requires_grad = False #* 不進行倒傳遞
                self.embeddings[key] = {}
            self.prepare_embeddings() #* 對 text、image 做encoding

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if self.opt.images is not None:
            self.pearson = PearsonCorrCoef().to(self.device)

        #* 定義optimizer
        self.optimizer = optimizer(self.model)

        #* 定義 lr
        self.lr_scheduler = lr_scheduler(self.optimizer)

        #* 使用移動平均
        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        #* 混和精度訓練，可以加快訓練並減少memory
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        #* 訓練階段相關參數
        self.total_train_t = 0
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        #* 設定儲存資料夾
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)                          #* 創建資料夾
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")     #* log檔案位置
            self.log_ptr = open(self.log_path, "a+")                            #* 開啟log檔案

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')        #* ckpt 資料夾
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"                #* ckpt 檔案位置
            os.makedirs(self.ckpt_path, exist_ok=True)                          #* 創建ckpt 資料夾

            # Save a copy of image_config in the experiment workspace
            if opt.image_config is not None:
                shutil.copyfile(opt.image_config, os.path.join(self.workspace, os.path.basename(opt.image_config)))

        self.log(f'[INFO] Cmdline: {self.argv}')
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')


        #* 讀取checkpoint
        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest": #* 預設這個
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    # calculate the text embs.
    @torch.no_grad()
    def prepare_embeddings(self):

        # text embeddings (stable-diffusion)
        if self.opt.text is not None:

            if 'SD' in self.guidance:
                #* 用clip 對 text prompt 進行encode 
                self.embeddings['SD']['default'] = self.guidance['SD'].get_text_embeds([self.opt.text])
                self.embeddings['SD']['uncond'] = self.guidance['SD'].get_text_embeds([self.opt.negative])

                #* 對不同相機角度分別做帶有方位的text prompt
                for d in ['front', 'side', 'back', 'bev']:
                    self.embeddings['SD'][d] = self.guidance['SD'].get_text_embeds([f"{self.opt.text}, {d} view"])

            if 'IF' in self.guidance:
                self.embeddings['IF']['default'] = self.guidance['IF'].get_text_embeds([self.opt.text])
                self.embeddings['IF']['uncond'] = self.guidance['IF'].get_text_embeds([self.opt.negative])

                for d in ['front', 'side', 'back']:
                    self.embeddings['IF'][d] = self.guidance['IF'].get_text_embeds([f"{self.opt.text}, {d} view"])

            if 'clip' in self.guidance:
                self.embeddings['clip']['text'] = self.guidance['clip'].get_text_embeds(self.opt.text)

        if self.opt.images is not None:

            h = int(self.opt.known_view_scale * self.opt.h)
            w = int(self.opt.known_view_scale * self.opt.w)

            # load processed image
            rgbas = [cv2.cvtColor(cv2.imread(image, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA) for image in self.opt.images]
            rgba_hw = np.stack([cv2.resize(rgba, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 for rgba in rgbas])
            rgb_hw = rgba_hw[..., :3] * rgba_hw[..., 3:] + (1 - rgba_hw[..., 3:])
            self.rgb = torch.from_numpy(rgb_hw).permute(0,3,1,2).contiguous().to(self.device)
            self.mask = torch.from_numpy(rgba_hw[..., 3] > 0.5).to(self.device)
            print(f'[INFO] dataset: load image prompt {self.opt.images} {self.rgb.shape}')

            # load depth
            depth_paths = [image.replace('_rgba.png', '_depth.png') for image in self.opt.images]
            depths = [cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) for depth_path in depth_paths]
            depth = np.stack([cv2.resize(depth, (w, h), interpolation=cv2.INTER_AREA) for depth in depths])
            self.depth = torch.from_numpy(depth.astype(np.float32) / 255).to(self.device)
            print(f'[INFO] dataset: load depth prompt {depth_paths} {self.depth.shape}')

            # load normal
            normal_paths = [image.replace('_rgba.png', '_normal.png') for image in self.opt.images]
            normals = [cv2.imread(normal_path, cv2.IMREAD_UNCHANGED) for normal_path in normal_paths]
            normal = np.stack([cv2.resize(normal, (w, h), interpolation=cv2.INTER_AREA) for normal in normals])
            self.normal = torch.from_numpy(normal.astype(np.float32) / 255).to(self.device)
            print(f'[INFO] dataset: load normal prompt {normal_paths} {self.normal.shape}')

            # encode embeddings for zero123
            if 'zero123' in self.guidance:
                rgba_256 = np.stack([cv2.resize(rgba, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 for rgba in rgbas])
                rgbs_256 = rgba_256[..., :3] * rgba_256[..., 3:] + (1 - rgba_256[..., 3:])
                rgb_256 = torch.from_numpy(rgbs_256).permute(0,3,1,2).contiguous().to(self.device)
                guidance_embeds = self.guidance['zero123'].get_img_embeds(rgb_256)
                self.embeddings['zero123']['default'] = {
                    'zero123_ws' : self.opt.zero123_ws,
                    'c_crossattn' : guidance_embeds[0],
                    'c_concat' : guidance_embeds[1],
                    'ref_polars' : self.opt.ref_polars,
                    'ref_azimuths' : self.opt.ref_azimuths,
                    'ref_radii' : self.opt.ref_radii,
                }

            if 'clip' in self.guidance:
                self.embeddings['clip']['image'] = self.guidance['clip'].get_img_embeds(self.rgb)


    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()


    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------

    def train_step(self, data, save_guidance_path:Path=None):
        """
            Args:
                save_guidance_path: an image that combines the NeRF render, the added latent noise,
                    the denoised result and optionally the fully-denoised image.
        """


        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        mvp = data['mvp'] # [B, 4, 4]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        just_depth = False

        if self.global_step < (self.opt.latent_iter_ratio * self.opt.iters):
            ambient_ratio = 1.0
            shading = 'normal' #* 猜測在訓練前期主要根據法線進行訓練，主要訓練物體形狀，而沒有訓練顏色的部分
            as_latent = True
            binarize = False
            bg_color = None
            just_depth = True #! 前期用來只訓練深度

        else:
            if self.global_step < (self.opt.albedo_iter_ratio * self.opt.iters):
                ambient_ratio = 1.0
                shading = 'albedo'
            else:
                # random shading
                ambient_ratio = 0.1 + 0.9 * random.random()
                rand = random.random()
                if rand > 0.8:
                    shading = 'textureless' #* 在訓練中後期，會開始訓練物體顏色，不過還是會random 去掉顏色來保持形狀的穩定
                else:
                    shading = 'lambertian'

                #! 固定ambient 為1，取消random對訓練的影響
                #! 固定 shading 為 lambertian，取消對mse 造成的影響
                shading = 'lambertian'
                ambient_ratio = 1.0

            as_latent = False

            # random weights binarization (like mobile-nerf) [NOT WORKING NOW]
            # binarize_thresh = min(0.5, -0.5 + self.global_step / self.opt.iters)
            # binarize = random.random() < binarize_thresh
            binarize = False

            # random background
            rand = random.random()
            if self.opt.bg_radius > 0 and rand > 0.5:
                bg_color = None # use bg_net
            else:
                bg_color = torch.rand(3).to(self.device) # single color random bg

        bg_color = 0 #*! 背景都先設定黑色來訓練

        '''
            predict 每個採樣點, 並使用volume render 影像出來 
            #!跳到render.py中
            
            input:
                rays_o 相機世界座標的位置
                rays_d 這個相機原點射向每個pixel的向量 (世界座標)
                mvp: 轉換矩陣用來將世界座標的點轉換到相機平面上
                H,W 
                ambient_ratio: 跟反照率有關
                bg_color: 是否有背景顏色
            
            output:
                預測出來的 
                image:          rgb影像 shape (B,N,3)
                depth:          深度圖 shape (B,N)
                weight_sum:     影像的透明度 shape (B,N)
                weight:         可能是每個點的透明度 shape (M)
                loss_orient:    某個神奇的loss, 可能跟維持物體形狀有關 (B)
                
        '''
        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=False, perturb=True, bg_color=bg_color, ambient_ratio=ambient_ratio, shading=shading, binarize=binarize)
        pred_depth = outputs['depth'].reshape(B, 1, H, W)
        pred_mask = outputs['weights_sum'].reshape(B, 1, H, W)
        if 'normal_image' in outputs:
            pred_normal = outputs['normal_image'].reshape(B, H, W, 3)

        if as_latent:
            #* 加上透明度
            # abuse normal & mask as latent code for faster geometry initialization (ref: fantasia3D)
            pred_rgb = torch.cat([outputs['image'], outputs['weights_sum'].unsqueeze(-1)], dim=-1).reshape(B, H, W, 4).permute(0, 3, 1, 2).contiguous() # [B, 4, H, W]
        else:
            #* 沒有透明度
            pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]


        # novel view loss

        loss = 0

        if 'SD' in self.guidance and self.opt.without_SDS==False:
            # interpolate text_z
            #* 圓心角
            azimuth = data['azimuth'] # [-180, 180]

            # ENHANCE: remove loop to handle batch size > 1
            text_z = [self.embeddings['SD']['uncond']] * azimuth.shape[0]
            for b in range(azimuth.shape[0]):
                #* 物體正面
                if azimuth[b] >= -90 and azimuth[b] < 90:
                    if azimuth[b] >= 0:
                        r = 1 - azimuth[b] / 90
                    else:
                        r = 1 + azimuth[b] / 90
                    start_z = self.embeddings['SD']['front']
                    end_z = self.embeddings['SD']['side']
                #* 物體背面
                else:
                    if azimuth[b] >= 0:
                        r = 1 - (azimuth[b] - 90) / 90
                    else:
                        r = 1 + (azimuth[b] + 90) / 90
                    start_z = self.embeddings['SD']['side']
                    end_z = self.embeddings['SD']['back']

                #* 根據相機角度對 text embedding 做插值
                text_z.append(r * start_z + (1 - r) * end_z) 

                #! 取消 前、後、側邊 視角描述, 增加bev 視角
                p=text_z.pop()
                if data['polar'] < -45: #* bev 視角, 圓頂0度 平視90度
                    text_z.append(self.embeddings['SD']['bev'])
                else:
                    text_z.append(self.embeddings['SD']['default'])

            
            #* concat uncondition 跟 condition(物體、方位) 的text embedding
            #* text_z shape (2,77,1024)
            text_z = torch.cat(text_z, dim=0) 

            #* 真實深度圖 numpy array
            depth_gt = data['depth_gt']
            image_gt = data['image_gt']

            #* 給定text prompt 進行 stable diffusion 的guidance 
            #* 計算sds loss
            #! 跳到 sd_utils.py
            loss = loss + self.guidance['SD'].train_step(text_z, pred_rgb,depth_gt, as_latent=as_latent, guidance_scale=self.opt.guidance_scale, grad_scale=self.opt.lambda_guidance,
                                                            save_guidance_path=save_guidance_path,just_depth = just_depth)

            depth_gt_tensor = torch.tensor(depth_gt).to(torch.float32).to(self.device)
            depth_gt_tensor = rearrange(depth_gt_tensor,'H W -> 1 1 H W')
            mse_loss = nn.MSELoss()
            depth_loss = mse_loss(depth_gt_tensor,pred_depth)

            loss = loss + depth_loss*self.opt.lambda_depth

            pred_rgb3 = pred_rgb[:,:3,...]
            image_gt_tensor = torch.tensor(image_gt).to(torch.float32).to(self.device)
            image_gt_tensor = rearrange(image_gt_tensor,'C H W -> 1 C H W')
            rgb_loss = mse_loss(image_gt_tensor,pred_rgb3)

            if just_depth: #! 前期的color只是用來表示法向量，不用來算mse Loss
                rgb_loss = 0

            # loss = loss + rgb_loss
            
            if self.use_tensorboardX:
                self.writer.add_scalar("train/loss_noise_mse", self.guidance['SD'].noise_mse, self.global_step)
                self.writer.add_scalar("train/loss_depth_mse", depth_loss, self.global_step)
                # self.writer.add_scalar("train/loss_rgb_mse", rgb_loss, self.global_step)
        
        if self.opt.without_SDS:
            pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
            init_img = self.guidance['SD'].init_image
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor()
            ])
            init_img = transform(init_img)
            init_img = torch.unsqueeze(init_img, 0).to(self.device)
            mse_loss = nn.MSELoss()
            loss = mse_loss(init_img, pred_rgb)

            if self.use_tensorboardX:
                self.writer.add_scalar("train/loss_mse", loss, self.global_step)


        # regularizations
        #* 加上幾個奇怪的loss
        if not self.opt.dmtet:

            if self.opt.lambda_opacity > 0:
                loss_opacity = (outputs['weights_sum'] ** 2).mean()
                loss = loss + self.opt.lambda_opacity * loss_opacity
                self.writer.add_scalar("train/loss_opacity", self.opt.lambda_opacity * loss_opacity, self.global_step)

            if self.opt.lambda_entropy > 0:
                alphas = outputs['weights'].clamp(1e-5, 1 - 1e-5)
                # alphas = alphas ** 2 # skewed entropy, favors 0 over 1
                loss_entropy = (- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()
                lambda_entropy = self.opt.lambda_entropy * min(1, 2 * self.global_step / self.opt.iters)
                loss = loss + lambda_entropy * loss_entropy
                self.writer.add_scalar("train/loss_lambda_entropy ", lambda_entropy * loss_entropy, self.global_step)

            if self.opt.lambda_2d_normal_smooth > 0 and 'normal_image' in outputs:
                loss_smooth = (pred_normal[:, 1:, :, :] - pred_normal[:, :-1, :, :]).square().mean() + \
                              (pred_normal[:, :, 1:, :] - pred_normal[:, :, :-1, :]).square().mean()
                loss = loss + self.opt.lambda_2d_normal_smooth * loss_smooth

            if self.opt.lambda_orient > 0 and 'loss_orient' in outputs:
                loss_orient = outputs['loss_orient']
                loss = loss + self.opt.lambda_orient * loss_orient
                self.writer.add_scalar("train/loss_orient", self.opt.lambda_orient * loss_orient, self.global_step)

            if self.opt.lambda_3d_normal_smooth > 0 and 'loss_normal_perturb' in outputs:
                loss_normal_perturb = outputs['loss_normal_perturb']
                loss = loss + self.opt.lambda_3d_normal_smooth * loss_normal_perturb
            
            # if self.use_tensorboardX:
            #     self.writer.add_scalar("train/loss_lambda_entropy ", lambda_entropy * loss_entropy, self.global_step)
            #     self.writer.add_scalar("train/loss_orient", self.opt.lambda_orient * loss_orient, self.global_step)
            #     self.writer.add_scalar("train/loss_opacity", self.opt.lambda_opacity * loss_opacity, self.global_step)
                
        return pred_rgb, pred_depth, depth_gt_tensor,loss, image_gt_tensor, self.guidance['SD'].non_zero_mask

    def post_train_step(self):

        # unscale grad before modifying it!
        # ref: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
        self.scaler.unscale_(self.optimizer)

        # clip grad
        if self.opt.grad_clip >= 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.opt.grad_clip)

        if not self.opt.dmtet and self.opt.backbone == 'grid':

            if self.opt.lambda_tv > 0:
                lambda_tv = min(1.0, self.global_step / (0.5 * self.opt.iters)) * self.opt.lambda_tv
                self.model.encoder.grad_total_variation(lambda_tv, None, self.model.bound)
            if self.opt.lambda_wd > 0:
                self.model.encoder.grad_weight_decay(self.opt.lambda_wd)

    def eval_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        mvp = data['mvp']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        bg_color = 0 #*! 背景都先設定黑色

        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=True, perturb=False, bg_color=bg_color, light_d=light_d, ambient_ratio=ambient_ratio, shading=shading)
        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        # dummy
        loss = torch.zeros([1], device=pred_rgb.device, dtype=pred_rgb.dtype)

        return pred_rgb, pred_depth, loss

    def test_step(self, data, bg_color=None, perturb=False):
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        mvp = data['mvp']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(rays_o.device)
        bg_color = 0 #*! 背景都先設定黑色

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=True, perturb=perturb, light_d=light_d, ambient_ratio=ambient_ratio, shading=shading, bg_color=bg_color)

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        return pred_rgb, pred_depth, None

    def save_mesh(self, loader=None, save_path=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'mesh')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(save_path, exist_ok=True)

        self.model.export_mesh(save_path, resolution=self.opt.mcubes_resolution, decimate_target=self.opt.decimate_target)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, test_loader, max_epochs):

        #* 使用tensorboard
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))
            
        #* 儲存初始畫面
        # if self.use_tensorboardX:
        #     init_img = self.guidance['SD'].init_image
        #     transform = transforms.ToTensor()
        #     init_img_tensor = transform(init_img)
        #     self.writer.add_image("init_img",init_img_tensor,0)

        start_t = time.time()

        #* 開始訓練 max_epoch 預設100
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch
            
            #* 訓練一個epoch
            self.train_one_epoch(train_loader, max_epochs)

            #* 儲存model
            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            #* 進行evaluation
            if self.epoch % self.opt.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

            if self.epoch % self.opt.test_interval == 0 or self.epoch == max_epochs:
                self.test(test_loader)

        end_t = time.time()

        self.total_train_t = end_t - start_t + self.total_train_t

        self.log(f"[INFO] training takes {(self.total_train_t)/ 60:.4f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []

        with torch.no_grad():

            for i, data in enumerate(loader):

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, _ = self.test_step(data)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                else:
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)

                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)

            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8, macro_block_size=1)

        self.log(f"==> Finished Test.")

    # [GUI] train text step.
    def train_gui(self, train_loader, step=16):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)

        loader = iter(train_loader)

        for _ in range(step):

            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                pred_rgbs, pred_depths, loss = self.train_step(data)

            self.scaler.scale(loss).backward()
            self.post_train_step()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss.detach()

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }

        return outputs


    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, mvp, W, H, bg_color=None, spp=1, downscale=1, light_d=None, ambient_ratio=1.0, shading='albedo'):

        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)
        mvp = torch.from_numpy(mvp).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)

        # from degree theta/phi to 3D normalized vec
        light_d = np.deg2rad(light_d)
        light_d = np.array([
            np.sin(light_d[0]) * np.sin(light_d[1]),
            np.cos(light_d[0]),
            np.sin(light_d[0]) * np.cos(light_d[1]),
        ], dtype=np.float32)
        light_d = torch.from_numpy(light_d).to(self.device)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'mvp': mvp,
            'H': rH,
            'W': rW,
            'light_d': light_d,
            'ambient_ratio': ambient_ratio,
            'shading': shading,
        }

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed!
                preds, preds_depth, _ = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        outputs = {
            'image': preds[0].detach().cpu().numpy(),
            'depth': preds_depth[0].detach().cpu().numpy(),
        }

        return outputs

    def train_one_epoch(self, loader, max_epochs):
        #! training 一個 epoch
        self.log(f"==> [{time.strftime('%Y-%m-%d_%H-%M-%S')}] Start Training {self.workspace} Epoch {self.epoch}/{max_epochs}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        #! 會進入到dataloader 的collate function 進行資料處理
        for data in loader:

            '''
                data = {
                    'H': self.H,                #* 64
                    'W': self.W,                #* 64
                    'rays_o': rays['rays_o'],   #* 相機原點 (世界座標)
                    'rays_d': rays['rays_d'],   #* 相機原點指向每個pixel的向量 (世界座標)
                    'dir': dirs,                #* 這個相機是看向物體的哪個位置 (前、後、兩側、上方)
                    'mvp': mvp,                 #* mvp轉換矩陣, 可以將世界座標點轉換到相機平面
                    'polar': delta_polar,       #* 與default theta的差距角度
                    'azimuth': delta_azimuth,   #* 與default phis的差距角度
                    'radius': delta_radius,     #* 與default 半徑的差距
                    'depth_gt': depth_np        #* 相機視角下的真實深度圖
                }
            '''

            #* update grid every 16 steps
            #* 更新 instant ngp 中的occupancy grid
            if (self.model.cuda_ray or self.model.taichi_ray) and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            #! 測試一個角度多訓練幾次 !!
            repeat_time = 0
            total_time_a_camera = 1

            while repeat_time<total_time_a_camera:

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    if self.opt.save_guidance and (self.global_step % self.opt.save_guidance_interval == 0):
                        save_guidance_path = save_guidance_folder / f'step_{self.global_step:07d}.png'
                    else:
                        save_guidance_path = None
                    
                    #! 訓練一個data, 也就是一個角度的相機
                    '''
                        input:
                            前面得到的data
                        output:
                            pred_rgbs: render出來的影像
                            pref_depth: render出來的深度圖
                            loss
                    '''
                    pred_rgbs, pred_depths, depth_gt,loss,image_gt, grad_map = self.train_step(data, save_guidance_path=save_guidance_path)

                #* 倒傳遞
                self.scaler.scale(loss).backward()

                self.post_train_step()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.scheduler_update_every_step:
                    self.lr_scheduler.step()
                
                repeat_time += 1
            
            #! 測試一個角度多訓練幾次 !!

            loss_val = loss.item() #* 這個相機的loss
            total_loss += loss_val  #* 整個epoch的loss

            if self.local_rank == 0:
                #* 儲存每個step 的資訊
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss_all", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.use_tensorboardX:
            pred_rgbs = pred_rgbs[0].detach().cpu().numpy()
            pred_rgbs = (pred_rgbs * 255).astype(np.uint8)
            pred_depths = pred_depths[0].detach().cpu().numpy()
            pred_depths = (pred_depths - pred_depths.min()) / (pred_depths.max() - pred_depths.min() + 1e-6)
            pred_depths = (pred_depths * 255).astype(np.uint8)

            depth_gt = depth_gt[0].detach().cpu().numpy()
            depth_gt = (depth_gt - depth_gt.min())/(depth_gt.max()-depth_gt.min() + 1e-6)
            depth_gt = (depth_gt * 255).astype(np.uint8)

            image_gt = image_gt[0].detach().cpu().numpy()
            image_gt = (image_gt * 255).astype(np.uint8)

            grad_map = rearrange(grad_map, "H W -> 1 H W")
            grad_map = grad_map.detach().cpu().numpy()
            grad_map = (grad_map - grad_map.min()) / (grad_map.max() - grad_map.min() + 1e6)
            grad_map = (grad_map * 255).astype(np.uint8)

            self.writer.add_image("render_image",pred_rgbs,self.global_step)
            self.writer.add_image("render_depth",pred_depths,self.global_step)
            self.writer.add_image("depth_GT",depth_gt,self.global_step)
            self.writer.add_image("image_GT",image_gt,self.global_step)
            self.writer.add_image("grad_map",grad_map,self.global_step)
            
        if self.ema is not None:
            self.ema.update()

        #* 這個epoch 平均loss
        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()

        cpu_mem, gpu_mem = get_CPU_mem(), get_GPU_mem()[0]
        self.log(f"==> [{time.strftime('%Y-%m-%d_%H-%M-%S')}] Finished Epoch {self.epoch}/{max_epochs}. CPU={cpu_mem:.1f}GB, GPU={gpu_mem:.1f}GB.")


    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            #* 預設size為8
            for data in loader:
                self.local_step += 1
                
                #* 跟training 差不多
                #* 只差在render 時，eval 會將已經算完的ray的資源分配給其他ray來加快速度
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size

                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                #* 不重要 它沒有額外去計算SDS loss
                #* 這裡它是隨便填1而已
                loss_val = loss.item()
                total_loss += loss_val

                #* 儲存這個epoch 8個角度相機render 出來的rgb 跟 depth
                if self.local_rank == 0:

                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_depth.png')

                    #self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)
                    pred = Image.fromarray(pred)

                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
                    pred_depth = (pred_depth * 255).astype(np.uint8)
                    pred_depth = Image.fromarray(pred_depth, mode='L')
                    
                    pred_depth.save(save_path_depth)
                    pred.save(save_path)
                    # cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)


        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_density'] = self.model.mean_density

        if self.opt.dmtet:
            state['tet_scale'] = self.model.tet_scale.cpu().numpy()

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:
            if len(self.stats["results"]) > 0:
                # always save best since loss cannot reflect performance.
                if True:
                    # self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    # self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if self.model.cuda_ray:
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']

        if self.opt.dmtet:
            if 'tet_scale' in checkpoint_dict:
                new_scale = torch.from_numpy(checkpoint_dict['tet_scale']).to(self.device)
                self.model.verts *= new_scale / self.model.tet_scale
                self.model.tet_scale = new_scale

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")


def get_CPU_mem():
    return psutil.Process(os.getpid()).memory_info().rss /1024**3


def get_GPU_mem():
    num = torch.cuda.device_count()
    mem, mems = 0, []
    for i in range(num):
        mem_free, mem_total = torch.cuda.mem_get_info(i)
        mems.append(int(((mem_total - mem_free)/1024**3)*1000)/1000)
        mem += mems[-1]
    return mem, mems
