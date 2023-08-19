"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
sys.path.append("./")
print(sys.path)

import argparse
import pathlib
import time
from einops import rearrange

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from datasets.nerf_synthetic import SubjectLoader
from lpips import LPIPS
from radiance_fields.mlp import VanillaNeRFRadianceField

import tensorboardX

import matplotlib.pyplot as plt

from examples.utils import (
    NERF_SYNTHETIC_SCENES,
    render_image_with_occgrid,
    set_random_seed,
)
from nerfacc.estimators.occ_grid import OccGridEstimator

from nerf.provider import NeRFDataset
from examples.datasets.utils import Rays

device = "cuda:0"
set_random_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    default=str(pathlib.Path.cwd() / "data/nerf_synthetic"),
    help="the root dir of the dataset",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train",
    choices=["train", "trainval"],
    help="which train split to use",
)
parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="the path of the pretrained model",
)
parser.add_argument(
    "--scene",
    type=str,
    default="lego",
    choices=NERF_SYNTHETIC_SCENES,
    help="which scene to use",
)
parser.add_argument(
    "--test_chunk_size",
    type=int,
    default=4096,
)
parser.add_argument(
    "--workspace",
    type=str,
    default="test",
)

parser.add_argument('--min_near', type=float, default=0.01, help="minimum near distance for camera")
parser.add_argument('--radius_range', type=float, nargs='*', default=[3.0, 3.5], help="training camera radius range")
parser.add_argument('--theta_range', type=float, nargs='*', default=[45, 105], help="training camera fovy range")
parser.add_argument('--phi_range', type=float, nargs='*', default=[-180, 180], help="training camera fovy range")
parser.add_argument('--fovy_range', type=float, nargs='*', default=[10, 30], help="training camera fovy range")
parser.add_argument('--default_radius', type=float, default=2, help="radius for the default view")
parser.add_argument('--default_polar', type=float, default=90, help="polar for the default view")
parser.add_argument('--default_azimuth', type=float, default=0, help="azimuth for the default view")
parser.add_argument('--default_fovy', type=float, default=20, help="fovy for the default view")
parser.add_argument('--for_test', action='store_true')
parser.add_argument('--heightmap_path', type=str,default='BEV/simple.png')
parser.add_argument('--uniform_sphere_rate', type=float, default=0)
parser.add_argument('--angle_overhead', type=float, default=30)
parser.add_argument('--angle_front', type=float, default=60)

args = parser.parse_args()

folder_path = f"examples/trial/{args.workspace}"
rgb_path = f"{folder_path}/rgb"
depth_path = f"{folder_path}/depth"
ckpt_path = f"{folder_path}/ckpt"
tb_path = f"{folder_path}/tb"
os.makedirs(rgb_path, exist_ok=True)
os.makedirs(depth_path, exist_ok=True)
os.makedirs(ckpt_path, exist_ok=True)
os.makedirs(tb_path, exist_ok=True)

writer = tensorboardX.SummaryWriter(tb_path)

# training parameters
max_steps = 50000 #! 50000
init_batch_size = 1024
target_sample_batch_size = 1 << 16
# scene parameters
aabb = torch.tensor([-1, -1, -1, 1, 1, 1], device=device)
near_plane = 0.0
far_plane = 1.0e10
# model parameters
grid_resolution = 128
grid_nlvl = 1
# render parameters
render_step_size = 5e-3

# setup the dataset
# train_dataset = SubjectLoader(
#     subject_id=args.scene,
#     root_fp=args.data_root,
#     split=args.train_split,
#     num_rays=init_batch_size,
#     device=device,
# )
# test_dataset = SubjectLoader(
#     subject_id=args.scene,
#     root_fp=args.data_root,
#     split="test",
#     num_rays=None,
#     device=device,
# )

estimator = OccGridEstimator(
    roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
).to(device)

# setup the radiance field we want to train.
radiance_field = VanillaNeRFRadianceField().to(device)
optimizer = torch.optim.Adam(radiance_field.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[
        max_steps // 2,
        max_steps * 3 // 4,
        max_steps * 5 // 6,
        max_steps * 9 // 10,
    ],
    gamma=0.33,
)

lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

if args.model_path is not None:
    checkpoint = torch.load(args.model_path)
    radiance_field.load_state_dict(checkpoint["radiance_field_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    estimator.load_state_dict(checkpoint["estimator_state_dict"])
    steps = checkpoint["step"]
else:
    steps = 0


def save_img(step,rgb_path,depth_path,rgb=None,depth=None):
    if rgb!=None:
        imageio.imwrite(
            f"{rgb_path}/rgb_{step}.png",
            (rgb.cpu().detach().numpy() * 255).astype(np.uint8),
        )
        rgb = rearrange(rgb,"H W C -> C H W")
        render_rgb = (rgb.cpu().detach().numpy() * 255).astype(np.uint8)
        writer.add_image("render_image",render_rgb,step)

    if depth!=None:
        depth = depth.cpu().detach().numpy()
        depth = depth[:,:,0]
        plt.imshow(depth, cmap='viridis', vmin=0, vmax=depth.max())
        plt.colorbar()
        plt.savefig(f"{depth_path}/depth_{step}.png")
        plt.close()

        depth = rearrange(depth,"H W -> 1 H W")
        render_depth = (depth - depth.min())/(depth.max()-depth.min() + 1e-6)
        render_depth = (render_depth * 255).astype(np.uint8)
        writer.add_image("render_depth",render_depth,step)


#! ====================  training ======================
tic = time.time()
pbar = tqdm.tqdm(range(steps,max_steps + 1))

total_iter = 50000
dataset_size = 100
total_epoch = int(total_iter/dataset_size)
now_epoch = int(steps/dataset_size)
step = steps

train_loader = NeRFDataset(args, device=device, type='train', H=64, W=64, size=dataset_size).dataloader(batch_size=1)
test_loader = NeRFDataset(args, device=device, type='test', H=256, W=256, size=dataset_size).dataloader(batch_size=1)

# for step in pbar:
for epoch in range(now_epoch,total_epoch):

    for data_bev in train_loader:

        step = step+1
        
        radiance_field.train()
        estimator.train()

        #! ---------- get ray -------------

        # i = torch.randint(0, len(train_dataset), (1,)).item()
        # data = train_dataset[i]

        # render_bkgd = data["color_bkgd"] #* (3) 
        # rays = data["rays"]     #*  typle[ ray_o (N,3), ray_d (N,3)]
        # pixels = data["pixels"] #* (N,3)

        pixels_bev = data_bev["image_gt"]                           #* (C,H,W)
        pixels_bev = rearrange(pixels_bev," C H W -> H W C")        #* (H,W,C)
        pixels_bev = torch.from_numpy(pixels_bev).to(device)        
        depth_gt_bev = data_bev["depth_gt"]                         #* (H,W)
        depth_gt_bev = torch.from_numpy(depth_gt_bev).to(device)

        rays_o = data_bev["rays_o"]                                 #* (1,HW,C)
        rays_d = data_bev["rays_d"]
        rays_o = torch.reshape(rays_o,(64,64,3))                    #* (H,W,C)
        rays_d = torch.reshape(rays_d,(64,64,3))                    #* (H,W,C)
        rays_bev = Rays(origins=rays_o, viewdirs=rays_d)
        render_bkgd_bev = torch.zeros(3, device=device) #* black

        #! --------- 更新occupancy grid -------------

        def occ_eval_fn(x):
            density = radiance_field.query_density(x)
            return density * render_step_size

        estimator.update_every_n_steps(
            step=step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=1e-2,
        )

        #! --------- render -------------
        
        rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
            radiance_field,
            estimator,
            rays_bev,
            # rendering options
            near_plane=near_plane,
            render_step_size=render_step_size,
            render_bkgd=render_bkgd_bev
        )
        if n_rendering_samples == 0:
            pbar.update()
            continue

        if target_sample_batch_size > 0:
            # dynamic batch size for rays to keep sample batch size constant.
            num_rays = len(pixels_bev)
            # num_rays = int(
            #     num_rays * (target_sample_batch_size / float(n_rendering_samples))
            # )
            # train_dataset.update_num_rays(num_rays)

        #! ---------- compute loss ----------
        
        loss = F.smooth_l1_loss(rgb, pixels_bev)

        writer.add_scalar("train/loss_mae", loss, step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            save_img(step,rgb_path,depth_path,rgb,depth)

        pbar.set_description(f"Iter {step} | MSE loss: {loss: .4f}")
        pbar.update()

    #! =============== 每500步進行測試 ======================

    if step % 500 == 0: #! 5000
        elapsed_time = time.time() - tic
        loss = F.mse_loss(rgb, pixels_bev)
        psnr = -10.0 * torch.log(loss) / np.log(10.0)
        print(
            f"elapsed_time={elapsed_time:.2f}s | step={step} | "
            f"loss={loss:.5f} | psnr={psnr:.2f} | "
            f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels_bev):d} | "
            f"max_depth={depth.max():.3f} | "
        )

        #! --------- 儲存 model 資訊 -------------

        model_save_path = f"{ckpt_path}/mlp_nerf_{step}"
        torch.save(
            {
                "step": step,
                "radiance_field_state_dict": radiance_field.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "estimator_state_dict": estimator.state_dict(),
            },
            model_save_path,
        )
        
        radiance_field.eval()
        estimator.eval()
            
        pbar_test = tqdm.tqdm(range(101))
        
        with torch.no_grad():

            all_depth = []
            all_rgb = []

            for data_bev in test_loader:

                # data = train_dataset[0]
                # render_bkgd = data["color_bkgd"]
                # rays = data["rays"]
                # pixels = data["pixels"]

                pixels_bev = data_bev["image_gt"]                           #* (C,H,W)
                pixels_bev = rearrange(pixels_bev," C H W -> H W C")        #* (H,W,C)
                # pixels_bev = pixels_bev.reshape(-1, pixels_bev.shape[-1])   #* (HW,C)
                pixels_bev = torch.from_numpy(pixels_bev).to(device)        
                depth_gt_bev = data_bev["depth_gt"]                         #* (H,W)
                depth_gt_bev = torch.from_numpy(depth_gt_bev).to(device)

                rays_o = data_bev["rays_o"]                              
                rays_d = data_bev["rays_d"]
                rays_o = torch.reshape(rays_o,(256,256,3))                    #* (H,W,C)
                rays_d = torch.reshape(rays_d,(256,256,3))                    #* (H,W,C)
                rays_bev = Rays(origins=rays_o, viewdirs=rays_d)
                render_bkgd_bev = torch.zeros(3, device=device) #* black


                #! ------------ rendering ----------------

                rgb, acc, depth, _ = render_image_with_occgrid(
                    radiance_field,
                    estimator,
                    rays_bev,
                    # rendering options
                    near_plane=near_plane,
                    render_step_size=render_step_size,
                    render_bkgd=render_bkgd_bev,
                    # test options
                    test_chunk_size=args.test_chunk_size,
                )

                #! ------------ 儲存rgb depth 圖像 ------------

                rgb = (rgb.cpu().detach().numpy() * 255).astype(np.uint8) # (H W 3)
                all_rgb.append(rgb)


                depth = depth.cpu().detach().numpy()
                # depth = depth[:,:,0]
                # depth = rearrange(depth,"H W -> 1 H W")
                render_depth = (depth - depth.min())/(depth.max()-depth.min() + 1e-6)
                render_depth = (render_depth * 255).astype(np.uint8) # (H W 1)

                all_depth.append(render_depth)

                pbar_test.update()

            all_depth = np.stack(all_depth, axis=0)
            all_rgb = np.stack(all_rgb, axis=0)
            imageio.mimwrite(f'{rgb_path}/rgb_{step}.mp4', all_rgb, fps=10, quality=8, macro_block_size=1)
            imageio.mimwrite(f'{depth_path}/depth_{step}.mp4', all_depth, fps=10, quality=8, macro_block_size=1)
        


    #! ============== 訓練完的 testing ===================
     
    if step > 0 and step % max_steps == 0:

        #! --------- 儲存 model 資訊 -------------

        model_save_path = f"{ckpt_path}/mlp_nerf_{step}"
        torch.save(
            {
                "step": step,
                "radiance_field_state_dict": radiance_field.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "estimator_state_dict": estimator.state_dict(),
            },
            model_save_path,
        )

        # evaluation
        radiance_field.eval()
        estimator.eval()

        psnrs = []
        lpips = []
        with torch.no_grad():
            for i in tqdm.tqdm(range(len(test_dataset))):
                if i%49 !=0:
                    continue
                data = test_dataset[i]
                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]

                # rendering
                rgb, acc, depth, _ = render_image_with_occgrid(
                    radiance_field,
                    estimator,
                    rays,
                    # rendering options
                    near_plane=near_plane,
                    render_step_size=render_step_size,
                    render_bkgd=render_bkgd,
                    # test options
                    test_chunk_size=args.test_chunk_size,
                )
                mse = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(mse) / np.log(10.0)
                psnrs.append(psnr.item())
                lpips.append(lpips_fn(rgb, pixels).item())
                # if i == 0:
                imageio.imwrite(
                    f"{rgb_path}/rgb_test{i}.png",
                    (rgb.cpu().numpy() * 255).astype(np.uint8),
                )
                imageio.imwrite(
                    f"{rgb_path}/rgb_error{i}.png",
                    (
                        (rgb - pixels).norm(dim=-1).cpu().numpy() * 255
                    ).astype(np.uint8),
                )
        psnr_avg = sum(psnrs) / len(psnrs)
        lpips_avg = sum(lpips) / len(lpips)
        print(f"evaluation: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}")