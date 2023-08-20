# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "9"

from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline
from diffusers import StableDiffusionDepth2ImgPipeline
from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile
from pathlib import Path
from PIL import Image
from einops import rearrange, repeat 

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from torch.cuda.amp import custom_bwd, custom_fwd

# from Midas import get_depth
import cv2
import numpy as np

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class StableDiffusion(nn.Module):
    def __init__(self, device, fp16, vram_O, sd_version='2.1', hf_key=None, t_range=[0.02, 0.98], text_prompt=""):
        super().__init__()

        self.device = device
        self.sd_version = sd_version
        self.text_prompt = text_prompt

        print(f'[INFO] loading stable diffusion...')

        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        #* 使用fp16 經度
        self.precision_t = torch.float16 if fp16 else torch.float32

        #* 創建 SD2.1 model
        # pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.precision_t)

        #* 創建 SD depth model
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth", torch_dtype=self.precision_t)

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        #* 創建初始影像
        # self.init_image = pipe(prompt=self.text_prompt,negative_prompt="elevation angle,from below,people").images[0]
        # self.init_image.save(f"test/init_image.png")
        # print(f'create init image ... text:{self.text_prompt}')

        #* 使用midas預測深度
        # self.init_image_depth = get_depth(self.init_image)
        # init_depth = self.init_image_depth.cpu().numpy()
        # init_depth = (init_depth - init_depth.min()) / (init_depth.max() - init_depth.min() + 1e-6)
        # init_depth = (init_depth * 255).astype(np.uint8)
        # cv2.imwrite(f"test/init_image_depth.png", init_depth)

        #* 提取 stable diffusion 中的各個小元件
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        #* 使用DDIM scheduler
        self.scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-depth", subfolder="scheduler", torch_dtype=self.precision_t)

        #* 從GPU mem 移除
        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps #* 預設1000步
        self.min_step = int(self.num_train_timesteps * t_range[0]) #* 20
        self.max_step = int(self.num_train_timesteps * t_range[1]) #* 980 只取 20~980 步
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt: [str]

        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        #* output shape [1,77,1024]

        return embeddings

    def get_depth_mask(self,depth_gt,depth_type):
        '''
            將輸入的真實depth map轉換為輸入的latent
            input:
                depth_gt: numpy array (H,W)
                depth_type: 跟text_embed 一樣的type
            output:
                disparity_map: tensor (1,1,H,W)
        '''

        #* 配合midas 將深度inverse，除了0(背景以外)都做 1/depth
        non_zero_mask = (depth_gt != 0)
        depth_gt[non_zero_mask] = 1.0 / depth_gt[non_zero_mask]
        disparity_map = torch.tensor(depth_gt).to(torch.float32)

        #* 擴增tensor以符合之後訓練的shape
        disparity_map = rearrange(disparity_map,'H W -> 1 1 H W')

        #! 儲存用來在util.py做 depth loss
        self.depth_gt = disparity_map

        #* resize (64,64) 應該是SD 預設的latent大小
        disparity_map = F.interpolate(
            disparity_map,
            size=(64, 64),
            mode='bilinear',
            align_corners=False,
        ).to(torch.float16)


        #* normalize 到 -1~1
        disparity_min = torch.amin(disparity_map, dim=[1, 2, 3], keepdim=True)
        disparity_max = torch.amax(disparity_map, dim=[1, 2, 3], keepdim=True)
        disparity_map = 2.0 * (disparity_map - disparity_min) / (disparity_max - disparity_min) - 1.0
        
        #* 疊起來做classifier free
        disparity_map = torch.cat([disparity_map] * 2)
        disparity_map = disparity_map.to(self.device)

        return disparity_map, non_zero_mask


    def train_step(self, text_embeddings, pred_rgb, depth_gt,guidance_scale=100, as_latent=False, grad_scale=1,
                   save_guidance_path:Path=None,just_depth=False):

        if as_latent:
            #* 縮放到 64x64 並把 (0,1) => (-1,1)
            #* 直接將render的影像當作latent code進行引導
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1 
        else:
            #* 將render影像放大到 512x512, 然後再通過vae encode成latent code
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        #* 獲取深度channel
        disparity_map, non_zero_mask = self.get_depth_mask(depth_gt, text_embeddings.dtype)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        #* 從 20~980 抽取一個 t
        t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t) #* 圖像加上t步的noise

            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2) #* condition + uncondition
            latent_model_input = torch.cat([latent_model_input, disparity_map], dim=1) #* 增加depth channel

            tt = torch.cat([t] * 2)
            noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample #* 預測出來的noise

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond) #* guidance 後的noise , guidance_scale 預設 100

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        
        #* 只計算前景部分的loss
        non_zero_mask = torch.tensor(non_zero_mask).to(self.device)
        non_zero_mask = rearrange(non_zero_mask, "H W -> 1 1 H W")
        non_zero_mask = repeat(non_zero_mask, f"1 1 H W -> 1 {noise.shape[1]} H W")
        noise_pred = noise_pred * non_zero_mask
        noise = noise * non_zero_mask

        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise) #* 根據原本的noise與pred_noise 計算gradient
        grad = torch.nan_to_num(grad)

        mse_loss = nn.MSELoss()
        self.noise_mse = mse_loss(noise_pred,noise)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        #* 不太懂 就先照他的做
        '''
            應該是因為中間有一部份沒有做gradient的計算
            所以我們不能拿最終的loss直接給pytorch 自動計算整個model的gradient
            而是要手動計算後半部分的gradient,與倒傳遞回來的loss
            再把這個loss 拿去給前半部分的model做倒傳遞
        '''
        loss = SpecifyGradient.apply(latents, grad)

        return loss

    @torch.no_grad()
    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None,bev=""):

        #! 處理 depth map
        depth_img = Image.open(f"terrain_render/{bev}/disparity/0.png")
        depth_img.save(f"sd_sample/{bev}/depth_cond.png")
        depth_map = torch.tensor(np.array(depth_img)).to(text_embeddings.dtype)
        depth_map = rearrange(depth_map,'H W -> 1 1 H W')
        depth_map = F.interpolate(
            depth_map,
            size=(height // 8, width // 8),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0
        depth_map = torch.cat([depth_map] * 2)
        depth_map = depth_map.to(self.device)

        #! 原本的

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels-1, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = torch.cat([latent_model_input, depth_map], dim=1)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None, bev=""):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts) # [1, 77, 768]
        neg_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(
            text_embeds, height=height, 
            width=width, latents=latents, 
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale,
            bev=bev
        ) # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':

    '''

    CUDA_VISIBLE_DEVICES=4 \
    python guidance/sd_utils.py \
    --prompt "a landscape,hdr,masterpiece,64k" \
    --bev heightmap
        
    '''

    import argparse
    import os
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default="stabilityai/stable-diffusion-2-depth", help="hugging face Stable diffusion model key")
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1210)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--bev', type=str,default="simple")
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    os.makedirs(f"sd_sample/{opt.bev}", exist_ok=True)

    sd = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps, bev=opt.bev)

    # visualize image
    plt.imshow(imgs[0])
    plt.imsave(f"sd_sample/{opt.bev}/sample_img.png",imgs[0])




