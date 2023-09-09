
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import imageio
from einops import rearrange

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRendererWithFragments,
    MeshRasterizer,
    TexturesVertex,
    look_at_view_transform,
    FoVPerspectiveCameras
)
from pytorch3d.renderer.mesh.shader import HardDepthShader, ShaderBase, BlendParams, HardPhongShader
from pytorch3d.renderer.blending import hard_rgb_blend, softmax_rgb_blend

# from nerf.provider import circle_poses, visualize_poses


class VertexColorShader(ShaderBase):
    def __init__(self, blend_soft=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.blend_soft = blend_soft

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        if self.blend_soft:
            return softmax_rgb_blend(texels, fragments, blend_params)
        else:
            return hard_rgb_blend(texels, fragments, blend_params)

class Terrain:
    def __init__(self,opt):
        self.opt = opt
        self.device = 'cuda'

        height_map = np.array(Image.open(self.opt.heightmap_path))

        height, width = height_map.shape

        x, y = np.meshgrid(range(width), range(height))

        z = height_map[:, :]

        # 將 x, y 和 z 座標縮放到 (-1, 1) 範圍內
        x = 2.0 * (x / (width - 1)) - 1.0
        y = 2.0 * (y / (height - 1)) - 1.0
        z = 2.0 * (z / 255.0) - 1.0  # 將 z 座標正規化至 (-1, 1)
        # z[z<0] = 0 #* 給定一個地板

        vertices = np.column_stack((x.flatten(), z.flatten(), y.flatten()))

        faces = []
        for i in range(height - 1):
            for j in range(width - 1):
                idx = i * width + j
                faces.append([idx, idx + width, idx + 1])
                faces.append([idx + 1, idx + width, idx + width + 1])
        faces = np.array(faces)

        color = []
        for i in range(height):
            for j in range(width):
                if i<height/2 and j<width/2:
                    color.append([0.1,0.3,0.5])
                elif i>=height/2 and j<width/2:
                    color.append([0.6,0.4,0.2])
                elif i<height/2 and j>=width/2:
                    color.append([0.8,0.8,0.3])
                else:
                    color.append([0.8,0.3,0.7])
        colors = np.array(color)
        colors = torch.tensor(colors, dtype=torch.float32)

        self.vertices = torch.tensor(vertices, dtype=torch.float32)
        self.faces = torch.tensor(faces, dtype=torch.int64)

        # 為每個頂點設置相同的顏色
        # num_vertices = len(vertices)
        # colors = torch.ones(num_vertices, 3, dtype=torch.float32, device=self.device) * torch.tensor([0.8, 0.6, 0.4], device=self.device)

        # 創建 TextureVertex 物件
        textures = TexturesVertex(verts_features=[colors])

        self.mesh = Meshes(verts=[self.vertices], faces=[self.faces])

        # 更新 mesh 的 texture 屬性
        self.mesh.textures = textures


        
    def render_from_camera(self,camera,H,W,blur_radius=0,faces_per_pixel=1):

        blend_params = BlendParams(1e-4, 1e-4, (0, 0, 0))

        raster_settings = RasterizationSettings(
            image_size=(H, W),
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel,
            clip_barycentric_coords=True
        )

        renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=camera,
                raster_settings=raster_settings
            ),
            shader=VertexColorShader(
                blend_soft=False,
                device=self.device,
                cameras=camera,
                blend_params=blend_params
            )
        )

        # renderer.shader.blend_params.background_color = torch.zeros_like(vertex_features[:,0])

        # Create a depth shader
        depth_shader = HardDepthShader(device=self.device, cameras=camera)

        # render RGB and depth, get mask
        mesh = self.mesh.to(self.device)
        images, fragments = renderer(mesh)
        mask = (fragments.pix_to_face[..., 0] < 0).squeeze()
        depth = depth_shader(fragments, mesh).squeeze()
        depth[mask] = 0

        return images[0].permute(2, 0, 1), depth, mask, fragments.pix_to_face, fragments.zbuf

if __name__ == '__main__':
    import argparse
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--bev', type=str,default='simple')
    parser.add_argument('--heightmap_path', type=str,default='')
    opt = parser.parse_args()

    opt.heightmap_path = f"BEV/{opt.bev}.png"
    terrain = Terrain(opt)

    disparity_path = f"terrain_render/{opt.bev}/disparity"
    rgb_path = f"terrain_render/{opt.bev}/rgb"

    os.makedirs(disparity_path, exist_ok=True)
    os.makedirs(rgb_path, exist_ok=True)

    size = 20
    device = 'cuda'

    all_depth = []
    tilt = 30

    for i in range(size):
        # circle pose
        thetas = torch.FloatTensor([90-tilt]).to(device)
        phis = torch.FloatTensor([(i/size)*360]).to(device)
        radius = torch.FloatTensor([2.2]).to(device)

        if phis>180:
            phis = -(360-phis)

        R2, T2 = look_at_view_transform(dist=2.2, elev=0+tilt, azim=phis)

        camera = FoVPerspectiveCameras(R=R2, T=T2, fov=20 , device='cuda')


        # 進行渲染
        H, W = 512, 512  # 定義渲染的圖像尺寸
        image, depth, mask, _, _ = terrain.render_from_camera(camera, H, W)

        #* 配合midas 將深度inverse，除了0(背景以外)都做 1/depth
        non_zero_mask = (depth != 0)
        depth[non_zero_mask] = 1.0 / depth[non_zero_mask]

        # 只保留RGB顏色通道並轉換形狀
        image_np = image[:3,...].cpu().numpy()
        image_np = image_np.transpose(1, 2, 0)

        plt.imshow(image_np)
        plt.imsave(f"{rgb_path}/{i}.png",image_np)

        # 儲存深度圖像
        depth_np = depth.cpu().numpy()
        
        depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-6)
        depth_np = (depth_np * 255).astype(np.uint8)
        all_depth.append(depth_np)
        depth_image = Image.fromarray(depth_np, mode='L')
        depth_image.save(f"{disparity_path}/{i}.png")
    
    all_depth = np.stack(all_depth, axis=0)
    imageio.mimwrite(f"{disparity_path}/video.mp4", all_depth, fps=10, quality=8, macro_block_size=1)