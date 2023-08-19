import imageio
import os

# 资料夹路径和文件名前缀
folder_path = 'trial/trial_simple/validation'  # 替换为实际的文件夹路径
file_prefix = '000'  # 文件名前缀，如'0000.png'的前缀为'000'

# 视频参数
output_video_path = 'output_video_simple.mp4'
fps = 30.0
frame_size = (800, 800)  # 图像大小，请根据实际情况修改

# 生成图像路径列表
image_paths = [os.path.join(folder_path, f"df_ep{i+1:04d}_0002_depth.png") for i in range(81)]

# 读取图像并合并成视频
images = [imageio.imread(path) for path in image_paths]

# 生成视频
imageio.mimsave(output_video_path, images, 'mp4', fps=fps)

