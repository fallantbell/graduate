import numpy as np
from PIL import Image

# 图像大小
image_size = 256

# 中心位置
center_x, center_y = image_size // 2, image_size // 2

# 方差（控制分布的扩散程度）
variance = 40.0
x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))
distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
gaussian_image = np.exp(-0.5 * (distance / variance)**2)
gaussian_image = gaussian_image / np.max(gaussian_image)
gaussian_image = (gaussian_image * 100).astype(np.uint8)+30
tl = np.clip(gaussian_image, 0, 130)

variance = 60.0
x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))
distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
gaussian_image = np.exp(-0.5 * (distance / variance)**2)
gaussian_image = gaussian_image / np.max(gaussian_image)
gaussian_image = (gaussian_image * 120).astype(np.uint8)+30
tr = np.clip(gaussian_image, 0, 150)

variance = 50.0
x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))
distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
gaussian_image = np.exp(-0.5 * (distance / variance)**2)
gaussian_image = gaussian_image / np.max(gaussian_image)
gaussian_image = (gaussian_image * 150).astype(np.uint8)+30
bl = np.clip(gaussian_image, 0, 180)

variance = 40.0
x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))
distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
gaussian_image = np.exp(-0.5 * (distance / variance)**2)
gaussian_image = gaussian_image / np.max(gaussian_image)
gaussian_image = (gaussian_image * 100).astype(np.uint8)+30
br = np.clip(gaussian_image, 0, 130)

full_image = np.zeros((512, 512), dtype=np.uint8)
# full_image[:image_size, :image_size] = tl
# full_image[:image_size, -image_size:] = tr
# full_image[-image_size:, :image_size] = bl
# full_image[-image_size:, -image_size:] = br
full_image[128+50:128*3-50, 128+50:128*3-50] = 150


# 使用Pillow库保存图像
pil_image = Image.fromarray(full_image, 'L')
pil_image.save('BEV/tree.png')

# 显示图像（可选）
# pil_image.show()
