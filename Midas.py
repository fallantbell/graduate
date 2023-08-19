import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

model_type = "DPT_Large"
device = torch.device("cuda")

model = torch.hub.load("intel-isl/MiDaS", model_type)
transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
    
def get_depth(img):
    model.to(device)
    model.eval()

    img = np.array(img)

    input_batch = transforms(img).to(device)

    with torch.no_grad():
        prediction = model(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    return prediction

