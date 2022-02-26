import urllib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.nn as nn
import torchvision
import json

from torchvision import transforms
from PIL import Image


import coremltools as ct

import time
import matplotlib.pyplot as plt

import os
from multiprocessing import cpu_count

cpu_num = cpu_count() # 自动获取最大核心数目
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

print ("cpu_num ", cpu_num)
# Load the model (deeplabv3)
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True).eval()
# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True).eval()


# model = torch.load("/Users/jin/Third_Party_Packages/DeepLabV3Plus-Pytorch/best_deeplabv3plus_mobilenet_voc_os16.pth", map_location='cpu') # no working
# model.eval()

preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
 ])


input_image = Image.open("/Users/jin/Desktop/test/tum_test.png")
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

with torch.no_grad():
    output = model(input_batch)['out'][0]
torch_predictions = output.argmax(0)


def display_segmentation(input_image, output_predictions):
    # Create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # Plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(
        output_predictions.byte().cpu().numpy()
    ).resize(input_image.size)
    r.putpalette(colors)

    # Overlay the segmentation mask on the original image
    alpha_image = input_image.copy()
    alpha_image.putalpha(255)
    r = r.convert("RGBA")
    r.putalpha(128)
    seg_image = Image.alpha_composite(alpha_image, r)
    # display(seg_image) -- doesn't work
    seg_image.show()

display_segmentation(input_image, torch_predictions)