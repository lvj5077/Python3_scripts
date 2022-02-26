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



def people_segmentation(input_path,out_path1,out_path2):
    input_img= Image.open(input_path)
    input_tensor = preprocess((input_img))
    input_batch = input_tensor.unsqueeze(0)

    t = time.time()

    with torch.no_grad():
        output = model(input_batch)['out']

    elapsed = time.time() - t
    print (elapsed*1000,"ms")

    scores = torch.nn.functional.softmax(output, dim=1)
    output_predictions = scores[ : , 15, : , : ].argmax(0)

    # print( output_predictions.size())
    # # you can do this
    output_predictions = scores[ : , : , : ].argmax(1) == 15
    # # or, if you want to setup a threshold
    # output_predictions = scores[ : , 15, : , : ] > 0.5 
    output_predictions = output_predictions[0]
    # print( output_predictions.size())


    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * (palette)
    colors = (colors % 512).numpy().astype("uint8")
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_img.size)
    r.putpalette(colors)


    # f, ax = plt.subplots(1, 2)
    # ax[0].set_title('input image')
    # ax[0].axis('off')
    # ax[0].imshow(input_img)
    # ax[1].set_title('output')
    # ax[1].axis('off')
    # ax[1].imshow(r)
    # plt.show()

    # r.save(out_path1)

    display_segmentation(input_img, output_predictions,out_path2)



def display_segmentation(input_image, output_predictions,out_path):
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
    # seg_image.save(out_path)

# input_path = "/Users/jin/Desktop/SemanticSeg/3508.png"
# out_path = "/Users/jin/Desktop/SemanticSeg/3508_people.png"
# people_segmentation(input_path,out_path)

# display_segmentation(input_img, output_predictions)
base_path = "/Users/jin/Q_Mac/Local_Data/02_03_2022/costco/2022-02-03T10-48-16/"
for i in range(1, 8021):
    print(i/8021*100)
    input_path = base_path+ "color/" + str(i) + ".png"
    out_path1 = base_path+ "people_mask2/" + str(i) + ".png"
    out_path2 = base_path+ "blend_people2/" + str(i) + ".png"
    people_segmentation(input_path,out_path1,out_path2)

