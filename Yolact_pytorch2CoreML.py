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

# Load the model (deeplabv3)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True).eval()

# model = torch.load("/Users/jin/Third_Party_Packages/DeepLabV3Plus-Pytorch/best_deeplabv3plus_mobilenet_voc_os16.pth",map_location=torch.device('cpu')) # no working
# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True).eval()

input_image = Image.open("/Users/jin/Desktop/6099.png")
# input_image.show()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)


# Wrap the Model to Allow Tracing*
class WrappedYolact_mobilenetv2(nn.Module):
    
    def __init__(self):
        super(WrappedYolact_mobilenetv2, self).__init__()
        self.model  = torch.load("/Users/jin/Third_Party_Packages/yolact_cpu/weights/yolact_mobilenetv2_54_800000.pth",map_location=torch.device('cpu')) 
        # self.model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True).eval()
    
    def forward(self, x):
        res = self.model(x)
        x = res["out"]
        return x
        
# Trace the Wrapped Model
traceable_model = WrappedYolact_mobilenetv2().eval()
trace = torch.jit.trace(traceable_model, input_batch)

# Convert the model
mlmodel = ct.convert(
    trace,
    inputs=[ct.TensorType(name="input", shape=input_batch.shape)],
)

# Save the model without new metadata
mlmodel.save("YolactModel_no_metadata.mlmodel")

# # Load the saved model
# mlmodel = ct.models.MLModel("SegmentationModel_no_metadata.mlmodel")