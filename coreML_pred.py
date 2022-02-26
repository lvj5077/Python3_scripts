import coremltools as ct
import numpy as np
import PIL.Image

# Load a model whose input type is "Image".
model = ct.models.MLModel('/Users/jin/Q_Mac/Local_Data/mlmodel/SegmentationModel_with_metadata.mlmodel')

Height = 480  # use the correct input image height
Width = 640  # use the correct input image width

model_expected_input_shape = (1, 3, Height, Width) # depending on the model description, this could be (3, Height, Width)

# Scenario 1: load an image from disk.
def load_image(path, resize_to=None):
    # resize_to: (Width, Height)
    img = PIL.Image.open(path)
    if resize_to is not None:
        img = img.resize(resize_to, PIL.Image.ANTIALIAS)
    img_np = np.array(img).astype(np.float32)
    return img_np, img


# Load the image and resize using PIL utilities.
_, img = load_image('/Users/jin/Desktop/6099.png', resize_to=(Width, Height))
out_dict = model.predict({'image': img})

# Scenario 2: load an image from a NumPy array.
shape = (Height, Width, 3)  # height x width x RGB
data = np.zeros(shape, dtype=np.uint8)
# manipulate NumPy data
pil_img = PIL.Image.fromarray(data)
out_dict = model.predict({'image': pil_img})

print( out_dict )


def load_image_as_numpy_array(path, resize_to=None):
    # resize_to: (Width, Height)
    img = PIL.Image.open(path)
    if resize_to is not None:
        img = img.resize(resize_to, PIL.Image.ANTIALIAS)
    img_np = np.array(img).astype(np.float32) # shape of this numpy array is (Height, Width, 3)
    return img_np

# Load the image and resize using PIL utilities.
img_as_np_array = load_image_as_numpy_array('/Users/jin/Desktop/6099.png', resize_to=(Width, Height)) # shape (Height, Width, 3)

# PIL returns an image in the format in which the channel dimension is in the end,
# which is different than Core ML's input format, so that needs to be modified.
img_as_np_array = np.transpose(img_as_np_array, (2,0,1)) # shape (3, Height, Width)

# Add the batch dimension if the model description has it.
img_as_np_array = np.reshape(img_as_np_array, model_expected_input_shape)

# Now call predict.
out_dict = model.predict({'image': img_as_np_array})