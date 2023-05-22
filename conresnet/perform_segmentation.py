import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
from torch import nn
import glob
import torch.nn.functional as F
import segmentation_refinement as refine

model = torch.load('CFExp/model.pt', map_location=torch.device('cpu'))
model.eval()

file_names = glob.glob('testset' + '/*.png')
image_buffer_list = []
image_res_list = []

for i in range(len(file_names) - 35):
    image_buffer = np.zeros((256, 256, 8), dtype=np.float32)
    k = 0
    for j in range(i, 40 + i, 5):
        file_name = file_root + "{:04d}".format(j) + '.png'
        img = cv2.imread(file_name)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_buffer[:, :, k] = img
        k = k + 1

    image_copy = np.zeros((256, 256, 8)).astype(np.float32)
    image_copy[:, :, 1:] = image_buffer[:, :, 0:8 - 1]
    image_res = image_buffer - image_copy
    image_res[:, :, 0] = 0
    image_res = np.abs(image_res)

    image_buffer_list.append(image_buffer)
    image_res_list.append(image_res)

preprocess = transforms.Compose([
    transforms.ToTensor(),
])

output_list = []
for i in range(len(image_buffer_list)):

    input_image = image_buffer_list[i]
    image_res = image_res_list[i]

    input_tensor = preprocess(input_image)
    image_res = preprocess(image_res)

    input_tensor = input_tensor.unsqueeze(0)
    image_res = image_res.unsqueeze(0)
    inputs = torch.cat((input_tensor, image_res), dim=0)
    input_batch = inputs.unsqueeze(0)

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)[0]
        output = output.squeeze(0)
        m = nn.Softmax(dim=0)
        output = m(output)

        output_predictions = (output[1] > 0.2).type(torch.uint8) * 255 + (output[2] > 0.5).type(torch.uint8) * 192 \
                             + (output[3] > 0.5).type(torch.uint8) * 128 + (output[4] > 0.5).type(torch.uint8) * 85 + (output[5] > 0.5).type(torch.uint8) * 64
    output_list.append(output_predictions)