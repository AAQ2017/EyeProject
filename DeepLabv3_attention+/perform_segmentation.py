import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch import nn
import glob
import os
import segmentation_refinement as refine

model = torch.load('CFExp/model.pt', map_location=torch.device('cpu'))
model.eval()
refiner = refine.Refiner(device='cuda:0')

file_path = "testset"
files = glob.glob(file_path + '/*.png')

for file_name in files:

    input_image = Image.open(file_name)
    input_image1 = np.array((input_image)).astype(np.uint8)  # change data format

    if len(input_image1.shape) == 2:
        input_image1 = cv2.cvtColor(input_image1, cv2.COLOR_GRAY2BGR)
    # input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    input_image = np.array(np.float32(input_image1))  # change data format

    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)[0]
        output = output.squeeze(0)
        m = nn.Softmax(dim=0)
        output = m(output)
        output_predictions = output.argmax(0)

    palette = torch.tensor([2 ** 13 - 1, 2 ** 15 - 1, 2 ** 15 - 1])
    colors = torch.as_tensor([i for i in range(4)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.shape[:2])
    r.putpalette(colors)

    output = np.uint8(r)
    output = np.where(output == 1, 255, output)
    output = np.where(output == 2, 128, output)
    output = np.where(output == 3, 85, output)
    output = np.where(output == 4, 72, output)
    output = np.where(output == 5, 64, output)
    output = np.uint8(output)

    cv2.imwrite(file_name, output)

