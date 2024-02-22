# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 01:38:05 2024

@author: user
"""

import torch
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch.nn as nn


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, skip_channels, dilation, dropout=0.0):
        super(UNetUp, self).__init__()
        # Adjust the input channels for the transposed convolution
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(out_size + skip_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size + skip_channels, out_size, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)
        )
        if dropout:
            self.conv.add_module("dropout", nn.Dropout(dropout))

    def forward(self, x, skip_input):
        x = self.up(x)
        x = torch.cat((x, skip_input), 1)
        x = self.conv(x)
        return x


class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        self.down1 = UNetDown(3, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 1024)
        self.up1 = UNetUp(in_size=1024, out_size=512, skip_channels=512, dilation=2, dropout=0.5)
        self.up2 = UNetUp(in_size=512, out_size=256, skip_channels=256, dilation=2, dropout=0.5)
        self.up3 = UNetUp(in_size=256, out_size=128, skip_channels=128, dilation=2, dropout=0.0)
        self.up4 = UNetUp(in_size=128, out_size=64, skip_channels=64, dilation=2, dropout=0.0)
        # Adjust the number of output channels in the final layer to match your input image
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x, mask):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        u4 = self.final(u4)
        return u4 * mask + x * (1 - mask)
# Load the trained generator model
def load_generator(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetGenerator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Prepare the input image and mask
def prepare_input(image_path, mask_path, image_size=(256, 256)):
    img_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    mask_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('1')

    image = img_transform(image)
    mask = mask_transform(mask)

    return image, mask

# Inpaint the masked image
def inpaint_image(model, image, mask, device):
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    mask = mask.unsqueeze(0).to(device)  # Add batch dimension and ensure binary mask
    mask = (mask > 0).float()
    masked_image = image * (1 - mask)  # Apply mask to image

    with torch.no_grad():  # Inference mode
        inpainted_image = model(masked_image, mask)

    inpainted_image = inpainted_image.squeeze(0).cpu()  # Remove batch dimension correctly
    inpainted_image = (inpainted_image * mask + image * (1 - mask)).squeeze(0)  # Apply mask and remove batch dimension

    return inpainted_image

# Display or save the inpainted image
def display_image(image_tensor):
    # Define the mean and std dev used for the initial normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    

    # Inverse normalization of the image tensor
    for t, m, s in zip(image_tensor, mean, std):
        t.mul_(s).add_(m)    # Multiply by std and add the mean for each channel
    t = torch.clamp(t, 0.0, 1)  # Clamp the values to the range [0, 1] to avoid display issues
    
    unloader = transforms.ToPILImage()
    image = image_tensor.clone()  # Clone the tensor to not do changes on it
    image = unloader(image)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Main function to test the model
def test_inpainting(model_path, image_path, mask_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = load_generator(model_path)
    image, mask = prepare_input(image_path, mask_path)
    inpainted_image = inpaint_image(generator, image, mask, device)
    display_image(inpainted_image)

# Example usage
model_path = 'E:/Thesis_Stuffs/Models/Self/generator_20_300.pth'

image_path = 'E:/Thesis_Stuffs/Dataset/Mask/i1.png'  # Replace with your test image path
mask_path = 'E:/Thesis_Stuffs/Dataset/Mask/image_with_hole.png'    # Replace with your test mask path

    
    

test_inpainting(model_path, image_path, mask_path)
