# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 01:47:46 2024

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 20:12:28 2024

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

ImageFile.LOAD_TRUNCATED_IMAGES = True
# Your UNetGenerator definition here
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
    def __init__(self, in_size, out_size, skip_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        # Use in_size for the transposed convolution, which does not yet include skip_input
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)

        # Initialize batch norm with out_size, which is the size after upsampling but before concatenating with skip_input
        self.conv = nn.Sequential(
            nn.BatchNorm2d(out_size + skip_channels),  # Notice the addition here
            nn.ReLU(inplace=True)
        )
        if dropout:
            self.conv.add_module("dropout", nn.Dropout(dropout))

    def forward(self, x, skip_input):
        x = self.up(x)
        x = torch.cat((x, skip_input), 1)  # Concatenate along the channel dimension
        x = self.conv(x)
        return x


class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()
# Downsampling
        self.down1 = UNetDown(3, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 1024)

        # Upsampling
        # The in_size for each upsampling layer is the sum of the out_size of the corresponding downsampling layer
        # and the out_size of the previous upsampling layer (which is the in_size for this layer).
        self.up1 = UNetUp(in_size=1024, out_size=512, skip_channels=512, dropout=0.5)
        self.up2 = UNetUp(in_size=1024, out_size=256, skip_channels=256, dropout=0.5)  # Corrected in_size (512 from up1 + 512 from down4)
        self.up3 = UNetUp(in_size=512, out_size=128, skip_channels=128, dropout=0.0)   # Corrected in_size (256 from up2 + 256 from down3)
        self.up4 = UNetUp(in_size=256, out_size=64, skip_channels=64, dropout=0.0)     # Corrected in_size (128 from up3 + 128 from down2)

        # Final layer to output 3 channel image
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),  # Corrected to match the channels (64 from up4 + 64 from down1)
            nn.Tanh()
        )

    def forward(self, x,mask):
        # Downsampling
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        
        u1 = self.up1(d5, d4)
        #print(u1.size(), d3.size())  # Add this to print tensor sizes before concatenation
        u2 = self.up2(u1, d3)  
        u3 = self.up3(u2, d2)  
        u4 = self.up4(u3, d1)

        u4 = self.final(u4)  # Generated part
        return u4 * mask + x * (1 - mask)


# Function to load the trained generator model
def load_generator(filepath, device):
    model = UNetGenerator().to(device)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()
    return model


def dilate_mask(mask, dilation_kernel_size=3):
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        dilated_mask = cv2.dilate(mask.numpy(), kernel, iterations=1)
        return torch.from_numpy(dilated_mask)

# def prepare_single_image(image_path, mask_path, image_transform, mask_transform, device):
#     image = Image.open(image_path).convert('RGB')
#     mask = Image.open(mask_path).convert('L')  # Convert to single-channel image

#     # Apply transformations
#     if image_transform:
#         image = image_transform(image)
#     if mask_transform:
#         mask = mask_transform(mask)

#     # Ensure mask is a binary tensor with the same size as image in the channel dimension
#     mask = mask.expand_as(image)

    
#     dilated_mask = dilate_mask(mask,3)

#     # Create a weight map
#     border_weight =2.0
#     border = dilated_mask - mask
#     weight_map = torch.ones_like(mask)
#     weight_map[border == 1] = border_weight

#     # Convert to PyTorch tensors
#     image = image.unsqueeze(0).to(device)
#     mask = mask.unsqueeze(0).to(device)
#     dilated_mask = dilated_mask.unsqueeze(0).to(device)
#     weight_map = weight_map.unsqueeze(0).to(device)

#     # Create masked image
#     masked_image = image*(1- mask)

#     return masked_image*weight_map


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
    
    # Ensure mask is a binary tensor with the same size as image in the channel dimension
    mask = mask.expand_as(image)

    
    dilated_mask = dilate_mask(mask,3)

    # Create a weight map
    border_weight =2.0
    border = dilated_mask - mask
    weight_map = torch.ones_like(mask)
    weight_map[border == 1] = border_weight

    # Convert to PyTorch tensors
    image = image.unsqueeze(0)
    mask = mask.unsqueeze(0)
    dilated_mask = dilated_mask.unsqueeze(0)
    weight_map = weight_map.unsqueeze(0)

    # Create masked image
    masked_image = image*(1- mask)

    return masked_image*weight_map, dilated_mask


# Inpaint the masked image
def inpaint_image(model, image, mask, device):
    mask = (mask > 0).float()
    masked_image = image * (1 - mask)  # Apply mask to image

    with torch.no_grad():  # Inference mode
        inpainted_image = model(masked_image.to(device), mask.to(device))

    inpainted_image = inpainted_image.squeeze(0).cpu()
    inpainted_image = (inpainted_image * mask + image * (1 - mask)).squeeze(0)# Remove batch dimension correctly
    return inpainted_image

# Display or save the inpainted image
def display_image(image_tensor):
    # Define the mean and std dev used for the initial normalization
    mean = np.array([0.555, 0.576, 0.556])
    std = np.array([0.259, 0.254, 0.255])
    

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


model_path = 'E:/Thesis_Stuffs/Models/Self/generator_99_300.pth'
image_path = 'E:/Thesis_Stuffs/Dataset/Mask/i1.png'  # Replace with your test image path
mask_path = 'E:/Thesis_Stuffs/Dataset/Mask/image_with_hole.png'

test_inpainting(model_path, image_path, mask_path)