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

def prepare_single_image(image_path, mask_path, image_transform, mask_transform, device):
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')  # Convert to single-channel image

    # Apply transformations
    if image_transform:
        image = image_transform(image)
    if mask_transform:
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
    image = image.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    dilated_mask = dilated_mask.unsqueeze(0).to(device)
    weight_map = weight_map.unsqueeze(0).to(device)

    # Create masked image
    masked_image = image*(1- mask)
    

    return masked_image*weight_map, mask, image

def inv_normalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    tensor = tensor * std.view(-1, 1, 1) + mean.view(-1, 1, 1)
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

def tensor_to_pil(tensor):

    tensor = inv_normalize(tensor)
    tensor = torch.clamp(tensor, 0.0, 1) 
    return transforms.ToPILImage()(tensor)

# Function to test the model on a single image and display/save the results
def test_single_image(generator, image_path, mask_path, image_transform, mask_transform, device):
    masked_image, mask, real_image = prepare_single_image(image_path, mask_path, image_transform, mask_transform, device)
    with torch.no_grad():
        inpainted_image = generator(masked_image,mask).to(device)

    masked_image = inv_normalize(masked_image)
    masked_img_pil =  transforms.ToPILImage()(masked_image.squeeze(0).cpu())
    inpainted_image = inv_normalize(inpainted_image)
    inpainted_img_pil = transforms.ToPILImage()(inpainted_image.squeeze(0))
    real_image=inv_normalize(real_image)
    real_img_pil =  transforms.ToPILImage()(real_image.squeeze(0).cpu())
    
    
    l1_loss = F.l1_loss(inpainted_image, real_image)
    l2_loss = F.mse_loss(inpainted_image, real_image)
    

    print("l1 loss:" ,l1_loss)
    print("l2 loss:", l2_loss)
    
    # Display the images
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(masked_img_pil)
    plt.title('Masked Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(inpainted_img_pil)
    plt.title('Inpainted Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(real_img_pil)
    plt.title('Real Image')
    plt.axis('off')
    plt.show()

    # Save the images
    #masked_img_pil.save('bird_m99.png')
    inpainted_img_pil.save('bird_inp99.png')
    #real_img_pil.save('bird_r99.png')
    

# Define transforms
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator_checkpoint = 'E:/Thesis_Stuffs/Models/Self/generator_99_300.pth' 
checkpoint = torch.load(generator_checkpoint, map_location=device)
generator = load_generator(generator_checkpoint, device)

image_path = 'E:/Thesis_Stuffs/Dataset/Mask/a.jpg'  # Replace with your test image path
mask_path = 'E:/Thesis_Stuffs/Dataset/Mask/b.png'    # Replace with your test mask path

    


test_single_image(generator, image_path, mask_path, image_transform, mask_transform, device)

image_path = 'E:/Thesis_Stuffs/Dataset/Mask/i1.png'  # Replace with your test image path
mask_path = 'E:/Thesis_Stuffs/Dataset/Mask/image_with_hole.png'    # Replace with your test mask path

    


# Test the model on a single image
test_single_image(generator, image_path, mask_path, image_transform, mask_transform, device)


image_path = 'E:/Thesis_Stuffs/Dataset/Mask/i1.png'  # Replace with your test image path
mask_path = 'E:/Thesis_Stuffs/Dataset/Mask/image_with_hole.png'    # Replace with your test mask path

    


# Test the model on a single image
test_single_image(generator, image_path, mask_path, image_transform, mask_transform, device)


image_path = 'E:/Thesis_Stuffs/Models/generative_inpainting-master/examples/places2/case1_raw.png'  # Replace with your test image path
mask_path = 'E:/Thesis_Stuffs/Models/generative_inpainting-master/examples/places2/case1_mask.png'    # Replace with your test mask path

    


# Test the model on a single image
test_single_image(generator, image_path, mask_path, image_transform, mask_transform, device)


image_path = 'E:/Thesis_Stuffs/Models/generative_inpainting-master/examples/places2/case4_raw.png'  # Replace with your test image path
mask_path = 'E:/Thesis_Stuffs/Models/generative_inpainting-master/examples/places2/case4_mask.png'    # Replace with your test mask path

    


# Test the model on a single image
test_single_image(generator, image_path, mask_path, image_transform, mask_transform, device)


image_path = 'E:/Thesis_Stuffs/Models/generative_inpainting-master/examples/places2/case2_raw.png'  # Replace with your test image path
mask_path = 'E:/Thesis_Stuffs/Models/generative_inpainting-master/examples/places2/case2_mask.png'    # Replace with your test mask path

    


# Test the model on a single image
test_single_image(generator, image_path, mask_path, image_transform, mask_transform, device)