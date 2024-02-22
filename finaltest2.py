# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:02:49 2024

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:49:30 2024

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 02:18:56 2024

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 00:01:58 2024

@author: user
"""

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
import glob
import random
from pytorch_msssim import ssim

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

def create_binary_mask_with_rectangles(width=256, height=256, max_rect_size=64, num_rects=10):
    # Create a black background
    mask = np.zeros((height, width), dtype=np.uint8)

    for _ in range(num_rects):
        # Randomly choose rectangle size
        rect_width = np.random.randint(1, max_rect_size + 1)
        rect_height = np.random.randint(1, max_rect_size + 1)

        # Randomly choose the position of the rectangle
        x_start = np.random.randint(0, width - rect_width)
        y_start = np.random.randint(0, height - rect_height)

        # Draw the rectangle on the mask
        mask[y_start:y_start+rect_height, x_start:x_start+rect_width] = 1

    return mask

def prepare_single_image(image_path, image_transform, mask_transform, device):
    image = Image.open(image_path).convert('RGB')
    
    mask = create_binary_mask_with_rectangles(256, 256, 128, 1)  # Convert to single-channel image
    
    # Convert mask from NumPy array to PIL Image
    mask = Image.fromarray(mask * 255)
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



def tensor_to_pil(tensor):
# Undo the normalization
    inv_normalize = transforms.Normalize(
        mean=[-0.425 / 0.53, -0.426 / 0.53, -0.426 / 0.53],
        std=[1 / 0.555, 1 / 0.555, 1 / 0.555]
        # mean=[-0.425 / 0.589, -0.426 / 0.584, -0.426 / 0.585],
        # std=[1/ 0.559, 1 / 0.554, 1/ 0.555]
        # mean=[-0.405 / 0.559, -0.406 / 0.554, -0.406 / 0.555],
        # std=[1 / 0.589, 1 / 0.584, 1 / 0.585]
        
        # mean=[-0.425 / 0.509, -0.426 / 0.504, -0.426 / 0.505],
        # std=[1 / 0.559, 1 / 0.554, 1 / 0.557]
        
        # mean=[-0.405 / 0.509, -0.406 / 0.504, -0.406 / 0.505],
        # std=[1 / 0.489, 1 / 0.484, 1 / 0.485]
        
        )
    tensor = inv_normalize(tensor)
    tensor = torch.clamp(tensor, 0.0, 1) 
    return transforms.ToPILImage()(tensor)

criterion_pixelwise = nn.L1Loss()
def psnr(target, prediction, max_pixel=1.0):
    mse = torch.mean((target - prediction) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))
# This function saves the masked and inpainted images, and returns paths for logging
def test_single_image_modified(generator, image_path, output_dir, device, image_transform, mask_transform):
    masked_image, mask, real_image = prepare_single_image(image_path,  image_transform, mask_transform, device)
    
    # Generate the inpainted image
    with torch.no_grad():
        inpainted_image = generator(masked_image, mask).to(device)
        inpainted_image = (inpainted_image * mask + real_image * (1 - mask)).squeeze(0)
    # Add a batch dimension to inpainted_image
    inpainted_image = inpainted_image.unsqueeze(0)  # This line is added
    
    inpainted_image = inpainted_image.to(device).float()
    real_image = real_image.to(device).float()
    
    # Calculate SSIM
    ssim_val = ssim(inpainted_image, real_image, data_range=1.0, size_average=True)
    
    l1_loss_val = F.l1_loss(inpainted_image, real_image)
    l2_loss_val = F.mse_loss(inpainted_image, real_image)
    psnr_val = psnr(inpainted_image, real_image)

    masked_img_pil = tensor_to_pil(masked_image.squeeze(0).cpu())
    inpainted_img_pil = tensor_to_pil(inpainted_image.squeeze(0).cpu())
    inpainted_img_np = np.array(inpainted_img_pil)
    
    
    
    # Convert RGB to BGR for OpenCV
    inpainted_img_np = inpainted_img_np[:, :, ::-1]
    
    # Apply a Gaussian blur to the inpainted image
    blurred = cv2.GaussianBlur(inpainted_img_np, (0, 0), 3)
    
    # Subtract the blurred image from the original (sharpening effect)
    sharpened = cv2.addWeighted(inpainted_img_np, 1.5, blurred, -0.5, 0)
    
    # Convert BGR to RGB for consistency
    sharpened = sharpened[:, :, ::-1]
    
    # Convert the numpy array back to PIL Image
    sharpened_img_pil = Image.fromarray(sharpened)
    # Make sure both images are on the same device and have the same datatype
    
    # Construct file paths
    base_filename = os.path.basename(image_path)
    output_image_path = os.path.join(output_dir, f"inpainted_{base_filename}")
    output_masked_image_path = os.path.join(output_dir, f"masked_{base_filename}")
    
    # Save images
    masked_img_pil.save(output_masked_image_path)
    sharpened_img_pil.save(output_image_path)
    
    return output_image_path, output_masked_image_path, l1_loss_val,l2_loss_val,psnr_val, ssim_val

def process_test_set(generator, images_dir, masks_dir, output_dir, log_file_path, device, image_transform, mask_transform):
    image_files = glob.glob(os.path.join(images_dir, '*.jpg'))  # Adjust the pattern as needed
    mask_files = glob.glob(os.path.join(masks_dir, '*.png'))  # Adjust the pattern as needed
    total_l1_loss = 0.0
    total_l2_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    if not mask_files:  # Check if mask_files is empty
        print(f"No mask files found in {masks_dir}. Please check the directory path and file type.")
        return 
    with open(log_file_path, 'w') as log_file:
        for i in range(200):
            image_path = random.choice(image_files)
            #mask_path = random.choice(mask_files)  # Randomly select a mask
            output_image_path, output_masked_image_path,l1_loss_val, l2_loss_val, psnr_val, ssim_val = test_single_image_modified(
                generator, image_path, output_dir, device, image_transform, mask_transform)
            
            total_l1_loss += l1_loss_val
            total_l2_loss += l2_loss_val
            total_psnr += psnr_val
            total_ssim += ssim_val
            
            log_file.write(f"{os.path.basename(image_path)}, {output_image_path}, "
                           f"{output_masked_image_path}, L1: {l1_loss_val:.4f}, L2: {l2_loss_val:.4f}, "
                           f"PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}\n")
            
    print(f"Processed {len(image_files)} images. Results and logs saved to {output_dir} and {log_file_path}.")
    avg_l1_loss = (total_l1_loss / 500) * 100  # Express as percentage
    avg_l2_loss = (total_l2_loss / 500) * 100  # Express as percentage
    avg_psnr = total_psnr / 500
    avg_ssim = total_ssim/500
    
    print(f"[Validation] [Average L1 loss: {avg_l1_loss:.4f}%] "
          f"[Average L2 loss: {avg_l2_loss:.4f}%] [Average PSNR: {avg_psnr}]"
          f"[Average SSIM: {avg_ssim: .4f}%]")
    
    
# Define transforms
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.505, 0.505, 0.505], std=[0.555, 0.555, 0.555]),
    
    # transforms.Normalize(mean=[0.484, 0.483, 0.486], std=[0.559, 0.554, 0.555]),
    # transforms.Normalize(mean=[0.554, 0.553, 0.556], std=[0.459, 0.454, 0.455]),
    #transforms.Normalize(mean=[0.454, 0.453, 0.456], std=[0.509, 0.504, 0.505]),
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the generator model
generator_checkpoint = 'E:/Thesis_Stuffs/Models/Self/generator_23_1700.pth' 
checkpoint = torch.load(generator_checkpoint, map_location=device)
 # Replace with your checkpoint path
generator = load_generator(generator_checkpoint, device)

# Define directories
images_dir = 'E:/Thesis_Stuffs/Dataset/Mask/testingimg'
masks_dir = 'E:/Thesis_Stuffs/Dataset/Mask/testingmask'
output_dir = 'E:/Thesis_Stuffs/Dataset/Mask/output6'
log_file_path = 'E:/Thesis_Stuffs/Dataset/Mask/output6.txt'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process the test set
process_test_set(generator, images_dir, masks_dir, output_dir, log_file_path, device, image_transform, mask_transform)