# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 02:54:26 2024

@author: CCC-PC
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 16:12:43 2023

@author: CCC-PC
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import tensorflow as tf
from PIL import Image, ImageFile
import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision
from torch.utils.tensorboard import SummaryWriter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device:
    print('Model is running on GPU:', device)
else:
    print('Model is running on CPU')

class InpaintingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = os.listdir(image_dir)
        self.mask_list = os.listdir(mask_dir)
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        # Ensure the lists are sorted so they correspond
        self.image_list.sort()
        self.mask_list.sort()

    def __len__(self):
        return len(self.image_list)
    
    def dilate_mask(self, mask, dilation_kernel_size=3):
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        dilated_mask = cv2.dilate(mask.numpy(), kernel, iterations=1)
        return torch.from_numpy(dilated_mask)

    def create_weight_map(self, mask, dilated_mask, border_weight=2.0):
        border = dilated_mask - mask
        weight_map = torch.ones_like(mask)
        weight_map[border == 1] = border_weight
        return weight_map  # Add this line


    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('1')

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Ensure mask is a binary tensor with the same size as image in the channel dimension
        mask = mask.expand_as(image)

        masked_image = image * (1 - mask)
        
        mask = (mask > 0).float()

        masked_image = image * (1 - mask)

        dilated_mask = self.dilate_mask(mask)
        weight_map = self.create_weight_map(mask, dilated_mask)
        
        
        
        weighted_masked_image = masked_image * weight_map

        return {
            'ground_truth': image, 
            'weighted_masked_image': weighted_masked_image, 
            'mask': dilated_mask
            }

image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

# Create dataset instances
train_dataset = InpaintingDataset(
    image_dir='E:/1807033/Dataset/TEMPTRAINIMG',
    mask_dir='E:/1807033/Dataset/TEMPTRAINMASK',
    # image_dir='E:/1807033/Dataset/sub_train_256',
    # mask_dir='E:/1807033/Dataset/mask/mask/training',
    image_transform=image_transform,
    mask_transform=mask_transform
)

val_dataset = InpaintingDataset(
    # image_dir='E:/1807033/Dataset/val_256/val_256',
    # mask_dir='E:/1807033/Dataset/mask/mask/validation',
    image_dir='E:/1807033/Dataset/TEMPVALIMG',
    mask_dir='E:/1807033/Dataset/TEMPVALMASK',
    image_transform=image_transform,
    mask_transform=mask_transform
)

from PIL import Image
import os


train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)



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
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)

        self.conv = nn.Sequential(
            nn.BatchNorm2d(out_size + skip_channels),  
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


        self.up1 = UNetUp(in_size=1024, out_size=512, skip_channels=512, dropout=0.5)
        self.up2 = UNetUp(in_size=1024, out_size=256, skip_channels=256, dropout=0.5)  
        self.up3 = UNetUp(in_size=512, out_size=128, skip_channels=128, dropout=0.0)   
        self.up4 = UNetUp(in_size=256, out_size=64, skip_channels=64, dropout=0.0)     
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),  
            nn.Tanh()
        )

    def forward(self, x, mask):
        # Downsampling
        
        
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        
        u1 = self.up1(d5, d4)
        #print(u1.size(), d3.size())  
        u2 = self.up2(u1, d3)  
        u3 = self.up3(u2, d2)  
        u4 = self.up4(u3, d1)

        u4 = self.final(u4)  # Generated part
        return u4 * mask + x * (1 - mask)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(3, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device:
    print('Model is running on GPU:', device)
else:
    print('Model is running on CPU')

generator = UNetGenerator().to(device)
discriminator = Discriminator().to(device)

generator_path = 'E:/1807033/Generator/generator_43_300.pth'
discriminator_path = 'E:/1807033/Discriminator/discriminator_43_300.pth'

generator.load_state_dict(torch.load(generator_path))
discriminator.load_state_dict(torch.load(discriminator_path))


optimizer_G = torch.optim.Adam(generator.parameters(), lr=3e-4, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=3e-4, betas=(0.5, 0.999))

criterion_GAN = nn.BCEWithLogitsLoss()
criterion_pixelwise = nn.L1Loss()
save_interval=100
print_interval=20
num_epochs = 100  # Set the number of epochs
lambda_pixel = 100  # Weight for pixel-wise loss relative to the adversarial loss



def psnr(target, prediction, max_pixel=1.0):
    mse = torch.mean((target - prediction) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))

def validate_model(generator,val_dataloader, criterion_pixelwise, psnr, device, metrics_file):
    generator.eval()  # Set the generator to evaluation mode
    discriminator.eval()
    total_l1_loss = 0.0
    total_l2_loss = 0.0
    total_psnr = 0.0
    total_tv_loss = 0.0
    step = 0
    with torch.no_grad():  # Disable gradient computation during validation
        for i, batch in enumerate(val_dataloader):
            if batch is None:
                continue

            real_images_val = batch['ground_truth'].to(device)
            masked_images_val = batch['weighted_masked_image'].to(device)
            masks_val = batch['mask'].to(device)
            
            
            
            # Generate fake images
            fake_images_val = generator(masked_images_val, masks_val)

            # Compute L1 loss
            l1_loss_val = criterion_pixelwise(fake_images_val, real_images_val).item()
            total_l1_loss += l1_loss_val

            # Compute L2 loss
            l2_loss_val = torch.mean((fake_images_val - real_images_val)**2).item()
            total_l2_loss += l2_loss_val

            # Compute PSNR
            psnr_val = psnr(real_images_val, fake_images_val).item()
            total_psnr += psnr_val

            # Compute TV loss

            # Optionally, you can save or visualize the generated images for inspection

    #calculate average metrics over the entire validation set
    avg_l1_loss = (total_l1_loss / len(val_dataloader)) * 100  # Express as percentage
    avg_l2_loss = (total_l2_loss / len(val_dataloader)) * 100  # Express as percentage
    avg_psnr = total_psnr / len(val_dataloader)

    print(f"[Validation] [Average L1 loss: {avg_l1_loss:.4f}%] "
          f"[Average L2 loss: {avg_l2_loss:.4f}%] [Average PSNR: {avg_psnr}]")
    
    metrics_file.write(f"[Validation] [Average L1 loss: {avg_l1_loss:.4f}%] "
                           f"[Average L2 loss: {avg_l2_loss:.4f}%] [Average PSNR: {avg_psnr}]\n")

    generator.train()
    discriminator.train()
    
start_epoch = 44  # Set the epoch where the pre-training left off
total_epochs = 100
accumulation_steps = 4
from torch.cuda.amp import GradScaler, autocast
validation_interval = 100
scaler = GradScaler()
with open('validation_metrics.txt', 'w') as validation_metrics_file:
    for epoch in range(start_epoch, total_epochs):
        for i, batch in enumerate(train_dataloader):
            if batch is None:
                continue
    
            real_images = batch['ground_truth'].to(device)
            masked_images = batch['weighted_masked_image'].to(device)
            masks = batch['mask'].to(device)
    
            valid_shape = (real_images.size(0), 1, 15, 15)
            valid = torch.ones(valid_shape, dtype=torch.float, device=device)
            fake = torch.zeros(valid_shape, dtype=torch.float, device=device)
    
            # Use autocast for mixed precision training
            with autocast():
                # Generate fake images
                fake_images = generator(masked_images, masks)
    
                # Compute loss for the generator
                pred_fake = discriminator(fake_images)
                gan_loss_fake = criterion_GAN(pred_fake, valid)
                pixel_loss = criterion_pixelwise(fake_images, real_images) * lambda_pixel
                g_loss = gan_loss_fake + pixel_loss
    
                # Compute loss for the discriminator
                pred_real = discriminator(real_images)
                pred_fake_detached = discriminator(fake_images.detach())
                real_loss = criterion_GAN(pred_real, valid)
                fake_loss = criterion_GAN(pred_fake_detached, fake)
                d_loss = (real_loss + fake_loss) / 2
                
                
    
            # Backpropagation
            scaler.scale(g_loss).backward()
            scaler.scale(d_loss).backward()
    
            # Only step and update if it's time, according to the gradient accumulation steps
            if i % accumulation_steps == 0:
                scaler.step(optimizer_G)
                scaler.step(optimizer_D)
                scaler.update()
                optimizer_G.zero_grad()
                optimizer_D.zero_grad()
            if i % save_interval == 0:
                generator_path = f'E:/1807033/Generator/generator_{epoch}_{i}.pth'
                discriminator_path = f'E:/1807033/Discriminator/discriminator_{epoch}_{i}.pth'
                torch.save(generator.state_dict(), generator_path)
                torch.save(discriminator.state_dict(), discriminator_path)
            if i % print_interval == 0:            
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(train_dataloader)}] "
                      f"[D loss: {d_loss.item()}] [G loss: {gan_loss_fake.item()}, pixel: {pixel_loss.item()}]")
            
            if i % validation_interval == 0 and i>0:
                validate_model(generator, val_dataloader, criterion_pixelwise, psnr, device,validation_metrics_file)
