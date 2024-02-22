# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 02:21:32 2024

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
from torch.optim.lr_scheduler import StepLR
from torchvision.models import vgg16
from torchvision.models.inception import inception_v3
import torchvision.transforms as transforms
from pytorch_msssim import ssim
from torch.nn.functional import l1_loss
from torch.nn.functional import mse_loss
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from torchvision.models import vgg16, inception_v3, VGG16_Weights, Inception_V3_Weights



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device:
    print('Model is running on GPU:', device)
else:
    print('Model is running on CPU')


class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []

        blocks.append(vgg16(pretrained=True).features[:4].eval())
        blocks.append(vgg16(pretrained=True).features[4:9].eval())
        blocks.append(vgg16(pretrained=True).features[9:16].eval())
        blocks.append(vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = nn.ModuleList(blocks)
        self.transform = nn.functional.interpolate
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)


    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += F.l1_loss(x, y)
        return loss

class InpaintingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None,mask_augmentations=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = os.listdir(image_dir)
        self.mask_list = os.listdir(mask_dir)
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.mask_augmentations = mask_augmentations

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
    
    def mask_augmentations(self, mask):
        if self.mask_augmentations:
            mask = self.mask_augmentations(mask)

        return mask

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('1')
        
        if self.mask_augmentations:
            mask = self.mask_augmentations(mask)
        
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

mask_augmentations = transforms.Compose([
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomRotation(degrees=30),
    # Add more mask augmentations as needed
])


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
    # image_dir='E:/1807033/Dataset/TEMPTRAINIMG',
    # mask_dir='E:/1807033/Dataset/TEMPTRAINMASK',
    image_dir='E:/1807033/Dataset/sub_train_256',
    mask_dir='E:/1807033/Dataset/mask/mask/training',
    image_transform=image_transform,
    mask_transform=mask_transform,
    mask_augmentations=mask_augmentations

)

val_dataset = InpaintingDataset(
    image_dir='E:/1807033/Dataset/val_256/val_256',
    mask_dir='E:/1807033/Dataset/mask/mask/validation',
    # image_dir='E:/1807033/Dataset/TEMPVALIMG',
    # mask_dir='E:/1807033/Dataset/TEMPVALMASK',
    image_transform=image_transform,
    mask_transform=mask_transform,
    mask_augmentations=mask_augmentations

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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            *self.discriminator_block(3, 64, normalization=False),
            *self.discriminator_block(64, 128),
            *self.discriminator_block(128, 256),
            *self.discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def discriminator_block(self, in_filters, out_filters, normalization=True):
        layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
        if normalization:
            layers.append(nn.BatchNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, img, return_features=False):
        features = []
        x = img
        for i, layer in enumerate(self.model):
            x = layer(x)
            if return_features and i < len(self.model) - 1:  # Exclude final layer from features
                features.append(x)
        return features if return_features else x


generator = UNetGenerator().to(device)
discriminator = Discriminator().to(device)

# generator_path = 'E:/1807033/Generator/generator_30_1700.pth'
# discriminator_path = 'E:/1807033/Discriminator/discriminator_30_1700.pth'

# generator.load_state_dict(torch.load(generator_path))
# discriminator.load_state_dict(torch.load(discriminator_path))




optimizer_G = torch.optim.Adam(generator.parameters(), lr=3e-4, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=3e-4, betas=(0.5, 0.999))

scheduler_G = StepLR(optimizer_G, step_size=30, gamma=0.1)
scheduler_D = StepLR(optimizer_D, step_size=30, gamma=0.1)

criterion_GAN = nn.BCEWithLogitsLoss()
criterion_pixelwise = nn.L1Loss()

# New perceptual loss
perceptual_loss_fn = VGGPerceptualLoss().to(device)
#perceptual_loss_fn = vgg16_model.to(device)

# Edge loss (sobel_operator function should be defined as shown in the previous message)
def sobel_operator(images):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(images.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(images.device)

    edges = torch.sqrt(F.conv2d(images[:, 0:1, :, :], sobel_x, padding=1) ** 2 +
                      F.conv2d(images[:, 0:1, :, :], sobel_y, padding=1) ** 2 +
                      F.conv2d(images[:, 1:2, :, :], sobel_x, padding=1) ** 2 +
                      F.conv2d(images[:, 1:2, :, :], sobel_y, padding=1) ** 2 +
                      F.conv2d(images[:, 2:3, :, :], sobel_x, padding=1) ** 2 +
                      F.conv2d(images[:, 2:3, :, :], sobel_y, padding=1) ** 2)

    return edges

# Weights for the losses
lambda_pixel = 100
lambda_feature = 0.5
lambda_edge = 1
lambda_perceptual = 0.1  # Weight for perceptual loss


save_interval=100
print_interval=20
num_epochs = 100  # Set the number of epochs

inception_model = inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
inception_model.eval()  # Set the model to evaluation mode
inception_model.to(device)

def psnr(target, prediction, max_pixel=1.0):
    mse = torch.mean((target - prediction) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))
def calculate_inception_score(batch, inception_model, num_classes=1000):
    """
    Calculate the Inception Score for a batch of images.
    
    Args:
    - batch (torch.Tensor): a batch of images (BxCxHxW) normalized in the range [0, 1].
    - inception_model (torch.nn.Module): pre-loaded Inception v3 model.
    - num_classes (int): number of classes, default is 1000 for ImageNet.
    
    Returns:
    - inception_score (float): The Inception Score for the batch.
    """
    # Ensure the model is in evaluation mode
    inception_model.eval()
    
    # Disable gradients
    with torch.no_grad():
        # If your images are not already resized to 299x299, you need to resize them
        batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Get model predictions
        preds = inception_model(batch)
        
        # Convert outputs to probabilities
        p_yx = F.softmax(preds, dim=1)
        
        # Compute the marginal distribution p(y)
        p_y = p_yx.mean(0).unsqueeze(0)  # Average over the batch to get the distribution
        
        # Compute the KL divergence for each image in the batch
        # Compute the KL divergence for each image in the batch
        kl_div = p_yx * (torch.log(p_yx) - torch.log(p_y))

        # Sum KL divergences over classes, and average over the batch
        kl_div = kl_div.sum(1).mean()

        # Compute the final score without converting kl_div to a float prematurely
        inception_score = torch.exp(-kl_div).item()

    
    return inception_score
def validate_model(generator, val_dataloader, device, metrics_file):
    generator.eval()
    total_l1_loss = 0.0
    total_l2_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_inception_score = 0.0
    num_batches = len(val_dataloader)

    with torch.no_grad():
        for batch in val_dataloader:
            real_images = batch['ground_truth'].to(device)
            masked_images = batch['weighted_masked_image'].to(device)
            masks = batch['mask'].to(device)
            fake_images = generator(masked_images, masks)

            # Compute L1 and L2 loss
            batch_l1_loss = l1_loss(fake_images, real_images, reduction='mean')
            batch_l2_loss = mse_loss(fake_images, real_images, reduction='mean')
            total_l1_loss += batch_l1_loss.item()
            total_l2_loss += batch_l2_loss.item()

            # Compute PSNR
            psnr_val = psnr(real_images, fake_images).item()
            total_psnr += psnr_val

            # Compute SSIM
            batch_ssim = ssim(fake_images, real_images, data_range=1, size_average=True)
            total_ssim += batch_ssim.item()

            # Compute Inception Score for the batch
            batch_inception_score = calculate_inception_score(fake_images, inception_model)
            total_inception_score += batch_inception_score

    # Calculate average metrics
    avg_l1_loss = total_l1_loss / num_batches * 100  # As a percentage
    avg_l2_loss = total_l2_loss / num_batches * 100  # As a percentage
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    avg_inception_score = total_inception_score / num_batches

    print(f"[Validation] [Average L1 loss: {avg_l1_loss:.2f}%] "
          f"[Average L2 loss: {avg_l2_loss:.2f}%] [Average PSNR: {avg_psnr:.2f}] "
          f"[Average SSIM: {avg_ssim:.4f}] [Average Inception Score: {avg_inception_score:.2f}]")

    # Write validation metrics to the file
    metrics_file.write(f"[Validation] [Average L1 loss: {avg_l1_loss:.2f}%] "
                       f"[Average L2 loss: {avg_l2_loss:.2f}%] [Average PSNR: {avg_psnr:.2f}] "
                       f"[Average SSIM: {avg_ssim:.4f}] [Average Inception Score: {avg_inception_score:.2f}]\n")

    generator.train()
    
start_epoch = 31  # Set the epoch where the pre-training left off
total_epochs = 100
accumulation_steps = 4
from torch.cuda.amp import GradScaler, autocast
validation_interval = 100
scaler = GradScaler()
with open('validation_metrics2.txt', 'w') as validation_metrics_file:
    for epoch in range(num_epochs):
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
                gan_loss_fake_global = criterion_GAN(pred_fake, valid)
    
                # # For local loss, you can select a local patch from the real and fake images
                # patch_size = 64
                # top_left_x = torch.randint(0, real_images.shape[2] - patch_size, size=(1,))
                # top_left_y = torch.randint(0, real_images.shape[3] - patch_size, size=(1,))
                # real_images_patch = real_images[:, :, top_left_x:top_left_x+patch_size, top_left_y:top_left_y+patch_size]
                # fake_images_patch = fake_images[:, :, top_left_x:top_left_x+patch_size, top_left_y:top_left_y+patch_size]
                
                # # Compute local loss for the generator
                # pred_fake_patch = discriminator(fake_images_patch)
                # valid_patch_shape = pred_fake_patch.shape
                # valid_patch = torch.ones(valid_patch_shape, dtype=torch.float, device=device)
                # gan_loss_fake_local = criterion_GAN(pred_fake_patch.view(-1), valid_patch.view(-1))
                
                
                #pixel loss L1
                pixel_loss = criterion_pixelwise(fake_images, real_images) * lambda_pixel
                
                # Compute edge loss
                real_edges = sobel_operator(real_images)
                fake_edges = sobel_operator(fake_images)
                edge_loss = F.l1_loss(fake_edges, real_edges) * lambda_edge

                # Compute perceptual loss
                perceptual_loss = perceptual_loss_fn(fake_images, real_images) * lambda_perceptual

                
                # Feature Matching Loss
                # Feature Matching Loss
                # real_features = discriminator(real_images, return_features=True)
                # fake_features = discriminator(fake_images, return_features=True)
                
                # feature_matching_loss = 0
                # for real_feature, fake_feature in zip(real_features, fake_features):
                #     if torch.isnan(real_feature).any() or torch.isinf(real_feature).any():
                #         print("NaN or inf value detected in real_feature")
                #     if torch.isnan(fake_feature).any() or torch.isinf(fake_feature).any():
                #         print("NaN or inf value detected in fake_feature")
                #     feature_matching_loss += F.l1_loss(fake_feature, real_feature.detach())
                # feature_matching_loss *= lambda_feature

                
                g_loss = (
                    gan_loss_fake_global + 
                    pixel_loss + 
                    #gan_loss_fake_local+
                    # feature_matching_loss + 
                    edge_loss + 
                    perceptual_loss
                )
    
                # Compute loss for the discriminator
                pred_real = discriminator(real_images)
                pred_fake_detached = discriminator(fake_images.detach())
                real_loss = criterion_GAN(pred_real, valid)
                fake_loss = criterion_GAN(pred_fake_detached, fake)
                d_loss = (real_loss + fake_loss) / 2
    
                # Backpropagationaaa
            
            
            scaler.scale(g_loss).backward()
            scaler.scale(d_loss).backward()
            
            
            # Only step and update if it's time, according to the gradient accumulation steps
            if i % accumulation_steps == 0:
                
                scaler.step(optimizer_D)
                scaler.step(optimizer_G)
                
                scaler.update()
                
                optimizer_G.zero_grad()
                optimizer_D.zero_grad()
                
                
            
            if i % save_interval == 0:
                generator_path = f'E:/1807033/Generator3/generator_{epoch}_{i}.pth'
                discriminator_path = f'E:/1807033/Discriminator3/discriminator_{epoch}_{i}.pth'
                torch.save(generator.state_dict(), generator_path)
                torch.save(discriminator.state_dict(), discriminator_path)
                print(f"Epoch {epoch}: Generator LR is {scheduler_G.get_last_lr()[0]}, Discriminator LR is {scheduler_D.get_last_lr()[0]}")

            if i % print_interval == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(train_dataloader)}] "
                      f"[D loss: {d_loss.item()}] "
                      f"[G loss: {g_loss.item()}] "
                      f"[GAN loss (global): {gan_loss_fake_global.item()}] "
                      f"[Pixel loss: {pixel_loss.item()}] "
                      f"[Edge loss: {edge_loss.item()}] "
                      f"[Perceptual loss: {perceptual_loss.item()}]")
            if i % validation_interval == 0 and i>0:
                validate_model(generator, val_dataloader,  device,validation_metrics_file)
    scheduler_G.step()  
    scheduler_D.step()
    