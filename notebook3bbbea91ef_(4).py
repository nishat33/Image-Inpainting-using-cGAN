# -*- coding: utf-8 -*-
"""notebook3bbbea91ef (4).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19fwGk5PpSKFYMQrBP1YtL7bZrD_Y_KMb
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
import torchvision.transforms.functional as TF
import random
import cv2
import numpy as np

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(target, prediction, max_pixel=1.0):
    mse = torch.mean((target - prediction) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))

def calculate_ssim(target, prediction, data_range=1.0, channel_axis=-1):
    # Convert tensors to numpy arrays
    target_np = target.cpu().detach().numpy()
    prediction_np = prediction.cpu().detach().numpy()
    # Calculate SSIM over the batch
    ssim_val = np.mean([ssim(t, p, data_range=data_range, channel_axis=channel_axis) for t, p in zip(target_np, prediction_np)])
    return ssim_val

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

# Define your mask transformations including random rotation, flip, and dilation
mask_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
])

# Create dataset instances
train_dataset = InpaintingDataset(
    image_dir='/kaggle/input/trainingimage/new_subdataset',
    mask_dir='/kaggle/input/mask-dataset/training',
    image_transform=image_transform,
    mask_transform=mask_transform,
)

val_dataset = InpaintingDataset(
    image_dir='/kaggle/input/valimages/ValPlaces2',
    mask_dir='/kaggle/input/validation-mask',
    image_transform=image_transform,
    mask_transform=mask_transform,
)

from PIL import Image
import os
import math
from torch.utils.data import DataLoader, random_split

total_size = len(train_dataset)



# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

from torchvision.models import vgg16, inception_v3, VGG16_Weights, Inception_V3_Weights

weights_path = '/kaggle/input/vgg16model/vgg16-397923af.pth'

model = vgg16()
model.load_state_dict(torch.load(weights_path))
model.eval()

class VGG16FeatureExtractor(nn.Module):
    def __init__(self, weights_path):
        super(VGG16FeatureExtractor, self).__init__()
        # Initialize VGG16 model
        vgg16_model = vgg16()
        # Load the pretrained weights manually
        vgg16_model.load_state_dict(torch.load(weights_path))
        # Extract the features portion of VGG16
        self.features = vgg16_model.features[:23]  # Adjust based on the layers you need
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x)

# Path to the manually downloaded weights
weights_path = '/kaggle/input/vgg16model/vgg16-397923af.pth'

# Correct instantiation of VGG16FeatureExtractor
vgg16_feature_extractor = VGG16FeatureExtractor(weights_path=weights_path).to(device)

# Updated ContentStyleLoss class initialization to accept weights_path
class ContentStyleLoss(nn.Module):
    def __init__(self, weights_path):
        super(ContentStyleLoss, self).__init__()
        self.feature_extractor = VGG16FeatureExtractor(weights_path=weights_path)

    def compute_gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size (=1 for simplicity)
        features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        return G.div(a * b * c * d)

    def forward(self, generated, target):
        gen_features = self.feature_extractor(generated)
        target_features = self.feature_extractor(target)
        content_loss = F.mse_loss(gen_features, target_features)

        # Compute style loss
        gen_gram = self.compute_gram_matrix(gen_features)
        target_gram = self.compute_gram_matrix(target_features)
        style_loss = F.mse_loss(gen_gram, target_gram)

        return content_loss, style_loss

# Correct instantiation of ContentStyleLoss with weights_path
content_style_loss = ContentStyleLoss(weights_path=weights_path).to(device)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg16_model = vgg16()
        # Load the pretrained weights manually
        vgg16_model.load_state_dict(torch.load(weights_path))
        # Extract the features portion of VGG16
        self.features = vgg16_model.features[:23]  # Adjust based on the layers you need
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, inpainted_image, target_image):
        perception_loss = nn.MSELoss()
        return perception_loss(self.vgg19(inpainted_image), self.vgg19(target_image))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Initial convolution layer
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Subsequent convolutional layers
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Adaptive pooling layer added to ensure the feature map is reduced to 1x1
            nn.AdaptiveAvgPool2d(1),

            # Final convolutional layer to produce a single scalar output
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Flatten(),  # Flatten the output to ensure it is a scalar
            nn.Sigmoid()  # Sigmoid activation to obtain a probability
        )

    def forward(self, img):
        validity = self.model(img)
        return validity

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.down1 = UNetDown(3, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x, mask):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        inpainted = self.final(u7)

        # Blend the inpainted output with the original image outside the masked region
        # This assumes mask is 1 for regions to inpaint and 0 elsewhere
        output = (1 - mask) * x + mask * inpainted
        return output

class RefinementNetwork(nn.Module):
    def __init__(self):
        super(RefinementNetwork, self).__init__()
        self.refinement_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Ensuring the output is within the same range as the U-Net generator output
        )

    def forward(self, x):
        return self.refinement_layers(x)

def masked_l1_loss(output, target, mask):
    """
    Calculate L1 loss only for the masked regions.

    Parameters:
    - output: the output from the generator (inpainted image).
    - target: the ground truth image.
    - mask: the binary mask indicating the regions to inpaint (1 for missing regions).

    Returns:
    - The L1 loss computed only for the masked regions.
    """
    # Ensure the mask is in the correct format (same size as output/target and binary)
    mask = mask.expand_as(target)  # Expanding the mask to match the target dimensions if needed

    # Calculate the difference only in the masked regions
    difference = (output - target) * mask  # Apply mask to the difference

    # Calculate the L1 loss only for the masked regions
    loss = torch.abs(difference).sum() / mask.sum()  # Normalize by the number of masked pixels

    return loss

"""Following is the implementation from 9th paper of my folder"""

generator = UNet().to(device)
discriminator_d1 = Discriminator().to(device)
vgg16_feature_extractor = VGG16FeatureExtractor(weights_path).to(device)  # Used within ContentStyleLoss
perceptual_loss = PerceptualLoss().to(device)  # Optional based on your preference
criterion_gan= nn.BCEWithLogitsLoss()
refinement_network = RefinementNetwork().to(device)


# Define an optimizer for the refinement network
optimizer_refinement = torch.optim.Adam(refinement_network.parameters(), lr=1e-4)
optimizer_g = torch.optim.Adam(generator.parameters(), lr=3e-4)
optimizer_d1 = torch.optim.Adam(discriminator_d1.parameters(), lr=3e-4)

# Learning rate scheduler for decay
scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=2, gamma=0.5)
scheduler_d1 = torch.optim.lr_scheduler.StepLR(optimizer_d1, step_size=2, gamma=0.5)

from PIL import Image
import os


train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Training Loop
num_epochs = 100
alpha = 0.1  # Weight for content loss
beta = 0.2   # Weight for style loss
lambda_gp = 10
save_interval=100
criterion_pixelwise = nn.L1Loss()
for epoch in range(num_epochs):
    generator.train()
    total_train_loss = 0.0
    for i, batch in enumerate(train_dataloader):
        if batch is None:
            continue
        real_images = batch['ground_truth'].to(device)
        masked_images = batch['weighted_masked_image'].to(device)
        masks = batch['mask'].to(device)
        # Generate fake images
        fake_imgs = generator(real_images, masks)
        refined_images = refinement_network(fake_imgs)
        # ---------------------
        #  Train Discriminator D1
        # ---------------------
        #optimizer_d1.zero_grad()


        #discriminator_d1.zero_grad()

        real_loss = criterion_gan(discriminator_d1(real_images), torch.ones(real_images.size(0), 1, device=real_images.device))
        fake_loss = criterion_gan(discriminator_d1(fake_imgs.detach()), torch.zeros(fake_imgs.size(0), 1, device=fake_imgs.device))

        #gradient_penalty = compute_gradient_penalty(discriminator_d1, real_images, fake_imgs)
        #d_loss = -(torch.mean(real_loss) - torch.mean(fake_loss)) + lambda_gp * gradient_penalty
        d_loss=(real_loss+fake_loss)/2

        optimizer_d1.zero_grad()
        d_loss.backward(retain_graph=True)
        optimizer_d1.step()

        # Train Generator


        g_loss = criterion_gan(discriminator_d1(fake_imgs), torch.ones(fake_imgs.size(0), 1, device=fake_imgs.device))

        #pixel_loss = criterion_pixelwise(fake_imgs, real_images)


        pixel_loss = masked_l1_loss(refined_images, real_images, masks) #L1 loss counted only on the masked region
        content_loss, style_loss = content_style_loss(refined_images, real_images)

        total_loss = (1 * pixel_loss) + (0.1 * content_loss) + (100 * style_loss) + (0.05 * g_loss)


        optimizer_g.zero_grad()
        optimizer_refinement.zero_grad()

        # Backpropagate total loss
        total_loss.backward()

        # Step both optimizers
        optimizer_g.step()
        optimizer_refinement.step()

        if i % save_interval == 0:
            generator_path = f'generator_{epoch}_{i}.pth'
            discriminator_path = f'discriminator_{epoch}_{i}.pth'
            torch.save(generator.state_dict(), generator_path)
            torch.save(discriminator_d1.state_dict(), discriminator_path)
            print(f"Epoch {epoch}/{num_epochs} - D1 Loss: {d_loss.item()} - G Loss: {total_loss.item()}")
            print(f"content loss {content_loss.item()}- style loss:{style_loss.item()}- pixel loss:{pixel_loss.item()}")

        if i % 100 == 0 and i>0:
            # Update learning rate
            generator.eval()
            total_val_psnr = 0.0
            total_val_ssim = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    real_images = batch['ground_truth'].to(device)
                    masked_images = batch['weighted_masked_image'].to(device)
                    masks = batch['mask'].to(device)

                    fake_images = generator(masked_images, masks)

                    # Normalize images if necessary
                    real_images = (real_images + 1) / 2
                    fake_images = (fake_images + 1) / 2

                    batch_psnr = calculate_psnr(real_images, fake_images)

                    total_val_psnr += batch_psnr

            avg_val_psnr = total_val_psnr / len(val_dataloader)
            print(f"Epoch {epoch}: Avg. PSNR: {avg_val_psnr:.2f}")
            print(f"Epoch {epoch}/{num_epochs} - D1 Loss: {d_loss.item()} - G Loss: {total_loss.item()}")
            print(f"content loss {content_loss.item()}- style loss:{style_loss.item()}- pixel loss:{pixel_loss.item()}")

    scheduler_g.step()
    scheduler_d1.step()