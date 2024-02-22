import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageFile
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
ImageFile.LOAD_TRUNCATED_IMAGES = True
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

        # Apply mask to get the masked image: element-wise multiplication
        masked_image = image * (1 - mask)

        return {'ground_truth': image, 'masked_image': masked_image, 'mask': mask}

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

# Create dataset instances
train_dataset = InpaintingDataset(
    image_dir='E:/Thesis_Stuffs/Dataset/Mask/Places2',
    mask_dir='E:/Thesis_Stuffs/Dataset/Mask/tempTraining',
    image_transform=image_transform,
    mask_transform=mask_transform
)

val_dataset = InpaintingDataset(
    image_dir='E:/Thesis_Stuffs/Dataset/Mask/ValPlaces2',
    mask_dir='E:/Thesis_Stuffs/Dataset/Mask/Val2',
    image_transform=image_transform,
    mask_transform=mask_transform
)

from PIL import Image
import os

def check_images(image_dir):
    corrupt_files = []
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        try:
            with Image.open(image_path) as img:
                img.verify()  # Verify the integrity of the image
        except (IOError, OSError) as e:
            print(f'Corrupted image detected: {image_path} - Error: {e}')
            corrupt_files.append(image_path)
    return corrupt_files

image_dir='E:/Thesis_Stuffs/Dataset/Mask/Places2'
mask_dir='E:/Thesis_Stuffs/Dataset/Mask/tempTraining'

# Check training images
corrupt_train_images = check_images(image_dir)
print(f"Found {len(corrupt_train_images)} corrupted images in training set.")

# Check validation images
corrupt_val_images = check_images(mask_dir)
print(f"Found {len(corrupt_val_images)} corrupted images in training mask set.")


image_dir='E:/Thesis_Stuffs/Dataset/Mask/ValPlaces2'
mask_dir='E:/Thesis_Stuffs/Dataset/Mask/Val2'

corrupt_train_images = check_images(image_dir)
print(f"Found {len(corrupt_train_images)} corrupted images in validation set.")

# Check validation images
corrupt_val_images = check_images(mask_dir)
print(f"Found {len(corrupt_val_images)} corrupted images in validation mask set.")

# Create DataLoaders
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

    def forward(self, x):
        # Downsampling
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        
        u1 = self.up1(d5, d4)
        print(u1.size(), d3.size())  # Add this to print tensor sizes before concatenation
        u2 = self.up2(u1, d3)  
        u3 = self.up3(u2, d2)  
        u4 = self.up4(u3, d1)

        return self.final(u4)


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

generator = UNetGenerator().to(device)
discriminator = Discriminator().to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion_GAN = nn.BCEWithLogitsLoss()
criterion_pixelwise = nn.L1Loss()
save_interval=10

num_epochs = 100  # Set the number of epochs
lambda_pixel = 100  # Weight for pixel-wise loss relative to the adversarial loss



for epoch in range(num_epochs):
    for i, batch in enumerate(train_dataloader):
        if batch is None:  # Skip the batch if any image or mask failed to load
            continue

        # Move tensors to the configured device
        real_images = batch['ground_truth'].to(device)
        masked_images = batch['masked_image'].to(device)
        masks = batch['mask'].to(device)

        # Adversarial ground truths
        # Initialize the tensors to the correct shape
        valid_shape = (real_images.size(0), 1, 15, 15)  # Shape based on discriminator output
        valid = torch.ones(valid_shape, dtype=torch.float, device=device)
        fake = torch.zeros(valid_shape, dtype=torch.float, device=device)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        # Generate a batch of images
        fake_images = generator(masked_images)

        # Discriminator's prediction on fake images
        pred_fake = discriminator(fake_images)
        pred_real = discriminator(real_images)

        # Calculate the generator's loss
        gan_loss_fake = criterion_GAN(pred_fake, valid)
        pixel_loss = criterion_pixelwise(fake_images, real_images) * lambda_pixel
        g_loss = gan_loss_fake + pixel_loss

        # Calculate the discriminator's loss
        real_loss = criterion_GAN(pred_real, valid)
        fake_loss = criterion_GAN(pred_fake, fake)
        d_loss = (real_loss + fake_loss) / 2

        # Backward pass for both generator and discriminator
        g_loss.backward(retain_graph=True)
        d_loss.backward()  # No need to retain the graph here

        # Update weights
        optimizer_G.step()
        optimizer_D.step()

        # Log progress

        # ----------------
        #  Log Progress
        # ----------------
        print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(train_dataloader)}] "
              f"[D loss: {d_loss.item()}] [G loss: {gan_loss_fake.item()}, pixel: {pixel_loss.item()}]")

        # (Optional) Save Models
        if i % save_interval == 0:
            torch.save(generator.state_dict(), f'generator_{epoch}_{i}.pth')
            torch.save(discriminator.state_dict(), f'discriminator_{epoch}_{i}.pth')
