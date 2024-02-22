# -*- coding: utf-8 -*-

import torch
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import os
from pytorch_msssim import ssim

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch.nn as nn

drawing = False  # True if the mouse is pressed
ix, iy = -1, -1  # Initial positions
img = None  # Original image
mask = None  # Mask image

def draw(event, x, y, flags, param):
    global ix, iy, drawing, img, mask
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.line(img, (ix, iy), (x, y), (255, 255, 255), 5)
        cv2.line(mask, (ix, iy), (x, y), (255, 255, 255), 5)
        ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (ix, iy), (x, y), (255, 255, 255), 5)
        cv2.line(mask, (ix, iy), (x, y), (255, 255, 255), 5)

def mask_save(image_path):
    global img, mask
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    if img is None:
        print("Error: Image not found. Please check the path.")
        return
    
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw)

    while True:
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27: 
            cv2.destroyAllWindows()
            break 
        elif k == ord('k'): 
            cv2.imwrite('E:/Thesis_Stuffs/Models/Testing/mask.png', mask)
            print("Mask saved as 'mask.png'.")
            mask_path = 'E:/Thesis_Stuffs/Models/Testing/mask.png' 
            test_single_image(generator, image_path, mask_path, image_transform, mask_transform, device)
            cv2.destroyAllWindows()

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


def load_refinement(filepath, device):
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

    
    dilated_mask = dilate_mask(mask,5)

    # Create a weight map
    border_weight =1.975
    border = dilated_mask - mask
    weight_map = torch.ones_like(mask)
    weight_map[border == 1] = border_weight

    # Convert to PyTorch tensors
    image = image.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    dilated_mask = dilated_mask.unsqueeze(0).to(device)
    border=border.unsqueeze(0).to(device)
    weight_map = weight_map.unsqueeze(0).to(device)

    # Create masked image
    masked_image = image*(1- mask)

    return masked_image*weight_map, mask, image, border
from torchvision.utils import save_image, make_grid
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_pil_image

def denormalize(tensor, means, stds):
    """Denormalize a tensor image with means and stds."""
    means = torch.tensor(means).reshape(1, -1, 1, 1)
    stds = torch.tensor(stds).reshape(1, -1, 1, 1)
    
    if tensor.is_cuda:
        means = means.cuda()
        stds = stds.cuda()
    
    denormalized = tensor * stds + means
    return denormalized
means = [0.475, 0.475, 0.475]  # The same as used for normalization
stds = [1.0, 1, 1]  # The same as used for normalization
# means = [.55, .55, .55]  # The same as used for normalization
# stds = [0.9, 0.9, 0.9]  # The same as used for normalization
# means=[0.2, 0.2, 0.2]
# stds=[0.8, 0.8, 0.80]

def tensor_to_pil(tensor):
    # Undo the normalization
    
    # Denormalize the tensor
    denormalized_tensor = denormalize(tensor, means, stds)
    
    # Remove batch dimension only if it exists
    if denormalized_tensor.dim() == 4:
        denormalized_tensor = denormalized_tensor.squeeze(0)
    
    # Move tensor to CPU if it's not already
    denormalized_tensor_cpu = denormalized_tensor.cpu()
    
    # Convert to PIL Image
    pil_image = to_pil_image(denormalized_tensor_cpu)
    return pil_image

def selective_unsharp_mask(image, mask, alpha=9.5, beta=-2.5):
    image_np = np.array(image)
    mask_np = np.array(mask)

    blurred = cv2.GaussianBlur(image_np, (0, 0), 3)
    
    sharpened = cv2.addWeighted(image_np, alpha, blurred, beta, 0)

    mask_expanded = np.expand_dims(mask_np, axis=2)
    result = np.where(mask_expanded == 255, sharpened, image_np)

    return Image.fromarray(result)

def psnr(target, prediction, max_pixel=1.0):
    mse = torch.mean((target - prediction) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))




def merge(image_path, mask_path):
    
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path)
    
    # Resize image2 to match the size of image1
    image1_resized = cv2.resize(img, (256, 256))
    image2_resized = cv2.resize(mask, (256, 256))

    # Merge the two images
    merged_image = cv2.addWeighted(image1_resized, 1, image2_resized, 1, 0)

    cv2.imwrite('merged_image.png', merged_image)
    
def poisson_blend(original_img_pil, inpainted_img_pil, mask_np):
    # Convert PIL Images to NumPy arrays and ensure RGB format
    original_img_np = np.array(original_img_pil.convert('RGB'))
    inpainted_img_np = np.array(inpainted_img_pil.convert('RGB'))
    
    # Check if mask is already single-channel; only convert if it's not
    if mask_np.ndim == 3 and mask_np.shape[2] == 3:
        mask_np = cv2.cvtColor(mask_np, cv2.COLOR_BGR2GRAY)
    elif mask_np.ndim == 3:
        # If the mask has an unexpected shape, ensure it's reduced to a single channel correctly
        mask_np = mask_np[:,:,0]  # Taking one channel could be a workaround
    
    mask_np = (mask_np > 0).astype(np.uint8) * 255  # Ensure it's binary (0 or 255)
    
    # Define the center for seamlessClone (adjust based on actual mask center if necessary)
    center = (int(original_img_np.shape[1] / 2), int(original_img_np.shape[0] / 2))
    
    # Apply Poisson blending using seamlessClone
    blended_img_np = cv2.seamlessClone(inpainted_img_np, original_img_np, mask_np, center, cv2.MIXED_CLONE)
    
    # Convert blended image back to PIL Image
    blended_img_pil = Image.fromarray(blended_img_np)
    return blended_img_pil

def test_single_image(generator, image_path, mask_path, image_transform, mask_transform, device):
    masked_image, mask, real_image,border = prepare_single_image(image_path, mask_path, image_transform, mask_transform, device)
    
    with torch.no_grad():
        inpainted_image = generator(masked_image,mask).to(device)
        inpainted_image = (inpainted_image * mask + real_image * (1 - mask))
    
    
    
    merge(image_path, mask_path)
    masked_img_pil = tensor_to_pil(masked_image.squeeze(0).cpu())
    inpainted_img_pil = tensor_to_pil(inpainted_image.squeeze(0).cpu(),)
    real_img_pil = tensor_to_pil(real_image.squeeze(0).cpu())
    
    mask_np = border.squeeze().cpu().numpy().astype(np.uint8)
    inpainted_img_np = np.array(inpainted_img_pil)        
    psnr_val = psnr(real_image, inpainted_image).item()
    
    inpainted_img_np = inpainted_img_np[:, :, ::-1]
    blurred = cv2.GaussianBlur(inpainted_img_np, (0, 0), 3) 
    sharpened = cv2.addWeighted(inpainted_img_np, 1.5, blurred, -0.5, 0)
    sharpened = sharpened[:, :, ::-1]
    sharpened_img_pil = Image.fromarray(sharpened)
    
    
    sharpened_img_tensor = transforms.ToTensor()(sharpened_img_pil).unsqueeze(0).to(device)
    # Normalize the sharpened image tensor
    sharpened_img_tensor = transforms.Normalize(mean=[0.505, 0.505, 0.505], std=[0.555, 0.555, 0.555])(sharpened_img_tensor)

    
    # l1_loss = F.l1_loss(inpainted_image, real_image)
    # l2_loss = F.mse_loss(inpainted_image, real_image)
    # inpainted_image = inpainted_image.to(device).float()
    # real_image = real_image.to(device).float()
    
    ssim_val = ssim(sharpened_img_tensor, real_image, data_range=1.0, size_average=True)

    l1_loss = F.l1_loss(sharpened_img_tensor, real_image)
    l2_loss = F.mse_loss(sharpened_img_tensor, real_image)
    
    # PSNR calculation remains the same as it uses the original real_image and inpainted_image tensors
    psnr_val = psnr(real_image, inpainted_image).item()
    

    
    # Calculate SSIM
    # ssim_val = ssim(inpainted_image, real_image, data_range=1.0, size_average=True)  # data_range depends on the range of your input tensors; if normalized to [0,1], it should be 1.0
    
    print("SSIM:", ssim_val.item())
    print("l1 loss:" ,l1_loss)
    print("l2 loss:", l2_loss)
    print("psnr loss:", psnr_val)
    
    # Display the images
   # Display the images
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(masked_img_pil)
    plt.title('Masked Image')
    plt.axis('off')
 
    plt.subplot(1, 3, 2)
    plt.imshow(sharpened_img_pil)
    plt.title('Inpainted Image')
    plt.axis('off')
 
    plt.subplot(1, 3, 3)
    plt.imshow(real_img_pil)
    plt.title('Real Image')
    plt.axis('off')
    plt.show()

    # # Save the images
    masked_img_pil.save('masked_image.png')
    inpainted_img_pil.save('output.png')
    real_img_pil.save('real_image.png')
    

# Define transforms
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=means, std=stds),
    
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

#43_1600 bhalo kaaj kortese so far

# Load the generator model

generator_checkpoint = 'E:/Thesis_Stuffs/Models/Self/generator_47_900.pth' 

checkpoint = torch.load(generator_checkpoint, map_location=device)
 # Replace with your checkpoint path
generator = load_generator(generator_checkpoint, device)



image_path = 'E:/Thesis_Stuffs/Dataset/Original Photos/b.png'  # Replace with your test image path
mask_save(image_path)
