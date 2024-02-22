# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:22:13 2024

@author: user
"""

#########################33 Inception Score ###########################3333
import torch
from torchvision.models import inception_v3
from torchvision.transforms import functional as TF
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


def load_inception_model():
    model = inception_v3(pretrained=True, init_weights=False)
    model.fc = torch.nn.Identity()  # Adjusting to return logits
    model.eval()  # Set the model to evaluation mode
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inception_model = load_inception_model().to(device)

def load_image(image_path):
    """
    Loads an image from a specified path and converts it to a PyTorch tensor.

    Args:
    - image_path (str): Path to the image file.

    Returns:
    - torch.Tensor: The image as a PyTorch tensor.
    """
    image = Image.open(image_path).convert('RGB')
    # Convert the PIL Image to a tensor
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to range [0, 1]
    ])
    image_tensor = transform(image)
    return image_tensor

def preprocess_single_image(image):
    # Assuming image is a PyTorch tensor of shape [3, H, W] and in range [0, 1]
    image = TF.resize(image, size=(299, 299))
    image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Add batch dimension
    image = image.unsqueeze(0)
    return image.to(device)


def calculate_inception_score_for_folder(folder_path, inception_model, splits=10):
    image_paths = glob.glob(os.path.join(folder_path, '*'))  # Adjust pattern as necessary
    probs = []
    for image_path in image_paths:
        image = load_image(image_path)  # Load the image
        image = preprocess_single_image(image)  # Preprocess the image
        with torch.no_grad():
            logits = inception_model(image)
            prob = F.softmax(logits, dim=1)
            probs.append(prob.cpu())

    # Concatenate all probability tensors
    probs = torch.cat(probs, dim=0)

    # Calculate the Inception Score
    scores = []
    for i in range(splits):
        part = probs[i * (probs.size(0) // splits): (i + 1) * (probs.size(0) // splits), :]
        p_yx = part
        p_y = p_yx.mean(dim=0).unsqueeze(0)
        kl_d = p_yx * (torch.log(p_yx + 1e-16) - torch.log(p_y + 1e-16))
        kl_d = kl_d.sum(dim=1)
        avg_kl_d = kl_d.mean().item()
        scores.append(np.exp(avg_kl_d))

    is_mean = np.mean(scores)
    is_std = np.std(scores)
    return is_mean, is_std



folder_path = 'E:/Thesis_Stuffs/Dataset/Mask/rectmaskwithoutborder'  # Update this path to your folder
inception_model = load_inception_model().to(device)  # Make sure the model is loaded

# Calculate the Inception Score for the entire folder
is_mean, is_std = calculate_inception_score_for_folder(folder_path, inception_model)
print(f"Inception Score: Mean = {is_mean}, Std = {is_std}")