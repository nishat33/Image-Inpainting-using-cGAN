# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 05:50:38 2024

@author: user
"""

import os
import random
import shutil

# Set the path to your dataset folder
source_folder = "E:/Thesis_Stuffs/train_256_places365standard/data_256"

# Set the path where you want to save the new subdataset
destination_folder = "E:/Thesis_Stuffs/new_subdataset"

# Define the minimum number of images you want to select from each main folder
min_images_per_main_folder = 135
def find_image_files_in_subfolder(subfolder_path):
    """Finds image files directly within a given subfolder."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif'}
    images = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path)
              if os.path.isfile(os.path.join(subfolder_path, f)) and
              os.path.splitext(f)[1].lower() in image_extensions]
    return images

def copy_selected_images(images, destination_subfolder, max_images=135):
    """Copies a selection of images to the destination subfolder."""
    if not images:
        return  # No images to copy

    selected_images = random.sample(images, min(len(images), max_images))

    os.makedirs(destination_subfolder, exist_ok=True)  # Ensure destination exists
    for image in selected_images:
        shutil.copy(image, destination_subfolder)
    print(f"Copied {len(selected_images)} images to {destination_subfolder}")

def process_subfolders(source_folder, destination_root):
    """Processes each subfolder at all nesting levels, copying images to a mirrored structure."""
    for root, subdirs, files in os.walk(source_folder):
        for subdir in subdirs:
            subfolder_path = os.path.join(root, subdir)
            images = find_image_files_in_subfolder(subfolder_path)
            # Construct a corresponding destination path
            relative_path = os.path.relpath(subfolder_path, source_folder)
            destination_subfolder = os.path.join(destination_root, relative_path)
            copy_selected_images(images, destination_subfolder)
    
process_subfolders(source_folder, destination_folder)

    