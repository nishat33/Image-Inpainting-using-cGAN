# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 03:54:32 2024

@author: CCC-PC
"""

import os
import shutil
import random

def copy_images(source_dir, destination_dir, num_images=1999):
    # Ensure the destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Get a list of all files in the source directory
    all_files = os.listdir(source_dir)

    # Filter out only image files (you can adjust the condition based on your image file extensions)
    image_files = [file for file in all_files if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    # Shuffle the list of image files
    random.shuffle(image_files)

    # Copy and rename the specified number of images to the destination directory
    for i in range(min(num_images, len(image_files))):
        source_path = os.path.join(source_dir, image_files[i])
        destination_path = os.path.join(destination_dir, f"{i + 1}.jpg")
        shutil.copyfile(source_path, destination_path)
        print(f"Copying and renaming {image_files[i]} to {destination_path}")

# Example usage
source_directory = 'E:/Thesis_Stuffs/Dataset/Mask/training'
destination_directory = 'E:/Thesis_Stuffs/Dataset/Mask/temptestmask'
copy_images(source_directory, destination_directory, num_images=1999)
