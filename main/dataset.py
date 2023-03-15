import os
import cv2
import time 
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import constants

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((constants.IMAGE_SIZE, constants.IMAGE_SIZE)),
    transforms.ToTensor()
])

def aug_scale_mat(height, width, scale_factor):

    centerX = (width) / 2
    centerY = (height) / 2

    tx = centerX - centerX * scale_factor
    ty = centerY - centerY * scale_factor

    scale_mat = np.array([[scale_factor, 0, tx], [0, scale_factor, ty], [0., 0., 1.]])

    return scale_mat

def aug_rotate_mat(height, width, angle):

    centerX = (width - 1) / 2
    centerY = (height - 1) / 2

    rotation_mat = cv2.getRotationMatrix2D((centerX, centerY), angle, 1.0)
    rotation_mat = np.vstack([rotation_mat, [0., 0., 1.]])

    return rotation_mat

def warp_image(image, homography, target_h, target_w):
    return cv2.warpPerspective(image, homography, dsize=tuple((target_w, target_h)))

def center_crop(image, h, w):
    center = image.shape
    x = center[1]/2 - w/2
    y = center[0]/2 - h/2

    crop_img = image[int(y):int(y+h), int(x):int(x+w)]
    return crop_img

class HierText(Dataset):
    def __init__(self, csv_file, data_dir, binary_data_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.binary_data_dir = binary_data_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data["image_name"][idx]
        img_dir = os.path.join(self.data_dir, image_name)
        binary_img_dir = os.path.join(self.binary_data_dir, image_name)

        image = cv2.imread(img_dir)
        binary_image = cv2.imread(binary_img_dir, cv2.IMREAD_GRAYSCALE)
        
        h, w, _ = image.shape
        scale_factor = round(random.uniform(0.8, 1.2), 2)
        rot_factor = random.randint(-45, 45)
        scale_mat = aug_scale_mat(h, w, scale_factor)
        rot_mat = aug_rotate_mat(h, w, rot_factor)
        homography = np.matmul(rot_mat, scale_mat)
        image = warp_image(image, homography, target_h=h, target_w=w)
        binary_image = warp_image(binary_image, homography, target_h=h, target_w=w)

        binary_image = cv2.resize(binary_image, (constants.IMAGE_SIZE, constants.IMAGE_SIZE))
        binary_image = np.array(binary_image)
        binary_image[binary_image>=0.5]=1
        binary_image[binary_image<0.5]=0
        
        weight_map = np.zeros_like(binary_image)
        number_of_white_pix = np.sum(binary_image == 1) 
        number_of_black_pix = np.sum(binary_image == 0) 
        
        total_pixels = number_of_white_pix + number_of_black_pix
        weight_of_white_pix = 1 - (number_of_white_pix / total_pixels)
        weight_of_black_pix = 1 - (number_of_black_pix / total_pixels)
        
        weight_map = np.where(binary_image==1, weight_of_white_pix, weight_map)
        weight_map = np.where(binary_image==0, weight_of_black_pix, weight_map)
        
        binary_image = torch.from_numpy(binary_image)
        weight_map = torch.from_numpy(weight_map)
        
        sample = {"image": image, "binary_image": binary_image.float().squeeze(), "image_name": image_name, "weight_map": weight_map.squeeze()}
        
        if self.transform:
            sample["image"] = self.transform(sample["image"])
        return sample
    
if __name__ == "__main__":
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((constants.IMAGE_SIZE, constants.IMAGE_SIZE)),
            transforms.ToTensor()
        ])
    img_path = os.path.abspath(os.path.join("data/train/"))
    binary_data_dir = os.path.abspath(os.path.join("data/edge_detected_1dilated_train/"))
    csv_file = os.path.abspath(os.path.join("data/gt/hiertext.csv"))

    train_dataset = HierText(csv_file=csv_file, data_dir=img_path, binary_data_dir=binary_data_dir, transform=transform)

    result = train_dataset[0]
    print(result)