import os
import cv2
import time 
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import augmentation as aug
import constants

transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize((constants.IMAGE_SIZE, constants.IMAGE_SIZE)),
    transforms.ToTensor()
])



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
        self.random_state = np.random.RandomState(27)

    def __len__(self):
        return len(self.data)
    
    def perspective_distortion_mat(self, height, width):

        height, width = height, width

        aug_choice = np.random.default_rng().choice(['rotate', 'shear', 'elation', 'translate'])

        if aug_choice == 'rotate':

            scale_aug_values = self.random_state.uniform(0.6, 1.1, 1)

            if scale_aug_values <= 0.7:
                rot_choice = np.random.default_rng().choice([(-30.0, -20.0), (20.0, 30.0)])
            elif scale_aug_values <= 0.8:
                rot_choice = np.random.default_rng().choice([(-25.0, -10.0), (10.0, 25.0)])
            elif scale_aug_values <= 0.9:
                rot_choice = np.random.default_rng().choice([(-15.0, -5.0), (5.0, 15.0)])
            elif scale_aug_values <= 1.0: 
                rot_choice = (-10.0, 10.0)
            else:
                rot_choice = (-5.0, 5.0)

            rotate_aug_values = self.random_state.uniform(rot_choice[0], rot_choice[1], 1)

            scale_aug_values = np.float64(scale_aug_values)
            rotate_aug_values = np.float64(rotate_aug_values) 

            scale_mat = aug.scale_mat(height, width, scale_aug_values)
            rot_mat = aug.rotate_mat(height, width, rotate_aug_values)

            homography = np.matmul(rot_mat, scale_mat)
            aug_type = {'rotation' : {'scaling': scale_aug_values, 'rotation': rotate_aug_values}}

        elif aug_choice == 'shear':

            scale_aug_values = self.random_state.uniform(0.8, 1.05, 1)

            if scale_aug_values <= 0.9:
                shear_choice = np.random.default_rng().choice([(-0.3, -0.1), (0.1, 0.3)])
            elif scale_aug_values <= 1.0: 
                shear_choice = (-0.1, 0.1)
            else:
                shear_choice = (-0.05, 0.05)

            shear_aug_values = self.random_state.uniform(shear_choice[0], shear_choice[1], 1)

            scale_aug_values = np.float64(scale_aug_values)
            shear_aug_values = np.float64(shear_aug_values) 

            scale_mat = aug.scale_mat(height, width, scale_aug_values)
            shear_mat = aug.shear_mat(height, width, shear_aug_values)

            homography = np.matmul(shear_mat, scale_mat)
            aug_type = {'shearing': {'scaling': scale_aug_values, 'shearing': shear_aug_values}}
 
        elif aug_choice == 'translate':

            scale_aug_values = self.random_state.uniform(0.8, 1.05, 1)

            horizontal_choice = (-30, 30)
            vertical_choice = (-30, 30)

            horizontal_aug_values = self.random_state.uniform(horizontal_choice[0], horizontal_choice[1], 1)
            vertical_aug_values = self.random_state.uniform(vertical_choice[0], vertical_choice[1], 1)

            scale_aug_values = np.float64(scale_aug_values)
            horizontal_aug_values = np.float64(horizontal_aug_values) 
            vertical_aug_values = np.float64(vertical_aug_values) 

            scale_mat = aug.scale_mat(height, width, scale_aug_values)
            translate_mat = aug.translate_mat(horizontal_aug_values, vertical_aug_values)

            homography = np.matmul(translate_mat, scale_mat)
            aug_type = {'translation': {'scaling': scale_aug_values, 'translation_x': horizontal_aug_values, 'translation_y': vertical_aug_values}}

        elif aug_choice == 'elation':

            scale_aug_values = self.random_state.uniform(0.8, 1.05, 1)

            if scale_aug_values <= 0.9:
                elation_x_choice = np.random.default_rng().choice([(-0.0003, -0.0002), (0.0002, 0.0003)])
                elation_y_choice = np.random.default_rng().choice([(-0.0003, -0.0002), (0.0002, 0.0003)])
            elif scale_aug_values <= 1.0: 
                elation_x_choice = np.random.default_rng().choice([(-0.0002, -0.00015), (0.00015, 0.0002)])
                elation_y_choice = np.random.default_rng().choice([(-0.0002, -0.00015), (0.00015, 0.0002)])
            else:
                elation_x_choice = (-0.00015, 0.000015)
                elation_y_choice = (-0.00015, 0.000015)

            elation_x_aug_values = self.random_state.uniform(elation_x_choice[0], elation_x_choice[1], 1)
            elation_y_aug_values = self.random_state.uniform(elation_y_choice[0], elation_y_choice[1], 1)

            scale_aug_values = np.float64(scale_aug_values)
            elation_x_aug_values = np.float64(elation_x_aug_values)
            elation_y_aug_values = np.float64(elation_y_aug_values)

            scale_mat = aug.scale_mat(height, width, scale_aug_values)
            elation_mat = aug.elation_mat(height, width, elation_x_aug_values, elation_y_aug_values)

            homography = np.matmul(elation_mat, scale_mat)
            aug_type = {'elation': {'scaling': scale_aug_values, 'elation_x': elation_x_aug_values, 'elation_y': elation_y_aug_values}}

        return homography, aug_type
    
    def compute_weight_map(self, binary_image):
        weight_map = np.zeros_like(binary_image)
        number_of_white_pix = np.sum(binary_image == 1) 
        number_of_black_pix = np.sum(binary_image == 0) 
        
        total_pixels = number_of_white_pix + number_of_black_pix
        weight_of_white_pix = 1 - (number_of_white_pix / total_pixels)
        weight_of_black_pix = 1 - (number_of_black_pix / total_pixels)
        
        weight_map = np.where(binary_image==1, weight_of_white_pix, weight_map)
        weight_map = np.where(binary_image==0, weight_of_black_pix, weight_map)
        
        return weight_map

    def __getitem__(self, idx):
        image_name = self.data["image_name"][idx]
        img_dir = os.path.join(self.data_dir, image_name)
        binary_img_dir = os.path.join(self.binary_data_dir, image_name)

        image = cv2.imread(img_dir)
        binary_image = cv2.imread(binary_img_dir, cv2.IMREAD_GRAYSCALE)
        
        h, w = image.shape[:2]
        
        homography, _ = self.perspective_distortion_mat(h, w)
        image = warp_image(image, homography, target_h=h, target_w=w)
        binary_image = warp_image(binary_image, homography, target_h=h, target_w=w)        

        binary_image = cv2.resize(binary_image, (constants.IMAGE_SIZE, constants.IMAGE_SIZE))
        
        if np.random.uniform() < 0.5:
            image = aug.apply_random_drop(image)

        if np.random.uniform() < 1:
            distort_transform = aug.gaussian_distortion()
            binary_image = distort_transform(Image.fromarray(binary_image))
            image = distort_transform(Image.fromarray(image))
        
        binary_image = np.array(binary_image)
        binary_image[binary_image>=0.5]=1
        binary_image[binary_image<0.5]=0

        weight_map = self.compute_weight_map(binary_image)
        
        binary_image = torch.from_numpy(binary_image)
        weight_map = torch.from_numpy(weight_map)
                
        sample = {"image": image, "binary_image": binary_image.float().squeeze(), "image_name": image_name, "weight_map": weight_map.squeeze()}
        
        if self.transform:
            sample["image"] = self.transform(np.array(sample["image"]))
        
        return sample
    
if __name__ == "__main__":
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((constants.IMAGE_SIZE, constants.IMAGE_SIZE)),
            transforms.ToTensor()
        ])
    img_path = os.path.abspath(os.path.join("data/train/"))
    binary_data_dir = os.path.abspath(os.path.join("data/train_data_each_word_map/"))
    csv_file = os.path.abspath(os.path.join("data/gt/hiertext.csv"))

    train_dataset = HierText(csv_file=csv_file, data_dir=img_path, binary_data_dir=binary_data_dir, transform=transform)

    result = train_dataset[0]
    print(result)
