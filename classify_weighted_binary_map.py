import os
import json
import cv2
import time 
import torch
import random
import shutil
import nafnet
import unet_with_gmlp
import dice_loss    
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
cwd = os.getcwd()

train_data_dir = f'{cwd}/dataset/train'
val_data_dir = f'{cwd}/dataset/validation'
binary_data_dir = f'{cwd}/dataset/edge_detected_1dilated_train/'
val_binary_data_dir = f'{cwd}/dataset/edge_detected_1dilated_val/'
csv_file = f'{cwd}/hiertext/gt/hiertext.csv' 
val_csv_file = f'{cwd}/hiertext/gt/val_hiertext.csv'
saved_images = f"{cwd}/test/"

image_size = 1024

writer = SummaryWriter()
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])
bs = 2

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
    # homography = np.linalg.inv(homography)
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

        binary_image = cv2.resize(binary_image, (image_size,image_size))
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
        
        sample = {"image": image, "binary_image": binary_image.float(), "image_name": image_name, "weight_map": weight_map}
        if self.transform:
            sample["image"] = self.transform(sample["image"])


        return sample

train_dataset = HierText(csv_file=csv_file, data_dir=train_data_dir, binary_data_dir=binary_data_dir, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=bs, num_workers=22, shuffle=False)
val_dataset = HierText(csv_file=val_csv_file, data_dir=val_data_dir, binary_data_dir=val_binary_data_dir, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=bs, num_workers=22, shuffle=False)


def train(e, model, optimizer, learning_rate, scheduler, device):
    print(f"Training started, epoch: {e}")
    model.train()
    optimizer.zero_grad()    
    total_loss = 0 
    #print(f"Start time: {time.asctime()}")
    for batch_idx, data in enumerate(train_dataloader):
        image, binary_image, weight_map = data["image"].to(device), data["binary_image"].to(device), data["weight_map"].to(device)
        pred_binary_image = model(image)    
        
        loss_fn = torch.nn.BCEWithLogitsLoss(weight=weight_map.squeeze())
        loss = loss_fn(pred_binary_image.squeeze(), binary_image.squeeze())
        total_loss += loss.item()
        loss.backward()
       # #print(f"Stop time: {time.asctime()}")
        if batch_idx % 4 == 0: 
            optimizer.step()
            optimizer.zero_grad()
        if batch_idx % 200 == 0:
            print(f'Loss={round(loss.item(), 5)} Time={time.asctime()} num_data: {len(train_dataloader.dataset)} batch_idx={batch_idx}', flush=True)
    scheduler.step()    
    epoch_loss = (total_loss*bs)/len(train_dataloader.dataset)
    #print("Finished training epoch {}".format(e))
    writer.add_scalar('Loss/train', epoch_loss, e)
    
    if e % 10 == 0:
        if os.path.exists('/mnt/researchteam/.local/share/Trash/'):
            shutil.rmtree('/mnt/researchteam/.local/share/Trash/')            
        if os.path.exists(f"saved_models/nafnet{e-10}.pth"):
            os.remove(f"saved_models/nafnet{e-10}.pth")
        torch.save(model.state_dict(), f"{cwd}/saved_models/nafnet{e}.pth")
        
def val(e, model, optimizer, learning_rate, device):
    print(f"Validation started {e}")
    model.eval()
    val_loss = 0

    for batch_idx, data in enumerate(val_dataloader):
        image, binary_image, weight_map = data["image"].to(device), data["binary_image"].to(device), data["weight_map"].to(device)
        pred_binary_image = model(image) 
        loss_fn = torch.nn.BCEWithLogitsLoss(weight=weight_map)
        loss = loss_fn(pred_binary_image.squeeze(), binary_image.squeeze())
        val_loss += loss.item() 
        if batch_idx % 100 == 0: 
            print(f'Loss={round(loss.item(), 5)} Time={time.asctime()} num_data: {len(val_dataloader.dataset)}', flush=True)        

    epoch_loss = (val_loss*bs)/len(val_dataloader.dataset)
    print(f"Epoch: {e}, num_data: {len(val_dataloader.dataset)}, Loss: {epoch_loss}")
    writer.add_scalar('Loss/val', epoch_loss, e)
   
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_rate = 0.0001

    img_channel = 3
    width = 32

    enc_blks = [1, 1, 2, 4]
    middle_blk_num = 2
    dec_blks = [1, 1, 1, 1]

    model = nafnet.NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blks, dec_blk_nums=dec_blks).to(device)
#     model = unet_with_gmlp.DVQAModel().to(device)

    model.load_state_dict(torch.load(f"{cwd}/saved_models/nafnet70.pth"))
    optimizers = torch.optim.Adam(model.parameters(), lr=learning_rate)
    decayRate = 0.96
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers, gamma= decayRate)

    epoch=300 
    for e in tqdm(range(70, epoch)): 
        train(e+1, model, optimizers, learning_rate, scheduler, device)

    #    if e % 20 == 0:
        #val(e+1, model, optimizers, learning_rate, device)
if __name__ == "__main__": 
    main()
