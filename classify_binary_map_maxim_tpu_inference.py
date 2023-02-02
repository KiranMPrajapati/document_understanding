import os
import json
import time 
import torch
import shutil
import m1_gmlp
import numpy as np
from torchvision.utils import make_grid
import torchvision
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch_xla.core.xla_model as xm 
from PIL import Image
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
from torchsummary import summary
import augmentation

cwd = os.getcwd()

train_data_dir = f'{cwd}/dataset/train'
val_data_dir = f'{cwd}/dataset/validation'
val_csv_file = f'{cwd}/hiertext/gt/validation.jsonl'
binary_data_dir = f'{cwd}/dataset/binary_train/'
val_binary_data_dir = f'{cwd}/dataset/binary_val/'
csv_file = f'{cwd}/hiertext/gt/hiertext.csv' 
val_csv_file = f'{cwd}/hiertext/gt/val_hiertext.csv'
saved_model = f'{cwd}/saved_models/'
save_image = f'{cwd}/aug_result/'

image_size = 64
writer = SummaryWriter()
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((image_size, image_size)), 
    transforms.ToTensor()
])
bs = 1

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
        scale_mat = augmentation.aug_scale_mat(h, w, scale_factor)
        rot_mat = augmentation.aug_rotate_mat(h, w, rot_factor)
        homography = np.matmul(rot_mat, scale_mat)
        image = augmentation.warp_image(image, homography, target_h=h, target_w=w)
        binary_image = augmentation.warp_image(binary_image, homography, target_h=h, target_w=w)

        binary_image = cv2.resize(binary_image, (image_size,image_size))
        binary_image = np.array(binary_image)

        binary_image[binary_image >= 0.5] = 1
        binary_image[binary_image < 0.5] = 0

        binary_image = torch.from_numpy(binary_image)
        sample = {"image": image, "binary_image": binary_image.float(), "image_name": image_name}
        if self.transform:
            sample["image"] = self.transform(sample["image"])


        return sample

    
hiertext_train_dataset = HierText(csv_file=csv_file, data_dir=train_data_dir, binary_data_dir=binary_data_dir, transform=transform)
hiertext_val_dataset = HierText(csv_file=val_csv_file, data_dir=val_data_dir, binary_data_dir=val_binary_data_dir, transform=transform)
train_dataloader = DataLoader(hiertext_train_dataset, batch_size=bs, num_workers=32, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(hiertext_val_dataset, batch_size=bs, num_workers=32, shuffle=True)
resize_t = transforms.Resize((300, 300))
def inference(e, model, optimizer, loss_fn, learning_rate, scheduler, device):
    print("Inference started")
    model.train()
    optimizer.zero_grad()    
    total_loss = 0 
    print(f"Start time: {time.asctime()}")
    for batch_idx, data in enumerate(train_dataloader):
        image, binary_image = data["image"].to(device), data["binary_image"].to(device)
        pred_binary_image = model(image) 
        resized_binary_image = resize_t(binary_image.unsqueeze(1)).to('cpu')
        resized_pred_binary_image = resize_t(pred_binary_image).to('cpu')
        final_image = torch.cat([resized_binary_image, resized_pred_binary_image])
        grid_img = torchvision.utils.make_grid(final_image, nrow=2)
        torchvision.utils.save_image(grid_img, f"{batch_idx}.jpg")
        if batch_idx == 10:
            break
def main():
    device = xm.xla_device()
    learning_rate = 0.0001 
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model = m1_gmlp.MAXIM_dns_3s().to(device)
    #print(summary(model, (1, 3, image_size, image_size)))
    model.load_state_dict(torch.load(f"{saved_model}model_scheduler5.pth"))
    optimizers = torch.optim.Adam(model.parameters(), lr=learning_rate)
    decayRate = 0.96
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers, gamma= decayRate)

    epoch=1
    for e in tqdm(range(epoch)): 
        inference(e+1, model, optimizers, loss_fn, learning_rate, scheduler, device)

if __name__ == "__main__":
    # 4x 1 chip (2 cores) per process:
    os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1"
    os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
    # Different per process:
    os.environ["TPU_VISIBLE_DEVICES"] = "0" # "1", "2", "3"
    # Pick a unique port per process
    os.environ["TPU_MESH_CONTROLLER_ADDRESS"] = "localhost:8476"
    os.environ["TPU_MESH_CONTROLLER_PORT"] = "8476"
    main()
