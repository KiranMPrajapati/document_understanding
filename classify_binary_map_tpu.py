import os
import json
import torch
import shutil
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch_xla.core.xla_model as xm 
from PIL import Image
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

cwd = os.getcwd()

train_data_dir = f'{cwd}/dataset/train'
val_data_dir = f'{cwd}/dataset/validation'
val_csv_file = f'{cwd}/hiertext/gt/validation.jsonl'
binary_data_dir = f'{cwd}/dataset/binary_train/'
val_binary_data_dir = f'{cwd}/dataset/binary_val/'
csv_file = f'{cwd}/hiertext/gt/hiertext.csv' 
val_csv_file = f'{cwd}/hiertext/gt/val_hiertext.csv'

writer = SummaryWriter()
transform = transforms.Compose([
    transforms.Resize((400,400)), 
    transforms.ToTensor()
])
bs =16*2


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
   
        image = Image.open(img_dir)
        binary_image = Image.open(binary_img_dir)
        
        binary_image = binary_image.resize((400,400))
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

### Model 
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, ceil_mode=True)
        
        self.fc = nn.Linear(in_features=256*25*25, out_features=512)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.maxpool(out)
        out = self.layer3(out)
        out = self.maxpool(out)
        out = self.layer5(out)
        out = self.maxpool(out)
        out = self.layer6(out)
        out = self.maxpool(out)
        out = self.layer8(out)
        return out
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(in_features=512, out_features=256*25*25)
        
        self.layer7 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
            )
        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        
    def forward(self, x):
        out = self.layer3(x)
        out = self.layer4(out)
        out = self.layer6(out)
        out = self.layer7(out)
        return out
    
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__() 
        self.encoder = encoder 
        self.decoder = decoder 
    
    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out 

def train(e, model, optimizer, loss_fn, learning_rate, scheduler, device):
    print("Training started")
    model.train()
    optimizer.zero_grad()    
    total_loss = 0 
    for batch_idx, data in enumerate(train_dataloader):
        image, binary_image = data["image"].to(device), data["binary_image"].to(device)
        pred_binary_image = model(image) 
        loss = loss_fn(pred_binary_image, binary_image.unsqueeze(1))
        total_loss += loss.item()
        loss.backward()
        optimizer.step() 
        xm.mark_step()
        optimizer.zero_grad()
        if batch_idx % 50 == 0:
            print(f"Epoch: {e}, batch_idx: {batch_idx}, num_data: {len(train_dataloader.dataset)}, Loss: {loss}")
    scheduler.step()    
    epoch_loss = (total_loss*bs)/len(train_dataloader.dataset)
    print(f"Epoch: {e}, Epoch Loss: {epoch_loss}")
    writer.add_scalar('Loss/train', epoch_loss, e)
    
    if e % 50 == 0:
        if os.path.exists('/mnt/researchteam/.local/share/Trash/'):
            shutil.rmtree('/mnt/researchteam/.local/share/Trash/')            
        if os.path.exists(f"saved_models/model_scheduler{e-50}.pth"):
            os.remove(f"saved_models/model_scheduler{e-50}.pth")
        torch.save(model.state_dict(), f"{cwd}/saved_models/model_scheduler{e}.pth")
    
def val(e, model, optimizer, loss_fn, learning_rate, device):
    print("Validation started")
    model.eval()
    val_loss = 0
    for batch_idx, data in enumerate(val_dataloader):
        image, binary_image = data["image"].to(device), data["binary_image"].to(device)
        pred_binary_image = model(image) 
        loss = loss_fn(pred_binary_image, binary_image.unsqueeze(1))
        val_loss += loss.item() 
        if batch_idx % 100 == 0: 
            print(f"Epoch: {e}, batch_idx: {batch_idx}, num_data: {len(val_dataloader.dataset)}, Loss: {loss}")
        
    epoch_loss = (val_loss*bs)/len(val_dataloader.dataset)
    print(f"Epoch: {e}, num_data: {len(val_dataloader.dataset)}, Loss: {epoch_loss}")
    writer.add_scalar('Loss/val', epoch_loss, e)
    
def main():
    device = xm.xla_device()
    learning_rate = 0.0001 
    loss_fn = torch.nn.BCEWithLogitsLoss()
    encoder_model = Encoder().to(device)
    decoder_model = Decoder().to(device)
    model = EncoderDecoder(encoder_model, decoder_model).to(device)
   # model.load_state_dict(torch.load("saved_models/model_scheduler350.pth"))
    optimizers = torch.optim.Adam(model.parameters(), lr=learning_rate)
    decayRate = 0.5
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers, gamma= decayRate)

    epoch=1000
    for e in tqdm(range(epoch)): 
        train(e+1, model, optimizers, loss_fn, learning_rate, scheduler, device)
        if e % 5 == 0:
            val(e+1, model, optimizers, loss_fn, learning_rate, device)
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
