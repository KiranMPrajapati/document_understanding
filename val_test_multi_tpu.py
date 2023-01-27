import os
import json
import torch
import shutil
import numpy as np
import pandas as pd
import time
import torch.nn as nn
import matplotlib.pyplot as plt
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.debug.profiler as xp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu
from PIL import Image
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

cwd = os.getcwd()

SERIAL_EXEC = xmp.MpSerialExecutor()

train_data_dir = f'{cwd}/dataset/train'
val_data_dir = f'{cwd}/dataset/validation'
val_binary_data_dir = f'{cwd}/hiertext/gt/binary_val/'
binary_data_dir = f'{cwd}/dataset/binary_train/'
csv_file = f'{cwd}/hiertext/gt/hiertext.csv' 
val_csv_file = f'{cwd}/hiertext/gt/val_hiertext.csv'

writer = SummaryWriter()
transform = transforms.Compose([
    transforms.Resize((400,400)), 
    transforms.ToTensor()
])
bs = 4*16


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
    
train_dataset = HierText(csv_file=csv_file, data_dir=train_data_dir, binary_data_dir=binary_data_dir, transform=transform)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True)
val_dataset = HierText(csv_file=val_csv_file, data_dir=val_data_dir, binary_data_dir=val_binary_data_dir, transform=transform)
val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=bs, sampler=train_sampler, num_workers=32, shuffle=False, pin_memory=True)
val_dataloader = DataLoader(hiertext_val_dataset, batch_size=bs, sampler=val_sampler, num_workers=32, shuffle=False)

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

def train(e, model, optimizer, loss_fn, learning_rate, scheduler, device, rank):
    xm.master_print("Training started")
    para_loader = pl.ParallelLoader(train_dataloader, [device])
    model.train()
    optimizer.zero_grad()    
    total_loss = 0 
    for batch_idx, data in enumerate(para_loader.per_device_loader(device)):
        image, binary_image = data["image"].to(device), data["binary_image"].to(device)
        pred_binary_image = model(image) 
        loss = loss_fn(pred_binary_image, binary_image.unsqueeze(1))
        total_loss += loss.item()
        loss.backward()
        xm.optimizer_step(optimizer) 
        optimizer.zero_grad()
        if batch_idx % 50 == 0:
            xm.master_print(f'Loss={round(loss.item(), 5)} Time={time.asctime()} num_data: {len(train_dataloader.dataset)}', flush=True)
    scheduler.step()    
    epoch_loss = (total_loss*bs)/len(train_dataloader.dataset)
    xm.master_print("Finished training epoch {}".format(e))
    if rank==0:
        writer.add_scalar('Loss/train', epoch_loss, e)
    
    if e % 50 == 0 and rank==0:
        if os.path.exists('/mnt/researchteam/.local/share/Trash/'):
            shutil.rmtree('/mnt/researchteam/.local/share/Trash/')            
        if os.path.exists(f"saved_models/model_scheduler{e-50}.pth"):
            os.remove(f"saved_models/model_scheduler{e-50}.pth")
        xm.save(model.state_dict(), f"{cwd}/saved_models/model_scheduler{e}.pth")
    
def val(e, model, optimizer, loss_fn, learning_rate, device):
    xm.master_print("Validation started")
    para_loader = pl.ParallelLoader(val_dataloader, [device])
    model.eval()
    val_loss = 0
    for batch_idx, data in enumerate(para_loader.per_device_loader(device)):
        image, binary_image = data["image"].to(device), data["binary_image"].to(device)
        pred_binary_image = model(image) 
        loss = loss_fn(pred_binary_image, binary_image)
        val_loss += loss.item() 
        if batch_idx % 50 == 0: 
            xm.master_print(f"Epoch: {e}, batch_idx: {batch_idx}, num_data: {len(val_dataloader.dataset)}, Loss: {loss}")
        
    epoch_loss = (val_loss*bs)/len(val_dataloader.dataset)
    xm.master_print(f"Epoch: {e}, num_data: {len(val_dataloader.dataset)}, Loss: {epoch_loss}")
    writer.add_scalar('Loss/val', epoch_loss, e)
    
def main(rank):
    device = xm.xla_device()
    learning_rate = 0.0001 * xm.xrt_world_size()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    encoder_model = Encoder()
    decoder_model = Decoder()
    encoder_decoder_model = EncoderDecoder(encoder_model, decoder_model)
    model = xmp.MpModelWrapper(encoder_decoder_model)
    model = model.to(device)
#     model.load_state_dict(torch.load("saved_models/model400.pth"))
    optimizers = torch.optim.Adam(model.parameters(), lr=learning_rate)
    decayRate = 0.96
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers, gamma= decayRate)

    epoch=2
    for e in tqdm(range(epoch)): 
        train(e+1, model, optimizers, loss_fn, learning_rate, scheduler, device, rank)
#         if e % 5 == 0:
#         val(e+1, model, optimizers, loss_fn, learning_rate, device)oo
if __name__ == "__main__":
    xmp.spawn(main, args=(), nprocs=8, start_method='fork')
