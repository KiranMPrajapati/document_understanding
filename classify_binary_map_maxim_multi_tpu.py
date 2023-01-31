import os
import json
import torch
import shutil
import maxim_test 
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu
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

SERIAL_EXEC = xmp.MpSerialExecutor()

image_size = 512 
writer = SummaryWriter()
transforms = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.Resize((image_size,image_size)), 
    transforms.ToTensor()
])
bs = 16


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

        binary_image = binary_image.resize((image_size, image_size))
        binary_image = np.array(binary_image)

        binary_image[binary_image >= 0.5] = 1
        binary_image[binary_image < 0.5] = 0

        binary_image = torch.from_numpy(binary_image)
        sample = {"image": image, "binary_image": binary_image.float(), "image_name": image_name}
        if self.transform:
            sample["image"] = self.transform(sample["image"])


        return sample


train_dataset = HierText(csv_file=csv_file, data_dir=train_data_dir, binary_data_dir=binary_data_dir, transform=transforms)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=xm.xrt_world_size(),rank=xm.get_ordinal(),shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=bs, sampler=train_sampler, num_workers=16, shuffle=False)
val_dataset = HierText(csv_file=val_csv_file, data_dir=val_data_dir, binary_data_dir=val_binary_data_dir, transform=transforms)
#val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=xm.xrt_world_size(),rank=xm.get_ordinal(),shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=bs, num_workers=16, shuffle=False)

def train(e, model, optimizer, loss_fn, learning_rate, scheduler, device, rank):
    xm.master_print("Training started, epoch: {e}")
    model.train()
    optimizer.zero_grad()    
    total_loss = 0 
    para_loader = pl.ParallelLoader(train_dataloader, [device])
    print(para_loader.per_device_loader(train_dataloader))
    for batch_idx, data in enumerate(para_loader.per_device_loader(train_dataloader)):
        print('data')
        image, binary_image = data["image"].to(device), data["binary_image"].to(device)
        pred_binary_image = model(image) 
        loss = loss_fn(pred_binary_image, binary_image)
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

    if e % 5 == 0 and rank==0:
        if os.path.exists('/mnt/researchteam/.local/share/Trash/'):
            shutil.rmtree('/mnt/researchteam/.local/share/Trash/')            
        if os.path.exists(f"saved_models/model{e-5}.pth"):
            os.remove(f"saved_models/model{e-5}.pth")
        torch.save(model.state_dict(), f"{cwd}/saved_models/model{e}.pth")
    
def val(e, model, optimizer, loss_fn, learning_rate, device, rank):
    print("Validation started")
    model.eval()
    val_loss = 0
    para_loader = pl.ParallelLoader(val_dataloader, [device])
    
    for batch_idx, data in enumerate(para_loader.per_device_loader(val_dataloader)):
        image, binary_image = data["image"].to(device), data["binary_image"].to(device)
        pred_binary_image = model(image) 
        loss = loss_fn(pred_binary_image, binary_image)
        val_loss += loss.item() 
        if batch_idx % 100 == 0:
            xm.master_print(f'Loss={round(loss.item(), 5)} Time={time.asctime()} num_data: {len(val_dataloader.dataset)}', flush=True)        
    epoch_loss = (val_loss*bs)/len(val_dataloader.dataset)
    xm.master_print(f"Epoch: {e}, num_data: {len(val_dataloader.dataset)}, Loss: {epoch_loss}")
    if rank==0:
        writer.add_scalar('Loss/val_lr:0.0001', epoch_loss, e)
    
def main(rank):
    device = xm.xla_device()
    learning_rate = 0.0001 * xm.xrt_world_size()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model = xmp.MpModelWrapper(maxim_test.MAXIM_dns_3s()).to(device)
    optimizers = torch.optim.Adam(model.parameters(), lr=learning_rate)
    decayRate = 0.96
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers, gamma= decayRate)

    epoch = 1
    for e in range(epoch): 
        train(e+1, model, optimizers, loss_fn, learning_rate, scheduler, device, rank)
#         if e % 5 == 0:
        val(e+1, model, optimizers, loss_fn, learning_rate, device, rank)

if __name__ == "__main__":
    xmp.spawn(main, args=(), nprocs=1, start_method='fork')
