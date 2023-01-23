import os
import cv2 
import json
import torch
import shutil
import numpy as np
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
train_csv_file = f'{cwd}/hiertext/gt/train.jsonl'
val_data_dir = f'{cwd}/dataset/validation'
val_csv_file = f'{cwd}/hiertext/gt/validation.jsonl'

SERIAL_EXEC = xmp.MpSerialExecutor()

writer = SummaryWriter()
transforms = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.Resize((400,400)), 
    transforms.ToTensor()
])
bs = 16


class HierText(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.data = json.load(open(csv_file, 'r'))["annotations"]
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def draw_mask(self, vertices, w, h):
        mask = np.zeros((h, w, 3), dtype=np.float32)
        mask = cv2.fillPoly(mask, [vertices], [1.] * 3)[:, :, 0]
        return mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_annotations = self.data[idx]
        img_name = os.path.join(self.data_dir, data_annotations['image_id'])
        image = cv2.imread(f"{img_name}.jpg")
        w = data_annotations['image_width']
        h = data_annotations['image_height']

        gt_word_masks = []
        gt_word_weights = []

        for paragraph in data_annotations['paragraphs']:
            for line in paragraph['lines']:
                for word in line['words']:
                    gt_word_weights.append(1.0 if word['legible'] else 0.0)
                    vertices = np.array(word['vertices'])
                    gt_word_mask = self.draw_mask(vertices, w, h)
                    gt_word_masks.append(gt_word_mask)

        n_mask = len(gt_word_masks)

        gt_masks = (np.stack(gt_word_masks, -1) if n_mask else np.zeros(((h + 1) // 2, (w + 1) // 2, 0), np.float32))
        gt_weights = (np.array(gt_word_weights) if n_mask else np.zeros((0,), np.float32))
        
        palette = [[1]]*n_mask
        colored = np.reshape(np.matmul(np.reshape(gt_masks, (-1, n_mask)), palette), (h, w, 1))
        dont_care_mask = (np.reshape(np.matmul(np.reshape(gt_masks, (-1, n_mask)), np.reshape(1.- gt_weights, (-1, 1))), (h, w, 1)) > 0).astype(np.float32)

        binary_image = np.clip(dont_care_mask * 1. + (1. - dont_care_mask) * colored, 0., 1.)
        
        sample = {"image": image.astype(np.uint8), "binary_image": binary_image.astype(np.uint8), "image_name": f"{img_name}.jpg"}

        if self.transform:
            sample["image"] = self.transform(sample["image"])
            sample["binary_image"] = self.transform(sample["binary_image"])

        return sample
    
hiertext_train_dataset = HierText(csv_file=train_csv_file, data_dir=train_data_dir, transform=transforms)
train_sampler = torch.utils.data.distributed.DistributedSampler(hiertext_train_dataset, num_replicas=xm.xrt_world_size(),rank=xm.get_ordinal(),shuffle=True)
hiertext_val_dataset = HierText(csv_file=val_csv_file, data_dir=val_data_dir, transform=transforms)
train_dataloader = DataLoader(hiertext_train_dataset, batch_size=bs, sampler=train_sampler, num_workers=4, shuffle=True)
val_dataloader = DataLoader(hiertext_val_dataset, batch_size=bs, shuffle=True)

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

def train(e, model, optimizer, loss_fn, learning_rate, device):
    xm.master_print("Training started, epoch: {e}")
    model.train()
    optimizer.zero_grad()    
    total_loss = 0 
    para_loader = pl.ParallelLoader(train_dataloader, [device])
    for batch_idx, data in enumerate(para_loader.per_device_loader(train_dataloader)):
        image, binary_image = data["image"].to(device), data["binary_image"].to(device)
        pred_binary_image = model(image) 
        loss = loss_fn(pred_binary_image, binary_image)
        total_loss += loss.item()
        loss.backward()
        xm.optimizer_step(optimizer) 
        optimizer.zero_grad()
        if batch_idx % 50 == 0:
            print(f"xla: {xm.get_ordinal()}|{batch_idx}, num_data: {len(train_dataloader.dataset)}, Loss: {loss}, Rate: {round(tracker.rate())}, GlobalRate: {round(tracker.global_rate())}, Time: {time.asctime()}, flush=True")
    epoch_loss = (total_loss*bs)/len(train_dataloader.dataset)
    xm.master_print(f"Epoch: {e}, Epoch Loss: {epoch_loss}, Finished Training")
    xm.master_print(met.metrics_report(), flush=True)
    writer.add_scalar('Loss/train_lr:0.0001', epoch_loss, e)
    if e % 5 == 0:
        if os.path.exists('/mnt/researchteam/.local/share/Trash/'):
            shutil.rmtree('/mnt/researchteam/.local/share/Trash/')            
        if os.path.exists(f"saved_models/model{e-5}.pth"):
            os.remove(f"saved_models/model{e-5}.pth")
        torch.save(model.state_dict(), f"{cwd}/saved_models/model{e}.pth")
    
def val(e, model, optimizer, loss_fn, learning_rate, device):
    print("Validation started")
    model.eval()
    val_loss = 0
    for batch_idx, data in enumerate(val_dataloader):
        image, binary_image = data["image"].to(device), data["binary_image"].to(device)
        pred_binary_image = model(image) 
        loss = loss_fn(pred_binary_image, binary_image)
        val_loss += loss.item() 
        if batch_idx % 100 == 0: 
            print(f"Epoch: {e}, batch_idx: {batch_idx}, num_data: {len(val_dataloader.dataset)}, Loss: {loss}")
        
    epoch_loss = (val_loss*bs)/len(val_dataloader.dataset)
    print(f"Epoch: {e}, num_data: {len(val_dataloader.dataset)}, Loss: {epoch_loss}")
    writer.add_scalar('Loss/val_lr:0.0001', epoch_loss, e)
    
def main():
    device = xm.xla_device()
    learning_rate = 0.0001 * xm.xrt_world_size()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    encoder_model = Encoder().to(device)
    decoder_model = Decoder().to(device)
    model = xmp.MpModelWrapper(EncoderDecoder(encoder_model, decoder_model).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch = 50
    for e in range(epoch): 
        train(e+1, model, optimizer, loss_fn, learning_rate, device)
#         if e % 5 == 0:
#         val(e+1, model, optimizer, loss_fn, learning_rate, device)

if __name__ == "__main__":
    main()
