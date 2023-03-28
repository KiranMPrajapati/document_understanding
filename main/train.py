import time 
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt 

from pathlib import Path
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn import functional as F 

import gmlp_unet_concat as gmlp 
import constants
from losses import FocalTverskyLoss, TverskyLoss, FocalBCELoss
from dataset import HierText

torch.manual_seed(27)
torch.cuda.manual_seed_all(27)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train(args):
    model_save_dir = args['run_dir'] / 'checkpoints'
    if not model_save_dir.exists():
        model_save_dir.mkdir(parents=True, exist_ok=True)
        
    save_train_image_dir = args['run_dir'] / 'train_images'
    save_val_image_dir = args['run_dir'] / 'val_images'
        
    if not save_train_image_dir.exists():
        save_train_image_dir.mkdir(parents=True, exist_ok=True)
    if not save_val_image_dir.exists():
        save_val_image_dir.mkdir(parents=True, exist_ok=True)
        
    writer = SummaryWriter(str(args['run_dir'] / 'logs'))
    
    device = args["device"]
    train_dataset = HierText(csv_file=Path(args['csv_file']), data_dir=Path(args['train_data_dir']), binary_data_dir=Path(args['binary_data_dir']), transform=args['transform'])
    val_dataset = HierText(csv_file=Path(args['val_csv_file']), data_dir=Path(args['val_data_dir']), binary_data_dir=Path(args['val_binary_data_dir']), transform=args['transform'])
    
    train_dataloader = DataLoader(train_dataset, batch_size=constants.BATCH_SIZE, num_workers=constants.NUM_WORKERS, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=constants.BATCH_SIZE, num_workers=constants.NUM_WORKERS, shuffle=False, pin_memory=True)
    
    model = gmlp.DVQAModel().to(device)
#     img_channel = 3
#     width = 32

#     enc_blks = [1, 1, 2, 4]
#     middle_blk_num = 2
#     dec_blks = [1, 1, 1, 1]

#     model = nafnet.NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blks, dec_blk_nums=dec_blks).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args["decayRate"])
    diceloss_fn = TverskyLoss()
    
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    model_save_path = model_save_dir / 'model.pth'
    start_epoch = 0
    
    if args['resume']:
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        
    for epoch in range(start_epoch, args['epochs']):
        model.train()
        train_loss = 0.0

        with tqdm(train_dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Train Epoch {epoch}")
                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=True):
                    image = batch['image'].to(device)
                    binary_image = batch['binary_image'].to(device)
                    weight_map = batch["weight_map"].to(device)

                    outputs = model(image)
                    loss = constants.BCE_LOSS_WEIGHT * F.binary_cross_entropy_with_logits(outputs.squeeze(), binary_image.squeeze(), weight=weight_map.squeeze()) + constants.DICE_LOSS_WEIGHT * diceloss_fn(outputs.squeeze(), binary_image.squeeze())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

                tepoch.set_postfix(loss=loss.item())

        train_loss /= len(train_dataloader)
        scheduler.step()
        
        writer.add_scalars("loss", {
                "train_loss": train_loss}, epoch)
        
       
        if (epoch+1) % 10 == 0:
            state = {'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'epoch': epoch}
            torch.save(state, model_save_path)
            
            with torch.no_grad():
                model.eval()
                val_loss = 0.0

                with tqdm(val_dataloader, unit="batch") as tepoch:
                    for batch in tepoch:
                        tepoch.set_description(f"Val Epoch {epoch}")
                        image = batch['image'].to(device)
                        binary_image = batch['binary_image'].to(device)
                        weight_map = batch["weight_map"].to(device)

                        outputs = model(image)
                        loss = constants.BCE_LOSS_WEIGHT * F.binary_cross_entropy_with_logits(outputs.squeeze(), binary_image.squeeze(), weight=weight_map.squeeze()) + constants.DICE_LOSS_WEIGHT * diceloss_fn(outputs.squeeze(), binary_image.squeeze())

                        val_loss += loss.item()
                   
                        tepoch.set_postfix(loss=loss.item())

            val_loss /= len(val_dataloader)

            print(f"Epoch: {epoch}")
            print(f"Train Loss: {train_loss}")
            print(f"Val Loss: {val_loss}")

            writer.add_scalars("loss", {
                "train_loss": train_loss,
                "val_loss": val_loss}, epoch)

            
        if (epoch+1) % 50 == 0:
            with torch.no_grad():
                model.eval()
                with tqdm(train_dataloader, unit="batch") as tepoch:
                        for idx, batch in enumerate(tepoch):
                            tepoch.set_description(f"Started Saving train Image")
                            image = batch['image'].to(device)
                            binary_image = batch['binary_image'].to(device)

                            outputs = model(image)
                            outputs = torch.sigmoid(outputs)
                            outputs[outputs>=0.5]=1
                            outputs[outputs<0.5]=0

                            final_image = torch.cat([binary_image[0], outputs.squeeze()[0]])
                            grid_image = torchvision.utils.make_grid(final_image, nrow=2) 

                            torchvision.utils.save_image(grid_image, f"{save_train_image_dir}/{idx}.jpg")
                            plt.imsave(f"{save_train_image_dir}/real{idx}.jpg", image.to('cpu')[0].detach().permute(1,2,0).numpy())
                            if idx == 100:
                                break


                with tqdm(val_dataloader, unit="batch") as tepoch:
                    for idx, batch in enumerate(tepoch):
                        tepoch.set_description(f"Started Saving val Image")
                        image = batch['image'].to(device)
                        binary_image = batch['binary_image'].to(device)

                        outputs = model(image)
                        outputs = torch.sigmoid(outputs)
                        outputs[outputs>=0.5]=1
                        outputs[outputs<0.5]=0

                        final_image = torch.cat([binary_image[0], outputs[0]])
                        grid_image = torchvision.utils.make_grid(final_image, nrow=2) 

                        torchvision.utils.save_image(grid_image, f"{save_val_image_dir}/{idx}.jpg")
                        plt.imsave(f"{save_val_image_dir}/real{idx}.jpg", image.to('cpu')[0].detach().permute(1,2,0).numpy())
                        if idx == 100:
                            break
            
        
        print('***********************')

if __name__ == "__main__":
    out_dir = Path('out/')
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        
    args = {}
    args['train_data_dir'] = 'data/train'
    args['val_data_dir'] = 'data/validation'
    args['binary_data_dir'] = 'data/edge_detected_1dilated_train'
    args['val_binary_data_dir'] = 'data/edge_detected_1dilated_val'
    args['csv_file'] = 'data/gt/hiertext.csv'
    args['val_csv_file'] = 'data/gt/val_hiertext.csv'
    
    args['transform'] = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((constants.IMAGE_SIZE, constants.IMAGE_SIZE)),
            transforms.ToTensor()
        ])
    
    args['epochs'] = 220
    args['learning_rate'] = 0.0001
    args['decayRate'] = 0.96
    
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=False, default='')

    arg_parser = parser.parse_args()

    if arg_parser.run_dir:
        run_dir = out_dir / arg_parser.run_dir
        resume = True
    else:
        run_dir = out_dir / f"{int(time.time())}"
        resume = False
        run_dir.mkdir(parents=True, exist_ok=True)

    args['resume'] = resume
    args['run_dir'] = run_dir
    
    train(args)