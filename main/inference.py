import time 
import torch
import argparse
import torchvision
import torchmetrics
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt 

from pathlib import Path
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import unet_with_gmlp
import nafnet
import constants
from dataset import HierText

torch.manual_seed(27)
torch.cuda.manual_seed_all(27)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train(args, dataloader, dataset_type):
    model_save_dir = args['run_dir'] / 'checkpoints'
    if not model_save_dir.exists():
        model_save_dir.mkdir(parents=True, exist_ok=True)
    
    if dataset_type == "train":
        save_image_dir = args['run_dir'] / 'train_images'
    else:
        save_image_dir = args['run_dir'] / 'val_images'
        
    if not save_image_dir.exists():
        save_image_dir.mkdir(parents=True, exist_ok=True)
        
    device = args["device"]
    model = unet_with_gmlp.DVQAModel().to(device)
#     img_channel = 3
#     width = 32

#     enc_blks = [1, 1, 2, 4]
#     middle_blk_num = 2
#     dec_blks = [1, 1, 1, 1]

#     model = nafnet.NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blks, dec_blk_nums=dec_blks).to(device)

    model_save_path = model_save_dir / 'model.pth'
    
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint['epoch'] + 1
    
    f1 = torchmetrics.classification.BinaryF1Score().to(device)
    precision = torchmetrics.classification.BinaryPrecision().to(device)
    recall = torchmetrics.classification.BinaryRecall().to(device)
    accuracy = torchmetrics.classification.BinaryAccuracy().to(device)
        
    m = nn.Sigmoid()
    
    with torch.no_grad():
        model.eval()
        total_f1_score = 0 
        total_precision_score = 0
        total_recall_score = 0
        total_accuracy_score = 0

        with tqdm(dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"{dataset_type} Inference Started")
                image = batch['image'].to(device)
                binary_image = batch['binary_image'].to(device)

                outputs = model(image)
                outputs = m(outputs)
                outputs[outputs>=0.5]=1
                outputs[outputs<0.5]=0
                
                precision_score = precision(outputs.squeeze(), binary_image.squeeze())
                recall_score = recall(outputs.squeeze(), binary_image.squeeze())
                accuracy_score = accuracy(outputs.squeeze(), binary_image.squeeze())
                f1_score = f1(outputs.squeeze(), binary_image.squeeze())  

                total_f1_score += f1_score 
                total_precision_score += precision_score
                total_recall_score += recall_score
                total_accuracy_score += accuracy_score
            avg_f1_score = total_f1_score / len(dataloader) 
            avg_precision_score = total_precision_score / len(dataloader)
            avg_recall_score = total_recall_score/ len(dataloader)
            avg_accuracy_score = total_accuracy_score / len(dataloader) 
            
            print(f"****{dataset_type}******")
            print(f"Precision: {avg_precision_score}")
            print(f"Recall: {avg_recall_score}")
            print(f"Accuracy: {avg_accuracy_score}")
            print(f"F1 score: {avg_f1_score}")
            print('\n')
            
#         with tqdm(dataloader, unit="batch") as tepoch:
#             for idx, batch in enumerate(tepoch):
#                 tepoch.set_description(f"Started Saving {dataset_type} Image")
#                 image = batch['image'].to(device)
#                 binary_image = batch['binary_image'].to(device)

#                 outputs = model(image)
#                 outputs = m(outputs)
#                 outputs[outputs>=0.5]=1
#                 outputs[outputs<0.5]=0
                
#                 final_image = torch.cat([binary_image[0], outputs[0]])
#                 grid_image = torchvision.utils.make_grid(final_image, nrow=2) 

#                 torchvision.utils.save_image(grid_image, f"{save_image_dir}/{idx}.jpg")
#                 plt.imsave(f"{save_image_dir}/real{idx}.jpg", image.to('cpu')[0].detach().permute(1,2,0).numpy())
#                 if idx == 100:
#                     break
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
    
    train_dataset = HierText(csv_file=Path(args['csv_file']), data_dir=Path(args['train_data_dir']), binary_data_dir=Path(args['binary_data_dir']), transform=args['transform'])
    val_dataset = HierText(csv_file=Path(args['val_csv_file']), data_dir=Path(args['val_data_dir']), binary_data_dir=Path(args['val_binary_data_dir']), transform=args['transform'])
    
    train_dataloader = DataLoader(train_dataset, batch_size=constants.BATCH_SIZE, num_workers=constants.NUM_WORKERS, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=constants.BATCH_SIZE, num_workers=constants.NUM_WORKERS, shuffle=False, pin_memory=True)
    
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=False, default='')

    arg_parser = parser.parse_args()

    args['run_dir'] = out_dir / arg_parser.run_dir
   
#     train(args, train_dataloader, "train")
    train(args, val_dataloader, "val")    