import einops
import torch
import argparse
import torch.nn as nn
import gmlp_cross as gmlp 

from pathlib import Path

    
class DVQAModel_pretrained(nn.Module):
    def __init__(self, channels=12, pretrained_model_path=None, use_bias=True):
        super().__init__()
        self.channels = channels
        self.bias = use_bias
        self.pretrained_model_path = pretrained_model_path
        self.model = gmlp.DVQAModel()
        new_state_dict = {} 
        
        if self.pretrained_model_path:
            model_save_dir = self.pretrained_model_path / 'checkpoints'
            model_save_path = model_save_dir / 'model.pth'
            checkpoint = torch.load(model_save_path)
#             del checkpoint['model']["dec_conv_1.1.weight"]
#             del checkpoint['model']["dec_conv_1.1.bias"]
            
#             del self.model.dec_conv_1
            self.model.load_state_dict(checkpoint['model'])
                
            for name, param in self.model.named_parameters(): 
                if "dec_conv_1.1" not in name:
                    param.requires_grad = False
            
#         self.conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
#                                         nn.Conv2d(self.channels, 1, kernel_size=(3,3), bias=self.bias, padding=1, stride=1))
        
    def forward(self, x):
        x = self.model(x)
#         x = self.conv(x)

        return x
    
if __name__ == "__main__":
    out_dir = Path('out/')
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_dir", type=str, required=False, default='')
    arg_parser = parser.parse_args()

    if arg_parser.pretrained_model_dir:
        pretrained_model_dir = out_dir / arg_parser.pretrained_model_dir
        resume = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = DVQAModel_pretrained(12, pretrained_model_dir).to(device)
    x = torch.randn(2, 3, 256, 256).to(device)
    outputs = net(x)
    print(outputs.shape)
