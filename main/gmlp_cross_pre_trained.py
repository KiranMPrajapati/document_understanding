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
        
        if self.pretrained_model_path:
            model_save_dir = self.pretrained_model_path / 'checkpoints'
            model_save_path = model_save_dir / 'model.pth'
            checkpoint = torch.load(model_save_path)
            self.model.load_state_dict(checkpoint['model'])
            
            for param in self.model.parameters():
                param.requires_grad = False
                
        self.features = nn.Sequential(*list(self.model.children())[:-1])
        self.number_of_modules = len(self.features)

        self.conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                        nn.Conv2d(self.channels, 1, kernel_size=(3,3), bias=self.bias, padding=1, stride=1))
        
    def forward(self, x):
        x = self.features[0](x)
        enc_1 = self.features[1](x)
        x = self.features[2](enc_1)
        enc_2 = self.features[3](x)
        x = self.features[4](enc_2)
        enc_3 = self.features[5](x)
        x = self.features[6](enc_3)
        enc_4 = self.features[7](x)
        x = enc_4
        x = self.features[8](x, enc_4)
        x = self.features[9](x)
        x = self.features[10](x, enc_3)
        x = self.features[11](x)
        x = self.features[12](x, enc_2)
        x = self.features[13](x)
        x = self.features[14](x, enc_1)
        x = self.conv(x)

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
