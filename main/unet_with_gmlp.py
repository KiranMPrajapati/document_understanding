import einops
import torch
import torch.nn as nn

class Layer_norm_process(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.zeros(c), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.ones(c), requires_grad=True)
        self.eps = eps

    def forward(self, feature):
        var_mean = torch.var_mean(feature, dim=-1, unbiased=False)
        mean = var_mean[1]
        var = var_mean[0]
        # layer norm process
        feature = (feature - mean[..., None]) / torch.sqrt(var[..., None] + self.eps)
        gamma = self.gamma.expand_as(feature)
        beta = self.beta.expand_as(feature)
        feature = feature * gamma + beta
        return feature
    
def block_images_einops(x, patch_size):
    batch, height, width, channels = x.shape
    grid_height = height // patch_size[0]
    grid_width = width // patch_size[1]

    x = einops.rearrange(
        x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
        gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
    
    return x

def unblock_images_einops(x, grid_size, patch_size):
    """patches to images."""
    x = einops.rearrange(
        x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
        gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
    return x

class BlockGatingUnit(nn.Module):  #input shape: n (gh gw) (fh fw) c
    """A SpatialGatingUnit as defined in the gMLP paper.
    The 'spatial' dim is defined as the second last.
    If applied on other dims, you should swapaxes first.
    """
    def __init__(self, c, n, use_bias=True):
        super().__init__()
        self.c = c
        self.n = n
        self.use_bias = use_bias
        self.Dense_0 = nn.Linear(self.n, self.n, self.use_bias)
        self.intermediate_layernorm = Layer_norm_process(self.c//2)
    def forward(self, x):
        c = x.size(-1)
        c = c // 2  #split size
        u, v  = torch.split(x, c, dim=-1)
        v = self.intermediate_layernorm(v)
        v = v.permute(0, 1, 3, 2)  #n, (gh gw), c/2, (fh fw)
        v = self.Dense_0(v)  #apply fc on the last dimension (fh fw)
        v = v.permute(0, 1, 3, 2)  #n (gh gw) (fh fw) c/2
        return u* (v + 1.)

class GridGatingUnit(nn.Module):  #input shape: n (gh gw) (fh fw) c
    """A SpatialGatingUnit as defined in the gMLP paper.
    The 'spatial' dim is defined as the second.
    If applied on other dims, you should swapaxes first.
    """
    def __init__(self, c, n, use_bias=True):
        super().__init__()
        self.c = c
        self.n = n
        self.use_bias = use_bias
        self.intermediate_layernorm = Layer_norm_process(self.c//2)
        self.Dense_0 = nn.Linear(self.n, self.n, self.use_bias)
    def forward(self, x):
        c = x.size(-1)
        c = c // 2  #split size
        u, v  = torch.split(x, c, dim=-1)
        v = self.intermediate_layernorm(v)
        v = v.permute(0, 3, 2, 1)  #n, c/2, (fh fw) (gh gw)
        v = self.Dense_0(v)  #apply fc on the last dimension (gh gw)
        v = v.permute(0, 3, 2, 1)  #n (gh gw) (fh fw) c/2
        return u* (v + 1.)

class GridGmlpLayer(nn.Module): # input shape: n, h, w, c
    def __init__(self, grid_size, num_channels, use_bias=True, factor=2, dropout_rate=0):
        super().__init__()
        self.grid_size = grid_size
        self.gh = grid_size[0]
        self.gw = grid_size[1]
        self.num_channels = num_channels
        self.use_bias = use_bias
        self.factor = factor
        self.dropout_rate = dropout_rate

        self.LayerNorm = Layer_norm_process(self.num_channels)
        self.in_project = nn.Linear(self.num_channels, self.num_channels*self.factor, self.use_bias)
        self.gelu = nn.GELU()
        self.GridGatingUnit = GridGatingUnit(self.num_channels*self.factor, n=self.gh*self.gw)
        self.out_project = nn.Linear(self.num_channels*self.factor//2, self.num_channels, self.use_bias) 
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        n, h, w, num_channels = x.shape
        fh, fw = h // self.gh, w // self.gw
        x = block_images_einops(x, patch_size=(fh, fw)) #n (gh gw) (fh fw) c

        y = self.LayerNorm(x)
        y = self.in_project(y)
        y = self.gelu(y)
        y = self.GridGatingUnit(y)
        y = self.out_project(y)
        y = self.dropout(y)
        x = x + y
        x = unblock_images_einops(x, grid_size=(self.gh, self.gw), patch_size=(fh, fw))
        return x

class BlockGmlpLayer(nn.Module):  #input shape: n, h, w, c
    """Block gMLP layer that performs local mixing of tokens."""
    def __init__(self, block_size, num_channels, use_bias=True, factor=2, dropout_rate=0):
        super().__init__()
        self.block_size = block_size
        self.fh = self.block_size[0]
        self.fw = self.block_size[1]
        self.num_channels = num_channels
        self.use_bias = use_bias
        self.factor = factor
        self.dropout_rate = dropout_rate
        self.LayerNorm = Layer_norm_process(self.num_channels)
        self.in_project = nn.Linear(self.num_channels, self.num_channels*self.factor, self.use_bias)   #c->c*factor
        self.gelu = nn.GELU()
        self.BlockGatingUnit = BlockGatingUnit(self.num_channels*self.factor, n=self.fh*self.fw)  #number of channels????????????????
        self.out_project = nn.Linear(self.num_channels*self.factor//2, self.num_channels, self.use_bias)   #c*factor->c
        self.dropout = nn.Dropout(self.dropout_rate)
    def forward(self, x):
        _, h, w, _ = x.shape
        gh, gw = h // self.fh, w // self.fw
        x = block_images_einops(x, patch_size=(self.fh, self.fw))  #n (gh gw) (fh fw) c
        # gMLP2: Local (block) mixing part, provides local block communication.
        y = self.LayerNorm(x)
        y = self.in_project(y)  #channel proj
        y = self.gelu(y)
        y = self.BlockGatingUnit(y)
        y = self.out_project(y)
        y = self.dropout(y)
        x = x + y
        x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(self.fh, self.fw))
        return x

class ResidualSplitHeadMultiAxisGmlpLayer(nn.Module):
    
    def __init__(self, block_size, grid_size, num_channels, input_proj_factor=2,block_gmlp_factor=2, grid_gmlp_factor=2, use_bias=True, dropout_rate=0.):
        super().__init__()

        self.block_size = block_size
        self.grid_size = grid_size
        self.num_channels = num_channels
        self.input_proj_factor = input_proj_factor
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.use_bias = use_bias
        self.drop = dropout_rate

        self.LayerNorm_in = Layer_norm_process(self.num_channels)
        self.in_project = nn.Linear(self.num_channels, self.num_channels*self.input_proj_factor, bias=self.use_bias)
        self.gelu = nn.GELU()
        self.GridGmlpLayer = GridGmlpLayer(grid_size=self.grid_size, num_channels=self.num_channels*self.input_proj_factor//2, 
                                            use_bias=self.use_bias, factor=self.grid_gmlp_factor)
        self.BlockGmlpLayer = BlockGmlpLayer(block_size=self.block_size, num_channels=self.num_channels*self.input_proj_factor//2, 
                                             use_bias=self.use_bias, factor=self.block_gmlp_factor)
        self.out_project = nn.Linear(self.num_channels*self.input_proj_factor, self.num_channels, bias=self.use_bias)
        self.dropout = nn.Dropout(self.drop)

    def forward(self, x):
        shortcut = x
        x = self.LayerNorm_in(x)
        x = self.in_project(x)
        x = self.gelu(x)
        c = x.size(-1)//2
        u, v = torch.split(x, c, dim=-1)
        #grid gMLP
        u = self.GridGmlpLayer(u)
        #block gMLP
        v = self.BlockGmlpLayer(v)
        # out projection
        x = torch.cat([u, v], dim=-1)
        x = self.out_project(x)
        x = self.dropout(x)
        x = x + shortcut
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, num_channels, use_bias=True, dropout_rate=0.):
        super().__init__()

        self.num_channels = num_channels
        self.use_bias = use_bias
        self.drop = dropout_rate

        self.LayerNorm_in = Layer_norm_process(self.num_channels)
        self.in_project = nn.Linear(self.num_channels, self.num_channels, bias=self.use_bias)
        self.gelu = nn.GELU()
        self.out_project = nn.Linear(self.num_channels, self.num_channels, bias=self.use_bias)
        self.dropout = nn.Dropout(self.drop)

    def forward(self, x, enc):
        shortcut = x
        x = self.LayerNorm_in(x)
        x = self.in_project(x)
        x = self.gelu(x)

        x = x * enc
        x = self.out_project(x)
        x = self.dropout(x)

        x = x + shortcut
        
        return x

class UNetEncoderBlock(nn.Module):
    def __init__(self, num_channels, block_size, grid_size, lrelu_slope=0.2,block_gmlp_factor=2, grid_gmlp_factor=2,
                input_proj_factor=2, channels_reduction=4, dropout_rate=0., use_bias=True):
      
        super().__init__()
   
        self.num_channels = num_channels
        self.block_size = block_size
        self.grid_size = grid_size
        self.lrelu_slope = lrelu_slope
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.reduction = channels_reduction
        self.drop = dropout_rate
        self.use_bias = use_bias

        self.SplitHeadMultiAxisGmlpLayer = ResidualSplitHeadMultiAxisGmlpLayer(block_size=self.block_size, 
                            grid_size=self.grid_size, num_channels=self.num_channels, input_proj_factor=self.input_proj_factor,
                            block_gmlp_factor=self.block_gmlp_factor, grid_gmlp_factor=self.grid_gmlp_factor, dropout_rate=self.drop, use_bias=self.use_bias)

    def forward(self, x):

        x = x.permute(0,2,3,1)  #n,h,w,c

        x = self.SplitHeadMultiAxisGmlpLayer(x)

        x = x.permute(0,3,1,2)  #n,c,h,w

        return x
    
class UNetDecoderBlock(nn.Module):
    def __init__(self, num_channels, block_size, grid_size, lrelu_slope=0.2,block_gmlp_factor=2, grid_gmlp_factor=2,
                input_proj_factor=2, channels_reduction=4, dropout_rate=0., use_bias=True):
      
        super().__init__()
   
        self.num_channels = num_channels
        self.block_size = block_size
        self.grid_size = grid_size
        self.lrelu_slope = lrelu_slope
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.reduction = channels_reduction
        self.drop = dropout_rate
        self.use_bias = use_bias
        
        self.SplitHeadMultiAxisGmlpLayer = ResidualSplitHeadMultiAxisGmlpLayer(block_size=self.block_size, 
                            grid_size=self.grid_size, num_channels=self.num_channels, input_proj_factor=self.input_proj_factor,
                            block_gmlp_factor=self.block_gmlp_factor, grid_gmlp_factor=self.grid_gmlp_factor, dropout_rate=self.drop, use_bias=self.use_bias)

        self.cross_attention = CrossAttention(self.num_channels)

    def forward(self, x, enc):
        
        x = x.permute(0,2,3,1)  #n,h,w,c
        enc = enc.permute(0,2,3,1) #n,h,w,c

        x = self.SplitHeadMultiAxisGmlpLayer(x)
        x = self.cross_attention(x, enc)

        x = x.permute(0,3,1,2)  #n,c,h,w

        return x
    
class DVQAModel(nn.Module):
    def __init__(self, channels=12, use_bias=True):
        super().__init__()

        self.channels = channels
        self.bias = use_bias

        self.enc_conv_1 = nn.Conv2d(3,self.channels, kernel_size=(7,7), bias=self.bias, padding=3, stride=2)
        self.enc_block_1 = UNetEncoderBlock(num_channels= self.channels, block_size=(32, 32), grid_size=(32, 32))

        self.enc_conv_2 = nn.Conv2d(self.channels, 2 * self.channels,kernel_size=(3,3),stride=2,padding=1)
        self.enc_block_2 = UNetEncoderBlock(num_channels= 2 * self.channels, block_size=(16, 16), grid_size=(16, 16))

        self.enc_conv_3 = nn.Conv2d(2 * self.channels, 4 * self.channels,kernel_size=(3,3),stride=2,padding=1)
        self.enc_block_3 = UNetEncoderBlock(num_channels= 4 * self.channels, block_size=(8, 8), grid_size=(8, 8))

        self.enc_conv_4 = nn.Conv2d(4 * self.channels, 8 * self.channels,kernel_size=(3,3),stride=2,padding=1)
        self.enc_block_4 = UNetEncoderBlock(num_channels= 8 * self.channels, block_size=(4, 4), grid_size=(4, 4))

        self.dec_embedding = nn.Parameter(torch.randn(1, 8 * self.channels, 64, 64))

        self.dec_block_4 = UNetDecoderBlock(num_channels= 8 * self.channels, block_size=(4, 4), grid_size=(4, 4))
        self.dec_conv_4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                        nn.Conv2d(8 * self.channels, 4 * self.channels, kernel_size=(3,3), bias=self.bias, padding=1, stride=1))
        
        self.dec_block_3 = UNetDecoderBlock(num_channels= 4 * self.channels, block_size=(8, 8), grid_size=(8, 8))
        self.dec_conv_3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                        nn.Conv2d(4 * self.channels, 2 * self.channels, kernel_size=(3,3), bias=self.bias, padding=1, stride=1))
        
        self.dec_block_2 = UNetDecoderBlock(num_channels= 2 * self.channels, block_size=(16, 16), grid_size=(16, 16))
        self.dec_conv_2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                        nn.Conv2d(2 * self.channels, self.channels, kernel_size=(3,3), bias=self.bias, padding=1, stride=1))

        self.dec_block_1 = UNetDecoderBlock(num_channels= self.channels, block_size=(32, 32), grid_size=(32, 32))
        self.dec_conv_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                        nn.Conv2d(self.channels, 1, kernel_size=(3,3), bias=self.bias, padding=1, stride=1))
        
    def forward(self, input):
        batch_size = input.shape[0]

        x = self.enc_conv_1(input)
        x_enc_block_1 = self.enc_block_1(x)

        x = self.enc_conv_2(x_enc_block_1)
        x_enc_block_2 = self.enc_block_2(x)
        
        x = self.enc_conv_3(x_enc_block_2)
        x_enc_block_3 = self.enc_block_3(x)

        x = self.enc_conv_4(x_enc_block_3)
        x_enc_block_4 = self.enc_block_4(x)
        
        x_dec = self.dec_embedding.repeat(batch_size, 1, 1, 1)
        x_dec = self.dec_block_4(x_dec, x_enc_block_4)
        x_dec = self.dec_conv_4(x_dec)

        x_dec = self.dec_block_3(x_dec, x_enc_block_3)
        x_dec = self.dec_conv_3(x_dec)

        x_dec = self.dec_block_2(x_dec, x_enc_block_2)
        x_dec = self.dec_conv_2(x_dec)

        x_dec = self.dec_block_1(x_dec, x_enc_block_1)
        x_dec = self.dec_conv_1(x_dec)

        return x_dec.squeeze()
