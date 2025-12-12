import deepinv
import torch
import torch.nn as nn
import math

class GPredictor(nn.Module):
    def __init__(self, backbone,physics_s, device):
        super(GPredictor, self).__init__()
        self.backbone = backbone
        self.physics_s = physics_s
        self.device = device
    
    def forward(self, x):
        # Forward pass through the backbone
        x = self.backbone(x)
        # Apply the physics model
        y = self.physics_s.A(x)
        return y




class ViT(nn.Module):
    """Vision Transformer for reconstruction."""
    
    def __init__(self, in_channels=1, embed_dim=64, num_heads=4, num_layers=2, 
                 patch_size=4, image_size=32):
        super(ViT, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Image projection
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Image reconstruction
        self.deproj = nn.ConvTranspose2d(embed_dim, in_channels, kernel_size=patch_size, stride=patch_size)
        
        
    def forward(self, x):
        x_proj = self.proj(x)
        b, c, h, w = x_proj.shape
        x_proj = x_proj.flatten(2).transpose(1,2)
        x_transformed = self.transformer(x_proj)
        x_transformed = x_transformed.transpose(1,2).view(b, c, h, w)
        x_reconstructed = self.deproj(x_transformed)
        
        return x_reconstructed
    


def double_conv(in_channels, out_channels):
    """Double convolution block for UNet."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """UNet implementation."""
    
    def __init__(self, n_channels, base_channel):
        super().__init__()

        self.dconv_down1 = double_conv(n_channels, base_channel)
        self.dconv_down2 = double_conv(base_channel, base_channel * 2)
        self.dconv_down3 = double_conv(base_channel * 2, base_channel * 4)
        self.dconv_down4 = double_conv(base_channel * 4, base_channel * 8)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.dconv_up3 = double_conv(base_channel * 12, base_channel * 4)
        self.dconv_up2 = double_conv(base_channel * 6, base_channel * 2)
        self.dconv_up1 = double_conv(base_channel * 3, base_channel)

        self.conv_last = nn.Conv2d(base_channel, n_channels, 1)

    def forward(self, x):
        # x = x.reshape(x.shape[0], 1, int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1])))
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        bootle = self.dconv_down4(x)

        x = self.upsample(bootle)
        x = torch.cat([x, conv3], dim=1)
        up1 = self.dconv_up3(x)

        x = self.upsample(up1)
        x = torch.cat([x, conv2], dim=1)
        up2 = self.dconv_up2(x)

        x = self.upsample(up2)
        x = torch.cat([x, conv1], dim=1)
        up3 = self.dconv_up1(x)

        out = self.conv_last(up3)
        return out


