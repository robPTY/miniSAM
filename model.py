from typing import Dict

import torch
from torch import nn, Tensor

class PatchLayer(nn.Module):
    def __init__(self, cfg: Dict[str, int]) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg['PATCH_SIZE']
        self.conv_layer = nn.Conv2d(
            in_channels=cfg['IN_CHANNELS'],
            out_channels=cfg['D'],
            kernel_size=cfg['PATCH_SIZE'],
            stride=cfg['PATCH_SIZE']
        )
        self.num_patches = (cfg["IMG_SIZE"] // cfg["PATCH_SIZE"]) ** 2
        self.class_token = nn.Parameter(torch.randn(cfg['BATCH_SIZE'], 1, cfg['D']), requires_grad=True)
        self.pos_embeds = nn.Parameter(torch.randn(1, self.num_patches+1, cfg['D']))
    
    def forward(self, x: Tensor) -> Tensor:
        '''Return a tensor containing all the flattened image patch embeddings with PE and class token'''
        x = self.conv_layer(x) # (B, C, IMG_SIZE, IMG_SIZE) -> (B, emb_dim, 14, 14)
        x = x.flatten(2) # (B, emb_dim, 196)
        x = x.transpose(1, 2) # (B, 196, emb_dim)
        x = torch.cat((self.class_token, x), dim=1) # (B, 197, emb_dim)
        x = x + self.pos_embeds # (B, 197, emb_dim)
        return x
    
class MLPBlock(nn.Module):
    def __init__(self, cfg: Dict[str, int]) -> None:
        super().__init__() 
        self.W1 = nn.Linear(cfg['D'], cfg['MLP_SIZE'])
        self.W2 = nn.Linear(cfg['MLP_SIZE'], cfg['D'])
        self.m = nn.GELU()
        self.dropout = nn.Dropout(cfg['p'])
    
    def forward(self, x: Tensor) -> Tensor:
        '''Return a tensor being passed through two MLP layers and GELU non-linearity'''
        xForward = self.W1(x)
        xGelud = self.m(xForward)
        return self.W2(self.dropout(xGelud))

# class Encoder(nn.Module):
#     def __init__(self, cfg: Dict[str, int]):
#         super().__init__()
#         self.MLP_block = MLPBlock(cfg)


class VisionTransformer(nn.Module):
    def __init__(self, cfg: Dict[str, int]) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg['PATCH_SIZE']
        self.image_size = cfg['IMG_SIZE']
        assert self.image_size % self.patch_size == 0, "Image size must be divisible by patch size"
        self.patch_layer = PatchLayer(cfg)

    def forward(self, x: Tensor):
        x = self.patch_layer(x)