from typing import Dict

import torch
from torch import nn, Tensor
from robertorch import LayerNorm, MLPBlockGeLU, ResidualLayer, MultiHeadAttention

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
        self.class_token = nn.Parameter(torch.randn(1, 1, cfg['D']), requires_grad=True)
        self.pos_embeds = nn.Parameter(torch.randn(1, self.num_patches+1, cfg['D']))
    
    def forward(self, x: Tensor) -> Tensor:
        '''Return a tensor containing all the flattened image patch embeddings with PE and class token'''
        x = self.conv_layer(x) # (B, C, IMG_SIZE, IMG_SIZE) -> (B, emb_dim, 14, 14)
        x = x.flatten(2) # (B, emb_dim, 196)
        x = x.transpose(1, 2) # (B, 196, emb_dim)

        B = x.shape[0]
        cls = self.class_token.expand(B, -1, -1) # (B, 1, D)

        x = torch.cat((cls, x), dim=1) # (B, 197, emb_dim)
        x = x + self.pos_embeds # (B, 197, emb_dim)
        return x


class Encoder(nn.Module):
    def __init__(self, cfg: Dict[str, int]):
        super().__init__()
        self.MLP_block = MLPBlockGeLU(cfg)
        self.layer_norm = LayerNorm(cfg['D'], cfg['EPS'])
        self.MHA_block = MultiHeadAttention(cfg)
        # Residual connection after MHA and after MLP block
        self.residuals = nn.ModuleList([ResidualLayer(cfg) for _ in range(2)])
    
    def forward(self, x: Tensor, mask: Tensor):
        # The first residual will call Res(x + Dropout(MHA(Norm(x), Mask))))
        x = self.residuals[0](x, lambda x: self.MHA_block(x, x, x, mask), post_norm=False)
        # The second residual will call Res(x + Dropout(MLP(Norm(x))))
        x = self.residuals[1](x, self.MLP_block, post_norm=False)
        return x

class MLPHead(nn.Module):
    def __init__(self, cfg: Dict[str, int]) -> None:
        super().__init__()
        self.linear_layer = nn.Linear(cfg['D'], cfg['NUM_CLASSES'])
    
    def forward(self, x: Tensor) -> Tensor:
        cls = x[:, 0]
        logits = self.linear_layer(cls)
        return logits


class VisionTransformer(nn.Module):
    def __init__(self, cfg: Dict[str, int]) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg['PATCH_SIZE']
        self.image_size = cfg['IMG_SIZE']
        assert self.image_size % self.patch_size == 0, "Image size must be divisible by patch size"
        self.patch_layer = PatchLayer(cfg)
        self.encoder_layers = nn.ModuleList([Encoder(cfg) for _ in range(cfg['L'])])
        self.mlp_head = MLPHead(cfg)
        self.mask = None

    def forward(self, x: Tensor):
        x = self.patch_layer(x) # embedded patches
        for layer in self.encoder_layers:
            x = layer(x, self.mask)
        # pre-MLP head, x should be of shape (B, N+1, D)
        return self.mlp_head(x)  