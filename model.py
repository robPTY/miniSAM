from typing import Dict

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
    
    def forward(self, x: Tensor) -> Tensor:
        '''Return a tensor containing all the flattened image patches'''
        x = self.conv_layer(x) # (B, C, IMG_SIZE, IMG_SIZE) -> (B, emb_dim, 14, 14)
        x = x.flatten(2) # (B, emb_dim, 196)
        x = x.transpose(1, 2) # (B, 196, emb_dim)
        return x


class Embedding(nn.Module):
    def __init__(self, cfg: Dict[str, int]) -> None:
        super().__init__()
        self.cfg = cfg 
        self.emb_dim = cfg['D']
        # self.emb_layer = nn.Embedding(, cfg['D']) 
    
    def forward(self, x: Tensor) -> Tensor:
        pass


class VisionTransformer(nn.Module):
    def __init__(self, cfg: Dict[str, int]) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg['PATCH_SIZE']
        self.image_size = cfg['IMAGAE_SIZE']
        assert self.image_size % self.patch_size == 0, "Image size must be divisible by patch size"
        self.patch_layer = PatchLayer(cfg['PATCH_SIZE'])
        self.embedding = Embedding()

    def forward(self, x: Tensor):
        x = self.patch_layer(x)