from typing import Dict

from torch import nn 

class Embedding(nn.Module):
    def __init__(self):
        super().__init__()

class VisionTransformer(nn.Module):
    def __init__(self, cfg: Dict[str]) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg['PATCH_SIZE']

    def forward(self):
        pass