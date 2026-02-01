from PIL import Image
from typing import Iterable, Iterator

import torch 
from torch import Tensor
from torchvision.transforms import v2
from torch.utils.data import IterableDataset

class ImageDataset(IterableDataset): 
    def __init__(self, images: Iterable[Image.Image], labels: Iterable[int]) -> None:
        self.images = images
        self.labels = labels
        self.transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def __iter__(self) -> Iterator[tuple[Tensor, int]]:
        '''
        Return an iterator object containing the respective image and label.
        The image object is transformed from a PIL (H, W, C) -> Tensor image (C, H, W)
        
        note to self: ideally move the transforms out to the DataLoader to apply per-batch
        '''
        transformed_images = map(self.transforms, self.images)
        return zip(transformed_images, self.labels)