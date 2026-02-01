from PIL import Image
from typing import Iterable, Iterator

from torch.utils.data import IterableDataset

class ImageDataset(IterableDataset): 
    def __init__(self, images: Iterable[Image.Image], labels: Iterable[int]) -> None:
        self.images = images
        self.labels = labels

    def __iter__(self) -> Iterator[tuple[Image.Image, int]]:
        '''Return an iterator object containing the respective image and label'''
        return zip(self.images, self.labels)