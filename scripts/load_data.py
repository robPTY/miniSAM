from PIL import Image

from datasets import load_dataset, Dataset
import matplotlib.pyplot as plt

from config import ViT_BASE_CFG as cfg

def resize_image(dataset: Dataset) -> Dataset:
    '''Return a dataset with re-sized images according to config's IMG_SIZE.'''
    target_size = (cfg["IMG_SIZE"], cfg["IMG_SIZE"])
    dataset['image'] = [img.convert("RGB").resize(target_size, Image.Resampling.BILINEAR) for img in dataset["image"]]
    return dataset

# Load streaming datasets
print("Loading datasets...")
train_data = load_dataset("imagenet-1k", streaming=True, split="train")
valid_data = load_dataset("imagenet-1k", streaming=True, split="validation")
test_data = load_dataset("imagenet-1k", streaming=True, split="test")

# Resize images
print("Resizing images...")
resized_train = train_data.map(resize_image, batched=True)
resized_valid = valid_data.map(resize_image, batched=True)
resized_test = test_data.map(resize_image, batched=True)