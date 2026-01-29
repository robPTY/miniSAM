from datasets import load_dataset
import matplotlib.pyplot as plt

# Load streaming dataset
dataset = load_dataset("imagenet-1k", streaming=True, split="train")

# Get first image
sample = next(iter(dataset))
plt.imshow(sample['image'])
plt.title(f"Label: {sample['label']}")
plt.axis('off')
plt.show()
