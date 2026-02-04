import wandb
from torch.utils.data import DataLoader

import config
from dataset import ImageDataset
from model import VisionTransformer
from scripts.load_data import resized_train, resized_valid, resized_test

def load_datasets() -> tuple[ImageDataset, ImageDataset, ImageDataset]:
    '''Load the streaming datasets from hf into respective IterableDatasets'''
    train_images, train_labels = resized_train['image'], resized_train['label']
    valid_images, valid_labels = resized_valid['image'], resized_valid['label']
    test_images, test_labels = resized_test['image'], resized_test['label']

    train_set = ImageDataset(train_images, train_labels)
    valid_set = ImageDataset(valid_images, valid_labels)
    test_set = ImageDataset(test_images, test_labels)

    return train_set, valid_set, test_set

def main():
    configs = config.ViT_BASE_CFG
    train_set, valid_set, test_set = load_datasets()

    # Shuffle is turned to False by default because its streamed
    # If i want to shuffle, should find a different way (ideally in Dataset)
    train_loader = DataLoader(train_set, batch_size=configs['BATCH_SIZE'])
    images, labels = next(iter(train_loader))

    model = VisionTransformer(configs)
    model(images)

    # Log in and start wandb run
    wandb.login()
    run = wandb.init(entity=config.entity, project=config.project, config=configs)
    
    return 1

if __name__ == "__main__":
    main()