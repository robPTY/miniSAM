from dataset import ImageDataset
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
    train_set, valid_set, test_set = load_datasets()
    return 1

if __name__ == "__main__":
    main()