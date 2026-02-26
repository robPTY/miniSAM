import wandb
from torch.utils.data import DataLoader

import torch
import torch.nn as nn

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

    model = VisionTransformer(configs)
    cost_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=configs['LR'], betas=(configs['BETA1'], configs['BETA2']), eps=configs['ADAM_EPS'])

    # Log in and start wandb run
    wandb.login()
    run = wandb.init(entity=config.entity, project=config.project, config=configs)
    wandb.watch(model, log="gradients", log_freq=100)

    global_step = 0
    curr_loss = 0.0
    curr_correct = 0
    curr_seen = 0

    for epoch in range(configs['EPOCHS']):
        model.train() 
        
        for images, labels in train_loader:
            # clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(images)
            loss = cost_function(logits, labels)
            # Backward pass
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            curr_loss += loss.item() * batch_size
            curr_correct += (logits.argmax(dim=-1) == labels).sum().item()
            curr_seen += batch_size
            global_step += 1

            if global_step % configs.get("LOG_EVERY", 50) == 0:
                train_loss = curr_loss / max(curr_seen, 1)
                train_acc = curr_correct / max(curr_seen, 1)

                wandb.log(
                    {"train/loss": train_loss, "train/acc": train_acc, "epoch": epoch},
                    step=global_step,
                )
                curr_loss = 0.0
                curr_correct = 0
                curr_seen = 0

        print(f"epoch {epoch+1}/{configs['EPOCHS']} done | step={global_step}")
    
    run.finish()
    return 1

if __name__ == "__main__":
    main()