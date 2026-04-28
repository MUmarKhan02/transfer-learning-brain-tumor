import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import config_brain as config

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=3),  # MRI is grayscale, convert to 3 channel
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

def get_subset(dataset, fraction):
    if fraction >= 1.0:
        return dataset
    targets = np.array([dataset.targets[i] for i in range(len(dataset))])
    indices = []
    classes = np.unique(targets)
    n = int(len(dataset) * fraction)
    per_class = max(1, n // len(classes))
    for c in classes:
        class_idx = np.where(targets == c)[0]
        chosen = np.random.choice(class_idx, min(per_class, len(class_idx)), replace=False)
        indices.extend(chosen.tolist())
    return Subset(dataset, indices)

def get_loaders(split_fraction=1.0):
    train_dataset = datasets.ImageFolder(
        root="data/brain_tumor/Training",
        transform=get_transforms(train=True)
    )
    val_dataset = datasets.ImageFolder(
        root="data/brain_tumor/Testing",
        transform=get_transforms(train=False)
    )
    train_subset = get_subset(train_dataset, split_fraction)
    return (
        DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True,
                   num_workers=config.NUM_WORKERS, pin_memory=True),
        DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                   num_workers=config.NUM_WORKERS, pin_memory=True),
    )