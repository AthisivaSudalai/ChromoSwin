import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=32):

    # same transforms for ALL three models — fair comparison
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(data_dir + '/train',
                                      transform=train_transforms)
    val_data   = datasets.ImageFolder(data_dir + '/val',
                                      transform=val_transforms)
    test_data  = datasets.ImageFolder(data_dir + '/test',
                                      transform=val_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_data,   batch_size=batch_size,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_data,  batch_size=batch_size,
                              shuffle=False, num_workers=0)

    print(f"Classes found: {train_data.classes}")
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    return train_loader, val_loader, test_loader