from parser_my import args
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.1),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.4179, 0.4867, 0.4081], [0.0542, 0.0461, 0.1009])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4179, 0.4867, 0.4081], [0.0542, 0.0461, 0.1009])
    ]),
}

dataset={
    'train': ImageFolder('./data/train',transform=data_transforms['train']),
    'val': ImageFolder('./data/val',transform=data_transforms['val']),
    'test': ImageFolder('./data/test',transform=data_transforms['val'])
}

dataloader = {
    'train': DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True),
    'val': DataLoader(dataset['val'], batch_size=args.batch_size, shuffle=False),
    'test': DataLoader(dataset['test'], batch_size=args.batch_size, shuffle=False)
}