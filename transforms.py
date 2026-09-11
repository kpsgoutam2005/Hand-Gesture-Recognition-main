from torchvision import transforms
from config import Config

def get_transforms():
    # Training transformations with augmentation
    train_transform = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.RandomRotation(Config.ROTATION_RANGE),
        transforms.RandomAffine(
            degrees=0,
            translate=(Config.TRANSLATE_RANGE, Config.TRANSLATE_RANGE),
            scale=(1-Config.ZOOM_RANGE, 1+Config.ZOOM_RANGE)
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Validation/test transformations
    val_test_transform = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return train_transform, val_test_transform