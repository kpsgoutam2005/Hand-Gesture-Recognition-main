import os
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

class HandGestureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.image_paths = []
        self.labels = []
        
        print(f"\nLoading dataset from: {root_dir}")
        for label, class_name in enumerate(tqdm(self.classes, desc="Processing classes")):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                raise FileNotFoundError(f"Class directory {class_dir} not found")
                
            for img_name in tqdm(os.listdir(class_dir), desc=f"Class {class_name}", leave=False):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.labels[idx]
