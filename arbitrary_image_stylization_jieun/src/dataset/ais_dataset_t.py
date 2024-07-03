import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class AisDataset(Dataset):
    def __init__(self, content_path, style_path, size=256):
        self.content_paths = self.read_file_list(content_path)
        self.style_paths = self.read_file_list(style_path)
        self.content_size = len(self.content_paths)
        self.style_size = len(self.style_paths)
        self.size = size

        self.content_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
        
        self.style_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.8, saturation=0.5, hue=0.2),
            transforms.ToTensor()
        ])
        
    def __getitem__(self, index):
        index1 = index % self.content_size
        index2 = index % self.style_size
        content_path = self.content_paths[index1]
        style_path = self.style_paths[index2]

        content_image = Image.open(content_path).convert('RGB')
        style_image = Image.open(style_path).convert('RGB')

        content_image = self.content_transform(content_image)
        style_image = self.style_transform(style_image)

        return content_image, style_image
    
    def read_file_list(self, path):
        """Read each image and its corresponding label from directory."""
        file_types = ['jpg', 'jpeg', 'png']
        images_path = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.split('.')[-1].lower() in file_types:
                    images_path.append(os.path.join(root, file))
        return images_path

    def __len__(self):
        return max(self.content_size, self.style_size)

def create_dataset(args):
    dataset = AisDataset(args.content_path, args.style_path, size=args.size)
    print("c path, s path: ", args.content_path, args.style_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=bool(args.shuffle), num_workers=args.num_workers, drop_last=True)

    return dataloader
