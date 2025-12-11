import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class MultiClassSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dirs, transform=None, resize=(256,256)):
        """
        image_dir : Ordner mit Originalbildern
        mask_dirs : dict, z.B. {'text': '...', 'curve': '...', 'grid': '...'}
        transform : Transformation auf Bilder (z.B. ToTensor)
        resize : Tuple (H, W) für Bild + Masken Resize
        """
        self.image_dir = image_dir
        self.mask_dirs = mask_dirs
        self.transform = transform
        self.resize = resize
        self.images = os.listdir(image_dir)
        self.class_names = list(mask_dirs.keys())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.resize:
            image = image.resize(self.resize)

        mask_arr = np.zeros((self.resize[1], self.resize[0]), dtype=np.uint8)  # HxW

        for class_idx, class_name in enumerate(self.class_names, start=1):
            mask_path = os.path.join(self.mask_dirs[class_name], img_name)
            if os.path.exists(mask_path):
                class_mask = Image.open(mask_path).convert("L")
                if self.resize:
                    class_mask = class_mask.resize(self.resize)
                mask_arr[np.array(class_mask) > 0] = class_idx

        mask = torch.as_tensor(mask_arr, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, mask


def get_dataloader(image_dir, mask_dirs, batch_size=1, shuffle=True, transform=None, resize=(256,256)):
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])
    dataset = MultiClassSegmentationDataset(image_dir, mask_dirs, transform=transform, resize=resize)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
