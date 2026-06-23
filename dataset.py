import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class PoolBallCountingDataset(Dataset):
    def __init__(self, json_path, img_dir, transforms=None, return_path=False):
        self.img_dir = img_dir
        self.transforms = transforms
        self.return_path = return_path

        with open(json_path, 'r') as f:
            self.coco_data = json.load(f)
            
        self.images = self.coco_data['images']

        self.image_id_to_count = {img['id']: 0 for img in self.images}
        for ann in self.coco_data['annotations']:
            if ann['image_id'] in self.image_id_to_count:
                self.image_id_to_count[ann['image_id']] += 1

        self.valid_images = []
        for img in self.images:
            count = self.image_id_to_count[img['id']]
            if count <= 16:
                self.valid_images.append(img)

        removed_count = len(self.images) - len(self.valid_images)
        if removed_count > 0:
            print(f"Removed {removed_count} noisy images from {json_path}")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_info = self.valid_images[idx]
        file_name = img_info['file_name'] 
        img_path = os.path.join(self.img_dir, file_name)

        image = Image.open(img_path).convert("RGB")

        count = float(self.image_id_to_count[img_info['id']])
        count_tensor = torch.tensor([count], dtype=torch.float32)
        
        if self.transforms:
            image = self.transforms(image)

        if self.return_path:
            return image, count_tensor, file_name  
          
        return image, count_tensor