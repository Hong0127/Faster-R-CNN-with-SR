import torch
from torch.utils.data import Dataset
import os
import json
from PIL import Image

class SOD4SBDataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        with open(annFile) as f:
            self.annotations = json.load(f)
        self.transform = transform

        self.images = {img['id']: img for img in self.annotations['images']}
        self.annotations = [ann for ann in self.annotations['annotations']]

    def __getitem__(self, index):
        ann = self.annotations[index]
        img_id = ann['image_id']
        img_info = self.images[img_id]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)

        boxes = torch.tensor([ann['bbox']], dtype=torch.float32)
        labels = torch.tensor([ann['category_id']], dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([img_id])
        target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        target['iscrowd'] = torch.zeros((1,), dtype=torch.int64)

        return img, target

    def __len__(self):
        return len(self.annotations)