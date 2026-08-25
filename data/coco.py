import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image

class COCODataset(Dataset):
    def __init__(self, root_dir, ann_file, transforms=None, is_train=True):
        self.root_dir = root_dir
        self.transforms = transforms
        self.is_train = is_train
        
        if os.path.exists(ann_file):
            with open(ann_file, 'r') as f:
                self.coco = json.load(f)
            self.images = {img['id']: img for img in self.coco['images']}
            
            cat_ids = sorted(c['id'] for c in self.coco['categories'])
            self.cat_id_to_label = {cid: i for i, cid in enumerate(cat_ids)}
            
            self.annotations = {}
            for ann in self.coco['annotations']:
                if ann.get('iscrowd', 0) == 1 or ann['bbox'][2] <= 1 or ann['bbox'][3] <= 1:
                    continue
                img_id = ann['image_id']
                if img_id not in self.annotations:
                    self.annotations[img_id] = []
                self.annotations[img_id].append(ann)
            self.img_ids = list(self.annotations.keys())
            self.dummy_mode = False
        else:
            self.dummy_mode = True
            self.img_ids = list(range(100))
            
    def __len__(self):
        return len(self.img_ids)
        
    def __getitem__(self, idx):
        if self.dummy_mode:
            img = torch.randn(3, 640, 640)
            targets = {
                'boxes': torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
                'labels': torch.tensor([0], dtype=torch.long)
            }
            if self.transforms:
                img, targets = self.transforms(img, targets)
            return img, targets
            
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        
        img = Image.open(img_path).convert("RGB")
        
        boxes = []
        labels = []
        for ann in self.annotations[img_id]:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_label[ann['category_id']])
            
        targets = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
        
        if self.transforms:
            img, targets = self.transforms(img, targets)
            
        return img, targets

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

def create_dataloader(root_dir, ann_file, batch_size=4, is_train=True, num_workers=0, transforms=None):
    dataset = COCODataset(root_dir, ann_file, transforms=transforms, is_train=is_train)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=is_train,
        num_workers=num_workers, 
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=is_train
    )
