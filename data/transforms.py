import torch
import torchvision.transforms.functional as TF
import random

class TrainTransforms:
    def __init__(self, img_size=(640, 640), p_flip=0.5):
        self.img_size = img_size
        self.p_flip = p_flip

    def __call__(self, image, targets):
        if not isinstance(image, torch.Tensor):
            image = TF.to_tensor(image)
        _, h0, w0 = image.shape
        
        # Color Jitter 증강 (밝기/대비 랜덤 조절)
        if random.random() < 0.5:
            image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image, contrast_factor=random.uniform(0.8, 1.2))
            
        # Resize
        image = TF.resize(image, list(self.img_size))
        sx = self.img_size[1] / w0
        sy = self.img_size[0] / h0
        
        if targets['boxes'].numel() > 0:
            targets['boxes'] = targets['boxes'] * torch.tensor([sx, sy, sx, sy])
            
        # 수평 뒤집기 증강 (Horizontal Flip)
        if random.random() < self.p_flip:
            image = TF.hflip(image)
            # Bounding Box도 함께 뒤집어줍니다 (x축 반전)
            if targets['boxes'].numel() > 0:
                boxes = targets['boxes']
                new_xmin = self.img_size[1] - boxes[:, 2]
                new_xmax = self.img_size[1] - boxes[:, 0]
                boxes[:, 0] = new_xmin
                boxes[:, 2] = new_xmax
                targets['boxes'] = boxes
                
        return image, targets

class ValTransforms:
    def __init__(self, img_size=(640, 640)):
        self.img_size = img_size

    def __call__(self, image, targets):
        if not isinstance(image, torch.Tensor):
            image = TF.to_tensor(image)
        _, h0, w0 = image.shape
        image = TF.resize(image, list(self.img_size))
        sx, sy = self.img_size[1] / w0, self.img_size[0] / h0
        if targets['boxes'].numel() > 0:
            targets['boxes'] = targets['boxes'] * torch.tensor([sx, sy, sx, sy])
        return image, targets

