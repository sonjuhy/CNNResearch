import torch
import torchvision.transforms.functional as TF
import random

def letterbox(image, size=(640, 640), pad_value=114/255):
    """
    비율을 유지하며 리사이즈하고 남은 여백을 회색 패딩(pad_value)으로 채우는 letterbox 변환.
    Returns:
        canvas: (3, target_h, target_w)
        r: 스케일 비율
        (pad_x, pad_y): 좌우/상하 패딩 (left, top)
    """
    if not isinstance(image, torch.Tensor):
        image = TF.to_tensor(image)
    _, h0, w0 = image.shape
    target_h, target_w = size if isinstance(size, (tuple, list)) else (size, size)
    
    r = min(target_h / h0, target_w / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    
    # 리사이즈
    resized = TF.resize(image, [nh, nw], antialias=True)
    
    pad_y = (target_h - nh) // 2
    pad_x = (target_w - nw) // 2
    
    canvas = torch.full((3, target_h, target_w), pad_value, dtype=image.dtype, device=image.device)
    canvas[:, pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    
    return canvas, r, (pad_x, pad_y)

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
            
        # Letterbox 패딩 리사이즈 (종횡비 보존)
        image, r, (pad_x, pad_y) = letterbox(image, self.img_size)
        
        if targets['boxes'].numel() > 0:
            boxes = targets['boxes'] * r
            boxes[:, [0, 2]] += pad_x
            boxes[:, [1, 3]] += pad_y
            targets['boxes'] = boxes
            
        # 수평 뒤집기 증강 (Horizontal Flip)
        if random.random() < self.p_flip:
            image = TF.hflip(image)
            # Bounding Box도 함께 뒤집어줍니다 (x축 반전)
            if targets['boxes'].numel() > 0:
                target_w = self.img_size[1] if isinstance(self.img_size, (tuple, list)) else self.img_size
                boxes = targets['boxes']
                new_xmin = target_w - boxes[:, 2]
                new_xmax = target_w - boxes[:, 0]
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
        
        # 원본 이미지 크기 및 박스 보존 (평가 시 역변환용)
        targets['orig_size'] = torch.tensor([w0, h0], dtype=torch.float32)
        targets['orig_boxes'] = targets['boxes'].clone()
        
        image, r, (pad_x, pad_y) = letterbox(image, self.img_size)
        targets['r'] = torch.tensor(r, dtype=torch.float32)
        targets['pad'] = torch.tensor([pad_x, pad_y], dtype=torch.float32)

        if targets['boxes'].numel() > 0:
            boxes = targets['boxes'] * r
            boxes[:, [0, 2]] += pad_x
            boxes[:, [1, 3]] += pad_y
            targets['boxes'] = boxes
            
        return image, targets

