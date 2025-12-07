# datasets/transforms_rgbt.py
import random
import torch
import torchvision.transforms.functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img_rgb, img_t, target):
        for t in self.transforms:
            img_rgb, img_t, target = t(img_rgb, img_t, target)
        return img_rgb, img_t, target

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_rgb, img_t, target):
        if random.random() < self.p:
            # RGB 和 Thermal 必须同时翻转
            img_rgb = F.hflip(img_rgb)
            img_t = F.hflip(img_t)
            # 翻转 BBox 坐标
            if "boxes" in target:
                bbox = target["boxes"]
                # bbox format: [cx, cy, w, h] normalized or [x0, y0, x1, y1]
                # 这里假设是 DETR/MOTR 标准的相对坐标 cx,cy,w,h (0-1)
                # 翻转只需改变 cx: cx_new = 1.0 - cx
                bbox[:, 0] = 1.0 - bbox[:, 0]
                target["boxes"] = bbox
        return img_rgb, img_t, target

class RandomSelect(object):
    """
    为了演示，这里只写了翻转。
    实际 MOTR 还需要 RandomResize, RandomCrop 等。
    关键在于：生成一个随机参数，然后分别应用到 rgb 和 t。
    """
    def __init__(self, transform1, transform2, p=0.5):
        self.transform1 = transform1
        self.transform2 = transform2
        self.p = p

    def __call__(self, img_rgb, img_t, target):
        if random.random() < self.p:
            return self.transform1(img_rgb, img_t, target)
        return self.transform2(img_rgb, img_t, target)

class ToTensor(object):
    def __call__(self, img_rgb, img_t, target):
        # 转换为 Tensor 并归一化 (0-1)
        img_rgb = F.to_tensor(img_rgb)
        img_t = F.to_tensor(img_t)
        return img_rgb, img_t, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img_rgb, img_t, target):
        # RGB 标准化
        img_rgb = F.normalize(img_rgb, mean=self.mean, std=self.std)
        # Thermal 标准化 (通常 Thermal 是单通道或伪彩色，这里假设我们也做标准化)
        # 如果 Thermal 是单通道，需要调整 mean/std 维度
        if img_t.shape[0] == 1:
             img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
        else:
             img_t = F.normalize(img_t, mean=self.mean, std=self.std)
        return img_rgb, img_t, target

# 构建基础的训练变换
def make_rgbt_transforms(image_set):
    normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    if image_set == 'train':
        return Compose([
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ])

    if image_set == 'val':
        return Compose([
            ToTensor(),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')