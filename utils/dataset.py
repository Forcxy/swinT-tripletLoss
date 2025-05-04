import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
import cv2

def apply_gray_clahe(img, clip_limit=2.0, grid_size=(8, 8)):
    """函数式实现：RGB→灰度→CLAHE→复制三通道、用于数据增强"""
    # PIL转OpenCV格式
    img_np = np.array(img)

    # RGB转灰度
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # CLAHE增强
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced = clahe.apply(gray)

    # 复制为三通道并转回PIL格式
    rgb = np.stack([enhanced] * 3, axis=-1)  # HWC格式
    return Image.fromarray(rgb)


class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        构建样式：
        样本 0: 路径=demo_data/cat/cat001.jpg, 标签=0
        样本 1: 路径=demo_data/cat/cat002.jpg, 标签=0
        样本 2: 路径=demo_data/dog/dog001.jpg, 标签=1
        样本 3: 路径=demo_data/dog/dog002.jpg, 标签=1
        """
        self.dataset = datasets.ImageFolder(root_dir)
        self.transform = transform
        self.labels = np.array([label for _, label in self.dataset.samples])
        self.label_to_indices = {
            label: np.where(self.labels == label)[0]
            for label in set(self.labels)
        }

    def __getitem__(self, index):
        # 获取anchor
        anchor_img, anchor_label = self.dataset[index]

        # 随机选择positive
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[anchor_label])
        positive_img, _ = self.dataset[positive_index]

        # 随机选择negative
        negative_label = np.random.choice(list(set(self.labels) - set([anchor_label])))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        negative_img, _ = self.dataset[negative_index]

        # 转换图像
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img

    def __len__(self):
        return len(self.dataset)


def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    # 数据转换
    transform = transforms.Compose([
        transforms.Lambda(apply_gray_clahe),  # 使用独立函数
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    train_dataset = TripletDataset(
        os.path.join(data_dir, "train"),
        transform=transform
    )

    val_dataset = TripletDataset(
        os.path.join(data_dir, "val"),
        transform=transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader