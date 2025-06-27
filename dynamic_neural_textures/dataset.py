import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

class VideoDataset(Dataset):
    """视频数据集 - 用于加载视频帧序列"""
    def __init__(self, video_path: str, transform=None):
        super().__init__()
        # 这里应该实现视频帧的读取逻辑
        # 简化起见，我们假设已经有帧图像存储在文件夹中
        self.frame_paths = sorted([os.path.join(video_path, f) for f in os.listdir(video_path) 
                                  if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transform
        
    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self, idx):
        # 读取图像
        img = Image.open(self.frame_paths[idx])
        
        # 应用变换
        if self.transform:
            img = self.transform(img)
        
        # 计算时间戳(归一化到0-1)
        time = idx / (len(self) - 1) if len(self) > 1 else 0.5
        
        return {
            'image': img,
            'time': time
        }

class ImageSequenceDataset(Dataset):
    """图像序列数据集 - 用于加载有序的图像序列"""
    def __init__(self, image_dir: str, time_interval: float = 1.0, transform=None):
        super().__init__()
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                                  if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.time_interval = time_interval
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 读取图像
        img = Image.open(self.image_paths[idx])
        
        # 应用变换
        if self.transform:
            img = self.transform(img)
        
        # 计算时间戳
        time = idx * self.time_interval / (len(self) - 1) if len(self) > 1 else 0.5
        
        return {
            'image': img,
            'time': time
        }    