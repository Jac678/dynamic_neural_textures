import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List

class PositionalEncoder(nn.Module):
    """位置编码器，将坐标映射到高维特征空间"""
    def __init__(self, input_dim: int, num_freqs: int, include_input: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.output_dim = 0
        
        if include_input:
            self.output_dim += input_dim
        
        self.output_dim += input_dim * 2 * num_freqs
        
        # 创建频率系数
        self.freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入坐标进行位置编码"""
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"输入必须是形状为 [N, {self.input_dim}] 的张量")
        
        output = [x] if self.include_input else []
        
        for freq in self.freq_bands:
            for func in [torch.sin, torch.cos]:
                output.append(func(x * freq))
                
        return torch.cat(output, dim=1)

class DynamicNeuralTexture(nn.Module):
    """动态神经纹理主类 - 学习随时间变化的纹理表示"""
    def __init__(
        self,
        spatial_resolution: Tuple[int, int] = (256, 256),
        temporal_resolution: int = 100,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4,
        use_positional_encoding: bool = True,
        pe_num_freqs: int = 6
    ):
        super().__init__()
        
        self.spatial_resolution = spatial_resolution
        self.temporal_resolution = temporal_resolution
        self.feature_dim = feature_dim
        
        # 位置编码器
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            # 2D空间坐标 + 1D时间坐标
            self.encoder = PositionalEncoder(3, pe_num_freqs)
            input_dim = self.encoder.output_dim
        else:
            input_dim = 3
        
        # 神经网络 - MLP
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, feature_dim + 1))  # +1 为alpha通道(透明度)
        self.mlp = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        前向传播，根据输入坐标(空间+时间)生成纹理特征和alpha值
        Args:
            coords: 形状为 [N, 3] 的张量，包含 [x, y, t] 坐标
        Returns:
            形状为 [N, feature_dim + 1] 的张量，包含特征和alpha值
        """
        if self.use_positional_encoding:
            encoded_coords = self.encoder(coords)
        else:
            encoded_coords = coords
            
        output = self.mlp(encoded_coords)
        features = output[:, :-1]  # 特征
        alpha = torch.sigmoid(output[:, -1:])  # alpha通道(0-1范围)
        
        return torch.cat([features, alpha], dim=1)
    
    def get_texture_at_time(self, time: float, resolution: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        获取指定时间点的纹理图像
        Args:
            time: 时间值(范围0-1)
            resolution: 输出纹理分辨率，默认为初始化时的分辨率
        Returns:
            形状为 [H, W, feature_dim + 1] 的纹理张量
        """
        if resolution is None:
            resolution = self.spatial_resolution
            
        h, w = resolution
        
        # 创建网格坐标
        y_coords = torch.linspace(0, 1, h)
        x_coords = torch.linspace(0, 1, w)
        
        # 创建 [x, y, t] 坐标网格
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='xy')
        t_grid = torch.full_like(x_grid, time)
        
        # 展平并组合坐标
        coords = torch.stack([x_grid.flatten(), y_grid.flatten(), t_grid.flatten()], dim=1)
        
        # 通过网络获取特征
        output = self(coords)
        
        # 重塑为图像形状
        return output.reshape(h, w, -1)
    
    def save(self, path: str):
        """保存模型到文件"""
        torch.save({
            'spatial_resolution': self.spatial_resolution,
            'temporal_resolution': self.temporal_resolution,
            'feature_dim': self.feature_dim,
            'model_state_dict': self.state_dict()
        }, path)

def load_dnt(path: str) -> DynamicNeuralTexture:
    """从文件加载动态神经纹理模型"""
    checkpoint = torch.load(path)
    
    dnt = DynamicNeuralTexture(
        spatial_resolution=checkpoint['spatial_resolution'],
        temporal_resolution=checkpoint['temporal_resolution'],
        feature_dim=checkpoint['feature_dim']
    )
    
    dnt.load_state_dict(checkpoint['model_state_dict'])
    return dnt    