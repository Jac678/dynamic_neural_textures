import torch
import torch.nn as nn
import torch.nn.functional as F
from .dynamic_texture import DynamicNeuralTexture

class DNTRenderer(nn.Module):
    """动态神经纹理渲染器 - 将神经纹理渲染为RGB图像"""
    def __init__(self, feature_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        
        self.render_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # RGB输出
            nn.Sigmoid()  # 将输出限制在0-1范围内
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        将特征渲染为RGB图像
        Args:
            features: 形状为 [..., feature_dim] 的特征张量
        Returns:
            形状为 [..., 3] 的RGB图像张量
        """
        # 展平除最后一维外的所有维度
        original_shape = features.shape
        flattened_features = features.reshape(-1, original_shape[-1])
        
        # 通过渲染网络
        rgb = self.render_net(flattened_features)
        
        # 重塑回原始形状
        return rgb.reshape(*original_shape[:-1], 3)
    
    def render_texture(self, dnt: DynamicNeuralTexture, time: float, resolution: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        渲染指定时间点的神经纹理
        Args:
            dnt: 动态神经纹理模型
            time: 时间值(范围0-1)
            resolution: 输出图像分辨率
        Returns:
            形状为 [H, W, 3] 的RGB图像张量
        """
        # 获取指定时间的纹理特征
        texture = dnt.get_texture_at_time(time, resolution)
        
        # 分离特征和alpha通道
        features = texture[..., :-1]
        alpha = texture[..., -1:]
        
        # 渲染RGB颜色
        rgb = self(features)
        
        # 应用alpha通道(如果需要)
        return rgb * alpha + (1 - alpha)  # 背景为白色    