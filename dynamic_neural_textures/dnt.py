import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

class DynamicNeuralTexture(nn.Module):
    """
    动态神经纹理 (Dynamic Neural Texture) 模型
    将时空坐标映射到纹理特征和透明度
    """
    def __init__(
        self,
        spatial_resolution: Tuple[int, int] = (256, 256),
        temporal_resolution: int = 100,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4,
        use_pos_encoding: bool = True,
        activation: str = 'relu',
        output_alpha: bool = True
    ):
        super().__init__()
        self.spatial_resolution = spatial_resolution
        self.temporal_resolution = temporal_resolution
        self.feature_dim = feature_dim
        self.output_alpha = output_alpha
        
        # 位置编码
        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            self.pos_encoder = PositionalEncoder(3, hidden_dim)  # x, y, t
            input_dim = hidden_dim
        else:
            input_dim = 3
        
        # 主网络
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = feature_dim + 1 if i == num_layers - 1 else hidden_dim  # +1 为 alpha 通道
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:  # 最后一层不使用激活函数
                layers.append(self._get_activation(activation))
        
        self.mlp = nn.Sequential(*layers)
        
    def _get_activation(self, name: str) -> nn.Module:
        """获取激活函数"""
        if name.lower() == 'relu':
            return nn.ReLU()
        elif name.lower() == 'gelu':
            return nn.GELU()
        elif name.lower() == 'sine':
            return Sine()
        else:
            raise ValueError(f"不支持的激活函数: {name}")
        
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        将时空坐标映射到纹理特征和透明度
        
        Args:
            coords: 输入坐标，形状为 [batch_size, 3] (x, y, t)
            
        Returns:
            output: 输出特征，形状为 [batch_size, feature_dim + 1]
                    最后一维为 alpha 通道 (透明度)
        """
        if self.use_pos_encoding:
            x = self.pos_encoder(coords)
        else:
            x = coords
            
        output = self.mlp(x)
        
        # 应用 sigmoid 到 alpha 通道
        if self.output_alpha:
            features = output[:, :-1]  # 特征
            alpha = torch.sigmoid(output[:, -1:])  # alpha 通道
            output = torch.cat([features, alpha], dim=1)
            
        return output
    
    def sample(self, x: float, y: float, t: float) -> torch.Tensor:
        """
        在特定时空点采样纹理特征
        
        Args:
            x, y: 空间坐标 (范围 [0, 1])
            t: 时间坐标 (范围 [0, 1])
            
        Returns:
            output: 采样的特征和透明度 [feature_dim + 1]
        """
        coords = torch.tensor([[x, y, t]], device=self.mlp[0].weight.device)
        return self(coords)
    
    def generate_texture_at_time(self, time: float, resolution: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        在指定时间生成完整纹理
        
        Args:
            time: 时间值 (范围: 0.0-1.0)
            resolution: 输出纹理分辨率，默认为初始化时的分辨率
            
        Returns:
            texture: 生成的纹理，形状为 [1, feature_dim+1, H, W]
                     最后一个通道是 alpha
        """
        if resolution is None:
            resolution = self.spatial_resolution
            
        # 创建网格坐标
        device = next(self.parameters()).device
        y_coords = torch.linspace(0, 1, resolution[0], device=device)
        x_coords = torch.linspace(0, 1, resolution[1], device=device)
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        # 添加时间维度
        t_grid = torch.full_like(x_grid, time)
        
        # 重塑为 [N, 3]
        coords = torch.stack([x_grid.flatten(), y_grid.flatten(), t_grid.flatten()], dim=1)
        
        # 生成特征
        with torch.no_grad():
            output = self(coords)
            
        # 重塑为图像格式 [1, feature_dim+1, H, W]
        output = output.reshape(1, self.feature_dim + 1, resolution[0], resolution[1])
        return output


class PositionalEncoder(nn.Module):
    """位置编码器，用于增强输入坐标的表达能力"""
    def __init__(self, input_dim: int, output_dim: int, num_frequencies: int = 6):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        
        # 创建频率系数
        frequencies = 2.0 ** torch.linspace(0.0, num_frequencies - 1, num_frequencies)
        self.register_buffer('frequencies', frequencies)
        
        # 线性投影层
        self.projection = nn.Linear(input_dim * (1 + 2 * num_frequencies), output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用位置编码
        
        Args:
            x: 输入坐标，形状为 [batch_size, input_dim]
            
        Returns:
            encoded: 编码后的特征，形状为 [batch_size, output_dim]
        """
        # 原始坐标
        encoding = [x]
        
        # 添加正弦和余弦编码
        for freq in self.frequencies:
            encoding.append(torch.sin(freq * x))
            encoding.append(torch.cos(freq * x))
            
        # 拼接所有编码
        encoding = torch.cat(encoding, dim=-1)
        
        # 线性投影
        encoded = self.projection(encoding)
        return encoded


class Sine(nn.Module):
    """Sine 激活函数: f(x) = sin(x)"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)

