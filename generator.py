import torch
import torch.nn as nn
from typing import Optional, Tuple

class NeuralTextureGenerator(nn.Module):
    """
    神经纹理生成器：将输入参数映射到神经纹理特征
    """
    def __init__(
        self,
        spatial_resolution: Tuple[int, int] = (256, 256),
        temporal_resolution: int = 100,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4,
        use_pos_encoding: bool = True,
        activation: str = 'relu'
    ):
        super().__init__()
        self.spatial_resolution = spatial_resolution
        self.temporal_resolution = temporal_resolution
        self.feature_dim = feature_dim
        self.use_pos_encoding = use_pos_encoding
        
        # 选择激活函数
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation.lower() == 'sine':
            self.activation = lambda x: torch.sin(x)
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 输入维度：2D空间坐标 + 时间
        input_dim = 3  # x, y, time
        
        # 位置编码（可选）
        if use_pos_encoding:
            self.pos_encoder = self._create_positional_encoder(input_dim, hidden_dim)
            mlp_input_dim = hidden_dim
        else:
            mlp_input_dim = input_dim
        
        # 主MLP网络
        layers = []
        for i in range(num_layers):
            in_dim = mlp_input_dim if i == 0 else hidden_dim
            out_dim = feature_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:  # 最后一层不使用激活函数
                layers.append(self.activation)
        
        self.mlp = nn.Sequential(*layers)
        
    def _create_positional_encoder(self, input_dim: int, hidden_dim: int) -> nn.Module:
        """创建位置编码器"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation
        )
        
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        从坐标生成神经纹理特征
        
        Args:
            coords: 输入坐标，形状为 [batch_size, 3] (x, y, time)
            
        Returns:
            features: 生成的神经纹理特征，形状为 [batch_size, feature_dim]
        """
        if self.use_pos_encoding:
            x = self.pos_encoder(coords)
        else:
            x = coords
            
        features = self.mlp(x)
        return features
    
    def generate_texture_at_time(self, time: float, resolution: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        在指定时间生成完整纹理
        
        Args:
            time: 时间值 (范围: 0.0-1.0)
            resolution: 输出纹理分辨率，默认为初始化时的分辨率
            
        Returns:
            texture: 生成的纹理，形状为 [1, feature_dim, H, W]
        """
        if resolution is None:
            resolution = self.spatial_resolution
            
        # 创建网格坐标
        y_coords = torch.linspace(0, 1, resolution[0], device=next(self.parameters()).device)
        x_coords = torch.linspace(0, 1, resolution[1], device=next(self.parameters()).device)
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        # 添加时间维度
        t_grid = torch.full_like(x_grid, time)
        
        # 重塑为 [N, 3]
        coords = torch.stack([x_grid.flatten(), y_grid.flatten(), t_grid.flatten()], dim=1)
        
        # 生成特征
        with torch.no_grad():
            features = self(coords)
            
        # 重塑为图像格式 [1, feature_dim, H, W]
        features = features.reshape(1, self.feature_dim, resolution[0], resolution[1])
        return features

