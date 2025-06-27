# optical_flow.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class RAFTFlowEstimator(nn.Module):
    """
    RAFT (Recurrent All Pairs Field Transforms) 光流估计器
    简化版本，实际应用中建议使用完整的 RAFT 实现
    """
    def __init__(self, iters=12, hidden_dim=128, input_dim=128):
        super().__init__()
        self.iters = iters
        self.hidden_dim = hidden_dim
        
        # 特征编码器（简化版）
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, input_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # 光流预测器（简化版）
        self.flow_predictor = nn.Sequential(
            nn.Conv2d(input_dim*2 + 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1)  # 2通道：u和v
        )
        
    def forward(self, image1, image2):
        """
        估计两帧之间的光流
        
        Args:
            image1: 第一帧图像 [B, 3, H, W]
            image2: 第二帧图像 [B, 3, H, W]
            
        Returns:
            flow: 光流场 [B, 2, H, W]
        """
        # 提取特征
        feat1 = self.feature_encoder(image1)
        feat2 = self.feature_encoder(image2)
        
        # 特征金字塔（简化版）
        feat1_pyramid = [feat1]
        feat2_pyramid = [feat2]
        
        # 初始光流为零
        flow = torch.zeros_like(image1[:, :2, :, :])
        
        # 迭代优化光流（简化版）
        for i in range(self.iters):
            # 对第二帧特征进行光流扭曲
            flow_up = F.interpolate(flow, scale_factor=0.5, mode='bilinear', align_corners=True)
            feat2_warped = self.warp(feat2_pyramid[0], flow_up)
            
            # 拼接特征并预测光流增量
            concat_features = torch.cat([feat1_pyramid[0], feat2_warped, flow_up], dim=1)
            flow_delta = self.flow_predictor(concat_features)
            
            # 更新光流
            flow = flow_up + flow_delta
            
        # 上采样到原始分辨率
        flow = F.interpolate(flow, scale_factor=8, mode='bilinear', align_corners=True)
        return flow
    
    def warp(self, x, flow):
        """
        使用光流扭曲特征
        
        Args:
            x: 输入特征 [B, C, H, W]
            flow: 光流 [B, 2, H, W]
            
        Returns:
            warped: 扭曲后的特征 [B, C, H, W]
        """
        B, C, H, W = x.size()
        
        # 创建网格坐标
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, device=x.device, dtype=torch.float32),
            torch.arange(0, W, device=x.device, dtype=torch.float32)
        )
        grid = torch.stack([grid_x, grid_y], dim=0)  # [2, H, W]
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2, H, W]
        
        # 添加光流
        vgrid = grid + flow
        
        # 归一化到 [-1, 1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        
        # 应用网格采样
        vgrid = vgrid.permute(0, 2, 3, 1)  # [B, H, W, 2]
        warped = F.grid_sample(x, vgrid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return warped

