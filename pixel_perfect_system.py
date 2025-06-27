import torch
import numpy as np
import matplotlib.pyplot as plt
from dynamic_neural_textures import DynamicNeuralTexture, DNTRenderer
from dynamic_neural_textures.utils import visualize_texture_evolution

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # 创建动态神经纹理模型
    dnt = DynamicNeuralTexture(
        spatial_resolution=(256, 256),  # 空间分辨率
        temporal_resolution=100,        # 时间分辨率
        feature_dim=64                  # 特征维度
    ).to(device)
    
    # 创建渲染器
    renderer = DNTRenderer(feature_dim=64).to(device)
    
    # 示例：生成并渲染特定时间的纹理
    time = 0.5  # 时间范围 [0, 1]
    resolution = (256, 256)  # 渲染分辨率
    
    # 生成网格坐标
    y_coords = torch.linspace(0, 1, resolution[0], device=device)
    x_coords = torch.linspace(0, 1, resolution[1], device=device)
    x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='xy')
    
    # 创建坐标张量 [N, 3] (x, y, time)
    t_grid = torch.full_like(x_grid, time)
    coords = torch.stack([x_grid.flatten(), y_grid.flatten(), t_grid.flatten()], dim=1)
    
    # 生成特征
    with torch.no_grad():
        output = dnt(coords)
        features = output[:, :-1]  # 特征 [N, feature_dim]
        alpha = output[:, -1:]     # 透明度 [N, 1]
        
        # 渲染 RGB 图像
        rgb = renderer(features)
        
        # 重塑为图像形状
        rgb_image = rgb.reshape(1, 3, resolution[0], resolution[1])
        alpha_image = alpha.reshape(1, 1, resolution[0], resolution[1])
        
        # 组合 RGB 和 alpha
        final_image = rgb_image * alpha_image + (1 - alpha_image)
    
    # 可视化结果
    plt.figure(figsize=(8, 8))
    plt.imshow(final_image[0].cpu().permute(1, 2, 0).numpy())
    plt.title(f'动态神经纹理 (时间: {time:.2f})')
    plt.axis('off')
    plt.savefig('texture_at_time_0.5.png', bbox_inches='tight')
    plt.show()
    
    # 可视化纹理随时间的演化
    visualize_texture_evolution(dnt, renderer, num_frames=20, device=device)

if __name__ == '__main__':
    main()

