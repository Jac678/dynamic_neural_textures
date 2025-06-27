import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dynamic_neural_textures import DynamicNeuralTexture, DNTRenderer
from dynamic_neural_textures.dataset import VideoDataset
import os

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据加载
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = VideoDataset('path/to/video/frames', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 模型初始化
dnt = DynamicNeuralTexture(
    spatial_resolution=(256, 256),
    temporal_resolution=len(dataset),
    feature_dim=64
).to(device)

renderer = DNTRenderer(feature_dim=64).to(device)

# 优化器
optimizer = optim.Adam(list(dnt.parameters()) + list(renderer.parameters()), lr=0.001)
criterion = nn.MSELoss()

# 训练循环
num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0.0
    
    for batch in dataloader:
        images = batch['image'].to(device)
        times = batch['time'].to(device)
        
        # 清除梯度
        optimizer.zero_grad()
        
        # 生成网格坐标
        h, w = images.shape[2:]
        y_coords = torch.linspace(0, 1, h, device=device)
        x_coords = torch.linspace(0, 1, w, device=device)
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        # 为每个图像创建完整的坐标
        batch_coords = []
        for t in times:
            t_grid = torch.full_like(x_grid, t)
            coords = torch.stack([x_grid.flatten(), y_grid.flatten(), t_grid.flatten()], dim=1)
            batch_coords.append(coords)
        
        batch_coords = torch.stack(batch_coords)
        
        # 前向传播
        batch_size = images.shape[0]
        predicted_images = []
        
        for i in range(batch_size):
            # 获取特征
            output = dnt(batch_coords[i])
            features = output[:, :-1]
            alpha = output[:, -1:]
            
            # 渲染RGB
            rgb = renderer(features)
            
            # 重塑为图像
            rgb_image = rgb.reshape(1, 3, h, w)
            alpha_image = alpha.reshape(1, 1, h, w)
            
            # 组合
            predicted_image = rgb_image * alpha_image + (1 - alpha_image)
            predicted_images.append(predicted_image)
        
        predicted_images = torch.cat(predicted_images, dim=0)
        
        # 计算损失
        loss = criterion(predicted_images, images)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # 打印训练信息
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.6f}')
    
    # 每10个epoch保存一次模型
    if (epoch + 1) % 10 == 0:
        dnt.save(f'dnt_epoch_{epoch+1}.pth')

print('训练完成!')    