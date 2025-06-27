import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
from .dynamic_texture import DynamicNeuralTexture
from .renderer import DNTRenderer

def visualize_texture_evolution(
    dnt: DynamicNeuralTexture,
    renderer: DNTRenderer,
    num_frames: int = 20,
    resolution: Tuple[int, int] = (256, 256),
    save_path: Optional[str] = None
):
    """
    可视化动态神经纹理随时间的演化
    Args:
        dnt: 动态神经纹理模型
        renderer: 渲染器
        num_frames: 帧数
        resolution: 输出分辨率
        save_path: 保存路径(可选)
    """
    # 创建画布
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title('Dynamic Neural Texture Evolution')
    ax.axis('off')
    
    # 预计算所有帧
    frames = []
    for i in range(num_frames):
        time = i / (num_frames - 1)
        texture = dnt.get_texture_at_time(time, resolution)
        rgb = renderer(texture[..., :-1]).detach().cpu().numpy()
        frames.append(rgb)
    
    # 动画更新函数
    def update(frame_idx):
        ax.imshow(frames[frame_idx])
        return [ax]
    
    # 创建动画
    ani = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=True)
    
    # 保存或显示动画
    if save_path:
        ani.save(save_path, writer='ffmpeg', fps=10)
    else:
        plt.show()
    
    return ani

def export_to_video(
    dnt: DynamicNeuralTexture,
    renderer: DNTRenderer,
    output_path: str,
    num_frames: int = 100,
    resolution: Tuple[int, int] = (512, 512),
    fps: float = 30.0
):
    """
    将动态神经纹理导出为视频
    Args:
        dnt: 动态神经纹理模型
        renderer: 渲染器
        output_path: 输出视频路径
        num_frames: 帧数
        resolution: 视频分辨率
        fps: 帧率
    """
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
    
    # 渲染每一帧
    for i in range(num_frames):
        time = i / (num_frames - 1)
        
        # 渲染纹理
        with torch.no_grad():
            rgb = renderer.render_texture(dnt, time, resolution)
        
        # 转换为OpenCV格式
        frame = rgb.detach().cpu().numpy()
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 写入视频
        out.write(frame)
    
    # 释放资源
    out.release()    