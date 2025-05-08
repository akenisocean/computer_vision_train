#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
可视化工具模块
用于车辆碰撞检测项目的可视化功能
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os


def draw_bounding_boxes(image, boxes, classes, scores=None, class_names=None, color_map=None):
    """
    在图像上绘制边界框
    
    参数:
        image (ndarray): 输入图像
        boxes (list): 边界框列表，格式 [[x1, y1, x2, y2], ...]
        classes (list): 类别ID列表
        scores (list): 置信度列表
        class_names (list): 类别名称列表
        color_map (dict): 类别颜色映射
    
    返回:
        ndarray: 绘制了边界框的图像
    """
    # 默认类别名称
    if class_names is None:
        class_names = ['car', 'truck', 'bus', 'motorcycle', 'collision']
    
    # 默认颜色映射
    if color_map is None:
        color_map = {
            0: (0, 255, 0),    # 汽车: 绿色
            1: (255, 0, 0),    # 卡车: 蓝色
            2: (0, 0, 255),    # 公交车: 红色
            3: (255, 255, 0),  # 摩托车: 青色
            4: (255, 0, 255)   # 碰撞: 紫色
        }
    
    # 复制图像
    output_image = image.copy()
    
    # 绘制每个边界框
    for i, (box, cls) in enumerate(zip(boxes, classes)):
        # 获取坐标
        x1, y1, x2, y2 = map(int, box)
        
        # 获取类别和颜色
        class_id = int(cls)
        color = color_map.get(class_id, (0, 255, 0))
        class_name = class_names[class_id] if class_id < len(class_names) else f"类别{class_id}"
        
        # 绘制边界框
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签背景
        label = class_name
        if scores is not None and i < len(scores):
            label = f"{class_name}: {scores[i]:.2f}"
        
        # 获取文本大小
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        # 绘制标签背景
        cv2.rectangle(
            output_image,
            (x1, y1 - label_height - baseline - 5),
            (x1 + label_width, y1),
            color,
            -1
        )
        
        # 绘制标签文本
        cv2.putText(
            output_image,
            label,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
    
    return output_image


def draw_collision_warning(image, has_collision, position=None, text_size=1.0):
    """
    在图像上绘制碰撞警告
    
    参数:
        image (ndarray): 输入图像
        has_collision (bool): 是否有碰撞
        position (tuple): 警告文本位置，默认为居中
        text_size (float): 文本大小
    
    返回:
        ndarray: 绘制了警告的图像
    """
    if not has_collision:
        return image
    
    # 复制图像
    output_image = image.copy()
    
    # 计算默认位置
    if position is None:
        h, w = output_image.shape[:2]
        position = (w // 2 - 100, 50)
    
    # 绘制警告文本
    cv2.putText(
        output_image,
        "碰撞警告!",
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        text_size,
        (0, 0, 255),
        3
    )
    
    return output_image


def draw_trajectories(image, trajectories, colors=None, thickness=2):
    """
    绘制车辆轨迹
    
    参数:
        image (ndarray): 输入图像
        trajectories (dict): 轨迹字典，格式 {id: [(x1, y1), (x2, y2), ...], ...}
        colors (dict): ID颜色映射
        thickness (int): 线条粗细
    
    返回:
        ndarray: 绘制了轨迹的图像
    """
    # 复制图像
    output_image = image.copy()
    
    # 为每个ID分配一个颜色（如果未提供）
    if colors is None:
        colors = {}
        for obj_id in trajectories.keys():
            # 为每个ID生成一个随机颜色
            colors[obj_id] = tuple(map(int, np.random.randint(0, 255, 3)))
    
    # 绘制每个轨迹
    for obj_id, points in trajectories.items():
        if len(points) < 2:
            continue
        
        # 获取颜色
        color = colors.get(obj_id, (0, 255, 0))
        
        # 绘制轨迹线
        for i in range(1, len(points)):
            pt1 = tuple(map(int, points[i-1]))
            pt2 = tuple(map(int, points[i]))
            cv2.line(output_image, pt1, pt2, color, thickness)
    
    return output_image


def plot_detections_on_image(image, boxes, classes, scores, class_names=None, save_path=None):
    """
    使用Matplotlib绘制检测结果
    
    参数:
        image (ndarray): 输入图像
        boxes (list): 边界框列表，格式 [[x1, y1, x2, y2], ...]
        classes (list): 类别ID列表
        scores (list): 置信度列表
        class_names (list): 类别名称列表
        save_path (str): 保存路径
    """
    # 默认类别名称
    if class_names is None:
        class_names = ['car', 'truck', 'bus', 'motorcycle', 'collision']
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # 定义颜色
    colors = ['g', 'b', 'r', 'c', 'm']
    
    # 绘制每个边界框
    ax = plt.gca()
    for box, cls, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        class_id = int(cls)
        color = colors[class_id % len(colors)]
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        
        # 绘制矩形
        rect = Rectangle((x1, y1), width, height, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # 添加标签
        plt.text(
            x1, y1 - 5,
            f"{class_name}: {score:.2f}",
            color='white',
            bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=1)
        )
    
    # 关闭坐标轴
    plt.axis('off')
    
    # 保存或显示
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def create_annotated_video(input_video, output_video, annotation_func, fps=None, codec='mp4v'):
    """
    创建带注释的视频
    
    参数:
        input_video (str): 输入视频路径
        output_video (str): 输出视频路径
        annotation_func (callable): 帧处理函数，格式: func(frame) -> annotated_frame
        fps (float): 输出视频的帧率，默认与输入视频相同
        codec (str): 视频编解码器
    
    返回:
        bool: 是否成功
    """
    # 打开输入视频
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {input_video}")
        return False
    
    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # 处理每一帧
    frame_count = 0
    success = True
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理帧
        try:
            annotated_frame = annotation_func(frame)
            writer.write(annotated_frame)
            frame_count += 1
        except Exception as e:
            print(f"处理第 {frame_count} 帧时出错: {str(e)}")
            success = False
            break
    
    # 释放资源
    cap.release()
    writer.release()
    
    if success:
        print(f"已成功处理 {frame_count} 帧，保存到 {output_video}")
    
    return success


def plot_collision_events(collision_events, video_duration, output_path=None):
    """
    绘制碰撞事件时间线
    
    参数:
        collision_events (list): 碰撞事件列表
        video_duration (float): 视频时长（秒）
        output_path (str): 保存路径
    """
    # 提取时间点
    times = [event['time'] for event in collision_events]
    
    # 创建图形
    plt.figure(figsize=(12, 4))
    
    # 绘制时间线
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # 绘制碰撞事件点
    plt.scatter(times, [0] * len(times), color='r', s=100, zorder=5)
    
    # 为每个点添加标签
    for i, event in enumerate(collision_events):
        vehicles = event['vehicles']
        plt.text(
            event['time'], 0.02,
            f"#{i+1}: 车辆 {vehicles}",
            rotation=45,
            ha='center',
            va='bottom'
        )
    
    # 设置坐标轴
    plt.xlim(0, video_duration)
    plt.ylim(-0.1, 0.1)
    plt.yticks([])
    
    # 添加标题和标签
    plt.title("碰撞事件时间线")
    plt.xlabel("时间 (秒)")
    
    # 网格线
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    
    # 保存或显示
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # 简单测试
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = [[100, 100, 300, 200], [400, 150, 500, 250]]
    classes = [0, 1]
    scores = [0.95, 0.87]
    
    # 测试边界框绘制
    annotated_image = draw_bounding_boxes(image, boxes, classes, scores)
    cv2.imshow("Bounding Boxes", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 