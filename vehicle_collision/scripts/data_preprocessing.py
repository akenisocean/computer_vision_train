#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据预处理脚本
将原始视频数据处理为适合模型训练的格式
"""

import os
import cv2
import argparse
import numpy as np
import shutil
import json
from pathlib import Path
from tqdm import tqdm
import random


def extract_frames(video_path, output_dir, frame_rate=5):
    """
    从视频中提取帧
    
    参数:
        video_path (str): 视频文件路径
        output_dir (str): 输出目录
        frame_rate (int): 每秒提取的帧数
    
    返回:
        list: 提取的帧的路径列表
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算提取频率
    frame_interval = max(1, int(fps / frame_rate))
    
    frame_paths = []
    frame_count = 0
    
    # 视频文件名（不包含扩展名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 提取帧
    pbar = tqdm(total=total_frames//frame_interval, desc=f"提取 {video_name} 的帧")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # 保存帧
            frame_path = os.path.join(output_dir, f"{video_name}_frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            pbar.update(1)
        
        frame_count += 1
    
    cap.release()
    pbar.close()
    
    print(f"从视频 {video_path} 中提取了 {len(frame_paths)} 帧")
    return frame_paths


def convert_labelstudio_to_yolo(labelstudio_json, output_dir, image_dir, class_map=None):
    """
    将Label Studio标注转换为YOLO格式
    
    参数:
        labelstudio_json (str): Label Studio导出的JSON文件路径
        output_dir (str): 输出目录，用于保存YOLO格式的标注
        image_dir (str): 图像目录
        class_map (dict): 类别映射，将Label Studio类别名映射到YOLO类别索引
    
    返回:
        dict: 处理结果统计
    """
    if class_map is None:
        # 默认类别映射
        class_map = {
            "car": 0,
            "truck": 1,
            "bus": 2,
            "motorcycle": 3,
            "collision": 4
        }
    
    # 创建输出目录
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    
    # 加载Label Studio JSON
    with open(labelstudio_json, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    stats = {"processed": 0, "skipped": 0, "annotations": 0}
    
    for item in tqdm(annotations, desc="转换标注"):
        # 获取图像信息
        if 'image' not in item:
            stats["skipped"] += 1
            continue
        
        image_name = os.path.basename(item['image'])
        image_path = os.path.join(image_dir, image_name)
        
        if not os.path.exists(image_path):
            print(f"警告: 图像 {image_path} 不存在，跳过")
            stats["skipped"] += 1
            continue
        
        # 读取图像尺寸
        img = cv2.imread(image_path)
        if img is None:
            print(f"警告: 无法读取图像 {image_path}，跳过")
            stats["skipped"] += 1
            continue
        
        img_height, img_width = img.shape[:2]
        
        # 创建YOLO标注文件
        label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + '.txt')
        
        with open(label_path, 'w') as f:
            # 检查是否有标注
            if 'annotations' not in item or not item['annotations']:
                # 创建空文件（表示没有标注）
                stats["processed"] += 1
                continue
            
            for ann in item['annotations']:
                if 'result' not in ann:
                    continue
                
                for result in ann['result']:
                    if 'value' not in result or 'rectanglelabels' not in result['value']:
                        continue
                    
                    # 获取标签
                    labels = result['value']['rectanglelabels']
                    if not labels:
                        continue
                    
                    label = labels[0]  # 使用第一个标签
                    if label not in class_map:
                        print(f"警告: 未知标签 '{label}'，跳过")
                        continue
                    
                    class_id = class_map[label]
                    
                    # 获取边界框坐标（相对值）
                    x = result['value']['x'] / 100.0
                    y = result['value']['y'] / 100.0
                    width = result['value']['width'] / 100.0
                    height = result['value']['height'] / 100.0
                    
                    # 转换为YOLO格式（中心点坐标和宽高）
                    center_x = x + width / 2
                    center_y = y + height / 2
                    
                    # 写入YOLO格式标注
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                    stats["annotations"] += 1
        
        stats["processed"] += 1
    
    print(f"处理完成: {stats['processed']} 个图像, {stats['annotations']} 个标注, {stats['skipped']} 个跳过")
    return stats


def split_dataset(data_dir, output_dir, split_ratio=(0.7, 0.2, 0.1)):
    """
    将数据集分割为训练集、验证集和测试集
    
    参数:
        data_dir (str): 数据目录，包含images和labels子目录
        output_dir (str): 输出目录
        split_ratio (tuple): 训练集、验证集、测试集的比例
    
    返回:
        dict: 各集合的图像数量
    """
    assert sum(split_ratio) == 1.0, "分割比例之和必须为1.0"
    
    # 创建输出目录结构
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)
    
    # 获取所有图像文件
    image_dir = os.path.join(data_dir, 'images')
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 随机打乱
    random.shuffle(image_files)
    
    # 计算各集合的大小
    train_size = int(len(image_files) * split_ratio[0])
    val_size = int(len(image_files) * split_ratio[1])
    
    # 分割数据集
    train_files = image_files[:train_size]
    val_files = image_files[train_size:train_size + val_size]
    test_files = image_files[train_size + val_size:]
    
    # 复制文件到相应目录
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    stats = {}
    
    for split, files in splits.items():
        for filename in tqdm(files, desc=f"复制{split}集文件"):
            # 复制图像
            src_img = os.path.join(image_dir, filename)
            dst_img = os.path.join(output_dir, split, 'images', filename)
            shutil.copy2(src_img, dst_img)
            
            # 复制标签（如果存在）
            label_filename = os.path.splitext(filename)[0] + '.txt'
            src_label = os.path.join(data_dir, 'labels', label_filename)
            if os.path.exists(src_label):
                dst_label = os.path.join(output_dir, split, 'labels', label_filename)
                shutil.copy2(src_label, dst_label)
        
        stats[split] = len(files)
    
    print(f"数据集分割完成: 训练集 {stats['train']}, 验证集 {stats['val']}, 测试集 {stats['test']}")
    return stats


def apply_augmentation(image_path, output_dir, num_augmentations=3):
    """
    对图像应用数据增强
    
    参数:
        image_path (str): 图像文件路径
        output_dir (str): 输出目录
        num_augmentations (int): 每张图像生成的增强版本数量
    
    返回:
        list: 增强后的图像路径列表
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"警告: 无法读取图像 {image_path}")
        return []
    
    # 图像文件名（不包含扩展名）
    basename = os.path.splitext(os.path.basename(image_path))[0]
    
    augmented_paths = []
    
    for i in range(num_augmentations):
        # 应用随机变换
        # 1. 旋转
        angle = random.uniform(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        rotated = cv2.warpAffine(img, M, (w, h))
        
        # 2. 亮度和对比度调整
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        adjusted = cv2.convertScaleAbs(rotated, alpha=contrast, beta=brightness)
        
        # 3. 水平翻转
        if random.random() > 0.5:
            adjusted = cv2.flip(adjusted, 1)
        
        # 保存增强后的图像
        output_path = os.path.join(output_dir, f"{basename}_aug_{i}.jpg")
        cv2.imwrite(output_path, adjusted)
        augmented_paths.append(output_path)
    
    return augmented_paths


def create_yolo_dataset_yaml(output_dir, class_names):
    """
    创建YOLO数据集配置文件
    
    参数:
        output_dir (str): 输出目录
        class_names (list): 类别名称列表
    """
    yaml_path = os.path.join(output_dir, 'data.yaml')
    
    content = f"""
# YOLO数据集配置
path: {os.path.abspath(output_dir)}  # 数据集根目录
train: train/images  # 训练图像相对路径
val: val/images  # 验证图像相对路径
test: test/images  # 测试图像相对路径

# 类别
nc: {len(class_names)}  # 类别数量
names: {class_names}  # 类别名称
"""
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已创建YOLO数据集配置文件: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description='车辆碰撞检测数据预处理工具')
    
    # 输入和输出
    parser.add_argument('--input', type=str, required=True, help='输入目录或文件')
    parser.add_argument('--output', type=str, default='../data/processed', help='输出目录')
    
    # 处理选项
    parser.add_argument('--extract-frames', action='store_true', help='从视频中提取帧')
    parser.add_argument('--frame-rate', type=int, default=5, help='每秒提取的帧数')
    parser.add_argument('--convert-annotations', action='store_true', help='转换Label Studio标注')
    parser.add_argument('--labelstudio-json', type=str, help='Label Studio导出的JSON文件')
    parser.add_argument('--split-dataset', action='store_true', help='分割数据集')
    parser.add_argument('--apply-augmentation', action='store_true', help='应用数据增强')
    parser.add_argument('--create-yaml', action='store_true', help='创建YOLO数据集配置文件')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 处理选项
    if args.extract_frames:
        if os.path.isdir(args.input):
            # 处理目录中的所有视频
            video_files = [f for f in os.listdir(args.input) 
                           if f.endswith(('.mp4', '.avi', '.mov'))]
            
            frames_dir = os.path.join(args.output, 'images')
            os.makedirs(frames_dir, exist_ok=True)
            
            for video_file in video_files:
                video_path = os.path.join(args.input, video_file)
                extract_frames(video_path, frames_dir, args.frame_rate)
        else:
            # 处理单个视频
            frames_dir = os.path.join(args.output, 'images')
            os.makedirs(frames_dir, exist_ok=True)
            extract_frames(args.input, frames_dir, args.frame_rate)
    
    if args.convert_annotations and args.labelstudio_json:
        image_dir = os.path.join(args.output, 'images')
        
        if not os.path.exists(image_dir):
            image_dir = os.path.join(args.input, 'images')
            if not os.path.exists(image_dir):
                raise ValueError("未找到图像目录，请先提取帧或指定正确的输入目录")
        
        convert_labelstudio_to_yolo(args.labelstudio_json, args.output, image_dir)
    
    if args.apply_augmentation:
        image_dir = os.path.join(args.output, 'images')
        
        if not os.path.exists(image_dir):
            image_dir = os.path.join(args.input, 'images')
            if not os.path.exists(image_dir):
                raise ValueError("未找到图像目录，请先提取帧或指定正确的输入目录")
        
        aug_dir = os.path.join(args.output, 'images_augmented')
        
        image_files = [f for f in os.listdir(image_dir) 
                       if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in tqdm(image_files, desc="应用数据增强"):
            img_path = os.path.join(image_dir, img_file)
            apply_augmentation(img_path, aug_dir)
    
    if args.split_dataset:
        data_dir = args.output
        
        if not os.path.exists(os.path.join(data_dir, 'images')):
            data_dir = args.input
            if not os.path.exists(os.path.join(data_dir, 'images')):
                raise ValueError("未找到图像目录，请先提取帧或指定正确的输入目录")
        
        split_dir = os.path.join(args.output, 'splits')
        split_dataset(data_dir, split_dir)
    
    if args.create_yaml:
        class_names = ["car", "truck", "bus", "motorcycle", "collision"]
        
        split_dir = os.path.join(args.output, 'splits')
        if not os.path.exists(split_dir):
            raise ValueError("未找到分割数据集目录，请先分割数据集")
        
        create_yolo_dataset_yaml(split_dir, class_names)


if __name__ == '__main__':
    main() 