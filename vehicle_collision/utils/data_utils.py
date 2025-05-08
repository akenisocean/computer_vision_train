#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据处理工具模块
用于车辆碰撞检测项目的数据处理功能
"""

import os
import cv2
import numpy as np
import random
import json
import yaml
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET


def video_to_frames(video_path, output_dir, frame_rate=5, max_frames=None):
    """
    将视频转换为帧图像
    
    参数:
        video_path (str): 视频文件路径
        output_dir (str): 输出目录
        frame_rate (int): 每秒提取的帧数
        max_frames (int): 最大提取帧数
    
    返回:
        list: 提取的帧的文件路径列表
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    # 计算提取频率
    frame_interval = max(1, int(fps / frame_rate))
    
    # 计算要提取的总帧数
    if max_frames is not None:
        total_frames = min(max_frames, frame_count // frame_interval)
    else:
        total_frames = frame_count // frame_interval
    
    # 提取帧
    frame_paths = []
    current_frame = 0
    extracted_count = 0
    
    # 视频文件名（不包含扩展名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    while cap.isOpened() and (max_frames is None or extracted_count < max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame % frame_interval == 0:
            # 保存帧
            frame_path = os.path.join(output_dir, f"{video_name}_frame_{current_frame:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            extracted_count += 1
        
        current_frame += 1
    
    cap.release()
    
    print(f"从视频 {video_path} 中提取了 {len(frame_paths)} 帧")
    return frame_paths


def augment_image(image, output_path=None, random_seed=None):
    """
    对图像进行数据增强
    
    参数:
        image: 输入图像(ndarray)或图像路径(str)
        output_path (str): 输出路径
        random_seed (int): 随机种子
    
    返回:
        ndarray: 增强后的图像
    """
    # 设置随机种子
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # 加载图像
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"无法读取图像: {image}")
    else:
        img = image.copy()
    
    # 应用随机变换
    # 1. 随机旋转
    angle = random.uniform(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    
    # 2. 随机亮度和对比度调整
    brightness = random.uniform(0.8, 1.2)
    contrast = random.uniform(0.8, 1.2)
    img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness*10)
    
    # 3. 随机水平翻转
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    
    # 保存结果
    if output_path:
        cv2.imwrite(output_path, img)
    
    return img


def split_dataset(data_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, copy_files=True):
    """
    分割数据集
    
    参数:
        data_dir (str): 数据目录，包含images和labels子目录
        output_dir (str): 输出目录
        train_ratio (float): 训练集比例
        val_ratio (float): 验证集比例
        test_ratio (float): 测试集比例
        copy_files (bool): 是否复制文件
    
    返回:
        dict: 分割结果统计
    """
    # 验证分割比例
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("分割比例之和必须为1.0")
    
    # 创建输出目录
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)
    
    # 获取图像文件
    image_dir = os.path.join(data_dir, 'images')
    label_dir = os.path.join(data_dir, 'labels')
    
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    # 随机打乱文件列表
    random.shuffle(image_files)
    
    # 计算分割点
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # 分割数据集
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]
    
    # 划分数据
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    statistics = {}
    
    # 处理每个分割
    for split, files in splits.items():
        statistics[split] = len(files)
        
        # 创建文件列表
        with open(os.path.join(output_dir, f'{split}.txt'), 'w') as f:
            for image_file in files:
                f.write(f"{os.path.join('images', image_file)}\n")
        
        # 复制文件
        if copy_files:
            for image_file in files:
                # 复制图像文件
                src_img = os.path.join(image_dir, image_file)
                dst_img = os.path.join(output_dir, split, 'images', image_file)
                shutil.copy2(src_img, dst_img)
                
                # 复制标签文件（如果存在）
                base_name = os.path.splitext(image_file)[0]
                label_file = f"{base_name}.txt"
                src_label = os.path.join(label_dir, label_file)
                
                if os.path.exists(src_label):
                    dst_label = os.path.join(output_dir, split, 'labels', label_file)
                    shutil.copy2(src_label, dst_label)
    
    print(f"数据集分割完成: 训练集 {statistics['train']}, 验证集 {statistics['val']}, 测试集 {statistics['test']}")
    
    return statistics


def create_yolo_config(output_path, dataset_path, class_names):
    """
    创建YOLO配置文件
    
    参数:
        output_path (str): 输出文件路径
        dataset_path (str): 数据集路径
        class_names (list): 类别名称列表
    """
    # 转换为绝对路径
    dataset_path = os.path.abspath(dataset_path)
    
    # 创建配置数据
    config = {
        'path': dataset_path,
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    # 写入YAML文件
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"YOLO配置文件已创建: {output_path}")


def convert_pascal_voc_to_yolo(xml_path, output_path, class_map):
    """
    将Pascal VOC格式的标注转换为YOLO格式
    
    参数:
        xml_path (str): Pascal VOC XML文件路径
        output_path (str): 输出的YOLO标注文件路径
        class_map (dict): 类别名称到ID的映射
    
    返回:
        list: 转换后的YOLO格式标注 [[class_id, x, y, w, h], ...]
    """
    # 解析XML文件
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 获取图像尺寸
    size = root.find('size')
    img_width = float(size.find('width').text)
    img_height = float(size.find('height').text)
    
    # 处理所有对象
    yolo_annotations = []
    
    for obj in root.findall('object'):
        # 获取类别名称
        class_name = obj.find('name').text
        
        # 跳过未知类别
        if class_name not in class_map:
            print(f"警告: 未知类别 '{class_name}'，已跳过")
            continue
        
        class_id = class_map[class_name]
        
        # 获取边界框
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # 转换为YOLO格式（归一化中心点坐标和宽高）
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        # 添加到标注列表
        yolo_annotations.append([class_id, x_center, y_center, width, height])
    
    # 保存YOLO格式标注
    if output_path:
        with open(output_path, 'w') as f:
            for annotation in yolo_annotations:
                f.write(' '.join([str(a) for a in annotation]) + '\n')
    
    return yolo_annotations


def convert_labelstudio_to_yolo(json_file, output_dir, image_dir, class_map=None):
    """
    将Label Studio导出的JSON转换为YOLO格式
    
    参数:
        json_file (str): Label Studio导出的JSON文件路径
        output_dir (str): 输出目录
        image_dir (str): 图像目录
        class_map (dict): 类别名称到ID的映射
    
    返回:
        dict: 处理结果统计
    """
    # 默认类别映射
    if class_map is None:
        class_map = {
            "car": 0,
            "truck": 1,
            "bus": 2,
            "motorcycle": 3,
            "collision": 4
        }
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    statistics = {'processed': 0, 'skipped': 0, 'annotations': 0}
    
    # 处理每个标注项
    for item in data:
        # 检查图像路径
        if 'image' not in item or not item['image']:
            statistics['skipped'] += 1
            continue
        
        # 获取图像名称
        image_path = item['image']
        if isinstance(image_path, str):
            image_name = os.path.basename(image_path)
        else:
            statistics['skipped'] += 1
            continue
        
        # 检查图像是否存在
        full_image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(full_image_path):
            print(f"警告: 图像不存在 {full_image_path}")
            statistics['skipped'] += 1
            continue
        
        # 读取图像尺寸
        img = cv2.imread(full_image_path)
        if img is None:
            print(f"警告: 无法读取图像 {full_image_path}")
            statistics['skipped'] += 1
            continue
        
        img_height, img_width = img.shape[:2]
        
        # 创建YOLO标注文件
        base_name = os.path.splitext(image_name)[0]
        output_file = os.path.join(output_dir, f"{base_name}.txt")
        
        with open(output_file, 'w') as f:
            # 检查是否有标注
            if 'annotations' not in item or not item['annotations']:
                statistics['processed'] += 1
                continue
            
            # 处理每个标注
            for annotation in item['annotations']:
                if 'result' not in annotation:
                    continue
                
                for result in annotation['result']:
                    if 'value' not in result or 'rectanglelabels' not in result['value']:
                        continue
                    
                    # 获取标签
                    labels = result['value']['rectanglelabels']
                    if not labels:
                        continue
                    
                    label = labels[0]  # 使用第一个标签
                    if label not in class_map:
                        print(f"警告: 未知标签 '{label}'")
                        continue
                    
                    class_id = class_map[label]
                    
                    # 获取边界框坐标（百分比）
                    x = result['value']['x'] / 100.0
                    y = result['value']['y'] / 100.0
                    width = result['value']['width'] / 100.0
                    height = result['value']['height'] / 100.0
                    
                    # 转换为YOLO格式（中心点坐标和宽高）
                    x_center = x + width / 2
                    y_center = y + height / 2
                    
                    # 写入YOLO格式
                    line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                    f.write(line)
                    statistics['annotations'] += 1
        
        statistics['processed'] += 1
    
    print(f"标注转换完成: 处理 {statistics['processed']} 个文件, {statistics['annotations']} 个标注, 跳过 {statistics['skipped']} 个文件")
    return statistics


def yolo_to_voc(yolo_file, image_path, output_file, class_names):
    """
    将YOLO格式的标注转换为Pascal VOC XML格式
    
    参数:
        yolo_file (str): YOLO标注文件路径
        image_path (str): 图像文件路径
        output_file (str): 输出XML文件路径
        class_names (list): 类别名称列表
    
    返回:
        bool: 是否成功
    """
    # 读取图像尺寸
    img = cv2.imread(image_path)
    if img is None:
        print(f"警告: 无法读取图像 {image_path}")
        return False
    
    img_height, img_width, img_depth = img.shape
    
    # 创建XML根元素
    root = ET.Element('annotation')
    
    # 添加基本信息
    ET.SubElement(root, 'folder').text = os.path.basename(os.path.dirname(image_path))
    ET.SubElement(root, 'filename').text = os.path.basename(image_path)
    ET.SubElement(root, 'path').text = os.path.abspath(image_path)
    
    # 添加图像信息
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(img_width)
    ET.SubElement(size, 'height').text = str(img_height)
    ET.SubElement(size, 'depth').text = str(img_depth)
    
    # 读取YOLO标注
    if not os.path.exists(yolo_file):
        print(f"警告: 标注文件不存在 {yolo_file}")
        return False
    
    with open(yolo_file, 'r') as f:
        lines = f.readlines()
    
    # 处理每个标注
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        
        class_id, x_center, y_center, width, height = map(float, parts)
        class_id = int(class_id)
        
        # 检查类别ID有效性
        if class_id >= len(class_names):
            print(f"警告: 无效的类别ID {class_id}")
            continue
        
        # 转换回绝对坐标
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        
        xmin = int(x_center - width / 2)
        ymin = int(y_center - height / 2)
        xmax = int(x_center + width / 2)
        ymax = int(y_center + height / 2)
        
        # 添加对象
        obj = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text = class_names[class_id]
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        
        bbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = str(max(0, xmin))
        ET.SubElement(bbox, 'ymin').text = str(max(0, ymin))
        ET.SubElement(bbox, 'xmax').text = str(min(img_width, xmax))
        ET.SubElement(bbox, 'ymax').text = str(min(img_height, ymax))
    
    # 创建XML树
    tree = ET.ElementTree(root)
    
    # 保存XML文件
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    return True


if __name__ == "__main__":
    # 测试代码
    print("数据工具模块") 