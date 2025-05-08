#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型工具模块
用于车辆碰撞检测项目的模型相关功能
"""

import os
import torch
import numpy as np
from ultralytics import YOLO
import cv2
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import time
import yaml


def load_model(model_path, device=None):
    """
    加载YOLO模型
    
    参数:
        model_path (str): 模型路径
        device (str): 设备选择 (None, 'cpu', '0', '0,1')
    
    返回:
        YOLO: 加载的模型
    """
    # 检查模型路径
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 自动检测设备
    if device is None:
        device = '0' if torch.cuda.is_available() else 'cpu'
    
    # 加载模型
    model = YOLO(model_path)
    
    # 设置设备
    model.to(device)
    
    print(f"已加载模型: {model_path}")
    return model


def model_info(model):
    """
    显示模型信息
    
    参数:
        model: YOLO模型
    
    返回:
        dict: 模型信息
    """
    info = {}
    
    try:
        # 获取模型名称
        info['name'] = model.name if hasattr(model, 'name') else 'Unknown'
        
        # 获取设备信息
        info['device'] = str(next(model.model.parameters()).device)
        
        # 获取类别信息
        if hasattr(model, 'names'):
            info['classes'] = model.names
            info['num_classes'] = len(model.names)
        
        # 获取模型类型
        if hasattr(model, 'task'):
            info['task'] = model.task
        
        # 显示信息
        print(f"模型名称: {info['name']}")
        print(f"运行设备: {info['device']}")
        print(f"任务类型: {info.get('task', 'Unknown')}")
        print(f"类别数量: {info.get('num_classes', 'Unknown')}")
        print(f"类别名称: {info.get('classes', 'Unknown')}")
    
    except Exception as e:
        print(f"获取模型信息时出错: {str(e)}")
    
    return info


def perform_inference(model, image_path, conf_threshold=0.25, img_size=640):
    """
    使用模型进行推理
    
    参数:
        model: YOLO模型
        image_path (str): 图像路径
        conf_threshold (float): 置信度阈值
        img_size (int): 图像大小
    
    返回:
        tuple: (原始图像, 结果对象)
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 运行推理
    results = model(image, conf=conf_threshold, imgsz=img_size)
    
    return image, results


def get_boxes_and_scores(results):
    """
    从结果中提取边界框和置信度
    
    参数:
        results: YOLO推理结果
    
    返回:
        tuple: (边界框列表, 类别ID列表, 置信度列表)
    """
    # 初始化列表
    boxes = []
    class_ids = []
    scores = []
    
    # 处理每个结果
    for r in results:
        # 提取边界框
        for box in r.boxes:
            # 获取坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 获取类别ID和置信度
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # 添加到列表
            boxes.append((x1, y1, x2, y2))
            class_ids.append(cls_id)
            scores.append(conf)
    
    return boxes, class_ids, scores


def calculate_fps(model, image, iterations=10, warmup=3):
    """
    计算模型推理的帧率
    
    参数:
        model: YOLO模型
        image (ndarray): 输入图像
        iterations (int): 迭代次数
        warmup (int): 预热迭代次数
    
    返回:
        float: 每秒帧数 (FPS)
    """
    # 预热
    for _ in range(warmup):
        _ = model(image)
    
    # 计时
    start_time = time.time()
    
    for _ in range(iterations):
        _ = model(image)
    
    # 计算平均时间
    avg_time = (time.time() - start_time) / iterations
    fps = 1.0 / avg_time
    
    return fps


def intersect_over_union(box1, box2):
    """
    计算两个边界框的IoU
    
    参数:
        box1 (tuple): 边界框1 (x1, y1, x2, y2)
        box2 (tuple): 边界框2 (x1, y1, x2, y2)
    
    返回:
        float: IoU值 (0~1)
    """
    # 计算交集区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 检查是否有交集
    if x2 < x1 or y2 < y1:
        return 0.0
    
    # 计算交集面积
    intersection_area = (x2 - x1) * (y2 - y1)
    
    # 计算两个框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集面积
    union_area = box1_area + box2_area - intersection_area
    
    # 计算IoU
    iou = intersection_area / union_area
    
    return iou


def non_max_suppression(boxes, scores, threshold=0.5):
    """
    非极大值抑制
    
    参数:
        boxes (list): 边界框列表 [(x1, y1, x2, y2), ...]
        scores (list): 置信度列表
        threshold (float): IoU阈值
    
    返回:
        tuple: (保留的索引列表, 保留的边界框列表, 保留的置信度列表)
    """
    # 检查输入
    if len(boxes) == 0:
        return [], [], []
    
    # 转换为numpy数组以便处理
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # 获取边界框的坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # 计算边界框的面积
    areas = (x2 - x1) * (y2 - y1)
    
    # 按置信度排序的索引
    idxs = np.argsort(scores)
    
    # 初始化保留的框索引
    keep = []
    
    # 循环处理所有边界框
    while idxs.size > 0:
        # 取出置信度最高的框
        last = idxs.size - 1
        i = idxs[last]
        keep.append(i)
        
        # 计算与其他框的IoU
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        # 计算交集宽高
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        # 计算交集面积
        inter = w * h
        
        # 计算IoU
        union = areas[i] + areas[idxs[:last]] - inter
        ious = inter / union
        
        # 保留IoU小于阈值的框
        idxs = idxs[np.where(ious < threshold)[0]]
    
    # 返回结果
    return keep, [boxes[i] for i in keep], [scores[i] for i in keep]


def convert_to_onnx(model_path, output_path, img_size=640, simplify=True):
    """
    将PyTorch模型转换为ONNX格式
    
    参数:
        model_path (str): PyTorch模型路径
        output_path (str): ONNX模型输出路径
        img_size (int): 图像大小
        simplify (bool): 是否简化模型
    
    返回:
        bool: 是否转换成功
    """
    try:
        # 加载模型
        model = YOLO(model_path)
        
        # 导出为ONNX
        model.export(format='onnx', imgsz=img_size, simplify=simplify, opset=14)
        
        # 自动生成的路径和请求的输出路径不同时，进行重命名
        auto_output = model_path.replace('.pt', '.onnx')
        if os.path.exists(auto_output) and auto_output != output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            shutil.move(auto_output, output_path)
        
        print(f"模型已转换为ONNX格式: {output_path}")
        return True
    
    except Exception as e:
        print(f"转换ONNX时出错: {str(e)}")
        return False


def optimize_model_size(model_path, output_path=None, int8=False, dynamic=True):
    """
    优化模型大小（量化或标记动态轴）
    
    参数:
        model_path (str): 模型路径
        output_path (str): 输出路径
        int8 (bool): 是否进行INT8量化
        dynamic (bool): 是否标记动态轴
    
    返回:
        str: 优化后的模型路径
    """
    try:
        import onnx
        from onnxsim import simplify
        
        # 设置输出路径
        if output_path is None:
            base, ext = os.path.splitext(model_path)
            output_path = f"{base}_optimized{ext}"
        
        # 加载模型
        model = onnx.load(model_path)
        
        # 执行优化
        if dynamic:
            # 将第一个维度标记为动态
            for input in model.graph.input:
                dim = input.type.tensor_type.shape.dim[0]
                dim.dim_param = "batch_size"
        
        # 简化模型
        model_simplified, check = simplify(model)
        
        if not check:
            print("警告: 模型简化后检查失败，可能有计算不匹配")
        
        # 保存优化后的模型
        onnx.save(model_simplified, output_path)
        
        print(f"模型已优化并保存到: {output_path}")
        return output_path
    
    except ImportError:
        print("警告: 需要安装onnx和onnx-simplifier包来优化模型")
        return model_path
    except Exception as e:
        print(f"优化模型时出错: {str(e)}")
        return model_path


def create_model_metadata(model_path, output_path=None, class_names=None, author=None, description=None):
    """
    创建模型元数据文件
    
    参数:
        model_path (str): 模型路径
        output_path (str): 元数据输出路径
        class_names (list): 类别名称列表
        author (str): 作者
        description (str): 描述
    
    返回:
        str: 元数据文件路径
    """
    # 设置默认输出路径
    if output_path is None:
        output_path = os.path.splitext(model_path)[0] + '_metadata.yaml'
    
    # 加载模型获取类别信息
    if class_names is None:
        try:
            model = YOLO(model_path)
            class_names = model.names
        except Exception:
            class_names = []
    
    # 创建元数据
    metadata = {
        'model': {
            'path': os.path.abspath(model_path),
            'type': 'YOLO',
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'framework': 'PyTorch'
        },
        'classes': {
            'names': class_names,
            'count': len(class_names)
        }
    }
    
    if author:
        metadata['author'] = author
    
    if description:
        metadata['description'] = description
    
    # 保存元数据
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    
    print(f"模型元数据已保存到: {output_path}")
    return output_path


def evaluate_model_performance(model, val_images_dir, ground_truth_dir, output_dir=None, conf_threshold=0.25):
    """
    评估模型性能
    
    参数:
        model: YOLO模型
        val_images_dir (str): 验证图像目录
        ground_truth_dir (str): 真实标签目录
        output_dir (str): 输出目录
        conf_threshold (float): 置信度阈值
    
    返回:
        dict: 性能评估结果
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 获取图像文件
    image_files = [f for f in os.listdir(val_images_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    # 初始化指标
    metrics = {
        'precision': [],
        'recall': [],
        'f1': [],
        'iou': []
    }
    
    # 处理每张图像
    for image_file in image_files:
        image_path = os.path.join(val_images_dir, image_file)
        
        # 获取对应的标签文件
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(ground_truth_dir, label_file)
        
        if not os.path.exists(label_path):
            print(f"警告: 找不到标签文件 {label_path}")
            continue
        
        # 加载图像并进行推理
        image, results = perform_inference(model, image_path, conf_threshold)
        
        # 获取预测结果
        pred_boxes, pred_classes, pred_scores = get_boxes_and_scores(results)
        
        # 加载真实标签
        gt_boxes, gt_classes = load_ground_truth(label_path, image.shape[:2])
        
        # 计算指标
        img_metrics = calculate_metrics(pred_boxes, pred_classes, pred_scores, 
                                       gt_boxes, gt_classes, iou_threshold=0.5)
        
        # 合并指标
        for k, v in img_metrics.items():
            if k in metrics:
                metrics[k].append(v)
        
        # 保存可视化结果
        if output_dir:
            # 绘制预测和真实标签
            result_img = results[0].plot()
            
            # 保存图像
            output_path = os.path.join(output_dir, f"eval_{image_file}")
            cv2.imwrite(output_path, result_img)
    
    # 计算平均指标
    avg_metrics = {k: np.mean(v) if v else 0 for k, v in metrics.items()}
    
    # 添加总体指标
    avg_metrics['image_count'] = len(image_files)
    
    # 生成评估报告
    if output_dir:
        # 保存指标
        metrics_path = os.path.join(output_dir, 'metrics.yaml')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            yaml.dump(avg_metrics, f, default_flow_style=False)
        
        # 绘制指标图表
        plt.figure(figsize=(10, 6))
        plt.bar(avg_metrics.keys(), avg_metrics.values())
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title('Model Performance Metrics')
        plt.savefig(os.path.join(output_dir, 'metrics.png'), dpi=300)
        plt.close()
    
    return avg_metrics


def load_ground_truth(label_path, image_shape):
    """
    加载真实标签
    
    参数:
        label_path (str): 标签文件路径
        image_shape (tuple): 图像尺寸 (height, width)
    
    返回:
        tuple: (边界框列表, 类别ID列表)
    """
    height, width = image_shape
    
    # 初始化列表
    boxes = []
    classes = []
    
    # 读取标签文件
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # 处理每个标签
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            # 解析YOLO格式标签
            class_id = int(parts[0])
            x_center = float(parts[1]) * width
            y_center = float(parts[2]) * height
            box_width = float(parts[3]) * width
            box_height = float(parts[4]) * height
            
            # 转换为边界框坐标
            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)
            
            # 添加到列表
            boxes.append((x1, y1, x2, y2))
            classes.append(class_id)
    
    return boxes, classes


def calculate_metrics(pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes, iou_threshold=0.5):
    """
    计算评估指标
    
    参数:
        pred_boxes (list): 预测边界框列表
        pred_classes (list): 预测类别ID列表
        pred_scores (list): 预测置信度列表
        gt_boxes (list): 真实边界框列表
        gt_classes (list): 真实类别ID列表
        iou_threshold (float): IoU阈值
    
    返回:
        dict: 指标结果
    """
    # 初始化指标
    metrics = {
        'precision': 0,
        'recall': 0,
        'f1': 0,
        'iou': []
    }
    
    # 边界情况处理
    if not gt_boxes:
        if not pred_boxes:
            # 无真实框，无预测框 -> 完美预测
            return {'precision': 1, 'recall': 1, 'f1': 1, 'iou': []}
        else:
            # 无真实框，有预测框 -> 全部是假阳性
            return {'precision': 0, 'recall': 1, 'f1': 0, 'iou': []}
    
    if not pred_boxes:
        # 有真实框，无预测框 -> 全部是假阴性
        return {'precision': 0, 'recall': 0, 'f1': 0, 'iou': []}
    
    # 计算所有预测框与真实框的IoU
    ious = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            ious[i, j] = intersect_over_union(pred_box, gt_box)
    
    # 确定匹配（贪心算法）
    matched_gt = set()
    true_positives = 0
    
    for i in range(len(pred_boxes)):
        # 找到最佳匹配
        best_iou = 0
        best_gt_idx = -1
        
        for j in range(len(gt_boxes)):
            if j in matched_gt:
                continue
            
            # 检查类别是否匹配
            if pred_classes[i] == gt_classes[j] and ious[i, j] > best_iou:
                best_iou = ious[i, j]
                best_gt_idx = j
        
        # 判断是否为真阳性
        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            true_positives += 1
            matched_gt.add(best_gt_idx)
            metrics['iou'].append(best_iou)
    
    # 计算指标
    if true_positives > 0:
        precision = true_positives / len(pred_boxes)
        recall = true_positives / len(gt_boxes)
        
        # 计算F1分数
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
    else:
        precision = 0
        recall = 0
        f1 = 0
    
    # 更新指标
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    # 计算平均IoU
    if metrics['iou']:
        metrics['avg_iou'] = np.mean(metrics['iou'])
    else:
        metrics['avg_iou'] = 0
    
    return metrics


if __name__ == "__main__":
    # 测试代码
    print("模型工具模块") 