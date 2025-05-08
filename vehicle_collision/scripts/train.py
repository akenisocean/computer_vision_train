#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
车辆碰撞检测模型训练脚本
使用YOLOv8进行训练
"""

import os
import argparse
import yaml
from ultralytics import YOLO
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import numpy as np
import time
import torch


def plot_training_results(results_dir):
    """
    绘制训练结果图表
    
    参数:
        results_dir (str): 结果目录路径
    """
    results_file = os.path.join(results_dir, 'results.csv')
    if not os.path.exists(results_file):
        print(f"警告: 未找到训练结果文件 {results_file}")
        return
    
    # 加载训练结果
    data = np.loadtxt(results_file, delimiter=',', skiprows=1)
    
    # 创建图表目录
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 提取指标
    epochs = data[:, 0]
    train_loss = data[:, 1]
    val_loss = data[:, 2]
    precision = data[:, 3]
    recall = data[:, 4]
    mAP50 = data[:, 5]
    mAP50_95 = data[:, 6]
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_loss, label='训练损失')
    plt.plot(epochs, val_loss, label='验证损失')
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'loss_curve.png'), dpi=300)
    plt.close()
    
    # 绘制性能指标
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, precision, label='精确率')
    plt.plot(epochs, recall, label='召回率')
    plt.plot(epochs, mAP50, label='mAP@0.5')
    plt.plot(epochs, mAP50_95, label='mAP@0.5:0.95')
    plt.xlabel('迭代次数')
    plt.ylabel('指标值')
    plt.title('性能指标')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'metrics_curve.png'), dpi=300)
    plt.close()
    
    print(f"训练结果图表已保存到 {plots_dir}")


def backup_best_model(runs_dir, output_dir):
    """
    备份最佳模型
    
    参数:
        runs_dir (str): 训练运行目录
        output_dir (str): 输出目录
    """
    # 查找最新的运行目录
    latest_run = None
    max_time = 0
    
    for run_dir in os.listdir(runs_dir):
        run_path = os.path.join(runs_dir, run_dir)
        if os.path.isdir(run_path):
            # 获取目录的修改时间
            mod_time = os.path.getmtime(run_path)
            if mod_time > max_time:
                max_time = mod_time
                latest_run = run_path
    
    if latest_run is None:
        print("未找到训练运行目录")
        return
    
    # 查找最佳模型
    weights_dir = os.path.join(latest_run, 'weights')
    best_model = os.path.join(weights_dir, 'best.pt')
    
    if not os.path.exists(best_model):
        print(f"未找到最佳模型 {best_model}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 复制最佳模型
    output_path = os.path.join(output_dir, 'vehicle_collision_best.pt')
    shutil.copy2(best_model, output_path)
    
    # 复制训练结果
    for result_file in ['results.csv', 'args.yaml']:
        src = os.path.join(latest_run, result_file)
        if os.path.exists(src):
            dst = os.path.join(output_dir, result_file)
            shutil.copy2(src, dst)
    
    print(f"最佳模型已保存到 {output_path}")
    return output_path


def validate_model(model_path, data_yaml, imgsz=640):
    """
    验证模型性能
    
    参数:
        model_path (str): 模型路径
        data_yaml (str): 数据配置文件路径
        imgsz (int): 图像大小
    
    返回:
        dict: 验证结果
    """
    model = YOLO(model_path)
    results = model.val(data=data_yaml, imgsz=imgsz)
    return results


def train(data_yaml, model_path, epochs=100, batch_size=16, imgsz=640, 
         device='', workers=4, project='../models', name='train', 
         resume=False, patience=50):
    """
    训练模型
    
    参数:
        data_yaml (str): 数据配置文件路径
        model_path (str): 预训练模型路径
        epochs (int): 训练轮数
        batch_size (int): 批次大小
        imgsz (int): 图像大小
        device (str): 设备选择 ('cpu', '0', '0,1,2,3')
        workers (int): 数据加载的工作线程数
        project (str): 项目保存目录
        name (str): 实验名称
        resume (bool): 是否继续上次训练
        patience (int): 早停耐心值
    
    返回:
        str: 训练结果路径
    """
    print(f"开始训练模型: {model_path}")
    print(f"数据配置: {data_yaml}")
    print(f"训练参数: epochs={epochs}, batch_size={batch_size}, imgsz={imgsz}")
    
    # 检查CUDA是否可用
    if device == '' and torch.cuda.is_available():
        print(f"检测到 {torch.cuda.device_count()} 个 CUDA 设备")
    
    # 加载模型
    model = YOLO(model_path)
    
    # 开始时间
    start_time = time.time()
    
    # 训练模型
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        workers=workers,
        project=project,
        name=name,
        exist_ok=True,
        patience=patience,
        pretrained=True,
        resume=resume
    )
    
    # 结束时间
    end_time = time.time()
    training_time = end_time - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"训练完成！用时: {int(hours)}小时 {int(minutes)}分 {int(seconds)}秒")
    
    # 获取训练结果路径
    results_dir = os.path.join(project, name)
    
    # 绘制训练结果图表
    plot_training_results(results_dir)
    
    return results_dir


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='YOLOv8车辆碰撞检测模型训练')
    
    # 数据和模型参数
    parser.add_argument('--data', type=str, required=True, 
                      help='数据配置文件路径 (YAML)')
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                      help='预训练模型路径或模型大小 (n, s, m, l, x)')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, 
                      help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, 
                      help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, 
                      help='图像大小')
    parser.add_argument('--device', type=str, default='', 
                      help='训练设备，例如 cpu 或 0 或 0,1,2,3')
    parser.add_argument('--workers', type=int, default=4, 
                      help='数据加载的工作线程数')
    
    # 输出参数
    parser.add_argument('--project', type=str, default='../models', 
                      help='项目保存目录')
    parser.add_argument('--name', type=str, default='train', 
                      help='实验名称')
    parser.add_argument('--output', type=str, default='../models/trained', 
                      help='最终模型保存目录')
    
    # 其他参数
    parser.add_argument('--resume', action='store_true', 
                      help='继续上次训练')
    parser.add_argument('--patience', type=int, default=50, 
                      help='早停耐心值，即性能不提升的轮数')
    parser.add_argument('--validate', action='store_true', 
                      help='训练后验证模型')
    
    args = parser.parse_args()
    
    # 检查数据配置文件
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"数据配置文件不存在: {args.data}")
    
    # 检查预训练模型
    if not args.model.endswith('.pt') and args.model not in ['n', 's', 'm', 'l', 'x']:
        # 如果只指定大小，转换为完整的模型名
        if args.model in ['n', 's', 'm', 'l', 'x']:
            args.model = f"yolov8{args.model}.pt"
        else:
            raise ValueError(f"无效的模型指定: {args.model}")
    
    if args.model.endswith('.pt') and not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件不存在: {args.model}")
    
    # 训练模型
    results_dir = train(
        data_yaml=args.data,
        model_path=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        patience=args.patience
    )
    
    # 备份最佳模型
    best_model_path = backup_best_model(results_dir, args.output)
    
    # 验证模型
    if args.validate and best_model_path:
        print("验证最佳模型性能...")
        validate_model(best_model_path, args.data, args.imgsz)


if __name__ == '__main__':
    main() 