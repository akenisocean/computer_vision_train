#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
车辆碰撞检测模型配置
定义模型结构、超参数和训练配置
"""

# 类别配置
CLASS_NAMES = ["car", "truck", "bus", "motorcycle", "collision"]
NUM_CLASSES = len(CLASS_NAMES)

# 模型配置
MODEL_CONFIG = {
    # 基本配置
    "name": "vehicle_collision_detector",
    "architecture": "YOLOv8",
    "variant": "n",  # 可选: n, s, m, l, x (从小到大)
    "task": "detect",  # 可选: detect, segment, pose, classify
    
    # 输入配置
    "input_size": 640,  # 图像输入大小
    "channels": 3,      # 图像通道数
    
    # 训练配置
    "train_settings": {
        "epochs": 100,
        "batch_size": 16,
        "patience": 50,  # 早停耐心值
        "warmup_epochs": 3,
        "initial_lr": 0.01,
        "final_lr": 0.001,
        "optimizer": "SGD",  # 可选: SGD, Adam, AdamW
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "augmentation": True,
    },
    
    # 数据增强配置
    "augmentation": {
        "hsv_h": 0.015,  # 色调调整
        "hsv_s": 0.7,    # 饱和度调整
        "hsv_v": 0.4,    # 亮度调整
        "degrees": 0.0,  # 旋转角度
        "translate": 0.1, # 平移
        "scale": 0.5,    # 缩放
        "shear": 0.0,    # 剪切
        "perspective": 0.0, # 透视变换
        "flipud": 0.0,   # 上下翻转概率
        "fliplr": 0.5,   # 左右翻转概率
        "mosaic": 1.0,   # 马赛克增强概率
        "mixup": 0.0,    # mixup增强概率
    },
    
    # 检测配置
    "detection": {
        "conf_threshold": 0.25,  # 置信度阈值
        "iou_threshold": 0.45,   # IoU阈值(NMS)
        "max_detections": 300,   # 最大检测数量
    },
    
    # 碰撞检测配置
    "collision_detection": {
        "iou_threshold": 0.1,     # 碰撞IoU阈值
        "velocity_change_threshold": 5.0,  # 速度变化阈值
        "track_history": 20,      # 跟踪历史长度
    },
    
    # 导出配置
    "export": {
        "format": "onnx",  # 可选: onnx, tflite, coreml, saved_model
        "dynamic": True,   # 动态批次大小
        "simplify": True,  # 简化模型
        "opset": 12,       # ONNX操作集版本
    }
}


# 数据集配置
DATASET_CONFIG = {
    "path": "dataset",  # 数据集根目录
    "train": "train/images",  # 训练图像相对路径
    "val": "val/images",      # 验证图像相对路径
    "test": "test/images",    # 测试图像相对路径
    
    # 类别配置
    "nc": NUM_CLASSES,  # 类别数量
    "names": CLASS_NAMES,  # 类别名称
    
    # 数据处理
    "cache": False,   # 是否缓存图像
    "rect": False,    # 是否使用矩形训练
    "single_cls": False,  # 是否作为单类别问题
}


# 跟踪器配置
TRACKER_CONFIG = {
    "tracker_type": "bytetrack",  # 可选: botsort, bytetrack
    "track_high_thresh": 0.5,     # 高置信度阈值
    "track_low_thresh": 0.1,      # 低置信度阈值
    "new_track_thresh": 0.6,      # 新轨迹阈值
    "track_buffer": 30,           # 跟踪缓冲区大小
    "match_thresh": 0.8,          # 匹配阈值
    "max_disappeared": 10,        # 最大消失帧数
    "max_distance": 50,           # 最大距离阈值
}


def get_config():
    """
    获取完整配置
    
    返回:
        dict: 配置字典
    """
    config = {
        "model": MODEL_CONFIG,
        "dataset": DATASET_CONFIG,
        "tracker": TRACKER_CONFIG,
    }
    
    return config


def save_config(path):
    """
    保存配置到YAML文件
    
    参数:
        path (str): 输出路径
    """
    import yaml
    
    config = get_config()
    
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"配置已保存到: {path}")


if __name__ == "__main__":
    # 当直接运行此脚本时，打印配置
    import json
    
    config = get_config()
    print(json.dumps(config, indent=4, ensure_ascii=False)) 