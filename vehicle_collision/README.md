# 车辆碰撞检测系统 (Vehicle Collision Detection System)

本项目是一个完整的车辆碰撞检测系统，涵盖了从数据标注到模型训练再到模型应用的全流程。该系统能够在视频或图像中实时检测车辆碰撞事件，可应用于智能交通监控、道路安全管理等场景。

## 目录结构

```
vehicle_collision/
├── data/                     # 数据存储目录
│   ├── raw/                  # 原始数据
│   ├── processed/            # 预处理后的数据
│   └── test/                 # 测试数据
├── models/                   # 模型存储目录
│   ├── pretrained/           # 预训练模型
│   └── trained/              # 训练好的模型
├── scripts/                  # 脚本目录
│   ├── data_preprocessing.py # 数据预处理脚本
│   ├── train.py              # 训练脚本
│   └── inference.py          # 推理脚本
├── utils/                    # 工具函数
│   ├── data_utils.py         # 数据处理工具
│   ├── model_utils.py        # 模型工具
│   └── visualization.py      # 可视化工具
├── configs/                  # 配置文件
│   └── model_config.py       # 模型配置
├── annotations/              # 标注数据目录
└── main.py                   # 主程序入口
```

## 项目流程

### 1. 数据采集与标注

使用Label Studio进行视频数据的标注，标记出以下内容：
- 车辆位置（边界框）
- 碰撞事件（时间戳和位置）
- 碰撞类型（前碰、侧碰、追尾等）

### 2. 数据预处理

- 视频分帧处理
- 标注数据转换为YOLO格式
- 数据增强（旋转、缩放、亮度调整等）
- 数据集划分（训练集、验证集、测试集）

### 3. 模型训练

基于YOLOv8模型进行训练，实现：
- 车辆检测
- 车辆跟踪
- 碰撞行为识别

### 4. 模型评估

使用以下指标评估模型性能：
- 准确率(Precision)
- 召回率(Recall)
- F1分数
- 平均精度(mAP)
- 碰撞检测成功率

### 5. 模型应用

- 实时视频流处理
- 碰撞事件检测与警报
- 结果可视化与保存

## 环境要求

```
Python 3.8+
PyTorch 2.0+
OpenCV 4.8+
Ultralytics 8.0+
Label Studio (用于数据标注)
NumPy 1.24+
```

## 使用方法

### 数据标注

```bash
# 启动Label Studio标注工具
cd vehicle_collision
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true label-studio start
```

### 数据预处理

```bash
python scripts/data_preprocessing.py --input data/raw --output data/processed
```

### 模型训练

```bash
python scripts/train.py --data configs/data.yaml --model yolov8n.pt --epochs 100
```

### 模型推理

```bash
python scripts/inference.py --model models/trained/best.pt --source [视频路径/摄像头]
```

### 运行完整系统

```bash
python main.py --mode [标注/训练/检测] --input [输入路径] --output [输出路径]
```

## 碰撞检测方法

本系统采用两种方法检测车辆碰撞：

1. **直接碰撞检测**：基于车辆边界框的IoU和相对速度变化识别碰撞
2. **行为序列分析**：通过分析车辆连续帧的行为模式识别碰撞事件

## 注意事项

1. 确保数据标注的一致性和准确性
2. 训练前进行充分的数据增强以提高模型泛化能力
3. 针对不同的道路场景和光照条件进行模型微调
4. 实际应用中需考虑系统延迟和处理速度 