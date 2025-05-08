# 基于yolo11的人群检测数据训练
**作者** : HappyGiraffe
**最后更新时间**： 2025年4月11日

# 环境准备

## 1. 创建一个python环境并安装所需要的依赖

   > ps: 这里为了依赖版本隔离，使用conda进行创建
```shell
conda create --name yolo11_train python=3.12
# 使用当前创建的环境
conda activate yolo11_train
# 下载所需要的依赖
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```
## 2. 浏览器访问label-studio的页面，进行图片标记
> ps: 如果已经有准备好的数据集合，可以跳过这一步
```shell
# 启动标记界面进行数据标记
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true label-studio start
```

浏览器访问：

```shell
http://localhost:8080
```

## 3. 导出yolo数据格式进行训练



















# YOLO人数统计系统

基于YOLOv11的人群密度检测与计数系统，可以准确识别图像中的人数，并提供可视化结果。

## 功能特点

- 支持单张图片和批量图片处理
- 实时显示每个检测目标的置信度
- 可视化检测结果，包含边界框标注
- 自动保存处理后的图片
- 支持多种图片格式（jpg、jpeg、png、bmp）

## 环境要求

```bash
pip install -r requirements.txt
```

主要依赖：
- ultralytics
- opencv-python
- numpy

## 使用方法

### 1. 处理单张图片

```bash
python main.py --input path/to/your/image.jpg
```

### 2. 处理整个目录

```bash
python main.py --input path/to/your/directory
```

### 3. 自定义输出目录

```bash
python main.py --input path/to/your/image.jpg --output custom_results
```

## 参数说明

- `--input`：必需参数，指定输入图片或目录的路径
- `--output`：可选参数，指定结果保存目录，默认为 'results'

## 输出结果

程序会在指定的输出目录中生成处理后的图片，包含以下信息：
- 绿色边界框标注检测到的人
- 每个检测目标的置信度标签
- 图片顶部显示总人数统计

## 注意事项

1. 确保 `best.pt` 模型文件在项目根目录下
2. 处理大量图片时，建议使用目录处理模式
3. 支持的图片格式：jpg、jpeg、png、bmp

## 示例输出

控制台输出示例：
```
图片 example.jpg 中检测到 5 人
结果已保存至: results/result_example.jpg
```

## 许可证

本项目采用 MIT 许可证

# 汽车和坑洼检测系统

这个项目使用YOLOv8模型来检测图片中的汽车和道路坑洼。

## 功能特点

- 支持图片中的汽车和坑洼检测
- 实时显示检测结果
- 支持图片保存功能
- 提供检测结果的置信度显示

## 环境要求

- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- NumPy

## 安装依赖

```bash
pip install ultralytics opencv-python numpy
```

## 使用方法

1. 确保模型文件 `car_pothole_yolo11-seg.pt` 在项目目录中
2. 运行主程序：
```bash
python car_pothole_detection.py
```

## 参数说明

- `image_path`: 输入图片的路径
- `conf_threshold`: 检测置信度阈值（默认0.25）
- `save_path`: 结果保存路径（可选）

## 返回值

- 返回处理后的图片，包含检测框和标签
- 同时显示检测结果窗口

# 用户路径追踪与分析系统

基于YOLOv8的用户路径追踪与行为分析系统，可以从视频中提取用户移动轨迹，并进行路径分析，以便对用户行为进行深入研究。

## 主要功能

- 从视频中检测和追踪人体
- 记录用户完整移动轨迹
- 生成路径可视化、热力图和停留点分析
- 分析用户活动模式和行为特征

## 使用方法

```bash
# 从视频中提取用户路径
python -m target_tracking.video_path_tracker --video PATH_TO_VIDEO --output results --analyze

# 可视化已保存的路径数据
python -m target_tracking.path_visualizer --data results/path_data_视频名称.json --type all
```

详细文档请参阅 [target_tracking/README.md](target_tracking/README.md)