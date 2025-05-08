# 基于图像的室内场景3D重建与位置定位系统

本系统通过多视角图像重建室内3D场景，并能够根据新图像实时定位用户在场景中的位置。

## 功能特点

- 图像采集与预处理：处理多角度室内场景图像
- 3D场景重建：利用Structure from Motion (SfM)和Multi-View Stereo (MVS)技术重建3D场景
- 实时位置定位：利用特征匹配技术确定新图像在3D场景中的位置
- 位置可视化：在2D图像上显示用户在3D场景中的位置和朝向

## 系统架构

1. **图像采集与预处理模块**：收集室内场景的多角度图像
2. **3D场景重建模块**：利用SfM和MVS技术重建3D场景
3. **实时位置定位模块**：利用特征匹配技术确定新图像在3D场景中的位置

## 使用方法

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 场景重建

收集并处理室内场景的多角度图像，生成3D场景模型：

```bash
python main.py --mode build --input path/to/images --output output_dir
```

### 3. 位置定位

使用当前位置的图像在已建立的3D场景中定位：

```bash
python main.py --mode locate --input query_image.jpg --model scene_model.ply --features features_db.npz
```

## 技术实现

- **图像特征提取**：使用SIFT特征提取算法
- **相机姿态估计**：通过特征匹配和基础矩阵/本质矩阵计算
- **3D重建**：结合SfM和MVS技术生成点云和网格模型
- **位置定位**：基于特征匹配和PnP算法

## 文件结构

```
position_test/
│
├── main.py                 # 主程序
├── requirements.txt        # 依赖库
├── image_preprocessor.py   # 图像预处理模块 
├── scene_reconstructor.py  # 3D场景重建模块
├── location_estimator.py   # 位置定位模块
└── README.md               # 项目说明文档
```

## 依赖库

- numpy
- opencv-python
- open3d
- scipy
- matplotlib

## 注意事项

1. 重建质量依赖于输入图像的质量和数量，建议使用20-50张覆盖场景各角度的高清图像
2. 位置定位精度受特征匹配质量影响，确保查询图像具有足够的特征点
3. 首次运行3D重建可能需要较长时间，请耐心等待 