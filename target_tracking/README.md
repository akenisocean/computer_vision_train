# 用户路径追踪与分析系统

这个模块用于从视频中提取用户移动轨迹，并进行路径分析，以便后续对用户行为进行深入分析。系统使用YOLOv8进行人体检测和追踪，记录用户在视频中的移动路径，并提供多种可视化和分析工具。

## 功能特点

- **路径追踪**：从视频中自动检测和追踪用户，记录完整移动轨迹
- **数据存储**：将轨迹数据保存为JSON格式，方便后续处理和分析
- **可视化工具**：
  - 轨迹可视化：显示用户完整移动路径
  - 热力图分析：识别场景中的热门区域
  - 停留点分析：检测并展示用户停留位置
- **行为分析**：
  - 用户移动速度、距离统计
  - 停留时间和频率分析
  - 用户路径模式识别

## 安装依赖

本系统依赖以下Python库：

```bash
pip install ultralytics>=8.0.0 opencv-python>=4.8.0 numpy>=1.24.0 matplotlib>=3.7.0
```

## 使用方法

### 1. 视频路径追踪

从视频中提取用户路径并保存：

```bash
python -m target_tracking.video_path_tracker --video PATH_TO_VIDEO --output OUTPUT_DIRECTORY
```

参数说明：
- `--video`: 视频文件路径
- `--output`: 输出目录(默认为results)
- `--model`: YOLO模型路径(默认为yolov8l-pose.pt)
- `--conf`: 检测置信度阈值(默认为0.3)
- `--no-vis`: 使用此参数关闭可视化
- `--analyze`: 使用此参数生成路径分析报告

例如：
```bash
python -m target_tracking.video_path_tracker --video sample.mp4 --output results --analyze
```

### 2. 路径数据可视化

可视化已保存的路径数据：

```bash
python -m target_tracking.path_visualizer --data PATH_TO_JSON_DATA --output-dir VISUALIZATION_DIR
```

参数说明：
- `--data`: 路径数据JSON文件
- `--output-dir`: 输出目录(默认为visualization)
- `--background`: 背景图像路径(可选)
- `--type`: 可视化类型(path/heatmap/stay/all)
- `--min-stay-time`: 最小停留时间(秒)，用于停留点分析

例如：
```bash
python -m target_tracking.path_visualizer --data results/path_data_sample.json --type all
```

## 输出文件说明

系统会生成以下文件：

1. **轨迹数据文件**: `path_data_视频名称.json`
   - 包含完整的轨迹信息，每个点的位置和时间戳
   
2. **轨迹可视化图**: `path_vis_视频名称.png`
   - 显示所有用户的完整移动轨迹
   
3. **分析结果文件**: `analysis_视频名称.json`
   - 包含用户移动速度、距离、停留点等详细分析
   
4. **分析可视化图**: `analysis_vis_视频名称.png`
   - 直观展示用户行为统计分析

## 数据格式

轨迹数据JSON格式说明：

```json
{
  "video_info": {
    "file_name": "视频文件名",
    "width": 1920,
    "height": 1080,
    "fps": 30,
    "frame_count": 900,
    "processed_date": "处理日期时间"
  },
  "tracks": {
    "1": [  // 用户ID为1的轨迹
      {
        "frame": 0,
        "timestamp": 0.0,
        "position": [x, y],
        "box": [x, y, w, h]
      },
      // 更多轨迹点...
    ],
    // 更多用户轨迹...
  }
}
```

## 应用场景

- 零售店顾客行为分析
- 商场人流动线研究
- 展馆访客路径优化
- 公共场所人员活动模式分析
- 安全监控和异常行为检测

## 注意事项

1. 确保视频质量良好，光线充足
2. 视频中人物遮挡过多可能导致追踪不连续
3. 处理高清视频时可能需要较高的计算资源
4. 第一次运行时会自动下载YOLOv8模型 