#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用户路径追踪与分析示例脚本
"""

import os
import argparse
from pathlib import Path
from target_tracking import UserPathTracker, PathVisualizer


def track_demo(video_path, output_dir="results"):
    """从视频中提取用户路径并进行分析"""
    print(f"开始处理视频: {video_path}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化追踪器
    tracker = UserPathTracker(
        model_path="yolov8l-pose.pt",
        conf_threshold=0.3,
        track_history_len=30
    )
    
    # 处理视频
    path_data = tracker.process_video(
        video_path=video_path,
        output_dir=output_dir,
        visualize=True  # 显示实时追踪结果
    )
    
    # 分析路径数据
    analysis_result = tracker.analyze_path_data(output_dir)
    
    print(f"视频处理完成! 结果已保存至: {output_dir}")
    print(f"共追踪到 {len(path_data['tracks'])} 个用户")
    
    return path_data


def visualize_demo(json_path, output_dir="visualization"):
    """可视化已保存的路径数据"""
    print(f"加载路径数据: {json_path}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化可视化器
    visualizer = PathVisualizer()
    
    # 加载数据
    path_data = visualizer.load_path_data(json_path)
    
    # 提取文件名
    filename = Path(path_data["video_info"]["file_name"]).stem
    
    # 生成各种可视化
    print("生成路径可视化...")
    path_img = visualizer.visualize_paths(
        path_data,
        output_path=os.path.join(output_dir, f"{filename}_paths.png")
    )
    
    print("生成热力图...")
    heatmap_img = visualizer.visualize_heatmap(
        path_data,
        output_path=os.path.join(output_dir, f"{filename}_heatmap.png")
    )
    
    print("生成停留点分析...")
    stay_img = visualizer.visualize_stay_points(
        path_data,
        output_path=os.path.join(output_dir, f"{filename}_stay_points.png"),
        min_stay_time=1.0
    )
    
    print(f"可视化完成! 结果已保存至: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="用户路径追踪与分析示例")
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # 追踪命令
    track_parser = subparsers.add_parser("track", help="从视频中提取用户路径")
    track_parser.add_argument("--video", type=str, required=True, help="视频文件路径")
    track_parser.add_argument("--output", type=str, default="results", help="输出目录")
    
    # 可视化命令
    vis_parser = subparsers.add_parser("visualize", help="可视化路径数据")
    vis_parser.add_argument("--data", type=str, required=True, help="路径数据JSON文件")
    vis_parser.add_argument("--output", type=str, default="visualization", help="输出目录")
    
    args = parser.parse_args()
    
    if args.command == "track":
        track_demo(args.video, args.output)
    elif args.command == "visualize":
        visualize_demo(args.data, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 