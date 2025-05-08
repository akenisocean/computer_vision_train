#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：演示如何使用握拳计数系统

此脚本提供了一个简单的测试界面，用于演示握拳计数系统的功能。
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
import traceback

# 确保可以导入pose模块
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.append(root_dir)

# 导入fist_counter模块
from pose.fist_counter import process_video


def check_dependencies():
    """检查必要的依赖是否安装"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        return True
    except ImportError:
        print("错误: 需要安装PIL/Pillow库以支持中文显示")
        print("请运行: pip install pillow")
        return False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="握拳计数系统测试脚本")
    
    # 视频源选择
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--camera', type=int, 
                            help='使用摄像头（指定索引，如0表示默认摄像头）')
    source_group.add_argument('--video', type=str,
                            help='使用视频文件')
    
    # 其他选项
    parser.add_argument('--output', type=str, default='pose/results',
                      help='结果保存目录，默认为 pose/results')
    parser.add_argument('--threshold', type=float, default=0.17,
                      help='握拳检测阈值，范围0-1，值越小越敏感，默认0.17')
    parser.add_argument('--no-save', dest='save', action='store_false',
                      help='不保存检测结果视频')
    parser.add_argument('--reset-key', type=str, default='r',
                      help='重置计数器的键，默认为 r')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)

    args = parse_args()
    
    # 准备参数
    class ProcessArgs:
        pass
    
    process_args = ProcessArgs()
    process_args.source = args.camera if args.camera is not None else args.video
    process_args.output = args.output
    process_args.threshold = args.threshold
    process_args.show = True
    process_args.save = args.save
    process_args.reset_key = args.reset_key
    
    print("===== 握拳计数系统测试 =====")
    
    if args.camera is not None:
        print(f"使用摄像头索引: {args.camera}")
    else:
        print(f"使用视频文件: {args.video}")
    
    print(f"握拳检测阈值: {args.threshold}")
    print(f"结果保存: {'开启' if args.save else '关闭'}")
    print(f"按 '{args.reset_key}' 键重置计数器")
    print(f"按 'q' 键退出程序")
    print("=========================")
    
    try:
        # 运行检测过程
        process_video(process_args)
    except Exception as e:
        print(f"错误: {str(e)}")
        traceback.print_exc()
    finally:
        print("测试结束")


if __name__ == "__main__":
    main() 