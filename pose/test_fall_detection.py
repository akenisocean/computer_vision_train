#!/usr/bin/env python3
"""
测试脚本：演示如何使用跌倒检测系统

此脚本提供了一个简单的测试界面，用于演示跌倒检测系统的功能。
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path

# 确保可以导入pose模块
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.append(root_dir)

# 导入fall_detection模块
from pose.fall_detection import process_video


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="跌倒检测系统测试脚本")
    
    # 视频源选择
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--camera', type=int, 
                            help='使用摄像头（指定索引，如0表示默认摄像头）')
    source_group.add_argument('--video', type=str,
                            help='使用视频文件')
    
    # 其他选项
    parser.add_argument('--output', type=str, default='pose/results',
                      help='结果保存目录，默认为 pose/results')
    parser.add_argument('--threshold', type=float, default=0.6,
                      help='跌倒检测阈值，范围0-1，值越小越敏感，默认0.6')
    parser.add_argument('--no-save', dest='save', action='store_false',
                      help='不保存检测结果视频')
    parser.add_argument('--alarm', action='store_true',
                      help='启用声音报警功能')
    
    return parser.parse_args()


def main():
    """主函数"""
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
    process_args.alarm = args.alarm
    
    print("===== 跌倒检测系统测试 =====")
    
    if args.camera is not None:
        print(f"使用摄像头索引: {args.camera}")
    else:
        print(f"使用视频文件: {args.video}")
    
    print(f"跌倒检测阈值: {args.threshold}")
    print(f"结果保存: {'开启' if args.save else '关闭'}")
    print(f"声音报警: {'开启' if args.alarm else '关闭'}")
    print("按 'q' 键退出程序")
    print("=========================")
    
    try:
        # 运行检测过程
        process_video(process_args)
    except Exception as e:
        print(f"错误: {str(e)}")
    finally:
        print("测试结束")


if __name__ == "__main__":
    main() 