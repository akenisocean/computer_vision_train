#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
握拳计数器：使用YOLOv8-Pose模型检测人体姿态，实时统计握拳次数
"""

import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入工具类
from pose.utils import GestureAnalyzer, VideoWriter


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='基于YOLOv8-Pose的握拳计数器')
    
    parser.add_argument('--source', type=str, required=True,
                      help='输入源，可以是摄像头索引(如 0)或视频文件路径')
    parser.add_argument('--output', type=str, default='pose/results',
                      help='结果保存目录，默认为 pose/results')
    parser.add_argument('--threshold', type=float, default=0.15,
                      help='握拳检测阈值，范围0-1，值越小越敏感，默认0.15')
    parser.add_argument('--show', action='store_true', default=True,
                      help='实时显示检测结果，默认开启')
    parser.add_argument('--no-show', dest='show', action='store_false',
                      help='不显示检测结果')
    parser.add_argument('--save', action='store_true', default=True,
                      help='保存检测结果视频，默认开启')
    parser.add_argument('--no-save', dest='save', action='store_false',
                      help='不保存检测结果视频')
    parser.add_argument('--reset-key', type=str, default='r',
                      help='重置计数器的键，默认为 r')
    
    return parser.parse_args()


def cv2_img_to_pil(cv_img):
    """将OpenCV图像转换为PIL图像"""
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def pil_img_to_cv2(pil_img):
    """将PIL图像转换为OpenCV图像"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def put_chinese_text(img, text, position, font_size, color):
    """
    在图像上绘制中文文本
    
    Args:
        img: 输入图像 (OpenCV格式)
        text: 要绘制的文本
        position: 文本位置 (x, y)
        font_size: 字体大小
        color: 文本颜色 (B, G, R)
        
    Returns:
        添加文本后的图像
    """
    # 转换为PIL图像
    pil_img = cv2_img_to_pil(img)
    draw = ImageDraw.Draw(pil_img)
    
    # 获取系统中的字体
    font_path = None
    if os.name == 'nt':  # Windows
        font_path = "C:/Windows/Fonts/simhei.ttf"  # 黑体
    elif sys.platform == 'darwin':  # macOS
        font_path = "/System/Library/Fonts/PingFang.ttc"  # PingFang 字体
    else:  # Linux
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # 通用字体
    
    # 如果找不到默认字体，尝试使用其他常见字体
    if not os.path.exists(font_path):
        possible_fonts = [
            # Windows 字体
            "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
            "C:/Windows/Fonts/simsun.ttc", # 宋体
            # macOS 字体
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            # Linux 字体
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        ]
        
        for possible_font in possible_fonts:
            if os.path.exists(possible_font):
                font_path = possible_font
                break
    
    # 如果还是找不到合适的字体，使用默认字体（不支持中文，但至少能运行）
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            # 使用默认字体
            font = ImageFont.load_default()
            print("警告: 未找到中文字体，将使用默认字体，可能出现中文乱码")
    except Exception as e:
        print(f"字体加载错误: {e}，使用默认字体")
        font = ImageFont.load_default()
    
    # 绘制文字背景
    x, y = position
    text_bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle([text_bbox[0]-5, text_bbox[1]-5, text_bbox[2]+5, text_bbox[3]+5], fill=(0, 0, 0))
    
    # 绘制文字
    rgb_color = (color[2], color[1], color[0])  # 转换BGR为RGB
    draw.text(position, text, font=font, fill=rgb_color)
    
    # 转回OpenCV格式
    return pil_img_to_cv2(pil_img)


def draw_fist_info(frame, is_fist, fist_count, fist_score=None, left_ratio=None, right_ratio=None, reset_key='r'):
    """
    在帧上绘制握拳信息
    
    Args:
        frame: 输入图像帧
        is_fist: 是否握拳
        fist_count: 握拳计数
        fist_score: 握拳得分
        left_ratio: 左手距离比例
        right_ratio: 右手距离比例
        reset_key: 重置计数器的按键
    
    Returns:
        添加信息后的图像
    """
    height, width = frame.shape[:2]
    
    # 握拳状态
    status_text = f"握拳状态: {'是' if is_fist else '否'}"
    frame = put_chinese_text(frame, status_text, (10, 30), 24, (0, 255, 255) if is_fist else (0, 255, 0))
    
    # 握拳计数
    count_text = f"握拳次数: {fist_count}"
    frame = put_chinese_text(frame, count_text, (10, 70), 24, (0, 255, 255))
    
    # 握拳得分
    if fist_score is not None:
        score_text = f"握拳得分: {fist_score:.2f}"
        frame = put_chinese_text(frame, score_text, (10, 110), 24, (0, 255, 255))
    
    # 左右手握拳信息
    y_offset = 150
    if left_ratio is not None:
        left_status = "握拳" if left_ratio < 0.15 else "张开"
        left_text = f"左手: {left_status} ({left_ratio:.2f})"
        frame = put_chinese_text(frame, left_text, (10, y_offset), 20, (255, 0, 255))
        y_offset += 40
    
    if right_ratio is not None:
        right_status = "握拳" if right_ratio < 0.15 else "张开"
        right_text = f"右手: {right_status} ({right_ratio:.2f})"
        frame = put_chinese_text(frame, right_text, (10, y_offset), 20, (255, 0, 255))
    
    # 操作提示
    hint_text = f"按 '{reset_key}' 重置计数器, 按 'q' 退出"
    frame = put_chinese_text(frame, hint_text, (10, height-30), 18, (255, 255, 255))
    
    return frame


def process_video(args):
    """
    处理视频进行握拳检测和计数
    
    Args:
        args: 命令行参数
    """
    # 加载YOLOv8-Pose模型
    model = YOLO("yolov8l-pose.pt")
    
    # 初始化握拳检测器
    gesture_analyzer = GestureAnalyzer(distance_threshold=args.threshold)
    
    # 打开视频源
    try:
        source = int(args.source)  # 尝试将source转换为整数（摄像头索引）
    except ValueError:
        source = args.source       # 如果转换失败，则认为是文件路径
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"错误: 无法打开视频源 {args.source}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # 默认帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 初始化视频写入器（如果需要保存）
    video_writer = None
    if args.save:
        # 确保输出目录存在
        os.makedirs(args.output, exist_ok=True)
        
        # 创建输出文件名
        if isinstance(source, int):
            output_filename = f"camera_{source}_fist_counter_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        else:
            output_filename = f"fist_counter_{Path(source).stem}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        
        output_path = os.path.join(args.output, output_filename)
        video_writer = VideoWriter(output_path, fps, (width, height))
        print(f"检测结果将保存至: {output_path}")
    
    # 处理视频帧
    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # 每3帧处理一次以提高性能 (可根据需要调整)
            if frame_count % 3 != 0 and frame_count > 1:
                # 显示上一帧的处理结果
                if args.show:
                    cv2.imshow("Fist Counter", annotated_frame)
                
                # 保存视频
                if args.save and video_writer:
                    video_writer.write(annotated_frame)
                    
                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(args.reset_key):
                    gesture_analyzer.reset_count()
                    print("计数器已重置")
                    
                continue
            
            # 使用YOLO模型进行姿态检测
            results = model.predict(frame, verbose=False)
            result = results[0]
            
            # 复制帧用于标注
            annotated_frame = frame.copy()
            
            # 绘制关键点和骨架
            annotated_frame = result.plot()
            
            # 获取姿态关键点
            if result.keypoints is not None and len(result.keypoints.data) > 0:
                # 只处理第一个人 (如果检测到多人)
                keypoints = result.keypoints.data[0].cpu().numpy()
                
                # 检测是否握拳
                is_fist, details = gesture_analyzer.detect_fist(keypoints)
                
                # 获取握拳统计信息
                fist_count = details["fist_count"]
                fist_score = details.get("fist_score", 0)
                left_ratio = details.get("left_distance_ratio")
                right_ratio = details.get("right_distance_ratio")
                
                # 绘制握拳信息
                annotated_frame = draw_fist_info(
                    annotated_frame, 
                    is_fist, 
                    fist_count, 
                    fist_score,
                    left_ratio,
                    right_ratio,
                    args.reset_key
                )
            else:
                # 如果没有检测到人，显示基本信息
                annotated_frame = put_chinese_text(annotated_frame, "未检测到人", (10, 30), 24, (0, 0, 255))
                # 使用自定义函数显示提示，而不是直接使用 args.reset_key
                annotated_frame = draw_fist_info(
                    annotated_frame,
                    False,
                    gesture_analyzer.fist_count,
                    reset_key=args.reset_key
                )
            
            # 显示帧
            if args.show:
                cv2.imshow("Fist Counter", annotated_frame)
            
            # 保存视频
            if args.save and video_writer:
                video_writer.write(annotated_frame)
                
            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(args.reset_key):
                gesture_analyzer.reset_count()
                print("计数器已重置")
    
    finally:
        # 清理资源
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()


def main():
    """
    主函数
    """
    # 全局变量
    global args
    args = parse_arguments()
    
    print("启动握拳计数系统...")
    print(f"视频源: {args.source}")
    print(f"握拳检测阈值: {args.threshold}")
    print(f"按 '{args.reset_key}' 键重置计数器")
    print(f"按 'q' 键退出程序")
    
    try:
        process_video(args)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序出错: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("握拳计数系统已关闭")


if __name__ == "__main__":
    main() 