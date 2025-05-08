import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入工具类
from pose.utils import PoseAnalyzer, AlarmSystem, VideoWriter


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='基于YOLOv8-Pose的跌倒检测与报警系统')
    
    parser.add_argument('--source', type=str, required=True,
                      help='输入源，可以是摄像头索引(如 0)或视频文件路径')
    parser.add_argument('--output', type=str, default='pose/results',
                      help='结果保存目录，默认为 pose/results')
    parser.add_argument('--threshold', type=float, default=0.6,
                      help='跌倒检测阈值，范围0-1，值越小越敏感，默认0.6')
    parser.add_argument('--show', action='store_true', default=True,
                      help='实时显示检测结果，默认开启')
    parser.add_argument('--no-show', dest='show', action='store_false',
                      help='不显示检测结果')
    parser.add_argument('--save', action='store_true', default=True,
                      help='保存检测结果视频，默认开启')
    parser.add_argument('--no-save', dest='save', action='store_false',
                      help='不保存检测结果视频')
    parser.add_argument('--alarm', action='store_true',
                      help='启用声音报警功能')
    
    return parser.parse_args()


def process_video(args):
    """
    处理视频进行跌倒检测
    
    Args:
        args: 命令行参数
    """
    # 加载YOLOv8-Pose模型
    model = YOLO("yolov8l-pose.pt")
    
    # 初始化跌倒检测器
    pose_analyzer = PoseAnalyzer(fall_threshold=args.threshold)
    
    # 初始化报警系统
    alarm_system = AlarmSystem(
        enable_sound=args.alarm,
        snapshot_dir=os.path.join(args.output, 'snapshots')
    )
    
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
            output_filename = f"camera_{source}_fall_detection_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        else:
            output_filename = f"fall_detection_{Path(source).stem}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        
        output_path = os.path.join(args.output, output_filename)
        video_writer = VideoWriter(output_path, fps, (width, height))
        print(f"检测结果将保存至: {output_path}")
    
    # 处理视频帧
    frame_count = 0
    fall_detected_count = 0  # 连续检测到跌倒的帧数
    fall_recovery_count = 0  # 连续未检测到跌倒的帧数
    is_fall_state = False    # 当前是否处于跌倒状态
    
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
                    cv2.imshow("Fall Detection", annotated_frame)
                
                # 保存视频
                if args.save and video_writer:
                    video_writer.write(annotated_frame)
                    
                # 检查按键
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                continue
            
            # 使用YOLO模型进行姿态检测
            results = model.predict(frame, verbose=False)
            result = results[0]
            
            # 复制帧用于标注
            annotated_frame = frame.copy()
            
            # 获取姿态关键点
            if result.keypoints is not None and len(result.keypoints.data) > 0:
                # 只处理第一个人 (如果检测到多人)
                keypoints = result.keypoints.data[0].cpu().numpy()
                
                # 检测是否跌倒
                is_falling, details = pose_analyzer.detect_fall(keypoints)
                
                # 状态管理 (防止误报和短暂误检)
                if is_falling:
                    fall_detected_count += 1
                    fall_recovery_count = 0
                    
                    # 连续多帧检测到跌倒才触发报警
                    if fall_detected_count >= 5 and not is_fall_state:
                        is_fall_state = True
                        alarm_system.trigger_alarm(annotated_frame, details)
                else:
                    fall_recovery_count += 1
                    fall_detected_count = 0
                    
                    # 连续多帧未检测到跌倒才解除跌倒状态
                    if fall_recovery_count >= 15 and is_fall_state:
                        is_fall_state = False
                
                # 绘制关键点和骨架
                annotated_frame = result.plot()
                
                # 显示跌倒状态和分数
                status_text = f"跌倒状态: {'是' if is_fall_state else '否'}"
                score_text = f"跌倒分数: {details['fall_score']:.2f}"
                
                cv2.putText(annotated_frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, 
                           (0, 0, 255) if is_fall_state else (0, 255, 0), 2)
                           
                cv2.putText(annotated_frame, score_text, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, 
                           (0, 0, 255) if is_fall_state else (0, 255, 0), 2)
            
            # 显示帧
            if args.show:
                cv2.imshow("Fall Detection", annotated_frame)
            
            # 保存视频
            if args.save and video_writer:
                video_writer.write(annotated_frame)
                
            # 检查按键
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
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
    args = parse_arguments()
    
    print("启动跌倒检测系统...")
    print(f"视频源: {args.source}")
    print(f"跌倒检测阈值: {args.threshold}")
    print(f"声音报警: {'开启' if args.alarm else '关闭'}")
    
    try:
        process_video(args)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序出错: {str(e)}")
    finally:
        print("跌倒检测系统已关闭")


if __name__ == "__main__":
    main() 