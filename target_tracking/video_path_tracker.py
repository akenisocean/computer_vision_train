import os
import cv2
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt


class UserPathTracker:
    def __init__(self, model_path="yolov8l-pose.pt", conf_threshold=0.3, track_history_len=30):
        """
        用户路径追踪器，用于从视频中提取用户移动轨迹
        
        参数:
            model_path (str): YOLO模型路径
            conf_threshold (float): 置信度阈值
            track_history_len (int): 轨迹历史长度
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.track_history_len = track_history_len
        self.track_history = {}  # 存储所有追踪ID的历史轨迹
        self.path_data = {}      # 存储完整路径数据用于保存
        
    def process_video(self, video_path, output_dir="results", visualize=True, save_interval=5):
        """
        处理视频并提取用户路径
        
        参数:
            video_path (str): 视频文件路径
            output_dir (str): 输出目录
            visualize (bool): 是否可视化结果
            save_interval (int): 保存间隔(秒)
            
        返回:
            path_data (dict): 用户路径数据
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
            
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 设置输出视频
        output_video_path = os.path.join(output_dir, f"tracked_{Path(video_path).stem}.mp4")
        output_video = None
        
        if visualize:
            output_video = cv2.VideoWriter(
                output_video_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )
        
        # 初始化计时器和帧计数器
        start_time = time.time()
        last_save_time = start_time
        frame_idx = 0
        
        # 初始化用户路径数据
        video_info = {
            "file_name": Path(video_path).name,
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "processed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.path_data = {
            "video_info": video_info,
            "tracks": {}
        }
        
        print(f"开始处理视频: {video_path}")
        print(f"视频信息: {width}x{height}, {fps}fps, {frame_count}帧")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 计算当前时间点(秒)
                timestamp = frame_idx / fps
                
                # 追踪当前帧中的人
                results = self.model.track(frame, persist=True, conf=self.conf_threshold, classes=0)
                
                if results[0].boxes is not None and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xywh.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    
                    # 更新每个ID的轨迹
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        
                        # 使用底部中心点作为位置
                        center_x, center_y = int(x), int(y + h/2)
                        
                        # 如果是新ID，初始化轨迹历史
                        if track_id not in self.track_history:
                            self.track_history[track_id] = []
                            self.path_data["tracks"][str(track_id)] = []
                        
                        # 更新轨迹历史
                        self.track_history[track_id].append((center_x, center_y))
                        if len(self.track_history[track_id]) > self.track_history_len:
                            self.track_history[track_id].pop(0)
                        
                        # 记录完整路径数据
                        self.path_data["tracks"][str(track_id)].append({
                            "frame": frame_idx,
                            "timestamp": timestamp,
                            "position": [float(center_x), float(center_y)],
                            "box": [float(x), float(y), float(w), float(h)]
                        })
                
                # 可视化结果
                if visualize:
                    annotated_frame = frame.copy()
                    
                    # 画出所有轨迹
                    for track_id, track in self.track_history.items():
                        # 跳过空轨迹
                        if not track:
                            continue
                            
                        # 为每个ID生成唯一颜色
                        color = self._get_color_by_id(track_id)
                        
                        # 绘制轨迹线
                        for i in range(1, len(track)):
                            cv2.line(
                                annotated_frame,
                                track[i - 1],
                                track[i],
                                color,
                                2
                            )
                            
                        # 在最新位置绘制ID标签
                        if track:
                            last_x, last_y = track[-1]
                            cv2.putText(
                                annotated_frame,
                                f"ID: {track_id}",
                                (last_x, last_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2
                            )
                    
                    # 显示帧信息
                    cv2.putText(
                        annotated_frame,
                        f"Frame: {frame_idx}/{frame_count} | Time: {timestamp:.2f}s",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
                    
                    # 显示追踪的人数
                    cv2.putText(
                        annotated_frame,
                        f"Tracked: {len(self.track_history)}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
                    
                    # 显示结果帧
                    cv2.imshow("User Path Tracking", annotated_frame)
                    
                    # 写入视频
                    if output_video:
                        output_video.write(annotated_frame)
                    
                    # 按q键退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # 定期保存轨迹数据
                current_time = time.time()
                if current_time - last_save_time >= save_interval:
                    self._save_path_data(output_dir, video_path)
                    last_save_time = current_time
                
                # 更新帧计数器
                frame_idx += 1
                
                # 打印进度
                if frame_idx % 30 == 0:
                    elapsed_time = time.time() - start_time
                    progress = frame_idx / frame_count * 100
                    print(f"进度: {progress:.1f}% | 已处理 {frame_idx}/{frame_count} 帧 | 耗时: {elapsed_time:.2f}秒")
        
        finally:
            # 释放资源
            cap.release()
            if output_video:
                output_video.release()
            cv2.destroyAllWindows()
            
            # 最后保存一次轨迹数据
            self._save_path_data(output_dir, video_path)
            
            # 生成轨迹可视化图像
            self._generate_path_visualization(output_dir, video_path, width, height)
            
            print(f"视频处理完成!")
            print(f"轨迹数据已保存至: {output_dir}")
            
        return self.path_data
    
    def _get_color_by_id(self, track_id):
        """为每个追踪ID生成唯一的颜色"""
        np.random.seed(int(track_id * 100))
        return tuple(map(int, np.random.randint(0, 255, 3)))
    
    def _save_path_data(self, output_dir, video_path):
        """保存轨迹数据为JSON文件"""
        file_name = f"path_data_{Path(video_path).stem}.json"
        file_path = os.path.join(output_dir, file_name)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.path_data, f, indent=2)
            
        return file_path
    
    def _generate_path_visualization(self, output_dir, video_path, width, height):
        """生成静态的轨迹可视化图像"""
        # 创建空白图像
        vis_img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # 绘制轨迹
        for track_id, track_data in self.path_data["tracks"].items():
            if not track_data:
                continue
                
            # 获取位置点
            points = [(d["position"][0], d["position"][1]) for d in track_data]
            
            # 获取颜色
            color = self._get_color_by_id(int(track_id))
            
            # 绘制轨迹线
            for i in range(1, len(points)):
                pt1 = (int(points[i-1][0]), int(points[i-1][1]))
                pt2 = (int(points[i][0]), int(points[i][1]))
                cv2.line(vis_img, pt1, pt2, color, 2)
            
            # 绘制起点和终点
            if len(points) > 0:
                # 起点(绿色)
                start_point = (int(points[0][0]), int(points[0][1]))
                cv2.circle(vis_img, start_point, 5, (0, 255, 0), -1)
                
                # 终点(红色)
                end_point = (int(points[-1][0]), int(points[-1][1]))
                cv2.circle(vis_img, end_point, 5, (0, 0, 255), -1)
                
                # 添加ID标签
                cv2.putText(
                    vis_img,
                    f"ID: {track_id}",
                    (end_point[0], end_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
        
        # 添加标题和信息
        cv2.putText(
            vis_img,
            f"用户路径 - {Path(video_path).stem}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        
        # 添加图例
        cv2.circle(vis_img, (width - 100, 30), 5, (0, 255, 0), -1)
        cv2.putText(
            vis_img,
            "起点",
            (width - 90, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
        
        cv2.circle(vis_img, (width - 100, 50), 5, (0, 0, 255), -1)
        cv2.putText(
            vis_img,
            "终点",
            (width - 90, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
        
        # 保存图像
        output_path = os.path.join(output_dir, f"path_vis_{Path(video_path).stem}.png")
        cv2.imwrite(output_path, vis_img)
        
        return output_path
        
    def analyze_path_data(self, output_dir, path_data=None):
        """
        分析路径数据并生成报告
        
        参数:
            output_dir (str): 输出目录
            path_data (dict): 路径数据(默认使用当前实例的数据)
        """
        if path_data is None:
            path_data = self.path_data
            
        if not path_data or "tracks" not in path_data:
            print("没有可用的路径数据进行分析")
            return
            
        # 创建分析结果目录
        analysis_dir = os.path.join(output_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # 初始化分析结果
        analysis_result = {
            "video_info": path_data["video_info"],
            "total_tracks": len(path_data["tracks"]),
            "track_analysis": {}
        }
        
        # 分析每个轨迹
        for track_id, track_data in path_data["tracks"].items():
            if not track_data:
                continue
                
            # 计算总距离
            total_distance = 0
            for i in range(1, len(track_data)):
                x1, y1 = track_data[i-1]["position"]
                x2, y2 = track_data[i]["position"]
                distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                total_distance += distance
                
            # 计算停留点(速度接近0的位置)
            stay_points = []
            for i in range(1, len(track_data)-1):
                x1, y1 = track_data[i-1]["position"]
                x2, y2 = track_data[i]["position"]
                x3, y3 = track_data[i+1]["position"]
                
                dist_before = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                dist_after = np.sqrt((x3-x2)**2 + (y3-y2)**2)
                
                if dist_before < 5 and dist_after < 5:
                    stay_points.append({
                        "frame": track_data[i]["frame"],
                        "timestamp": track_data[i]["timestamp"],
                        "position": track_data[i]["position"]
                    })
            
            # 计算平均速度(像素/秒)
            if len(track_data) > 1:
                start_time = track_data[0]["timestamp"]
                end_time = track_data[-1]["timestamp"]
                duration = end_time - start_time
                avg_speed = total_distance / duration if duration > 0 else 0
            else:
                avg_speed = 0
                
            # 获取轨迹持续时间
            track_duration = track_data[-1]["timestamp"] - track_data[0]["timestamp"] if len(track_data) > 1 else 0
            
            # 轨迹分析
            track_analysis = {
                "frames": [data["frame"] for data in track_data],
                "start_frame": track_data[0]["frame"],
                "end_frame": track_data[-1]["frame"],
                "start_time": track_data[0]["timestamp"],
                "end_time": track_data[-1]["timestamp"],
                "duration": track_duration,
                "total_distance": total_distance,
                "average_speed": avg_speed,
                "path_length": len(track_data),
                "start_position": track_data[0]["position"],
                "end_position": track_data[-1]["position"],
                "stay_points": stay_points,
                "stay_count": len(stay_points)
            }
            
            analysis_result["track_analysis"][track_id] = track_analysis
        
        # 保存分析结果
        analysis_path = os.path.join(analysis_dir, f"analysis_{Path(path_data['video_info']['file_name']).stem}.json")
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2)
        
        # 生成分析可视化
        self._generate_analysis_visualization(analysis_dir, analysis_result)
        
        print(f"路径分析完成，结果已保存至: {analysis_path}")
        return analysis_result
    
    def _generate_analysis_visualization(self, output_dir, analysis_result):
        """生成分析可视化图表"""
        if not analysis_result or "track_analysis" not in analysis_result:
            return
            
        plt.figure(figsize=(15, 10))
        
        # 1. 速度分布图
        plt.subplot(2, 2, 1)
        speeds = [data["average_speed"] for _, data in analysis_result["track_analysis"].items()]
        plt.hist(speeds, bins=10, color='skyblue', edgecolor='black')
        plt.title('用户移动速度分布')
        plt.xlabel('速度 (像素/秒)')
        plt.ylabel('数量')
        
        # 2. 停留点分布
        plt.subplot(2, 2, 2)
        stay_counts = [data["stay_count"] for _, data in analysis_result["track_analysis"].items()]
        plt.hist(stay_counts, bins=max(10, max(stay_counts) if stay_counts else 1), color='lightgreen', edgecolor='black')
        plt.title('停留点分布')
        plt.xlabel('停留点数量')
        plt.ylabel('用户数量')
        
        # 3. 轨迹持续时间
        plt.subplot(2, 2, 3)
        durations = [data["duration"] for _, data in analysis_result["track_analysis"].items()]
        plt.hist(durations, bins=10, color='salmon', edgecolor='black')
        plt.title('轨迹持续时间分布')
        plt.xlabel('持续时间 (秒)')
        plt.ylabel('数量')
        
        # 4. 轨迹距离
        plt.subplot(2, 2, 4)
        distances = [data["total_distance"] for _, data in analysis_result["track_analysis"].items()]
        plt.hist(distances, bins=10, color='mediumpurple', edgecolor='black')
        plt.title('轨迹距离分布')
        plt.xlabel('距离 (像素)')
        plt.ylabel('数量')
        
        plt.tight_layout()
        
        # 保存图表
        vis_path = os.path.join(output_dir, f"analysis_vis_{Path(analysis_result['video_info']['file_name']).stem}.png")
        plt.savefig(vis_path)
        plt.close()
        
        return vis_path


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='从视频中提取用户路径')
    parser.add_argument('--video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--output', type=str, default='results', help='输出目录，默认为results')
    parser.add_argument('--model', type=str, default='yolov8l-pose.pt', help='YOLO模型路径')
    parser.add_argument('--conf', type=float, default=0.3, help='置信度阈值')
    parser.add_argument('--no-vis', action='store_true', help='不显示可视化结果')
    parser.add_argument('--analyze', action='store_true', help='生成路径分析报告')
    
    args = parser.parse_args()
    
    # 初始化用户路径追踪器
    tracker = UserPathTracker(
        model_path=args.model,
        conf_threshold=args.conf
    )
    
    # 处理视频
    path_data = tracker.process_video(
        video_path=args.video,
        output_dir=args.output,
        visualize=not args.no_vis
    )
    
    # 分析路径数据(如果需要)
    if args.analyze:
        tracker.analyze_path_data(args.output)
    

if __name__ == '__main__':
    main() 