#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
车辆碰撞检测推理脚本
使用训练好的YOLOv8模型进行车辆碰撞检测
"""

import os
import cv2
import time
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import deque


class VehicleTracker:
    """车辆跟踪器，用于跟踪视频中的车辆"""
    
    def __init__(self, max_disappeared=10, max_distance=50):
        """
        初始化车辆跟踪器
        
        参数:
            max_disappeared (int): 最大消失帧数，超过此值则删除跟踪
            max_distance (int): 最大距离，用于关联检测框和跟踪对象
        """
        self.next_object_id = 0
        self.objects = {}  # 格式: {object_id: centroid}
        self.disappeared = {}  # 格式: {object_id: count}
        self.trajectories = {}  # 格式: {object_id: deque(最近的几个中心点)}
        self.velocities = {}  # 格式: {object_id: (dx, dy)}
        self.bboxes = {}  # 格式: {object_id: bbox}
        self.classes = {}  # 格式: {object_id: class_id}
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.collisions = set()  # 存储已检测到碰撞的车辆对
    
    def register(self, centroid, bbox, class_id):
        """
        注册新的跟踪对象
        
        参数:
            centroid (tuple): 中心点坐标 (x, y)
            bbox (tuple): 边界框坐标 (x1, y1, x2, y2)
            class_id (int): 类别ID
        """
        # 存储对象
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.trajectories[self.next_object_id] = deque(maxlen=20)
        self.trajectories[self.next_object_id].append(centroid)
        self.velocities[self.next_object_id] = (0, 0)
        self.bboxes[self.next_object_id] = bbox
        self.classes[self.next_object_id] = class_id
        
        # 增加ID计数
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """
        注销跟踪对象
        
        参数:
            object_id (int): 对象ID
        """
        # 删除相关数据
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.trajectories[object_id]
        del self.velocities[object_id]
        del self.bboxes[object_id]
        del self.classes[object_id]
    
    def update(self, bboxes, class_ids, confidences):
        """
        更新跟踪器状态
        
        参数:
            bboxes (list): 边界框列表 [(x1, y1, x2, y2), ...]
            class_ids (list): 类别ID列表
            confidences (list): 置信度列表
        
        返回:
            dict: 更新后的跟踪对象 {id: (bbox, class_id, centroid)}
        """
        # 如果没有检测框，则所有已跟踪对象的消失计数增加
        if len(bboxes) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # 如果消失太久，则删除
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return self.get_tracked_objects()
        
        # 计算当前帧检测到的对象的中心点
        centroids = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            centroids.append((cx, cy))
        
        # 如果当前没有跟踪对象，则注册所有检测到的对象
        if len(self.objects) == 0:
            for i in range(len(centroids)):
                self.register(centroids[i], bboxes[i], class_ids[i])
        
        # 否则，尝试匹配现有的跟踪对象
        else:
            # 获取当前跟踪的对象ID
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # 计算每个中心点之间的距离
            distances = {}
            for i, object_id in enumerate(object_ids):
                distances[object_id] = {}
                for j, centroid in enumerate(centroids):
                    # 计算欧氏距离
                    d = np.sqrt(
                        (object_centroids[i][0] - centroid[0]) ** 2
                        + (object_centroids[i][1] - centroid[1]) ** 2
                    )
                    distances[object_id][j] = d
            
            # 分配检测框给跟踪对象（贪心算法）
            used_centroids = set()
            
            # 按距离排序并分配
            for object_id in object_ids:
                if object_id not in self.disappeared:
                    continue
                    
                # 对当前对象的所有距离排序
                sorted_dists = sorted(
                    distances[object_id].items(), 
                    key=lambda x: x[1]
                )
                
                assigned = False
                for centroid_idx, dist in sorted_dists:
                    # 如果距离太大或中心点已被使用，则跳过
                    if dist > self.max_distance or centroid_idx in used_centroids:
                        continue
                    
                    # 分配中心点给对象
                    old_centroid = self.objects[object_id]
                    self.objects[object_id] = centroids[centroid_idx]
                    self.bboxes[object_id] = bboxes[centroid_idx]
                    self.classes[object_id] = class_ids[centroid_idx]
                    self.disappeared[object_id] = 0
                    
                    # 更新轨迹
                    self.trajectories[object_id].append(centroids[centroid_idx])
                    
                    # 计算速度（如果有足够的轨迹点）
                    if len(self.trajectories[object_id]) >= 2:
                        prev = self.trajectories[object_id][-2]
                        curr = self.trajectories[object_id][-1]
                        self.velocities[object_id] = (
                            curr[0] - prev[0],
                            curr[1] - prev[1]
                        )
                    
                    used_centroids.add(centroid_idx)
                    assigned = True
                    break
                
                # 如果没有分配，则增加消失计数
                if not assigned:
                    self.disappeared[object_id] += 1
            
            # 注册新的对象（没有被分配的检测框）
            for i in range(len(centroids)):
                if i not in used_centroids:
                    self.register(centroids[i], bboxes[i], class_ids[i])
        
        # 移除长时间消失的对象
        for object_id in list(self.disappeared.keys()):
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)
        
        # 检测碰撞
        self.detect_collisions()
        
        return self.get_tracked_objects()
    
    def get_tracked_objects(self):
        """
        获取当前跟踪的所有对象
        
        返回:
            dict: 跟踪对象 {id: (bbox, class_id, centroid, velocity)}
        """
        tracked_objects = {}
        
        for object_id in self.objects.keys():
            if object_id in self.bboxes:
                tracked_objects[object_id] = (
                    self.bboxes[object_id],
                    self.classes[object_id],
                    self.objects[object_id],
                    self.velocities[object_id]
                )
        
        return tracked_objects
    
    def detect_collisions(self):
        """
        检测车辆之间的碰撞
        """
        # 获取所有车辆对象ID
        vehicle_ids = [
            obj_id for obj_id, class_id in self.classes.items()
            if class_id in [0, 1, 2, 3]  # 车辆类别
        ]
        
        # 计算所有车辆对之间的碰撞
        new_collisions = set()
        
        for i in range(len(vehicle_ids)):
            id1 = vehicle_ids[i]
            bbox1 = self.bboxes[id1]
            
            for j in range(i+1, len(vehicle_ids)):
                id2 = vehicle_ids[j]
                bbox2 = self.bboxes[id2]
                
                # 计算IoU
                iou = self.calculate_iou(bbox1, bbox2)
                
                # 计算速度变化
                v1 = self.velocities[id1]
                v2 = self.velocities[id2]
                
                # 碰撞条件：IoU大于阈值并且有相对速度变化
                if iou > 0.1 and self.has_velocity_change(id1, id2):
                    collision_pair = tuple(sorted([id1, id2]))
                    new_collisions.add(collision_pair)
        
        # 更新全局碰撞列表
        self.collisions.update(new_collisions)
        
        return new_collisions
    
    def calculate_iou(self, bbox1, bbox2):
        """
        计算两个边界框的IoU
        
        参数:
            bbox1 (tuple): 边界框1 (x1, y1, x2, y2)
            bbox2 (tuple): 边界框2 (x1, y1, x2, y2)
        
        返回:
            float: IoU值
        """
        # 计算交集矩形
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        # 检查是否有交集
        if x2 < x1 or y2 < y1:
            return 0.0
        
        # 计算交集面积
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # 计算两个边界框的面积
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        # 计算并集面积
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # 计算IoU
        iou = intersection_area / union_area
        
        return iou
    
    def has_velocity_change(self, id1, id2, threshold=5):
        """
        检查两个对象之间是否有显著的速度变化
        
        参数:
            id1 (int): 对象1的ID
            id2 (int): 对象2的ID
            threshold (float): 速度变化阈值
        
        返回:
            bool: 是否有显著速度变化
        """
        # 检查是否有足够的轨迹点
        if (len(self.trajectories[id1]) < 3 or
            len(self.trajectories[id2]) < 3):
            return False
        
        # 计算id1的速度变化
        traj1 = self.trajectories[id1]
        v1_prev = (traj1[-3][0] - traj1[-2][0], traj1[-3][1] - traj1[-2][1])
        v1_curr = (traj1[-2][0] - traj1[-1][0], traj1[-2][1] - traj1[-1][1])
        
        # 计算id2的速度变化
        traj2 = self.trajectories[id2]
        v2_prev = (traj2[-3][0] - traj2[-2][0], traj2[-3][1] - traj2[-2][1])
        v2_curr = (traj2[-2][0] - traj2[-1][0], traj2[-2][1] - traj2[-1][1])
        
        # 计算速度变化量
        dv1 = np.sqrt((v1_curr[0] - v1_prev[0])**2 + (v1_curr[1] - v1_prev[1])**2)
        dv2 = np.sqrt((v2_curr[0] - v2_prev[0])**2 + (v2_curr[1] - v2_prev[1])**2)
        
        # 判断是否超过阈值
        return dv1 > threshold or dv2 > threshold


def process_video(model_path, video_path, output_path=None, conf_threshold=0.25, show=True):
    """
    处理视频进行车辆碰撞检测
    
    参数:
        model_path (str): 模型路径
        video_path (str): 视频路径
        output_path (str): 输出视频路径
        conf_threshold (float): 置信度阈值
        show (bool): 是否显示处理过程
    """
    # 加载模型
    model = YOLO(model_path)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return
    
    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 初始化视频写入器
    video_writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 创建车辆跟踪器
    tracker = VehicleTracker(max_disappeared=10, max_distance=100)
    
    # 帧计数
    frame_count = 0
    
    # 碰撞事件记录
    collision_events = []
    collision_frames = set()
    
    print(f"开始处理视频: {video_path}")
    
    # 逐帧处理
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 运行模型推理
        results = model(frame, conf=conf_threshold)
        
        # 提取检测结果
        bboxes = []
        class_ids = []
        confidences = []
        
        for r in results:
            boxes = r.boxes
            
            for box in boxes:
                # 获取边界框
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 获取类别ID和置信度
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # 只处理车辆类别 (0: car, 1: truck, 2: bus, 3: motorcycle)
                if cls_id in [0, 1, 2, 3]:
                    bboxes.append((x1, y1, x2, y2))
                    class_ids.append(cls_id)
                    confidences.append(conf)
        
        # 更新跟踪器
        tracked_objects = tracker.update(bboxes, class_ids, confidences)
        
        # 检查是否有新的碰撞
        new_collisions = tracker.detect_collisions()
        
        if new_collisions:
            collision_frames.add(frame_count)
            for collision in new_collisions:
                collision_events.append({
                    'frame': frame_count,
                    'time': frame_count / fps,
                    'vehicles': collision
                })
        
        # 可视化
        # 绘制跟踪的车辆
        for obj_id, (bbox, class_id, centroid, velocity) in tracked_objects.items():
            x1, y1, x2, y2 = bbox
            
            # 获取类别名称
            class_names = ['car', 'truck', 'bus', 'motorcycle']
            class_name = class_names[class_id]
            
            # 绘制边界框
            color = (0, 255, 0)  # 绿色
            
            # 如果车辆参与碰撞，则用红色标记
            for event in collision_events:
                if obj_id in event['vehicles'] and abs(frame_count - event['frame']) < 30:
                    color = (0, 0, 255)  # 红色
                    break
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 绘制ID和类别
            label = f"ID:{obj_id} {class_name}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 绘制轨迹
            if obj_id in tracker.trajectories:
                points = list(tracker.trajectories[obj_id])
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], color, 2)
        
        # 显示碰撞警告
        if frame_count in collision_frames or any(abs(frame_count - event['frame']) < 30 for event in collision_events):
            cv2.putText(frame, "碰撞警告!", (width // 2 - 100, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        # 显示帧计数和FPS
        fps_text = f"Frame: {frame_count}"
        cv2.putText(frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 显示视频
        if show:
            cv2.imshow("Vehicle Collision Detection", frame)
            
            # 按q键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 写入输出视频
        if video_writer:
            video_writer.write(frame)
    
    # 释放资源
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    
    # 打印碰撞事件统计
    print(f"处理完成，共检测到 {len(collision_events)} 次碰撞事件")
    for i, event in enumerate(collision_events):
        print(f"碰撞事件 {i+1}: 帧 {event['frame']} ({event['time']:.2f}秒), "
              f"车辆ID: {event['vehicles']}")
    
    return collision_events


def process_webcam(model_path, camera_id=0, output_path=None, conf_threshold=0.25):
    """
    处理网络摄像头视频进行车辆碰撞检测
    
    参数:
        model_path (str): 模型路径
        camera_id (int): 摄像头ID
        output_path (str): 输出视频路径
        conf_threshold (float): 置信度阈值
    """
    # 开启摄像头
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头 {camera_id}")
        return
    
    video_path = f"摄像头 {camera_id}"
    return process_video(model_path, video_path, output_path, conf_threshold, show=True)


def process_image(model_path, image_path, output_path=None, conf_threshold=0.25, show=True):
    """
    处理单张图像进行车辆碰撞检测
    
    参数:
        model_path (str): 模型路径
        image_path (str): 图像路径
        output_path (str): 输出图像路径
        conf_threshold (float): 置信度阈值
        show (bool): 是否显示处理过程
    """
    # 加载模型
    model = YOLO(model_path)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像 {image_path}")
        return
    
    # 运行模型推理
    results = model(image, conf=conf_threshold)
    
    # 提取检测结果
    result = results[0]
    annotated_img = result.plot()
    
    # 保存结果
    if output_path:
        cv2.imwrite(output_path, annotated_img)
        print(f"结果已保存至: {output_path}")
    
    # 显示结果
    if show:
        cv2.imshow("Vehicle Detection", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return results


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='车辆碰撞检测推理工具')
    
    # 模型和输入参数
    parser.add_argument('--model', type=str, required=True, 
                      help='模型路径')
    parser.add_argument('--source', type=str, required=True, 
                      help='输入源，可以是图像、视频路径，或摄像头索引')
    
    # 输出参数
    parser.add_argument('--output', type=str, default=None, 
                      help='输出路径')
    parser.add_argument('--show', action='store_true', 
                      help='显示处理过程')
    
    # 其他参数
    parser.add_argument('--conf', type=float, default=0.25, 
                      help='置信度阈值')
    parser.add_argument('--webcam', action='store_true', 
                      help='使用网络摄像头')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在 {args.model}")
        return
    
    # 处理输入源
    if args.webcam:
        try:
            camera_id = int(args.source)
        except ValueError:
            camera_id = 0
            print(f"警告: 无效的摄像头ID，使用默认值 0")
        
        process_webcam(args.model, camera_id, args.output, args.conf)
    
    elif os.path.isfile(args.source):
        # 判断是图像还是视频
        file_ext = os.path.splitext(args.source)[1].lower()
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            # 处理图像
            process_image(args.model, args.source, args.output, args.conf, args.show)
        
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            # 处理视频
            process_video(args.model, args.source, args.output, args.conf, args.show)
        
        else:
            print(f"错误: 不支持的文件类型 {file_ext}")
    
    else:
        print(f"错误: 输入源 {args.source} 不存在")


if __name__ == '__main__':
    main() 