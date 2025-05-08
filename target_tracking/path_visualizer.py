import os
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


class PathVisualizer:
    def __init__(self):
        """初始化路径可视化器"""
        pass
        
    def load_path_data(self, json_path):
        """
        加载用户路径数据
        
        参数:
            json_path (str): JSON文件路径
            
        返回:
            path_data (dict): 路径数据
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"找不到路径数据文件: {json_path}")
            
        with open(json_path, 'r', encoding='utf-8') as f:
            path_data = json.load(f)
            
        return path_data
        
    def visualize_paths(self, path_data, output_path=None, background_image=None):
        """
        可视化用户路径
        
        参数:
            path_data (dict): 路径数据
            output_path (str, optional): 输出图像路径
            background_image (str, optional): 背景图像路径
            
        返回:
            vis_img (numpy.ndarray): 可视化图像
        """
        if "video_info" not in path_data or "tracks" not in path_data:
            raise ValueError("无效的路径数据格式")
            
        width = path_data["video_info"]["width"]
        height = path_data["video_info"]["height"]
        
        # 创建图像
        if background_image and os.path.exists(background_image):
            vis_img = cv2.imread(background_image)
            if vis_img.shape[:2] != (height, width):
                vis_img = cv2.resize(vis_img, (width, height))
        else:
            vis_img = np.ones((height, width, 3), dtype=np.uint8) * 255
            
        # 绘制每个用户的路径
        for track_id, track_data in path_data["tracks"].items():
            if not track_data:
                continue
                
            # 获取位置点
            points = [(d["position"][0], d["position"][1]) for d in track_data]
            
            # 为每个ID生成唯一颜色
            np.random.seed(int(track_id) * 100)
            color = tuple(map(int, np.random.randint(0, 255, 3)))
            
            # 绘制轨迹线
            for i in range(1, len(points)):
                pt1 = (int(points[i-1][0]), int(points[i-1][1]))
                pt2 = (int(points[i][0]), int(points[i][1]))
                cv2.line(vis_img, pt1, pt2, color, 2)
            
            # 绘制起点和终点
            if points:
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
            f"用户路径 - {Path(path_data['video_info']['file_name']).stem}",
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
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, vis_img)
            print(f"图像已保存至: {output_path}")
            
        return vis_img
        
    def visualize_heatmap(self, path_data, output_path=None, background_image=None, sigma=15):
        """
        生成用户活动热力图
        
        参数:
            path_data (dict): 路径数据
            output_path (str, optional): 输出图像路径
            background_image (str, optional): 背景图像路径
            sigma (int): 高斯模糊的标准差，控制热点扩散范围
            
        返回:
            heatmap_img (numpy.ndarray): 热力图图像
        """
        if "video_info" not in path_data or "tracks" not in path_data:
            raise ValueError("无效的路径数据格式")
            
        width = path_data["video_info"]["width"]
        height = path_data["video_info"]["height"]
        
        # 创建热力图
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # 累积所有路径点
        for track_id, track_data in path_data["tracks"].items():
            for data in track_data:
                x, y = map(int, data["position"])
                if 0 <= x < width and 0 <= y < height:
                    heatmap[y, x] += 1
        
        # 应用高斯模糊使热力图平滑
        if np.max(heatmap) > 0:  # 防止空热力图
            heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigma)
            heatmap = heatmap / np.max(heatmap)  # 归一化
            
        # 转换为彩色热力图
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        # 添加背景图像(如果有)
        if background_image and os.path.exists(background_image):
            bg_img = cv2.imread(background_image)
            if bg_img.shape[:2] != (height, width):
                bg_img = cv2.resize(bg_img, (width, height))
                
            # 叠加热力图
            alpha = 0.7
            heatmap_img = cv2.addWeighted(bg_img, 1-alpha, heatmap_colored, alpha, 0)
        else:
            heatmap_img = heatmap_colored
            
        # 添加标题
        cv2.putText(
            heatmap_img,
            f"用户活动热力图 - {Path(path_data['video_info']['file_name']).stem}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # 保存图像
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, heatmap_img)
            print(f"热力图已保存至: {output_path}")
            
        return heatmap_img
        
    def visualize_stay_points(self, path_data, output_path=None, background_image=None, min_stay_time=1.0):
        """
        可视化用户停留点
        
        参数:
            path_data (dict): 路径数据
            output_path (str, optional): 输出图像路径
            background_image (str, optional): 背景图像路径
            min_stay_time (float): 最小停留时间(秒)
            
        返回:
            stay_points_img (numpy.ndarray): 停留点图像
        """
        if "video_info" not in path_data or "tracks" not in path_data:
            raise ValueError("无效的路径数据格式")
            
        width = path_data["video_info"]["width"]
        height = path_data["video_info"]["height"]
        fps = path_data["video_info"]["fps"]
        
        # 创建图像
        if background_image and os.path.exists(background_image):
            stay_points_img = cv2.imread(background_image)
            if stay_points_img.shape[:2] != (height, width):
                stay_points_img = cv2.resize(stay_points_img, (width, height))
        else:
            stay_points_img = np.ones((height, width, 3), dtype=np.uint8) * 255
            
        # 查找所有停留点
        stay_points = []
        for track_id, track_data in path_data["tracks"].items():
            if len(track_data) < 3:
                continue
                
            # 计算速度变化
            speeds = []
            for i in range(1, len(track_data)):
                x1, y1 = track_data[i-1]["position"]
                x2, y2 = track_data[i]["position"]
                t1 = track_data[i-1]["timestamp"]
                t2 = track_data[i]["timestamp"]
                
                # 计算距离和时间差
                dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                dt = t2 - t1
                
                speed = dist / dt if dt > 0 else 0
                speeds.append(speed)
            
            # 找出速度很低的连续帧段
            stay_segment = []
            for i, speed in enumerate(speeds):
                if speed < 5:  # 移动很慢，可能是停留
                    stay_segment.append(i + 1)  # 索引调整
                else:
                    # 检查当前停留段
                    if stay_segment:
                        start_idx = stay_segment[0]
                        end_idx = stay_segment[-1]
                        start_time = track_data[start_idx]["timestamp"]
                        end_time = track_data[end_idx]["timestamp"]
                        
                        # 如果停留时间够长，记录这个停留点
                        if end_time - start_time >= min_stay_time:
                            # 使用停留段的中心点
                            mid_idx = stay_segment[len(stay_segment)//2]
                            stay_points.append({
                                "track_id": track_id,
                                "position": track_data[mid_idx]["position"],
                                "start_time": start_time,
                                "end_time": end_time,
                                "duration": end_time - start_time
                            })
                        
                        stay_segment = []
            
            # 处理最后一个可能的停留段
            if stay_segment:
                start_idx = stay_segment[0]
                end_idx = stay_segment[-1]
                start_time = track_data[start_idx]["timestamp"]
                end_time = track_data[end_idx]["timestamp"]
                
                if end_time - start_time >= min_stay_time:
                    mid_idx = stay_segment[len(stay_segment)//2]
                    stay_points.append({
                        "track_id": track_id,
                        "position": track_data[mid_idx]["position"],
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": end_time - start_time
                    })
        
        # 绘制停留点
        for point in stay_points:
            track_id = point["track_id"]
            x, y = map(int, point["position"])
            duration = point["duration"]
            
            # 根据停留时间确定圆的大小
            radius = min(30, max(10, int(duration * 2)))
            
            # 为每个ID生成颜色
            np.random.seed(int(track_id) * 100)
            color = tuple(map(int, np.random.randint(0, 255, 3)))
            
            # 绘制停留点
            cv2.circle(stay_points_img, (x, y), radius, color, -1, cv2.LINE_AA)
            
            # 添加ID和停留时间标签
            cv2.putText(
                stay_points_img,
                f"ID: {track_id}",
                (x, y - radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            
            cv2.putText(
                stay_points_img,
                f"{duration:.1f}s",
                (x, y + radius + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        # 添加标题和信息
        cv2.putText(
            stay_points_img,
            f"用户停留点 - {Path(path_data['video_info']['file_name']).stem}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        
        cv2.putText(
            stay_points_img,
            f"停留点数量: {len(stay_points)} | 最小停留时间: {min_stay_time}秒",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
        
        # 保存图像
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, stay_points_img)
            print(f"停留点可视化已保存至: {output_path}")
            
        return stay_points_img


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='可视化用户路径数据')
    parser.add_argument('--data', type=str, required=True, help='路径数据JSON文件')
    parser.add_argument('--output-dir', type=str, default='visualization', help='输出目录')
    parser.add_argument('--background', type=str, help='背景图像路径(可选)')
    parser.add_argument('--type', type=str, default='all', 
                     choices=['path', 'heatmap', 'stay', 'all'], 
                     help='可视化类型: path-路径, heatmap-热力图, stay-停留点, all-全部')
    parser.add_argument('--min-stay-time', type=float, default=1.0, help='最小停留时间(秒)')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化可视化器
    visualizer = PathVisualizer()
    
    # 加载路径数据
    try:
        path_data = visualizer.load_path_data(args.data)
        print(f"已加载路径数据: {args.data}")
        
        # 提取文件名(不带扩展名)
        filename = Path(path_data["video_info"]["file_name"]).stem
        
        # 根据类型生成可视化
        if args.type in ['path', 'all']:
            output_path = os.path.join(args.output_dir, f"{filename}_paths.png")
            visualizer.visualize_paths(path_data, output_path, args.background)
            
        if args.type in ['heatmap', 'all']:
            output_path = os.path.join(args.output_dir, f"{filename}_heatmap.png")
            visualizer.visualize_heatmap(path_data, output_path, args.background)
            
        if args.type in ['stay', 'all']:
            output_path = os.path.join(args.output_dir, f"{filename}_stay_points.png")
            visualizer.visualize_stay_points(
                path_data, 
                output_path, 
                args.background,
                args.min_stay_time
            )
            
        print("可视化完成!")
            
    except Exception as e:
        print(f"错误: {str(e)}")


if __name__ == '__main__':
    main() 