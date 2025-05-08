import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Any


class PoseAnalyzer:
    """
    姿态分析器：用于检测人体跌倒的工具类
    """
    
    def __init__(self, 
                 fall_threshold: float = 0.6, 
                 history_size: int = 10, 
                 angle_threshold: float = 45.0,
                 height_ratio_threshold: float = 0.6):
        """
        初始化姿态分析器
        
        Args:
            fall_threshold: 跌倒检测的综合阈值 (0-1)，值越小越敏感
            history_size: 历史帧数据存储大小，用于运动分析
            angle_threshold: 躯干与地面夹角阈值，小于该角度视为可能跌倒
            height_ratio_threshold: 身体高度比例变化阈值，小于该值视为可能跌倒
        """
        self.fall_threshold = fall_threshold
        self.history_size = history_size
        self.angle_threshold = angle_threshold
        self.height_ratio_threshold = height_ratio_threshold
        
        # 历史关键点位置，用于计算运动速度
        self.keypoints_history = []
        
        # 关键点索引定义 (基于YOLOv8-Pose的17个关键点)
        self.NOSE = 0
        self.LEFT_EYE = 1
        self.RIGHT_EYE = 2
        self.LEFT_EAR = 3
        self.RIGHT_EAR = 4
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.LEFT_ELBOW = 7
        self.RIGHT_ELBOW = 8
        self.LEFT_WRIST = 9
        self.RIGHT_WRIST = 10
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 12
        self.LEFT_KNEE = 13
        self.RIGHT_KNEE = 14
        self.LEFT_ANKLE = 15
        self.RIGHT_ANKLE = 16
    
    def update_keypoints_history(self, keypoints: np.ndarray) -> None:
        """
        更新关键点历史记录
        
        Args:
            keypoints: 形状为 [17, 3] 的数组，表示当前帧的关键点
                       每个关键点包含 [x, y, confidence]
        """
        self.keypoints_history.append(keypoints)
        
        # 保持历史记录在指定大小范围内
        if len(self.keypoints_history) > self.history_size:
            self.keypoints_history.pop(0)
    
    def calculate_body_angle(self, keypoints: np.ndarray) -> float:
        """
        计算躯干与地面的夹角
        
        Args:
            keypoints: 形状为 [17, 3] 的数组，表示当前帧的关键点
                       
        Returns:
            躯干与地面的夹角（度数）
        """
        # 使用肩部和髋部中点计算躯干角度
        shoulders_midpoint = self._get_midpoint(keypoints, self.LEFT_SHOULDER, self.RIGHT_SHOULDER)
        hips_midpoint = self._get_midpoint(keypoints, self.LEFT_HIP, self.RIGHT_HIP)
        
        # 如果关键点缺失，返回90度（默认直立）
        if shoulders_midpoint is None or hips_midpoint is None:
            return 90.0
        
        # 计算躯干与垂直线的夹角
        dx = shoulders_midpoint[0] - hips_midpoint[0]
        dy = shoulders_midpoint[1] - hips_midpoint[1]
        
        # 防止除以零的情况
        if dx == 0:
            return 90.0
            
        # 计算角度 (弧度)
        angle_rad = math.atan2(abs(dx), abs(dy))
        # 转换为角度
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg
    
    def calculate_height_ratio_change(self) -> Optional[float]:
        """
        计算人体高度比例变化
        
        Returns:
            身体高度比例的变化率，如果历史数据不足则返回None
        """
        if len(self.keypoints_history) < 2:
            return None
            
        # 获取当前帧和历史帧的高度
        current_height = self._get_body_height(self.keypoints_history[-1])
        prev_height = self._get_body_height(self.keypoints_history[-2])
        
        # 如果任一高度计算失败，返回None
        if current_height is None or prev_height is None or prev_height == 0:
            return None
            
        # 计算高度比例变化
        height_ratio = current_height / prev_height
        
        return height_ratio
    
    def calculate_vertical_velocity(self) -> Optional[float]:
        """
        计算头部的垂直运动速度
        
        Returns:
            头部的垂直速度（像素/帧），如果历史数据不足则返回None
        """
        if len(self.keypoints_history) < 2:
            return None
            
        # 获取当前帧和上一帧的头部位置
        current_head_y = self._get_head_position(self.keypoints_history[-1])
        prev_head_y = self._get_head_position(self.keypoints_history[-2])
        
        # 如果任一头部位置计算失败，返回None
        if current_head_y is None or prev_head_y is None:
            return None
            
        # 计算垂直速度 (正值表示向下运动)
        vertical_velocity = current_head_y - prev_head_y
        
        return vertical_velocity
    
    def detect_fall(self, keypoints: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """
        检测是否发生跌倒
        
        Args:
            keypoints: 形状为 [17, 3] 的数组，表示当前帧的关键点
                       
        Returns:
            (is_falling, details): 是否跌倒的布尔值和详细信息字典
        """
        # 更新关键点历史
        self.update_keypoints_history(keypoints)
        
        # 初始化跌倒分数和详细信息
        fall_score = 0.0
        details = {
            "angle": None,
            "height_ratio": None,
            "vertical_velocity": None,
            "fall_score": 0.0
        }
        
        # 1. 分析躯干角度
        body_angle = self.calculate_body_angle(keypoints)
        details["angle"] = body_angle
        
        if body_angle < self.angle_threshold:
            # 角度越小，跌倒可能性越大
            angle_score = 1.0 - (body_angle / self.angle_threshold)
            fall_score += angle_score * 0.5  # 角度占总分的50%
        
        # 2. 分析高度比例变化
        height_ratio = self.calculate_height_ratio_change()
        details["height_ratio"] = height_ratio
        
        if height_ratio is not None and height_ratio < self.height_ratio_threshold:
            # 高度比例越小，跌倒可能性越大
            height_score = 1.0 - (height_ratio / self.height_ratio_threshold)
            fall_score += height_score * 0.3  # 高度比例占总分的30%
        
        # 3. 分析垂直运动速度
        vertical_velocity = self.calculate_vertical_velocity()
        details["vertical_velocity"] = vertical_velocity
        
        if vertical_velocity is not None and vertical_velocity > 5.0:  # 阈值可调
            # 垂直速度越大，跌倒可能性越大
            velocity_score = min(vertical_velocity / 20.0, 1.0)  # 归一化到0-1范围
            fall_score += velocity_score * 0.2  # 垂直速度占总分的20%
        
        details["fall_score"] = fall_score
        
        # 判断是否跌倒
        is_falling = fall_score > self.fall_threshold
        
        return is_falling, details
    
    def _get_midpoint(self, keypoints: np.ndarray, idx1: int, idx2: int) -> Optional[Tuple[float, float]]:
        """
        计算两个关键点的中点
        
        Args:
            keypoints: 关键点数组
            idx1: 第一个关键点索引
            idx2: 第二个关键点索引
            
        Returns:
            中点坐标 (x, y) 或 None（如果关键点不可用）
        """
        # 检查关键点置信度
        if (keypoints[idx1][2] < 0.5) or (keypoints[idx2][2] < 0.5):
            return None
            
        x1, y1 = keypoints[idx1][0], keypoints[idx1][1]
        x2, y2 = keypoints[idx2][0], keypoints[idx2][1]
        
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _get_body_height(self, keypoints: np.ndarray) -> Optional[float]:
        """
        计算人体高度（从头部到脚踝）
        
        Args:
            keypoints: 关键点数组
            
        Returns:
            人体高度或None（如果关键点不可用）
        """
        # 使用鼻子作为头部位置
        nose = keypoints[self.NOSE]
        
        # 使用脚踝平均位置作为底部
        left_ankle = keypoints[self.LEFT_ANKLE]
        right_ankle = keypoints[self.RIGHT_ANKLE]
        
        # 检查关键点可用性
        if nose[2] < 0.5:
            return None
            
        # 如果两个脚踝都不可用，尝试使用膝盖
        if left_ankle[2] < 0.5 and right_ankle[2] < 0.5:
            left_knee = keypoints[self.LEFT_KNEE]
            right_knee = keypoints[self.RIGHT_KNEE]
            
            # 如果膝盖也不可用，返回None
            if left_knee[2] < 0.5 and right_knee[2] < 0.5:
                return None
                
            # 使用可用的膝盖
            if left_knee[2] >= 0.5:
                bottom_y = left_knee[1]
            else:
                bottom_y = right_knee[1]
        else:
            # 使用可用的脚踝
            if left_ankle[2] >= 0.5:
                bottom_y = left_ankle[1]
            else:
                bottom_y = right_ankle[1]
        
        # 计算高度 (y值在图像中通常是从上到下递增)
        height = bottom_y - nose[1]
        
        return max(height, 1.0)  # 确保高度为正
    
    def _get_head_position(self, keypoints: np.ndarray) -> Optional[float]:
        """
        获取头部位置的y坐标
        
        Args:
            keypoints: 关键点数组
            
        Returns:
            头部的y坐标或None（如果关键点不可用）
        """
        # 使用鼻子作为头部位置
        nose = keypoints[self.NOSE]
        
        if nose[2] < 0.5:
            # 如果鼻子不可用，尝试使用眼睛的平均位置
            left_eye = keypoints[self.LEFT_EYE]
            right_eye = keypoints[self.RIGHT_EYE]
            
            if left_eye[2] >= 0.5 and right_eye[2] >= 0.5:
                return (left_eye[1] + right_eye[1]) / 2
            elif left_eye[2] >= 0.5:
                return left_eye[1]
            elif right_eye[2] >= 0.5:
                return right_eye[1]
            else:
                return None
        
        return nose[1] 