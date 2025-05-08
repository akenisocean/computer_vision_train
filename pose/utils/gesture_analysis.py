import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Any


class GestureAnalyzer:
    """
    手势分析器：用于检测握拳动作的工具类
    """
    
    def __init__(self, 
                 history_size: int = 10,
                 distance_threshold: float = 0.15,
                 angle_threshold: float = 60.0,
                 velocity_threshold: float = 10.0,
                 cooldown_frames: int = 15):
        """
        初始化手势分析器
        
        Args:
            history_size: 历史帧数据存储大小，用于避免重复计数
            distance_threshold: 拳头闭合距离阈值，手腕到手指距离比例
            angle_threshold: 手腕-肘部-肩部角度阈值
            velocity_threshold: 手腕移动速度阈值
            cooldown_frames: 连续检测间的冷却帧数，防止重复计数
        """
        self.history_size = history_size
        self.distance_threshold = distance_threshold
        self.angle_threshold = angle_threshold
        self.velocity_threshold = velocity_threshold
        self.cooldown_frames = cooldown_frames
        
        # 历史关键点位置
        self.keypoints_history = []
        
        # 握拳计数器
        self.fist_count = 0
        
        # 冷却计数器
        self.cooldown_counter = 0
        
        # 是否为握拳状态
        self.is_fist_state = False
        
        # 握拳检测连续帧计数
        self.continuous_fist_frames = 0
        self.continuous_nonfist_frames = 0
        
        # 握拳确认阈值（连续几帧）
        self.fist_confirmation_threshold = 3
        self.nonfist_confirmation_threshold = 5
        
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
        self.keypoints_history.append(keypoints.copy())
        
        # 保持历史记录在指定大小范围内
        if len(self.keypoints_history) > self.history_size:
            self.keypoints_history.pop(0)
    
    def calculate_arm_angle(self, keypoints: np.ndarray, is_left: bool) -> Optional[float]:
        """
        计算手臂弯曲角度
        
        Args:
            keypoints: 形状为 [17, 3] 的数组，表示当前帧的关键点
            is_left: 是否是左手臂
            
        Returns:
            手臂弯曲角度（度数）或 None（如果关键点不可用）
        """
        if is_left:
            shoulder_idx = self.LEFT_SHOULDER
            elbow_idx = self.LEFT_ELBOW
            wrist_idx = self.LEFT_WRIST
        else:
            shoulder_idx = self.RIGHT_SHOULDER
            elbow_idx = self.RIGHT_ELBOW
            wrist_idx = self.RIGHT_WRIST
        
        # 检查关键点是否可用
        if (keypoints[shoulder_idx][2] < 0.5 or 
            keypoints[elbow_idx][2] < 0.5 or 
            keypoints[wrist_idx][2] < 0.5):
            return None
        
        # 获取关键点坐标
        shoulder = keypoints[shoulder_idx][:2]
        elbow = keypoints[elbow_idx][:2]
        wrist = keypoints[wrist_idx][:2]
        
        # 计算向量
        vec1 = shoulder - elbow
        vec2 = wrist - elbow
        
        # 计算角度
        cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        # 防止数值误差导致的计算问题
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        
        return angle
    
    def calculate_wrist_velocity(self, is_left: bool) -> Optional[float]:
        """
        计算手腕移动速度
        
        Args:
            is_left: 是否是左手
            
        Returns:
            手腕移动速度或None（如果历史数据不足）
        """
        if len(self.keypoints_history) < 2:
            return None
            
        wrist_idx = self.LEFT_WRIST if is_left else self.RIGHT_WRIST
        
        # 获取当前帧和上一帧的手腕位置
        current = self.keypoints_history[-1][wrist_idx]
        previous = self.keypoints_history[-2][wrist_idx]
        
        # 检查关键点是否可用
        if current[2] < 0.5 or previous[2] < 0.5:
            return None
            
        # 计算速度（欧几里得距离）
        velocity = np.linalg.norm(current[:2] - previous[:2])
        
        return velocity
    
    def detect_fist(self, keypoints: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """
        检测握拳动作
        
        Args:
            keypoints: 形状为 [17, 3] 的数组，表示当前帧的关键点
            
        Returns:
            (is_fist, details): 是否握拳的布尔值和详细信息字典
        """
        # 更新关键点历史
        self.update_keypoints_history(keypoints)
        
        # 初始化握拳状态和详细信息
        details = {
            "left_arm_angle": None,
            "right_arm_angle": None,
            "left_distance_ratio": None,
            "right_distance_ratio": None,
            "left_velocity": None,
            "right_velocity": None,
            "fist_score": 0.0,
            "fist_count": self.fist_count
        }
        
        # 如果在冷却期内，减少冷却计数器
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return self.is_fist_state, details
        
        # 计算特征并收集到详细信息中
        
        # 1. 计算手臂角度
        left_arm_angle = self.calculate_arm_angle(keypoints, True)
        right_arm_angle = self.calculate_arm_angle(keypoints, False)
        
        details["left_arm_angle"] = left_arm_angle
        details["right_arm_angle"] = right_arm_angle
        
        # 2. 计算手腕与肘部的距离比例
        left_distance_ratio = self._calculate_wrist_movement_ratio(keypoints, True)
        right_distance_ratio = self._calculate_wrist_movement_ratio(keypoints, False)
        
        details["left_distance_ratio"] = left_distance_ratio
        details["right_distance_ratio"] = right_distance_ratio
        
        # 3. 计算手腕移动速度
        left_velocity = self.calculate_wrist_velocity(True)
        right_velocity = self.calculate_wrist_velocity(False)
        
        details["left_velocity"] = left_velocity
        details["right_velocity"] = right_velocity
        
        # 计算握拳得分
        fist_score = 0.0
        score_count = 0
        
        # 评分规则1: 基于手臂角度 (握拳时手臂角度通常小于阈值)
        if left_arm_angle is not None:
            # 小于阈值时，角度越小得分越高
            if left_arm_angle < self.angle_threshold:
                arm_angle_score = 1.0 - (left_arm_angle / self.angle_threshold)
                fist_score += arm_angle_score
                score_count += 1
        
        if right_arm_angle is not None:
            if right_arm_angle < self.angle_threshold:
                arm_angle_score = 1.0 - (right_arm_angle / self.angle_threshold)
                fist_score += arm_angle_score
                score_count += 1
        
        # 评分规则2: 基于距离比例 (握拳时前臂比例通常减小)
        if left_distance_ratio is not None:
            if left_distance_ratio < self.distance_threshold:
                distance_score = 1.0 - (left_distance_ratio / self.distance_threshold)
                fist_score += distance_score * 1.5  # 赋予更高权重
                score_count += 1.5
        
        if right_distance_ratio is not None:
            if right_distance_ratio < self.distance_threshold:
                distance_score = 1.0 - (right_distance_ratio / self.distance_threshold)
                fist_score += distance_score * 1.5  # 赋予更高权重
                score_count += 1.5
        
        # 评分规则3: 基于手腕速度 (握拳前通常有明显运动)
        if left_velocity is not None and left_velocity > self.velocity_threshold:
            velocity_score = min(left_velocity / (self.velocity_threshold * 2), 1.0)
            fist_score += velocity_score * 0.5  # 赋予较低权重
            score_count += 0.5
        
        if right_velocity is not None and right_velocity > self.velocity_threshold:
            velocity_score = min(right_velocity / (self.velocity_threshold * 2), 1.0)
            fist_score += velocity_score * 0.5  # 赋予较低权重
            score_count += 0.5
        
        # 归一化得分
        if score_count > 0:
            fist_score /= score_count
        
        details["fist_score"] = fist_score
        
        # 确定握拳状态
        # 使用状态机和连续帧确认，减少误检
        fist_detected_in_current_frame = fist_score > 0.6
        
        if fist_detected_in_current_frame:
            self.continuous_fist_frames += 1
            self.continuous_nonfist_frames = 0
        else:
            self.continuous_nonfist_frames += 1
            self.continuous_fist_frames = 0
        
        # 状态转换逻辑
        if not self.is_fist_state and self.continuous_fist_frames >= self.fist_confirmation_threshold:
            # 从非握拳转为握拳状态
            self.is_fist_state = True
            self.fist_count += 1
            details["fist_count"] = self.fist_count
            # 设置冷却计数器
            self.cooldown_counter = self.cooldown_frames
        elif self.is_fist_state and self.continuous_nonfist_frames >= self.nonfist_confirmation_threshold:
            # 从握拳转为非握拳状态
            self.is_fist_state = False
        
        return self.is_fist_state, details
    
    def _calculate_wrist_movement_ratio(self, keypoints: np.ndarray, is_left: bool) -> Optional[float]:
        """
        计算手腕移动比例，用于估计握拳状态
        
        Args:
            keypoints: 关键点数组
            is_left: 是否是左手
            
        Returns:
            手腕移动比例或None（如果关键点不可用）
        """
        if is_left:
            shoulder_idx = self.LEFT_SHOULDER
            elbow_idx = self.LEFT_ELBOW
            wrist_idx = self.LEFT_WRIST
        else:
            shoulder_idx = self.RIGHT_SHOULDER
            elbow_idx = self.RIGHT_ELBOW
            wrist_idx = self.RIGHT_WRIST
        
        # 检查关键点是否可用
        if (keypoints[shoulder_idx][2] < 0.5 or 
            keypoints[elbow_idx][2] < 0.5 or 
            keypoints[wrist_idx][2] < 0.5):
            return None
        
        # 获取关键点坐标
        shoulder = keypoints[shoulder_idx][:2]
        elbow = keypoints[elbow_idx][:2]
        wrist = keypoints[wrist_idx][:2]
        
        # 计算手臂长度（肩膀到肘部）
        arm_length = np.linalg.norm(shoulder - elbow)
        
        # 计算前臂长度（肘部到手腕）
        forearm_length = np.linalg.norm(elbow - wrist)
        
        # 计算手腕到前臂的距离比例
        # 当握拳时，这个比例通常会减小
        distance_ratio = forearm_length / (arm_length + 1e-6)  # 防止除零
        
        return distance_ratio
    
    def reset_count(self) -> None:
        """
        重置握拳计数器
        """
        self.fist_count = 0
        self.continuous_fist_frames = 0
        self.continuous_nonfist_frames = 0
        self.is_fist_state = False
        self.cooldown_counter = 0 