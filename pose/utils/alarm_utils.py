import time
import threading
import os
import datetime
from typing import Optional, Callable, Dict, Any
import cv2
import numpy as np


class AlarmSystem:
    """
    报警系统：提供多种报警方法，用于跌倒检测系统
    """
    
    def __init__(self, 
                 alarm_cooldown: float = 5.0, 
                 enable_sound: bool = False,
                 save_snapshots: bool = True,
                 snapshot_dir: str = 'pose/results/snapshots'):
        """
        初始化报警系统
        
        Args:
            alarm_cooldown: 报警冷却时间（秒），防止频繁报警
            enable_sound: 是否启用声音报警
            save_snapshots: 是否保存跌倒快照图片
            snapshot_dir: 快照保存目录
        """
        self.alarm_cooldown = alarm_cooldown
        self.enable_sound = enable_sound
        self.save_snapshots = save_snapshots
        self.snapshot_dir = snapshot_dir
        
        # 上次报警时间
        self.last_alarm_time = 0
        
        # 报警状态
        self.is_alarming = False
        
        # 报警线程
        self.alarm_thread = None
        
        # 创建快照目录
        if self.save_snapshots and not os.path.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir, exist_ok=True)
    
    def trigger_alarm(self, frame: np.ndarray, details: Dict[str, Any]) -> bool:
        """
        触发报警
        
        Args:
            frame: 当前视频帧
            details: 跌倒检测详细信息
            
        Returns:
            是否成功触发报警
        """
        current_time = time.time()
        
        # 检查是否在冷却期
        if current_time - self.last_alarm_time < self.alarm_cooldown:
            return False
            
        # 更新上次报警时间
        self.last_alarm_time = current_time
        
        # 保存跌倒快照
        if self.save_snapshots:
            self._save_snapshot(frame, details)
        
        # 如果已经在报警，不再启动新的报警
        if self.is_alarming:
            return False
            
        # 启动报警线程
        self.is_alarming = True
        self.alarm_thread = threading.Thread(target=self._alarm_procedure)
        self.alarm_thread.daemon = True
        self.alarm_thread.start()
        
        return True
    
    def _alarm_procedure(self) -> None:
        """
        报警过程，在独立线程中运行
        """
        try:
            # 控制台输出警报
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] 警报！检测到跌倒！")
            
            # 如果启用了声音报警
            if self.enable_sound:
                self._sound_alarm()
            
            # 报警持续一段时间，可以根据需要调整
            time.sleep(3.0)
        finally:
            # 报警结束
            self.is_alarming = False
    
    def _sound_alarm(self) -> None:
        """
        播放声音报警
        """
        # 在不同平台使用不同的命令播放声音
        if os.name == 'nt':  # Windows
            import winsound
            # 播放系统警报声音
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            time.sleep(0.5)
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        else:  # macOS / Linux
            for _ in range(5):
                print('\a')  # 控制台响铃
                time.sleep(0.3)
    
    def _save_snapshot(self, frame: np.ndarray, details: Dict[str, Any]) -> None:
        """
        保存跌倒检测的快照
        
        Args:
            frame: 当前视频帧
            details: 跌倒检测详细信息
        """
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.snapshot_dir, f"fall_{now}.jpg")
        
        # 在图像上添加跌倒检测信息
        snapshot = frame.copy()
        
        # 添加时间戳
        cv2.putText(snapshot, 
                   f"检测到跌倒! {now}", 
                   (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, 
                   (0, 0, 255), 
                   2)
        
        # 添加详细信息
        y_offset = 70
        for key, value in details.items():
            if value is not None:
                if isinstance(value, float):
                    text = f"{key}: {value:.2f}"
                else:
                    text = f"{key}: {value}"
                cv2.putText(snapshot, 
                           text, 
                           (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, 
                           (0, 0, 255), 
                           1)
                y_offset += 30
        
        # 保存图像
        cv2.imwrite(filename, snapshot)
        print(f"跌倒快照已保存: {filename}")


class VideoWriter:
    """
    视频写入器：用于保存带有跌倒检测信息的视频
    """
    
    def __init__(self, 
                 output_path: str, 
                 fps: float, 
                 frame_size: tuple,
                 fourcc: str = 'mp4v'):
        """
        初始化视频写入器
        
        Args:
            output_path: 输出视频文件路径
            fps: 帧率
            frame_size: 帧大小 (width, height)
            fourcc: 视频编码格式
        """
        self.output_path = output_path
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
        self.writer = cv2.VideoWriter(output_path, fourcc_code, fps, frame_size)
        
        self.is_open = True
    
    def write(self, frame: np.ndarray) -> None:
        """
        写入一帧
        
        Args:
            frame: 视频帧
        """
        if self.is_open:
            self.writer.write(frame)
    
    def release(self) -> None:
        """
        释放视频写入器
        """
        if self.is_open:
            self.writer.release()
            self.is_open = False
    
    def __del__(self):
        """
        析构函数，确保资源被释放
        """
        self.release() 