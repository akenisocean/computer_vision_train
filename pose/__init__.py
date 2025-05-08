"""
基于YOLOv8-Pose的跌倒检测与报警系统

此包含提供用于检测人体跌倒并发出警报的功能。
"""

# 版本信息
__version__ = '0.1.0'

# 导出主要模块
from .fall_detection import process_video

__all__ = ['process_video'] 