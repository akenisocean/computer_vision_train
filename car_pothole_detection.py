import cv2
import numpy as np
from ultralytics import YOLO
from typing import Optional, Tuple, List, Dict
import os

class CarPotholeDetector:
    """汽车、摩托车和坑洼检测器类"""
    
    def __init__(self, model_path: str = "car_pothole_yolo11-seg.pt", conf_threshold: float = 0.10):
        """
        初始化检测器
        
        Args:
            model_path: YOLO模型路径
            conf_threshold: 检测置信度阈值
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
        print(f"正在加载模型: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        print(f"模型加载完成，检测阈值设置为: {conf_threshold}")
        
    def detect_and_show(self, image_path: str, save_path: Optional[str] = None) -> np.ndarray:
        """
        检测图片中的汽车、摩托车和坑洼并保存结果
        
        Args:
            image_path: 输入图片路径
            save_path: 结果保存路径（可选）
            
        Returns:
            处理后的图片数组
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"输入图片不存在: {image_path}")
            
        print(f"正在处理图片: {image_path}")
        # 读取图片
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图片: {image_path}")
        except Exception as e:
            print(f"图片读取失败: {str(e)}")
            return None
        
        print(f"图片尺寸: {image.shape}")
        # 执行检测
        try:
            results = self.model(image, conf=self.conf_threshold)[0]
        except Exception as e:
            print(f"模型检测失败: {str(e)}")
            return None
        
        # 统计各类别数量
        class_counts = {}
        for box in results.boxes:
            class_name = results.names[int(box.cls[0])]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # 打印检测结果统计
        print("\n检测结果统计:")
        for class_name, count in class_counts.items():
            print(f"{class_name}: {count}个")
        
        # 在图片上绘制检测结果
        try:
            annotated_image = results.plot()
            print("检测结果绘制成功")
        except Exception as e:
            print(f"绘制检测结果失败: {str(e)}")
            return None
            
        # 保存结果
        try:
            if save_path:
                cv2.imwrite(save_path, annotated_image)
                print(f"结果已保存至: {save_path}")
            else:
                # 如果没有指定保存路径，使用默认路径
                default_save_path = "detection_result.jpg"
                cv2.imwrite(default_save_path, annotated_image)
                print(f"结果已保存至: {default_save_path}")
        except Exception as e:
            print(f"图片保存失败: {str(e)}")
            return None
            
        return annotated_image
    
    def get_detection_info(self, image_path: str) -> List[dict]:
        """
        获取检测详细信息
        
        Args:
            image_path: 输入图片路径
            
        Returns:
            包含检测信息的列表
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"输入图片不存在: {image_path}")
            
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
            
        results = self.model(image, conf=self.conf_threshold)[0]
        
        detection_info = []
        for box in results.boxes:
            info = {
                "class": results.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()
            }
            detection_info.append(info)
            
        return detection_info

def main():
    """主函数"""
    try:
        # 创建检测器实例
        detector = CarPotholeDetector()
        
        # 示例使用
        image_path = "车辆碰撞.jpg"  # 替换为实际的图片路径
        if os.path.exists(image_path):
            print(f"找到输入图片: {image_path}")
            # 执行检测并保存结果
            detector.detect_and_show(image_path, save_path="result.jpg")
            
            # 获取检测信息
            detection_info = detector.get_detection_info(image_path)
            print("\n检测结果详情:")
            for i, info in enumerate(detection_info, 1):
                print(f"\n目标 {i}:")
                print(f"类别: {info['class']}")
                print(f"置信度: {info['confidence']:.2f}")
                print(f"边界框坐标: {info['bbox']}")
        else:
            print(f"错误：找不到输入图片 {image_path}")
            
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main() 