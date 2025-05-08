#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像预处理器：用于采集和预处理输入图像
"""

import os
import cv2
import numpy as np
from pathlib import Path

class ImagePreprocessor:
    """图像预处理器，用于采集和预处理输入图像"""
    
    def __init__(self, input_dir="input_images", output_dir="processed_images"):
        """
        初始化预处理器
        
        Args:
            input_dir: 输入图像目录
            output_dir: 处理后图像保存目录
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def load_images(self, file_pattern="*.jpg"):
        """
        加载指定目录下的所有图像
        
        Args:
            file_pattern: 文件匹配模式
            
        Returns:
            图像列表及其文件名
        """
        image_paths = list(Path(self.input_dir).glob(file_pattern))
        images = []
        filenames = []
        
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
                filenames.append(img_path.name)
                
        print(f"成功加载{len(images)}张图像")
        return images, filenames
    
    def enhance_images(self, images):
        """
        增强图像质量
        
        Args:
            images: 输入图像列表
            
        Returns:
            增强后的图像列表
        """
        enhanced_images = []
        
        for img in images:
            # 调整亮度和对比度
            alpha = 1.2  # 对比度因子
            beta = 10    # 亮度增益
            enhanced = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            
            # 应用锐化滤镜
            kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            enhanced_images.append(enhanced)
            
        return enhanced_images
    
    def detect_features(self, images):
        """
        检测图像中的特征点
        
        Args:
            images: 输入图像列表
            
        Returns:
            特征点和描述符列表
        """
        sift = cv2.SIFT_create()
        keypoints_list = []
        descriptors_list = []
        
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            keypoints_list.append(keypoints)
            descriptors_list.append(descriptors)
            
        return keypoints_list, descriptors_list
    
    def save_processed_images(self, images, filenames):
        """
        保存处理后的图像
        
        Args:
            images: 处理后的图像列表
            filenames: 文件名列表
        """
        for img, filename in zip(images, filenames):
            output_path = os.path.join(self.output_dir, f"processed_{filename}")
            cv2.imwrite(output_path, img)
            
        print(f"已保存{len(images)}张处理后的图像到{self.output_dir}")


if __name__ == "__main__":
    # 简单测试代码
    preprocessor = ImagePreprocessor("test_images", "test_processed")
    
    # 检查测试目录是否存在，不存在则创建
    if not os.path.exists("test_images"):
        os.makedirs("test_images")
        print("创建了测试目录，请添加一些测试图像后再运行")
    else:
        # 加载图像
        images, filenames = preprocessor.load_images()
        
        if len(images) > 0:
            # 增强图像
            enhanced = preprocessor.enhance_images(images)
            
            # 检测特征
            keypoints, descriptors = preprocessor.detect_features(enhanced)
            
            # 保存处理后的图像
            preprocessor.save_processed_images(enhanced, filenames)
            
            print(f"处理完成，检测到的特征点数量:")
            for i, kps in enumerate(keypoints):
                print(f"  图像 {filenames[i]}: {len(kps)} 个特征点") 