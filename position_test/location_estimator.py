#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
位置估计器：用于定位用户在3D场景中的位置
"""

import numpy as np
import cv2
import open3d as o3d

class LocationEstimator:
    """位置估计器，用于定位用户在3D场景中的位置"""
    
    def __init__(self, scene_model=None, features_db=None):
        """
        初始化位置估计器
        
        Args:
            scene_model: 场景3D模型
            features_db: 特征数据库
        """
        self.scene_model = scene_model  # 场景3D模型
        self.features_db = features_db  # 特征数据库
        self.camera_matrix = None       # 相机内参矩阵
        
    def build_features_database(self, images, keypoints_list, descriptors_list, camera_poses):
        """
        构建特征数据库
        
        Args:
            images: 图像列表
            keypoints_list: 特征点列表
            descriptors_list: 描述符列表
            camera_poses: 相机姿态列表
            
        Returns:
            特征数据库
        """
        db = {
            'descriptors': descriptors_list,
            'keypoints': keypoints_list,
            'camera_poses': camera_poses,
            'images': images
        }
        self.features_db = db
        return db
    
    def calibrate_camera(self, image_size):
        """
        校准相机参数
        
        Args:
            image_size: 图像尺寸(宽, 高)
            
        Returns:
            相机内参矩阵
        """
        # 这里使用估计的相机参数，实际应用中应进行相机标定
        focal_length = 1000  # 估计的焦距
        principal_point = (image_size[0]/2, image_size[1]/2)  # 主点位于图像中心
        
        K = np.array([[focal_length, 0, principal_point[0]],
                      [0, focal_length, principal_point[1]],
                      [0, 0, 1]])
        
        self.camera_matrix = K
        return K
    
    def locate_position(self, query_image):
        """
        定位查询图像在3D场景中的位置
        
        Args:
            query_image: 查询图像
            
        Returns:
            估计的相机姿态和置信度
        """
        if self.features_db is None:
            raise ValueError("特征数据库尚未构建，请先调用build_features_database")
        
        # 提取查询图像的特征
        sift = cv2.SIFT_create()
        gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        # 初始化最佳匹配参数
        best_match_idx = -1
        max_good_matches = 0
        best_matches = None
        
        # 匹配查询图像与数据库中的所有图像
        matcher = cv2.BFMatcher()
        for i, db_descriptors in enumerate(self.features_db['descriptors']):
            if db_descriptors is None or len(db_descriptors) < 2:
                continue
                
            matches = matcher.knnMatch(descriptors, db_descriptors, k=2)
            
            # 应用Lowe's比率测试
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            
            # 更新最佳匹配
            if len(good_matches) > max_good_matches:
                max_good_matches = len(good_matches)
                best_match_idx = i
                best_matches = good_matches
        
        if best_match_idx == -1:
            print("未找到匹配的图像")
            return None, 0
        
        # 计算查询图像相对于最佳匹配图像的姿态
        db_keypoints = self.features_db['keypoints'][best_match_idx]
        
        # 提取匹配点坐标
        query_pts = np.float32([keypoints[m.queryIdx].pt for m in best_matches])
        db_pts = np.float32([db_keypoints[m.trainIdx].pt for m in best_matches])
        
        # 使用PnP算法估计相机姿态
        if self.camera_matrix is None:
            self.calibrate_camera((query_image.shape[1], query_image.shape[0]))
            
        # 这里需要知道3D点的坐标，可以通过之前的三角测量获得
        # 简化版本：使用本质矩阵估计相对姿态
        E, mask = cv2.findEssentialMat(query_pts, db_pts, self.camera_matrix, 
                                      method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        _, R, t, mask = cv2.recoverPose(E, query_pts, db_pts, self.camera_matrix)
        
        # 构建相对姿态矩阵
        relative_pose = np.eye(4)
        relative_pose[:3, :3] = R
        relative_pose[:3, 3] = t.ravel()
        
        # 计算全局姿态
        db_camera_pose = self.features_db['camera_poses'][best_match_idx]
        global_pose = db_camera_pose @ relative_pose
        
        # 计算置信度
        confidence = len(np.where(mask.ravel() == 1)[0]) / len(mask)
        
        return global_pose, confidence
    
    def visualize_position(self, query_image, estimated_pose, confidence):
        """
        可视化用户位置
        
        Args:
            query_image: 查询图像
            estimated_pose: 估计的相机姿态
            confidence: 位置估计的置信度
            
        Returns:
            可视化结果图像
        """
        # 创建结果图像
        result_img = query_image.copy()
        
        # 在图像上显示位置信息和置信度
        pos_text = f"位置: X={estimated_pose[0,3]:.2f}, Y={estimated_pose[1,3]:.2f}, Z={estimated_pose[2,3]:.2f}"
        conf_text = f"置信度: {confidence:.2f}"
        
        cv2.putText(result_img, pos_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result_img, conf_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 绘制指示方向的箭头
        height, width = result_img.shape[:2]
        center = (width // 2, height // 2)
        
        # 从姿态矩阵提取方向信息
        rotation = estimated_pose[:3, :3]
        forward = rotation @ np.array([0, 0, 1])  # 前方方向
        up = rotation @ np.array([0, 1, 0])      # 上方方向
        
        # 绘制方向箭头
        arrow_length = 50
        forward_end = (int(center[0] + forward[0] * arrow_length), 
                      int(center[1] - forward[2] * arrow_length))
        
        cv2.arrowedLine(result_img, center, forward_end, (0, 0, 255), 2)
        
        return result_img
    
    def overlay_on_3d_model(self, query_image, estimated_pose):
        """
        在3D模型上显示当前位置
        
        Args:
            query_image: 查询图像
            estimated_pose: 估计的相机姿态
            
        Returns:
            覆盖了当前位置的3D场景视图
        """
        if self.scene_model is None:
            raise ValueError("场景模型未加载")
            
        # 创建可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # 添加场景模型
        vis.add_geometry(self.scene_model)
        
        # 添加相机位置标记
        camera_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        camera_marker.paint_uniform_color([1, 0, 0])  # 红色标记
        camera_marker.transform(estimated_pose)
        vis.add_geometry(camera_marker)
        
        # 添加相机方向指示
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        camera_frame.transform(estimated_pose)
        vis.add_geometry(camera_frame)
        
        # 设置视图
        vis.get_render_option().point_size = 2.0
        vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])
        
        # 捕获视图图像
        vis.poll_events()
        vis.update_renderer()
        overlay_image = vis.capture_screen_float_buffer(True)
        
        # 关闭可视化器
        vis.destroy_window()
        
        # 转换为OpenCV格式
        overlay_image = np.asarray(overlay_image) * 255
        overlay_image = overlay_image.astype(np.uint8)
        overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)
        
        # 创建并返回组合图像（原图和3D模型视图并排）
        h, w = query_image.shape[:2]
        oh, ow = overlay_image.shape[:2]
        
        # 调整3D视图大小以匹配原图
        overlay_image = cv2.resize(overlay_image, (w, h))
        
        # 拼接图像
        combined = np.hstack((query_image, overlay_image))
        
        return combined


if __name__ == "__main__":
    print("位置定位模块 - 请从main.py运行完整流程")
    print("此模块需要场景重建结果和特征数据库支持") 