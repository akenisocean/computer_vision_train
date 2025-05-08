#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于图像的室内场景3D重建与位置定位系统
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path
import open3d as o3d

# 导入自定义模块
from image_preprocessor import ImagePreprocessor
from scene_reconstructor import SceneReconstructor
from location_estimator import LocationEstimator

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='基于图像的室内场景3D重建与位置定位系统')
    
    parser.add_argument('--mode', type=str, required=True, choices=['build', 'locate'],
                      help='运行模式: "build"构建3D场景, "locate"定位用户位置')
    parser.add_argument('--input', type=str, required=True,
                      help='输入文件夹(构建模式)或单张图片(定位模式)的路径')
    parser.add_argument('--output', type=str, default='output',
                      help='输出目录，默认为output')
    parser.add_argument('--model', type=str, default='scene_model.ply',
                      help='3D模型文件路径(定位模式需要)')
    parser.add_argument('--features', type=str, default='features_db.npz',
                      help='特征数据库文件路径(定位模式需要)')
    parser.add_argument('--visualize', action='store_true',
                      help='启用3D可视化')
    
    return parser.parse_args()

def build_3d_scene(args):
    """
    构建3D场景
    
    Args:
        args: 命令行参数
    """
    print("\n===== 开始3D场景重建 =====")
    
    # 初始化图像预处理器
    preprocessor = ImagePreprocessor(args.input, os.path.join(args.output, 'processed'))
    
    # 加载和处理图像
    print("\n1. 加载和预处理图像...")
    images, filenames = preprocessor.load_images()
    if len(images) < 2:
        print(f"错误: 至少需要2张图像进行3D重建，当前只有{len(images)}张")
        return
        
    enhanced_images = preprocessor.enhance_images(images)
    keypoints_list, descriptors_list = preprocessor.detect_features(enhanced_images)
    preprocessor.save_processed_images(enhanced_images, filenames)
    
    # 初始化场景重建器
    reconstructor = SceneReconstructor()
    
    # 估计相机姿态
    print("\n2. 估计相机姿态...")
    camera_poses = reconstructor.estimate_camera_poses(keypoints_list, descriptors_list)
    
    # 生成匹配列表
    print("\n3. 特征匹配...")
    matcher = cv2.BFMatcher()
    matches_list = []
    for i in range(len(descriptors_list) - 1):
        matches = matcher.knnMatch(descriptors_list[i], descriptors_list[i+1], k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        matches_list.append(good_matches)
        print(f"  图像对 {i+1}-{i+2}: 找到 {len(good_matches)} 个有效匹配")
    
    # 三角测量生成点云
    print("\n4. 三角测量生成点云...")
    pcd = reconstructor.triangulate_points(keypoints_list, matches_list)
    print(f"  生成点云包含 {len(pcd.points)} 个点")
    
    # 生成3D网格
    print("\n5. 生成3D网格...")
    mesh = reconstructor.generate_mesh(pcd)
    print(f"  生成网格包含 {len(mesh.triangles)} 个三角面")
    
    # 保存重建结果
    model_file = os.path.join(args.output, 'scene_model.ply')
    mesh_file = os.path.join(args.output, 'scene_mesh.obj')
    reconstructor.save_reconstruction(model_file, mesh_file)
    
    # 构建特征数据库
    print("\n6. 构建特征数据库...")
    estimator = LocationEstimator(pcd)
    features_db = estimator.build_features_database(
        enhanced_images, keypoints_list, descriptors_list, camera_poses)
    
    # 保存特征数据库
    features_file = os.path.join(args.output, 'features_db.npz')
    
    # 转换关键点为可序列化格式
    serializable_keypoints = []
    for kp_list in keypoints_list:
        serializable_kp = []
        for kp in kp_list:
            serializable_kp.append((kp.pt[0], kp.pt[1]))
        serializable_keypoints.append(serializable_kp)
    
    np.savez(features_file, 
             descriptors=descriptors_list,
             keypoints=serializable_keypoints,
             camera_poses=camera_poses)
    
    print(f"\n===== 3D场景重建完成！=====")
    print(f"模型已保存为:\n  - {model_file}\n  - {mesh_file}")
    print(f"特征数据库已保存为:\n  - {features_file}")
    
    # 可视化重建结果
    if args.visualize:
        print("\n正在可视化3D重建结果...")
        reconstructor.visualize_reconstruction()

def locate_position(args):
    """
    定位用户在3D场景中的位置
    
    Args:
        args: 命令行参数
    """
    print("\n===== 开始位置定位 =====")
    
    # 检查输入文件是否存在
    if not os.path.isfile(args.input):
        print(f"错误: 输入图片 {args.input} 不存在")
        return
        
    if not os.path.isfile(args.model):
        print(f"错误: 3D模型文件 {args.model} 不存在")
        return
        
    if not os.path.isfile(args.features):
        print(f"错误: 特征数据库文件 {args.features} 不存在")
        return
    
    # 加载3D模型
    print(f"\n1. 加载3D模型: {args.model}")
    scene_model = o3d.io.read_point_cloud(args.model)
    
    # 加载特征数据库
    print(f"\n2. 加载特征数据库: {args.features}")
    features_data = np.load(args.features, allow_pickle=True)
    
    # 重建特征数据库
    descriptors_list = features_data['descriptors']
    serialized_keypoints = features_data['keypoints']
    camera_poses = features_data['camera_poses']
    
    # 重构特征点列表
    keypoints_list = []
    for i in range(len(serialized_keypoints)):
        kps = []
        for pt in serialized_keypoints[i]:
            kp = cv2.KeyPoint(float(pt[0]), float(pt[1]), 1)
            kps.append(kp)
        keypoints_list.append(kps)
    
    # 创建位置估计器
    estimator = LocationEstimator(scene_model)
    estimator.features_db = {
        'descriptors': descriptors_list,
        'keypoints': keypoints_list,
        'camera_poses': camera_poses
    }
    
    # 加载查询图像
    print(f"\n3. 加载查询图像: {args.input}")
    query_image = cv2.imread(args.input)
    if query_image is None:
        print(f"错误: 无法读取图片 {args.input}")
        return
        
    # 校准相机
    estimator.calibrate_camera((query_image.shape[1], query_image.shape[0]))
    
    # 估计位置
    print("\n4. 正在定位...")
    estimated_pose, confidence = estimator.locate_position(query_image)
    
    if estimated_pose is None:
        print("位置定位失败: 未找到匹配图像")
        return
        
    # 可视化结果
    print("\n5. 正在可视化位置...")
    result_image = estimator.visualize_position(query_image, estimated_pose, confidence)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 保存结果
    result_file = os.path.join(args.output, 'location_result.jpg')
    cv2.imwrite(result_file, result_image)
    
    # 在3D模型上叠加显示位置
    if args.visualize:
        print("\n6. 在3D模型上显示位置...")
        combined_image = estimator.overlay_on_3d_model(query_image, estimated_pose)
        combined_file = os.path.join(args.output, 'location_3d_overlay.jpg')
        cv2.imwrite(combined_file, combined_image)
        print(f"3D叠加视图已保存为: {combined_file}")
    
    print(f"\n===== 位置定位完成 =====")
    print(f"位置坐标: X={estimated_pose[0,3]:.2f}, Y={estimated_pose[1,3]:.2f}, Z={estimated_pose[2,3]:.2f}")
    print(f"置信度: {confidence:.2f}")
    print(f"可视化结果已保存为: {result_file}")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 根据运行模式选择功能
    if args.mode == 'build':
        build_3d_scene(args)
    elif args.mode == 'locate':
        locate_position(args)
    else:
        print(f"错误: 未知的运行模式 {args.mode}")

if __name__ == "__main__":
    main() 