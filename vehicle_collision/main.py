#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
车辆碰撞检测系统主程序
实现从数据标注到模型训练到最后模型应用的完整流程
"""

import os
import argparse
import subprocess
import sys
import time
from pathlib import Path
import yaml

# 添加当前目录到搜索路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入工具模块
try:
    from utils.data_utils import video_to_frames, convert_labelstudio_to_yolo, split_dataset, create_yolo_config
    from utils.model_utils import load_model, model_info, create_model_metadata
    from utils.visualization import draw_bounding_boxes, plot_collision_events, create_annotated_video
    from scripts.inference import process_video, process_image, process_webcam
except ImportError as e:
    print(f"导入模块时出错: {e}")
    print("请确保已安装所有必要的依赖，并且当前目录是项目根目录")
    sys.exit(1)


def check_requirements():
    """检查环境要求"""
    try:
        import torch
        import ultralytics
        import opencv_python
        
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 设备数量: {torch.cuda.device_count()}")
            print(f"CUDA 设备名称: {torch.cuda.get_device_name(0)}")
        
        print(f"Ultralytics 版本: {ultralytics.__version__}")
        
        return True
    except ImportError as e:
        print(f"环境检查失败: {e}")
        print("请安装所需依赖: pip install -r requirements.txt")
        return False


def annotation_mode(args):
    """数据标注模式"""
    print("=== 数据标注模式 ===")
    
    # 检查输入目录
    if not os.path.exists(args.input):
        print(f"错误: 输入目录不存在 {args.input}")
        return False
    
    # 判断输入是视频还是图像目录
    if os.path.isfile(args.input) and args.input.lower().endswith(('.mp4', '.avi', '.mov')):
        # 如果是视频，先提取帧
        print(f"检测到视频文件，正在提取帧...")
        frames_dir = os.path.join(args.output, 'images')
        os.makedirs(frames_dir, exist_ok=True)
        
        frames = video_to_frames(args.input, frames_dir, frame_rate=args.frame_rate)
        print(f"已提取 {len(frames)} 帧到 {frames_dir}")
        
        # 设置Label Studio的数据路径
        data_path = frames_dir
    else:
        # 如果是图像目录，直接使用
        data_path = args.input
    
    # 启动Label Studio
    print("\n要启动标注界面，请在新的终端窗口运行以下命令:")
    print("--------------------------------------------------")
    print(f"cd {os.path.abspath(os.path.dirname(__file__))}")
    print("LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true label-studio start")
    print("--------------------------------------------------")
    print("然后在浏览器中访问: http://localhost:8080")
    print("\n1. 创建新项目")
    print("2. 选择'目标检测'任务")
    print("3. 添加标签: car, truck, bus, motorcycle, collision")
    print("4. 在'添加数据来源'中选择'本地文件'，并指向:")
    print(f"   {os.path.abspath(data_path)}")
    print("5. 标注完成后，从项目菜单中导出为'JSON'格式\n")
    
    if args.auto_start and not args.no_browser:
        try:
            # 尝试自动启动Label Studio
            env = os.environ.copy()
            env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
            
            print("正在启动Label Studio...")
            process = subprocess.Popen(
                ["label-studio", "start"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # 等待启动
            time.sleep(5)
            if process.poll() is None:
                print("Label Studio已启动，请在浏览器中访问: http://localhost:8080")
            else:
                stdout, stderr = process.communicate()
                print(f"启动Label Studio失败: {stderr.decode()}")
        
        except Exception as e:
            print(f"自动启动Label Studio失败: {str(e)}")
            print("请手动启动Label Studio")
    
    return True


def prepare_dataset(args):
    """准备训练数据集"""
    print("=== 准备训练数据集 ===")
    
    # 检查JSON标注文件
    if not args.annotation_file:
        print("错误: 需要提供标注JSON文件")
        return False
    
    if not os.path.exists(args.annotation_file):
        print(f"错误: 标注文件不存在 {args.annotation_file}")
        return False
    
    # 创建输出目录
    data_dir = os.path.join(args.output, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # 转换标注为YOLO格式
    print("正在转换标注为YOLO格式...")
    
    # 确定图像目录
    if args.images_dir:
        images_dir = args.images_dir
    else:
        # 尝试查找已提取的帧
        images_dir = os.path.join(args.output, 'images')
        if not os.path.exists(images_dir):
            images_dir = args.input
    
    # 转换标注
    labels_dir = os.path.join(data_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)
    
    # 类别映射
    class_map = {
        "car": 0,
        "truck": 1,
        "bus": 2,
        "motorcycle": 3,
        "collision": 4
    }
    
    # 运行转换
    stats = convert_labelstudio_to_yolo(args.annotation_file, labels_dir, images_dir, class_map)
    
    # 将图像复制到数据目录（如果不在同一位置）
    if images_dir != os.path.join(data_dir, 'images'):
        images_output_dir = os.path.join(data_dir, 'images')
        os.makedirs(images_output_dir, exist_ok=True)
        
        # 为每个标注文件复制对应的图像
        for label_file in os.listdir(labels_dir):
            base_name = os.path.splitext(label_file)[0]
            
            # 查找对应的图像（支持多种扩展名）
            image_found = False
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                img_file = base_name + ext
                img_path = os.path.join(images_dir, img_file)
                
                if os.path.exists(img_path):
                    # 复制图像
                    import shutil
                    shutil.copy2(img_path, os.path.join(images_output_dir, img_file))
                    image_found = True
                    break
            
            if not image_found:
                print(f"警告: 找不到标签 {label_file} 对应的图像")
    
    # 分割数据集
    split_dir = os.path.join(args.output, 'dataset')
    print(f"正在分割数据集...")
    split_stats = split_dataset(data_dir, split_dir)
    
    # 创建YOLO配置文件
    config_path = os.path.join(split_dir, 'data.yaml')
    class_names = ["car", "truck", "bus", "motorcycle", "collision"]
    create_yolo_config(config_path, split_dir, class_names)
    
    print(f"数据集准备完成！配置文件保存在 {config_path}")
    print(f"训练集: {split_stats['train']} 图像")
    print(f"验证集: {split_stats['val']} 图像")
    print(f"测试集: {split_stats['test']} 图像")
    
    return config_path


def training_mode(args):
    """模型训练模式"""
    print("=== 模型训练模式 ===")
    
    # 检查数据配置
    if not args.data_config:
        # 尝试查找已生成的配置
        default_config = os.path.join(args.output, 'dataset', 'data.yaml')
        if os.path.exists(default_config):
            args.data_config = default_config
            print(f"使用发现的数据配置文件: {default_config}")
        else:
            # 尝试先准备数据集
            if args.annotation_file:
                config_path = prepare_dataset(args)
                if config_path:
                    args.data_config = config_path
                else:
                    print("错误: 无法自动准备数据集")
                    return False
            else:
                print("错误: 请提供数据配置文件或标注文件")
                return False
    
    # 导入训练模块
    try:
        # 动态导入以避免不必要的依赖
        from scripts.train import train, backup_best_model, validate_model
    except ImportError as e:
        print(f"导入训练模块失败: {str(e)}")
        return False
    
    # 设置训练参数
    model_path = args.pretrained_model if args.pretrained_model else 'yolov8n.pt'
    epochs = args.epochs
    batch_size = args.batch_size
    imgsz = args.img_size
    device = args.device if args.device else ''
    
    # 创建模型目录
    model_dir = os.path.join(args.output, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # 训练模型
    print(f"开始训练模型...")
    print(f"数据配置: {args.data_config}")
    print(f"预训练模型: {model_path}")
    print(f"训练参数: epochs={epochs}, batch_size={batch_size}, img_size={imgsz}")
    
    try:
        # 运行训练
        results_dir = train(
            data_yaml=args.data_config,
            model_path=model_path,
            epochs=epochs,
            batch_size=batch_size,
            imgsz=imgsz,
            device=device,
            project=model_dir,
            name='train',
            resume=args.resume,
            patience=args.patience
        )
        
        # 备份最佳模型
        trained_dir = os.path.join(model_dir, 'trained')
        os.makedirs(trained_dir, exist_ok=True)
        
        best_model_path = backup_best_model(results_dir, trained_dir)
        
        if best_model_path:
            print(f"训练完成！最佳模型保存在 {best_model_path}")
            
            # 验证模型
            if args.validate:
                print("正在验证模型性能...")
                validate_model(best_model_path, args.data_config, imgsz)
            
            # 创建模型元数据
            try:
                create_model_metadata(
                    best_model_path,
                    os.path.join(trained_dir, 'model_info.yaml'),
                    class_names=["car", "truck", "bus", "motorcycle", "collision"],
                    author="车辆碰撞检测系统"
                )
            except Exception as e:
                print(f"创建模型元数据时出错: {str(e)}")
            
            return best_model_path
        else:
            print("训练完成，但未找到最佳模型")
            return False
    
    except Exception as e:
        print(f"训练过程出错: {str(e)}")
        return False


def inference_mode(args):
    """模型推理模式"""
    print("=== 模型推理模式 ===")
    
    # 检查模型路径
    if not args.model:
        # 尝试查找训练好的模型
        default_model = os.path.join(args.output, 'models', 'trained', 'vehicle_collision_best.pt')
        if os.path.exists(default_model):
            args.model = default_model
            print(f"使用发现的模型: {default_model}")
        else:
            print("错误: 请提供模型路径")
            return False
    
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在 {args.model}")
        return False
    
    # 检查输入源
    if not args.source:
        print("错误: 请提供输入源（图像、视频或摄像头索引）")
        return False
    
    # 加载模型
    try:
        model = load_model(args.model, device=args.device)
        model_info(model)
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return False
    
    # 设置输出目录
    if not args.inference_output:
        inference_dir = os.path.join(args.output, 'inference_results')
        os.makedirs(inference_dir, exist_ok=True)
        args.inference_output = inference_dir
    else:
        os.makedirs(args.inference_output, exist_ok=True)
    
    # 根据输入类型执行不同的处理
    try:
        if args.webcam:
            # 处理摄像头
            try:
                camera_id = int(args.source)
            except ValueError:
                camera_id = 0
                print(f"警告: 无效的摄像头ID，使用默认值 0")
            
            output_path = None
            if args.save_results:
                output_path = os.path.join(args.inference_output, f"webcam_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
            
            print(f"正在处理摄像头 {camera_id}...")
            events = process_webcam(
                args.model, 
                camera_id, 
                output_path, 
                conf_threshold=args.conf_threshold
            )
        
        elif os.path.isfile(args.source):
            # 判断是图像还是视频
            file_ext = os.path.splitext(args.source)[1].lower()
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                # 处理图像
                output_path = None
                if args.save_results:
                    base_name = os.path.basename(args.source)
                    output_path = os.path.join(args.inference_output, f"result_{base_name}")
                
                print(f"正在处理图像 {args.source}...")
                results = process_image(
                    args.model, 
                    args.source, 
                    output_path, 
                    conf_threshold=args.conf_threshold, 
                    show=args.show_results
                )
            
            elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
                # 处理视频
                output_path = None
                if args.save_results:
                    base_name = os.path.basename(args.source)
                    output_path = os.path.join(args.inference_output, f"result_{base_name}")
                
                print(f"正在处理视频 {args.source}...")
                events = process_video(
                    args.model, 
                    args.source, 
                    output_path, 
                    conf_threshold=args.conf_threshold, 
                    show=args.show_results
                )
                
                # 保存碰撞事件报告
                if events and len(events) > 0:
                    # 保存事件列表
                    events_path = os.path.join(args.inference_output, "collision_events.yaml")
                    with open(events_path, 'w', encoding='utf-8') as f:
                        yaml.dump(events, f, default_flow_style=False)
                    
                    # 生成时间线图表
                    import cv2
                    cap = cv2.VideoCapture(args.source)
                    video_fps = cap.get(cv2.CAP_PROP_FPS)
                    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    
                    video_duration = video_frames / video_fps
                    
                    plot_path = os.path.join(args.inference_output, "collision_timeline.png")
                    plot_collision_events(events, video_duration, plot_path)
                    
                    print(f"检测到 {len(events)} 个碰撞事件，结果保存在 {events_path}")
            
            else:
                print(f"错误: 不支持的文件类型 {file_ext}")
                return False
        
        else:
            print(f"错误: 输入源 {args.source} 不存在")
            return False
        
        print("推理完成！")
        return True
    
    except Exception as e:
        print(f"推理过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='车辆碰撞检测系统')
    
    # 基本参数
    parser.add_argument('--mode', type=str, choices=['annotation', 'training', 'inference', 'all'],
                      default='all', help='运行模式')
    parser.add_argument('--input', type=str, default='data/raw', 
                      help='输入目录或文件')
    parser.add_argument('--output', type=str, default='output', 
                      help='输出目录')
    
    # 数据标注参数
    parser.add_argument('--auto-start', action='store_true', 
                      help='自动启动Label Studio')
    parser.add_argument('--no-browser', action='store_true', 
                      help='禁止自动打开浏览器')
    parser.add_argument('--frame-rate', type=int, default=5,
                      help='视频帧提取率（帧/秒）')
    
    # 数据准备参数
    parser.add_argument('--annotation-file', type=str, 
                      help='Label Studio导出的JSON标注文件')
    parser.add_argument('--images-dir', type=str, 
                      help='图像目录路径')
    
    # 训练参数
    parser.add_argument('--data-config', type=str, 
                      help='数据集配置文件')
    parser.add_argument('--pretrained-model', type=str, 
                      help='预训练模型路径')
    parser.add_argument('--epochs', type=int, default=100, 
                      help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, 
                      help='批次大小')
    parser.add_argument('--img-size', type=int, default=640, 
                      help='图像大小')
    parser.add_argument('--device', type=str, 
                      help='训练设备，例如 cpu 或 0 或 0,1,2,3')
    parser.add_argument('--resume', action='store_true', 
                      help='继续上次训练')
    parser.add_argument('--patience', type=int, default=50, 
                      help='早停耐心值')
    parser.add_argument('--validate', action='store_true', 
                      help='训练后验证模型')
    
    # 推理参数
    parser.add_argument('--model', type=str, 
                      help='模型路径')
    parser.add_argument('--source', type=str, 
                      help='输入源（图像、视频路径或摄像头索引）')
    parser.add_argument('--inference-output', type=str, 
                      help='推理结果输出目录')
    parser.add_argument('--conf-threshold', type=float, default=0.25, 
                      help='置信度阈值')
    parser.add_argument('--save-results', action='store_true', 
                      help='保存结果')
    parser.add_argument('--show-results', action='store_true', 
                      help='显示结果')
    parser.add_argument('--webcam', action='store_true', 
                      help='使用网络摄像头')
    
    # 解析参数
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 检查环境
    if not check_requirements():
        return
    
    # 打印版本信息
    print("\n=== 车辆碰撞检测系统 ===")
    print(f"运行模式: {args.mode}")
    print(f"输出目录: {os.path.abspath(args.output)}")
    
    # 根据模式运行相应的功能
    try:
        if args.mode == 'annotation' or args.mode == 'all':
            annotation_mode(args)
        
        if args.mode == 'training' or args.mode == 'all':
            training_mode(args)
        
        if args.mode == 'inference' or args.mode == 'all':
            inference_mode(args)
    
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行时出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 