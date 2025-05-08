import os
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import argparse

def count_people(image_path, save_dir='results'):
    """
    统计图片中的人数并可视化结果
    :param image_path: 图片路径
    :param save_dir: 结果保存目录
    :return: 检测到的人数
    """
    # 加载模型
    model = YOLO('car_pothole_yolo11-seg.pt')
    
    # 读取图片
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 运行检测
    results = model(image)
    
    # 获取人物检测结果
    people_count = 0
    result = results[0]
    
    # 创建结果保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 在图像上绘制检测结果
    annotated_img = image.copy()
    for box in result.boxes:
        if box.cls == 0:  # 假设类别0为人
            people_count += 1
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # 获取置信度
            conf = float(box.conf[0])
            
            # 绘制边界框
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 添加置信度标签
            label = f'Person {conf:.2f}'
            cv2.putText(annotated_img, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 在图像顶部添加总人数
    cv2.putText(annotated_img, f'Total People: {people_count}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 保存结果图像
    output_path = os.path.join(save_dir, f'result_{Path(image_path).name}')
    cv2.imwrite(output_path, annotated_img)
    
    return people_count, output_path

def process_directory(directory_path, save_dir='results'):
    """
    处理目录中的所有图片
    :param directory_path: 图片目录路径
    :param save_dir: 结果保存目录
    """
    directory = Path(directory_path)
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for image_path in directory.glob('*'):
        if image_path.suffix.lower() in supported_formats:
            try:
                count, output_path = count_people(image_path, save_dir)
                print(f"图片 {image_path.name} 中检测到 {count} 人")
                print(f"结果已保存至: {output_path}")
            except Exception as e:
                print(f"处理图片 {image_path.name} 时出错: {str(e)}")

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='使用YOLO模型进行人数统计')
    parser.add_argument('--input', type=str, required=True,
                      help='输入图片路径或目录路径')
    parser.add_argument('--output', type=str, default='results',
                      help='结果保存目录，默认为results')
    
    args = parser.parse_args()
    input_path = args.input
    save_dir = args.output
    
    if os.path.isfile(input_path):
        # 处理单个文件
        try:
            count, output_path = count_people(input_path, save_dir)
            print(f"图片中检测到 {count} 人")
            print(f"结果已保存至: {output_path}")
        except Exception as e:
            print(f"处理图片时出错: {str(e)}")
    elif os.path.isdir(input_path):
        # 处理整个目录
        print(f"开始处理目录 {input_path} 中的图片...")
        process_directory(input_path, save_dir)
        print("处理完成！")
    else:
        print(f"错误：输入路径 {input_path} 不存在")

if __name__ == '__main__':
    main() 