# yolo11_crowd_human_train
基于yolo11的人群检测数据训练


## 环境准备
1. 创建一个python环境并安装所需要的依赖
```shell
conda create --name yolo11_train python=3.12
# 使用当前创建的环境
conda activate yolo11_train
# 下载所需要的依赖
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 使用label-studio进行图片标记
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true label-studio start
```
2. 浏览器访问label-studio的页面，进行图片标记
```shell
http://localhost:8080
```