import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from torchvision.ops import nms
import torch
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
import uuid
import pandas as pd
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns

#export CUDA_VISIBLE_DEVICES=1

def train(yaml_path):
    start_time = time.time()
    yolo = YOLO('/home/zww/2t_ssd1_ww/zww/ultralytics-yolov11/ultralytics/cfg/models/v3/yolov3-tiny.yaml')
    yolo.train(data=yaml_path, imgsz=1280, patience=500, device=0, epochs=500, batch=8, single_cls=True, close_mosaic=500)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

def process_image(model_path, img_path, pre_dir, original_dir, txt_dir, json_dir):
    start_time = time.time()
    try:
        # 加载模型
        model = YOLO(model_path)
        
        # 获取模型预测结果
        results = model(img_path, iou=0.45)
        
        # 处理并保存每个结果
        for r in results:
            save_image = False
            boxes_info = []
            shapes = []
            
            # 获取图像尺寸
            img_width, img_height = Image.open(img_path).size
            
            # 检查每个检测框的概率
            for box in r.boxes:
                if box.conf > 0.5:  # 检查概率是否大于0.9
                    # 获取检测框信息
                    x1, y1, x2, y2 = box.xyxy[0]
                    area = (x2 - x1) * (y2 - y1)
                    if area < 400:  # 如果面积小于400，则跳过
                        continue
                    
                    save_image = True
                    x_center = (x1 + x2) / 2 / img_width
                    y_center = (y1 + y2) / 2 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    boxes_info.append(f"{int(box.cls)} {x_center} {y_center} {width} {height}")
                    
                    # 添加到 shapes 列表
                    shapes.append({
                        "label": str(int(box.cls)),
                        "points": [
                            [x1.item(), y1.item()],
                            [x2.item(), y2.item()]
                        ],
                        "group_id": None,
                        "description": "",
                        "shape_type": "rectangle",
                        "flags": {}
                    })
            
            if save_image:
                # 保存预测结果图像
                im_array = r.plot(line_width=1, font_size=1)  # 绘制预测结果的 BGR numpy 数组
                im = Image.fromarray(im_array[..., ::-1])  # 转换为 RGB PIL 图像
                
                # 在图像上绘制检测框和概率
                draw = ImageDraw.Draw(im)
                font = ImageFont.load_default()
                for box in r.boxes:
                    if box.conf > 0.5:
                        x1, y1, x2, y2 = box.xyxy[0]
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
                        draw.text((x1, y1), f"{box.conf.item():.2f}", fill="red", font=font)
                
                output_path = os.path.join(pre_dir, os.path.splitext(os.path.basename(img_path))[0] + '.png')
                im.save(output_path)
                
                # 保存原始图像
                original_output_path = os.path.join(original_dir, os.path.splitext(os.path.basename(img_path))[0] + '.png')
                original_im = Image.open(img_path)
                original_im.save(original_output_path)
                
                # 保存检测框信息到txt文件
                txt_output_path = os.path.join(txt_dir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
                with open(txt_output_path, 'w') as f:
                    f.write('\n'.join(boxes_info))
                
                # 保存检测框信息到json文件
                json_output_path = os.path.join(json_dir, os.path.splitext(os.path.basename(img_path))[0] + '.json')
                json_data = {
                    "version": "5.3.1",
                    "flags": {},
                    "shapes": shapes,
                    "imagePath": os.path.basename(img_path),  # 修改为图像文件名
                    "imageData": None,
                    "imageHeight": img_height,
                    "imageWidth": img_width
                }
                with open(json_output_path, 'w') as f:
                    json.dump(json_data, f, indent=4)
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
    end_time = time.time()
    print(f"Processed image {img_path} in {end_time - start_time:.2f} seconds")

def predict(model_path):
    start_time = time.time()
    # 定义目录
    input_dir = '/home/zww/8t_sdc/zww/demo/images'
    output_dir = '/home/zww/8t_sdc/zww/demo/png_output'
    pre_dir = os.path.join(output_dir, 'pre_images')
    original_dir = os.path.join(output_dir, 'original_images')
    txt_dir = os.path.join(output_dir, 'labels')
    json_dir = os.path.join(output_dir, 'json_labels')

    # 如果输出目录不存在，则创建它们
    os.makedirs(pre_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    # 获取输入目录中的所有图像路径
    img_paths = [os.path.join(input_dir, img) for img in os.listdir(input_dir)]
    total_imgs = len(img_paths)

    # 使用多进程并行处理图像
    with ProcessPoolExecutor(max_workers=2) as executor:  # 增加最大并行进程数到8
        futures = [executor.submit(process_image, model_path, img_path, pre_dir, original_dir, txt_dir, json_dir) for img_path in img_paths]
        for i, future in enumerate(as_completed(futures)):
            try:
                future.result()  # 等待每个任务完成
                print(f"Processed {i + 1}/{total_imgs} images")
            except Exception as e:
                print(f"Error in processing image {i + 1}: {e}")
    end_time = time.time()
    print(f"Test completed in {end_time - start_time:.2f} seconds")


def process_sub_img(model, sub_img, x_offset, y_offset):
    sub_img_path = f'temp_sub_img_{uuid.uuid4().hex}.png'
    
    try:
        sub_img.save(sub_img_path)
    except Exception as e:
        print(f"Error saving sub-image {sub_img_path}: {e}")
        return [], [], []
    
    try:
        results = model(sub_img_path, iou=0.45)
    except Exception as e:
        print(f"Error processing sub-image {sub_img_path}: {e}")
        return [], [], []
    
    boxes, scores, classes = [], [], []
    for r in results:
        for box in r.boxes:
            if box.conf > 0.9:
                x1, y1, x2, y2 = box.xyxy[0]
                area = (x2 - x1) * (y2 - y1)
                if area < 400:
                    continue
                boxes.append([x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset])
                scores.append(box.conf)
                classes.append(box.cls)
    
    # 删除临时文件
    os.remove(sub_img_path)
    
    return boxes, scores, classes

def predict_imgs_big(model_path, large_img_dir, output_dir):
    start_time = time.time()
    # 加载模型
    model = YOLO(model_path)
    large_img_paths = [os.path.join(large_img_dir, img) for img in os.listdir(large_img_dir)]
    
    # 创建输出文件夹
    output_img_dir = os.path.join(output_dir, 'png')
    output_json_dir = os.path.join(output_dir, 'json')
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_json_dir, exist_ok=True)
    
    for large_img_path in large_img_paths:
        # 切割大图
        img = Image.open(large_img_path)
        img_width, img_height = img.size
        stride = 640  # 重叠50%
        sub_imgs = []
        for y in range(0, img_height, stride):
            for x in range(0, img_width, stride):
                sub_img = img.crop((x, y, x + 1280, y + 1280))
                sub_imgs.append((sub_img, x, y))
        
        all_boxes = []
        all_scores = []
        all_classes = []
        
        # 使用多线程获取每个子图的预测结果
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_sub_img, model, sub_img, x_offset, y_offset) for sub_img, x_offset, y_offset in sub_imgs]
            for future in as_completed(futures):
                boxes, scores, classes = future.result()
                all_boxes.extend(boxes)
                all_scores.extend(scores)
                all_classes.extend(classes)
        
        # 使用NMS去除重叠的检测框
        all_boxes = torch.tensor(all_boxes)
        all_scores = torch.tensor(all_scores)
        all_classes = torch.tensor(all_classes)
        indices = nms(all_boxes, all_scores, iou_threshold=0.5)
        
        final_boxes = all_boxes[indices]
        final_scores = all_scores[indices]
        final_classes = all_classes[indices]
        
        # 保存结果
        shapes = []
        for i in range(len(final_boxes)):
            x1, y1, x2, y2 = final_boxes[i]
            shapes.append({
                "label": str(int(final_classes[i])),
                "points": [
                    [x1.item(), y1.item()],
                    [x2.item(), y2.item()]
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {}
            })
        
        json_data = {
            "version": "5.3.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": os.path.basename(large_img_path),  # 修改为图像文件名
            "imageData": None,
            "imageHeight": img_height,
            "imageWidth": img_width
        }
        
        output_json_path = os.path.join(output_json_dir, os.path.basename(large_img_path).replace('.png', '.json'))
        with open(output_json_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        
        # 在原图上绘制红色框框和概率
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        for i in range(len(final_boxes)):
            x1, y1, x2, y2 = final_boxes[i]
            score = final_scores[i].item()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1), f"{score:.2f}", fill="red", font=font)
        
        # 保存带有框框的图像
        output_img_path = os.path.join(output_img_dir, os.path.basename(large_img_path))
        img.save(output_img_path)
    
    end_time = time.time()
    print(f"Test images completed in {end_time - start_time:.2f} seconds")

def val(model_path):
    start_time = time.time()
    model = YOLO(model_path)
    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps   # a list contains map50-95 of each category
    end_time = time.time()
    print(f"Validation completed in {end_time - start_time:.2f} seconds")

def test_index(model_path):
    model = YOLO(model_path)
    
    # Validate the model
    metrics = model.val(split='test')  # no arguments needed, dataset and settings remembered
    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps   # a list contains map50-95 of each category

    
if __name__ == '__main__':
    predict('runs/detect/Ablation experiment_v10s/dataset_all_models/train/train_yolov10s-SEA-Detect_FASFF-C3K2_2-C2f_EMA4/weights/best.pt')
    #val('runs/detect/train_yolov10m-SEAttention/weights/best.pt')
    #train('data_demo.yaml')  #train_demo
    #train('data.yaml')
    #predict_imgs_big('runs/detect/train_yolov8-c2f_Faster/weights/best.pt', '/home/zww/8t_sdc/zww/demo/qupath', '/home/zww/8t_sdc/zww/demo/qupath_test')
    #test_index('runs/detect/train_yolov11/train_yolov3-tiny/weights/best.pt')