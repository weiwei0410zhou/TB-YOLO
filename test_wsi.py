import openslide
import h5py
import numpy as np
import json
from ultralytics import YOLO
from torchvision.ops import nms  
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
from PIL import Image
import os
from functools import partial
import threading

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# 定义路径
wsi_dir = '/home/zww/8t_sdc/zww/dataset/IHC'
model_path = '/home/zww/2t_ssd1_ww/zww/yolov10-SEAttention/runs/detect/train_yolov10m-SEA-c2f_att-c2f_faster-SA/weights/best.pt'
tumor_save_dir = '/home/zww/2t_ssd1_ww/zww/zww/CK_data_color_decon_output/tumor_budding_json'
h5_dir = '/home/zww/2t_ssd1_ww/zww/zww/CK_data_color_decon_output/invasive_h5'

# 切割参数
tile_size = 1280
overlap = 0.3
magnification = 128
step_size = int(tile_size * (1 - overlap))
temp_size = tile_size // 128
model = YOLO(model_path)

def process_sub_img(coords, common_param):
    boxes, scores, classes = [], [], []
    tile = common_param.read_region(coords, 0, (tile_size, tile_size)).convert('RGB')
    try:
        results = model(tile, iou=0.45, verbose=False)
    except Exception as e:
        print(f"Error processing sub-image {coords}: {e}")
        return None
    
    for r in results:
        for box in r.boxes:
            if box.conf > 0.5:
                x1, y1, x2, y2 = box.xyxy[0]
                area = (x2 - x1) * (y2 - y1)
                if area < 400:
                    continue
                boxes.append([x1 + coords[0], y1 + coords[1], x2 + coords[0], y2 + coords[1]])
                scores.append(box.conf)
                classes.append(box.cls)

    print(f"Finished processing tile at {coords}")        
    return boxes, scores, classes

def process_wsi(wsi_path, h5_path, tumor_save_path, magnification, step_size):
    # 加载WSI图像
    slide = openslide.OpenSlide(wsi_path)
    process_image_with_param = partial(process_sub_img, common_param=slide)
    # 加载h5文件
    with h5py.File(h5_path, 'r') as hdf5_file:
        coords = hdf5_file['coords'][:]
        coords = coords.T
        print(f"Dataset shape: {coords.shape}")
        print(f"Total coordinates: {len(coords)}")
    mask = np.zeros((slide.dimensions[1] // magnification, slide.dimensions[0] // magnification), dtype=np.uint8)
    # 更新 mask
    for coord in coords:
        x, y = coord[:2]  # 只取前两个值
        mask[x, y] = 1
    print(f"Slide dimensions: {slide.dimensions}")

    # 保存结果
    all_boxes = []
    all_scores = []
    all_classes = []
    overlap_area_coords = []
    for x in range(0, slide.dimensions[0] // magnification, step_size // magnification):
        for y in range(0, slide.dimensions[1] // magnification, step_size // magnification):
            #print(f"Processing tile at ({x}, {y})")
            region_tile = mask[y:y+temp_size, x:x+temp_size]
            overlap_area = np.sum(region_tile == 1) / (temp_size * temp_size)
            if overlap_area > 0.2:
                print(f"Submitting tile at ({x}, {y}) with overlap area {overlap_area:.2f}")
                overlap_area_coords.append((x * magnification, y * magnification))

    # 切割图像并进行预测
    futures = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for c in overlap_area_coords:
            future = executor.submit(process_image_with_param, c)
            futures[future] = c
    for future in as_completed(futures):
        try:
            result = future.result()
            if result is None:
                continue
            boxes, scores, classes = result
            all_boxes.extend(boxes)
            all_scores.extend(scores)
            all_classes.extend(classes)
            print(f"Finished processing tile")
        except Exception as e:
            print(f"Error in future result: {e}")
    # 使用NMS去除重叠的检测框
    try:
        if len(all_boxes) > 0:
            all_boxes = torch.tensor(all_boxes)
            all_scores = torch.tensor(all_scores)
            all_classes = torch.tensor(all_classes)
            indices = nms(all_boxes, all_scores, iou_threshold=0.3)

            final_boxes = all_boxes[indices]
            final_scores = all_scores[indices]
            final_classes = all_classes[indices]
        else:
            final_boxes = torch.tensor([])
            final_scores = torch.tensor([])
            final_classes = torch.tensor([])
    except Exception as e:
        print(f"Error during NMS: {e}")
        return

    # 输出JSON文件
    annotations = []
    for i in range(len(final_boxes)):
        x1, y1, x2, y2 = final_boxes[i]
        tumor_budding_annotation = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [x1.item(), y1.item()],
                    [x2.item(), y1.item()],
                    [x2.item(), y2.item()],
                    [x1.item(), y2.item()],
                    [x1.item(), y1.item()]
                ]]
            },
            "properties": {
                "object_type": "annotation",
                "color": [255, 0, 0],
                "classification": {
                    "name": "tumor_budding",
                    "colorRGB": 256
                },
                "isLocked": False
            }
        }
        annotations.append(tumor_budding_annotation)

    try:
        with open(tumor_save_path, 'w') as f:
            json.dump(annotations, f)
        print(f"JSON file saved successfully at {tumor_save_path}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")

# 遍历wsi文件夹中的所有WSI文件
for wsi_file in os.listdir(wsi_dir):
    if wsi_file.endswith('.ndpi'):
        wsi_path = os.path.join(wsi_dir, wsi_file)
        h5_file = wsi_file.replace('.ndpi', '.h5')
        h5_path = os.path.join(h5_dir, h5_file)
        tumor_save_path = os.path.join(tumor_save_dir, wsi_file.replace('.ndpi', '.json'))
        process_wsi(wsi_path, h5_path, tumor_save_path, magnification, step_size)