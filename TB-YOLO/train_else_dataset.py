import os
from ultralytics import YOLO
import torch
import time

#export CUDA_VISIBLE_DEVICES=1
def train_Brain_tumor(yaml_path):
    start_time = time.time()
    yolo = YOLO('/home/zww/2t_ssd1_ww/zww/yolov10-SEAttention/ultralytics/cfg/yolov11 and yolov10/yolov10n-SEA-Detect_FASFF-C3K2_2-C2f_EMA4.yaml')
    yolo.train(data=yaml_path, imgsz=640, patience=500, device=0, epochs=500, batch=16, close_mosaic=500)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

def train_BCCD(yaml_path):
    start_time = time.time()
    yolo = YOLO('/home/zww/2t_ssd1_ww/zww/yolov10-SEAttention/ultralytics/cfg/yolov11 and yolov10/yolov10s-SEA-Detect_FASFF-C3K2_2-C2f_EMA4.yaml')
    yolo.train(data=yaml_path, imgsz=640, patience=500, device=0, epochs=500, batch=16, close_mosaic=500)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

def train_ColonDetection(yaml_path):
    start_time = time.time()
    yolo = YOLO('/home/zww/2t_ssd1_ww/zww/yolov10-SEAttention/ultralytics/cfg/models/v8/yolov8m.yaml')
    yolo.train(data=yaml_path, imgsz=640, patience=500, device=1, epochs=500, batch=16, close_mosaic=500)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

def train_colon_cancer_new_8154(yaml_path):
    start_time = time.time()
    yolo = YOLO('/home/zww/2t_ssd1_ww/zww/yolov10-SEAttention/ultralytics/cfg/yolov11 and yolov10/yolov10s-SEA-Detect_FASFF-C3K2_2-C2f_EMA4.yaml')
    yolo.train(data=yaml_path, imgsz=640, patience=500, device=1, epochs=500, batch=16, close_mosaic=500)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

def train_colon_cancer_new_26623(yaml_path):
    start_time = time.time()
    yolo = YOLO('/home/zww/2t_ssd1_ww/zww/yolov10-SEAttention/ultralytics/cfg/yolov11 and yolov10/yolov10m-SEA-Detect_FASFF-C3K2_2-C2f_EMA4.yaml')
    yolo.train(data=yaml_path, imgsz=640, patience=500, device=1, epochs=500, batch=16, close_mosaic=500)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

def train_PASCAL_VOC_2007(yaml_path):
    start_time = time.time()
    yolo = YOLO('/home/zww/2t_ssd1_ww/zww/yolov10-SEAttention/ultralytics/cfg/models/v8/yolov8s.yaml')
    yolo.train(data=yaml_path, imgsz=640, patience=500, device=0, epochs=500, batch=16, close_mosaic=500)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

def test_index(model_path):
    model = YOLO(model_path)
    
    # Validate the model
    metrics = model.val(split='test')  # no arguments needed, dataset and settings remembered
    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps   # a list contains map50-95 of each category

if __name__ == '__main__':
    #train_BCCD('bccd.yaml')
    #train_ColonDetection('ColonDetection.yaml')
    #train_colon_cancer_new_8154('colon_cancer_new_8154.yaml')
    #train_colon_cancer_new_26623('colon_cancer_new_26623.yaml')
    #train_Brain_tumor('Brain_tumor.yaml')
    #train_PASCAL_VOC_2007('PASCAL_VOC_2007.yaml')
    test_index('runs/detect/new_brain_tumor/train/train_yolov10s-SEA-Detect_FASFF-C3K2_2-C2f_EMA4/weights/best.pt')