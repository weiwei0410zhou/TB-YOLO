import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import glob
import h5py
from openslide import OpenSlide

def TILs_outer(tumor_roi):
    depth = 20
    if depth % 2 == 0:
        depth = depth + 1
    # 使用高斯模糊处理肿瘤区域
    tumor_roi_blurred_outer = cv2.GaussianBlur(tumor_roi, (depth, depth), 0)
    
    tumor_roi_blurred_outer = cv2.GaussianBlur(tumor_roi_blurred_outer, (3, 3), 0) 
    # 找到肿瘤区域的轮廓
    contours, _ = cv2.findContours(tumor_roi_blurred_outer, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    annotations = []

    # 构建标注信息
    for contour in contours:
        contour = contour.reshape(-1, 2)
        if contour.shape[0] < 3:
            continue
        # 将轮廓坐标点放大128倍
        contour = contour * 8
        coords = contour.tolist()
        coords.append(coords[0])
        annotation = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords]
            },
            "properties": {
                "object_type": "annotation",
                "color": color_list[1],
                "classification": {
                    "name": "invasive 500μm",
                    "colorRGB": 256
                },
                "isLocked": False
            }
        }
        annotations.append(annotation)

    return tumor_roi_blurred_outer, annotations

save_dir = "/home/zww/2t_ssd1_ww/zww/zww/CK_data_color_decon_output_True_WSI"
result_npy_dir = '/home/zww/2t_ssd1_ww/zww/zww/CK_data_color_decon_imgs_True_WSI/npy'
thumbnail_path = "/home/zww/2t_ssd1_ww/zww/zww/CK_data_imgs_True_WSI/thumbnail"
wsis_path = "/home/zww/8t_hdd1_ww/zww/tiff_out"

invasive_out_dir = os.path.join(save_dir, "invasive")
tumor_dir = os.path.join(save_dir, "tumor")
invasive_heatmap_dir = os.path.join(save_dir, "invasive_heatmap")
invasive_h5_dir = os.path.join(save_dir, "invasive_h5")

os.makedirs(invasive_out_dir, exist_ok=True)
os.makedirs(tumor_dir, exist_ok=True)
os.makedirs(invasive_heatmap_dir, exist_ok=True)
os.makedirs(invasive_h5_dir, exist_ok=True)
color_list = [(0, 255, 0), (0, 0, 255)]

img_names = os.listdir(result_npy_dir)
for img_name in img_names:
    img_path = os.path.join(result_npy_dir, img_name)
    wsi_path = os.path.join(wsis_path, img_name.replace("npy", "tiff"))
    tumor_save_path = os.path.join(tumor_dir, img_name.replace("npy", "json"))
    invasive_out_save_path = os.path.join(invasive_out_dir, img_name.replace("npy", "json"))
    invasive_heatmap_path = os.path.join(invasive_heatmap_dir, img_name.replace("npy", "png"))
    invasive_h5_path = os.path.join(invasive_h5_dir, img_name.replace("npy", "h5"))
    # 读取灰度图像
    img_gray = np.load(img_path)
    tumor_roi = cv2.GaussianBlur(img_gray, (3, 3), 0) 

    # 提取轮廓
    contours, _ = cv2.findContours(tumor_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 构建标注信息
    annotations = []
    for contour in contours:
        contour = contour.reshape(-1, 2)
        contour = contour * 8
        coords = contour.tolist()
        coords.append(coords[0])
        red_boundary_annotation = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords]
            },
            "properties": {
                "object_type": "annotation",
                "color": color_list[0],
                "classification": {
                    "name": "tumor",
                    "colorRGB": 256
                },
                "isLocked": False
            }
        }
        annotations.append(red_boundary_annotation)
    
    with open(tumor_save_path, 'w') as f:
        json.dump(annotations, f)
    
    tumor_roi_outer_500, invasive_outer_annotations = TILs_outer(tumor_roi)

    if len(invasive_outer_annotations) != 0:
        with open(invasive_out_save_path, 'w') as f:
            json.dump(invasive_outer_annotations, f)

    tumor_roi_outer_500[tumor_roi_outer_500 > 0] = 1
    tumor_roi[tumor_roi > 0] = 1
    overlap_region = tumor_roi_outer_500.astype('bool') ^ tumor_roi.astype('bool')
    all_coords = np.where(overlap_region == True)

    overlap_region1 = (overlap_region * 255).astype('uint8')

    every_thumbnail_path = os.path.join(thumbnail_path, img_name.replace('npy', 'png'))
    img1 = cv2.imread(every_thumbnail_path)
    for y in range(overlap_region1.shape[0]):
        for x in range(overlap_region1.shape[1]):
            if overlap_region1[y, x] != 0:  # 检查像素是否不为黑色
                img1[y, x] = 255
    cv2.imwrite(invasive_heatmap_path, img1)   

    with h5py.File(invasive_h5_path, 'w') as f:
        f.create_dataset('coords', data=all_coords)   
        print("H5 文件已保存到:", invasive_h5_path)

    print("finish")