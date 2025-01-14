import cv2
import numpy as np
import os
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity
from skimage import img_as_ubyte
from scipy.ndimage import binary_fill_holes
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def color_deconvolution(image, intensity):
    """
    对输入图像进行颜色反卷积，并且反卷积强度可调。
    
    参数:
    - image: 输入的RGB图像。
    - intensity: 用于调节反卷积强度的浮点数值,范围为0.0到2.0。默认值为1.0。
    
    返回值:
    - hematoxylin: 反卷积后的8位赫马托克林通道图像。
    """
    # 将图像从RGB空间转换到HED空间
    ihc_hed = rgb2hed(image)
    
    DAB = ihc_hed[:, :, 2] * intensity
    DAB = np.clip(DAB, 0, 1)
    DAB = rescale_intensity(DAB, out_range=(0, 1))
    DAB = img_as_ubyte(DAB)

    hematoxylin = ihc_hed[:, :, 0] * intensity

    # 裁剪值以保持在有效范围内
    hematoxylin = np.clip(hematoxylin, 0, 1)
    
    # 重新调整强度值
    hematoxylin = rescale_intensity(hematoxylin, out_range=(0, 1))
    
    # 转换为8位图像
    hematoxylin = img_as_ubyte(hematoxylin)
    
    return hematoxylin, DAB

def generate_mask(hematoxylin_image):
    # Apply threshold to generate binary image mask
    _, mask1 = cv2.threshold(hematoxylin_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(hematoxylin_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Remove small regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 1000:  # Remove regions smaller than 500 pixels
            cv2.drawContours(mask, [contour], -1, 0, -1)
    mask2 = binary_fill_holes(mask // 255).astype(np.uint8) * 255

    kernel_erode = np.ones((3, 3), np.uint8)  # Adjust kernel size as needed
    HE_mask = cv2.erode(mask, kernel_erode, iterations=1)
    #mask = binary_fill_holes(mask // 255).astype(np.uint8) * 255
    HE_mask = binary_fill_holes(HE_mask // 255).astype(np.uint8) * 255
    return mask1, mask2, HE_mask

def generate_DAB_mask(DAB_image):
    # 应用阈值生成二值图像mask
    _, mask = cv2.threshold(DAB_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 去除小区域
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 100 or cv2.contourArea(contour) > 4096:  # 小于64像素的区域被去除
            #-1：表示绘制所有的轮廓，而不是某个特定的轮廓。0：表示绘制颜色为黑色（在掩码图像中，0表示黑色）。-1：表示填充轮廓内部区域，而不仅仅是绘制轮廓的边界。
            cv2.drawContours(mask, [contour], -1, 0, -1)
    mask = binary_fill_holes(mask // 255).astype(np.uint8) * 255
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask

def process_images(a, b):
    # 假设 a 和 b 是同样大小的二值图像
    result = np.zeros_like(a)
    
    # 条件1：a=255且a-b=0，则a这个部分继续等于255
    condition1 = (a == 255) & ((a - b) == 0)
    result[condition1] = 255
    
    # 条件2：a=255且a-b>0,则a这个部分等于0
    condition2 = (a == 255) & ((a - b) > 0)
    result[condition2] = 0

    condition3 = (a == 0).all() or (b == 0).all()
    result[condition3] = 0
    
    return result

def overlay_contours(image, mask, color=(255, 0, 0), thickness=2):
    # Find contours in the binary image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Overlay contours on the original image
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, color, thickness)
    
    return image_with_contours

def draw_bounding_boxes(image, mask, color=(0, 0, 255), thickness=2):
    # Find contours in the binary image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes on the original image
    image_with_boxes = image.copy()
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), color, thickness)
        boxes.append([x, y, x + w, y + h])
    
    return image_with_boxes, boxes

def read_yolo_labels(label_path):
    boxes = []
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            x_center, y_center, width, height = map(float, parts[1:])
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)
            boxes.append([x1, y1, x2, y2])
    return boxes

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou

def calculate_ap(ious, iou_threshold):
    tp = sum([1 for iou in ious if iou >= iou_threshold])
    fp = len(ious) - tp
    fn = len(all_true_boxes) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall

# Paths
input_dir = r"D:\qupath\dataset\test\images"  # 输入图片文件夹路径
label_dir = r"D:\qupath\dataset\test\labels"  # 输入标签文件夹路径
output_dir = r"D:\qupath\demo\color_decon_output_worst_imgs-YOLO"

imgs_dir = os.path.join(output_dir, "imgs")
DAB_mask_dir = os.path.join(output_dir, "DAB_mask")
mask_origin_dir = os.path.join(output_dir, "mask_origin")
HE_mask_dir = os.path.join(output_dir, "HE_mask")
nuclei_combine_dir = os.path.join(output_dir, "nuclei_combine")
nuclei_mask_dir = os.path.join(output_dir, "nuclei_mask")
mask_with_contours_dir = os.path.join(output_dir, "mask_with_contours")
last_mask_with_contours_dir = os.path.join(output_dir, "last_mask_with_contours")
last_mask_dir = os.path.join(output_dir, "last_mask")

# Create directories if they don't exist
os.makedirs(imgs_dir, exist_ok=True)
os.makedirs(mask_origin_dir, exist_ok=True)
os.makedirs(DAB_mask_dir, exist_ok=True)
os.makedirs(HE_mask_dir, exist_ok=True)
os.makedirs(nuclei_combine_dir, exist_ok=True)
os.makedirs(nuclei_mask_dir, exist_ok=True)
os.makedirs(mask_with_contours_dir, exist_ok=True)
os.makedirs(last_mask_with_contours_dir, exist_ok=True)
os.makedirs(last_mask_dir, exist_ok=True)

# Process each image in the input directory
files_dir = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]

all_true_boxes = []
all_pred_boxes = []

for file_name in files_dir:
    img_path = os.path.join(input_dir, file_name)
    label_path = os.path.join(label_dir, file_name.replace('.png', '.txt'))
    img = cv2.imread(img_path)
    img_height, img_width = img.shape[:2]
    intensity = 1.5  # 可以根据需要调整这个值
    hematoxylin_image, DAB_image = color_deconvolution(img, intensity=intensity)
    mask, dilate_mask, HE_mask = generate_mask(hematoxylin_image)
    DAB_mask = generate_DAB_mask(DAB_image) 
    
    cv2.imwrite(os.path.join(HE_mask_dir, file_name), HE_mask)
    cv2.imwrite(os.path.join(mask_origin_dir, file_name), mask)
    cv2.imwrite(os.path.join(DAB_mask_dir, file_name), DAB_mask)
    nuclei_combine = process_images(mask, DAB_mask)
    cv2.imwrite(os.path.join(nuclei_combine_dir, file_name), nuclei_combine)

    mask_nuclei = np.where((HE_mask == 255) & (nuclei_combine == 255), 255, np.where((HE_mask == 255) & (nuclei_combine != 255), 127, 0)).astype(np.uint8)

    mask_with_contours = overlay_contours(img, HE_mask)

    contours_mask, _ = cv2.findContours(HE_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(mask_nuclei)

    for contour_a in contours_mask:
    # 第一步：筛选白色轮廓的大小
        contour_area = cv2.contourArea(contour_a)
        if 600 < contour_area < 4090:
            mask_a = np.zeros_like(mask_nuclei)
            cv2.drawContours(mask_a, [contour_a], -1, 255, thickness=cv2.FILLED)
            
            # Corrected line here
            mask_b = np.zeros_like(mask_nuclei)
            mask_b[mask_a == 255] = mask_nuclei[mask_a == 255]
            
            masked_c = np.zeros_like(mask_nuclei)
            masked_c[mask_b != 127] = 0
            masked_c[mask_b == 127] = 255

            contours_c_mask, _ = cv2.findContours(masked_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            num_black_labels = 0
            for contour_c in contours_c_mask:
                if 600 < cv2.contourArea(contour_c) < 4090:
                    num_black_labels += 1
            if 1 <= num_black_labels <= 10:
                cv2.drawContours(result, [contour_a], -1, 255, thickness=cv2.FILLED)
    last_mask = result

    last_mask_with_contours, pred_boxes = draw_bounding_boxes(img, last_mask, color=(0, 0, 255), thickness=4)
    true_boxes = read_yolo_labels(label_path)

    all_true_boxes.extend(true_boxes)
    all_pred_boxes.extend(pred_boxes)

    cv2.imwrite(os.path.join(mask_with_contours_dir, file_name), mask_with_contours)
    cv2.imwrite(os.path.join(nuclei_mask_dir, file_name), mask_nuclei)
    cv2.imwrite(os.path.join(last_mask_dir, file_name), last_mask)
    cv2.imwrite(os.path.join(last_mask_with_contours_dir, file_name), last_mask_with_contours)

    print(f"img have been saved to {img_path}")

# Calculate metrics
ious = [calculate_iou(pred, true) for pred in all_pred_boxes for true in all_true_boxes]

y_true = []
y_pred = []

for true_box in all_true_boxes:
    matched = False
    for pred_box in all_pred_boxes:
        iou = calculate_iou(pred_box, true_box)
        if iou >= 0.5:
            y_true.append(1)
            y_pred.append(1)
            matched = True
            break
    if not matched:
        y_true.append(1)
        y_pred.append(0)

for pred_box in all_pred_boxes:
    matched = False
    for true_box in all_true_boxes:
        iou = calculate_iou(pred_box, true_box)
        if iou >= 0.5:
            matched = True
            break
    if not matched:
        y_true.append(0)
        y_pred.append(1)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

iou_thresholds = np.arange(0.5, 1.0, 0.05)
aps = []
for iou_threshold in iou_thresholds:
    precision, recall = calculate_ap(ious, iou_threshold)
    aps.append(precision)

ap_50_95 = np.mean(aps)
print(f"AP_50_95: {ap_50_95}")

conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)