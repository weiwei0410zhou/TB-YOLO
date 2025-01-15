import numpy as np
import cv2
import os
from openslide import OpenSlide
from scipy.ndimage import binary_fill_holes
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity
from skimage import img_as_ubyte

# 文件路径设置
thumbnail_dir = "/home/zww/2t_ssd1_ww/zww/zww/CK_data_imgs_True_WSI/thumbnail"
save_path = "/home/zww/2t_ssd1_ww/zww/zww/CK_data_color_decon_imgs_True_WSI"

mask_dir = os.path.join(save_path, "mask")
output_dir = os.path.join(save_path, "output")
npy_dir = os.path.join(save_path, "npy")  # 新增路径用于保存npy文件
overlay_mask_dir = os.path.join(save_path, "overlay_mask")
overlay_filter_img_dir = os.path.join(save_path, "overlay_filter_img")

os.makedirs(mask_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(npy_dir, exist_ok=True)  # 创建npy文件目录
os.makedirs(overlay_mask_dir, exist_ok=True)
os.makedirs(overlay_filter_img_dir, exist_ok=True)

# 颜色反卷积函数
def color_deconvolution(image, base_name):
    ihc_hed = rgb2hed(image)
    hematoxylin_1 = ihc_hed[:, :, 0]
    eosin_1 = ihc_hed[:, :, 1]
    dab_1 = ihc_hed[:, :, 2]

    # 对每个通道进行适当的缩放
    hematoxylin_1 = rescale_intensity(hematoxylin_1, out_range=(0, 255)).astype(np.uint8)
    eosin_1 = rescale_intensity(eosin_1, out_range=(0, 255)).astype(np.uint8)
    dab_1 = rescale_intensity(dab_1, out_range=(0, 255)).astype(np.uint8)

    hematoxylin = rescale_intensity(hematoxylin_1, out_range=(0, 1))
    eosin = rescale_intensity(eosin_1, out_range=(0, 1))
    dab = rescale_intensity(dab_1, out_range=(0, 1))

    hematoxylin = img_as_ubyte(hematoxylin)
    eosin = img_as_ubyte(eosin)
    dab = img_as_ubyte(dab)

    return dab

# 生成mask的函数
def generate_mask(hematoxylin_image):
    _, mask = cv2.threshold(hematoxylin_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    """contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            cv2.drawContours(mask, [contour], -1, 0, -1)"""
    
    mask = binary_fill_holes(mask // 255).astype(np.uint8) * 255
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = binary_fill_holes(mask // 255).astype(np.uint8) * 255

    num_labels, labels = cv2.connectedComponents(mask, connectivity=4)
    region_sizes = np.bincount(labels.flatten())

    # 检查是否有足够的连通区域
    if len(region_sizes) <= 1:
        return mask, mask  # 如果没有足够的连通区域，直接返回原mask

    region_sizes_sorted = sorted(region_sizes, reverse=True)
    filter_image_connected = np.zeros_like(mask)
    max_region_size = region_sizes_sorted[1]  # 第0个是背景
    threshold_size = max_region_size / 20

    for region_label in range(1, num_labels):
        if region_sizes[region_label] > threshold_size:
            filter_image_connected[labels == region_label] = 255   
    return mask, filter_image_connected

# 叠加轮廓的函数
def overlay_contours(image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image = np.array(image)
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
    return image_with_contours

# 处理图像
img_dirs = os.listdir(thumbnail_dir)

for name in img_dirs:
    base_name = os.path.splitext(name)[0]
    mask_save_path = os.path.join(mask_dir, name)
    output_save_path = os.path.join(output_dir, name)
    npy_save_path = os.path.join(npy_dir, name.replace("png", "npy"))
    overlay_mask_path = os.path.join(overlay_mask_dir, name)
    overlay_filter_img_path = os.path.join(overlay_filter_img_dir, name)

    image_path = os.path.join(thumbnail_dir, name)
    image = cv2.imread(image_path)
    dab_image = color_deconvolution(image, base_name)
    mask, filter_img = generate_mask(dab_image)
    mask_with_contours = overlay_contours(image, mask)
    filter_img_with_contours = overlay_contours(image, filter_img)
    
    cv2.imwrite(overlay_mask_path, mask_with_contours)
    cv2.imwrite(overlay_filter_img_path, filter_img_with_contours)
    cv2.imwrite(mask_save_path, mask)
    cv2.imwrite(output_save_path, filter_img)
    np.save(npy_save_path, filter_img)

    print("finish")
print(f"Processing complete. Results saved in the specified directories.")