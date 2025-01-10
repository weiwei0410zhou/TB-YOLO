# TB-YOLO: Automated Tumor Budding Detection in Immunohistochemistry Slides of Colorectal Cancer

## This folder contains the implementation of the codes.

## Requirements:
The details of environment such as python package can reference 'requirements.txt'.

## Data:
It can be downloaded at:

Google Drive:  
[https://drive.google.com/file/d/1VsIzDFIb8Qv7nfMlCDYpFT2C3Bbor5TN/view](https://drive.google.com/file/d/1VsIzDFIb8Qv7nfMlCDYpFT2C3Bbor5TN/view)  

Baidu Netdisk:  
[https://pan.baidu.com/s/1o3A-uM2VZtpIRGKTwqpZeA?pwd=iraq](https://pan.baidu.com/s/1o3A-uM2VZtpIRGKTwqpZeA?pwd=iraq) 提取码: `iraq`
runs/detect
## Repository Structure
### Below are the main directories in the repository:
- **DMT_images_processing/**: The DMT method integrates deconvolution, morphological processing, and thresholding to enhance tumor budding detection
- **TB-YOLO/**: Contains the main code for the TB-YOLO model, including training and inference scripts
- **Tumor invasive front detection/**: Scripts for detecting the invasive front of tumors and calculating relevant metrics
- **runs/detect/**: Stores the results of detection runs, including logs and output files

### Below are the main executable scripts in the repository:
- **color_decon_test_imgs.py**: Script for testing color deconvolution on images
- **test.py**: main testing script
- **test_wsi.py**: Extracts patches from whole slide images (WSI) for processing
- **train.py**: main training script
- **train_else_dataset.py**: main training script for else dataset


