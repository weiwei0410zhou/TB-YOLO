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
- **DMT_images_processing/**: the data loader and augmentation pipeline
- **TB-YOLO/**: figures/GIFs used in the repo
- **Tumor invasive front detection/**: scripts for metric calculation
- **runs/detect/**: scripts for metric calculation

### Below are the main executable scripts in the repository:
- **color_decon_test_imgs.py**: configuration file
- **test.py**: defines the dataset classes
- **test_wsi.py**: extracts patches from original images
- **train.py**: main metric computation script
- **train_else_dataset.py**: main training script


