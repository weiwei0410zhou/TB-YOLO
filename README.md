# TB-YOLO: Automated Tumor Budding Detection in Immunohistochemistry Slides of Colorectal Cancer

## This folder contains the implementation of the codes.

## Requirements:
The details of environment such as python package can reference 'requirements.txt'.

## Data:
It can be downloaded at:

google drive: https://drive.google.com/drive/my-drive?hl=zh-CN](https://drive.google.com/file/d/1VsIzDFIb8Qv7nfMlCDYpFT2C3Bbor5TN/view?usp=drive_link


baidu：通过网盘分享的文件：TB-dataset
链接: https://pan.baidu.com/s/1TStAr1AOmqb9EYJm019VXA?pwd=1234 提取码: 1234

## Repository Structure
### Below are the main directories in the repository:
- **DMT_images_processing/**: the DMT method integrates deconvolution, morphological processing, and thresholding to enhance tumor budding detection
- **TB-YOLO/**: contains the main code for the TB-YOLO model, including training and inference scripts
- **Tumor invasive front detection/**: scripts for detecting the invasive front of tumors 
- **runs/detect/**: stores the results of detection runs, including logs and output files

### Below are the main executable scripts in the repository:
- **color_decon_test_imgs.py**: script for testing color deconvolution on images
- **color_decon_test_imgs_index.py**： script for testing color deconvolution index on images
- **test.py**: the main testing script for the TB-YOLO model
- **test_wsi.py**: directly tests a whole slide image (WSI) by processing it and extracting relevant patches
- **train.py**: the main training script for the TB-YOLO model
- **train_else_dataset.py**: main training script for else dataset

## Start to train
1.set up environment.

2.configure the dataset path and model.yaml file in 'train.py'.

3.run 'train.py'.

## Test model
1.run 'test.py' to generate prediction results.

2.calculate related evaluation metircs.

## Process a large-scale pathology image such as WSI
run 'test_wsi.py'

## Citation
@misc{yolo-project-for_dataset,
  author = {Tumor detection YOLO},
  title = {Yolo-project-for Dataset},
  year = {2024},
  note = {Dataset available at Roboflow Universe, accessed: 2025-01-03},
  howpublished = {\url{https://universe.roboflow.com/tumor-detection-yolo/yolo-project-for}}
}

@misc{colon_cancer_new_dataset,
  author = {school},
  title = {colon\_cancer\_new Dataset},
  year = {2023},
  note = {Dataset available at Roboflow Universe, accessed: 2025-01-03},
  howpublished = {\url{https://universe.roboflow.com/school-sd6by/colon_cancer_new}}
}


