# TB-YOLO: Automated Tumor Budding Detection in Immunohistochemistry Slides of Colorectal Cancer

## This folder contains the implementation of the codes.

## Requirements:
The details of environment such as python package can reference 'environment.txt'.

## Data:
It can be downloaded at:

Google Drive:  
[https://drive.google.com/file/d/1VsIzDFIb8Qv7nfMlCDYpFT2C3Bbor5TN/view](https://drive.google.com/file/d/1VsIzDFIb8Qv7nfMlCDYpFT2C3Bbor5TN/view)  

Baidu Netdisk:  
[https://pan.baidu.com/s/1o3A-uM2VZtpIRGKTwqpZeA?pwd=iraq](https://pan.baidu.com/s/1o3A-uM2VZtpIRGKTwqpZeA?pwd=iraq) 提取码: `iraq`

## Repository Structure
### Below are the main directories in the repository:
- **dataloader/**: the data loader and augmentation pipeline
- **docs/**: figures/GIFs used in the repo
- **metrics/**: scripts for metric calculation
- **misc/**: utils that are
- **models/**: model definition, along with the main run step and hyperparameter settings
- **run_utils/**: defines the train/validation loop and callbacks

### Below are the main executable scripts in the repository:
- **config.py**: configuration file
- **dataset.py**: defines the dataset classes
- **extract_patches.py**: extracts patches from original images
- **compute_stats.py**: main metric computation script
- **run_train.py**: main training script
- **run_infer.py**: main inference script for tile and WSI processing
- **convert_chkpt_tf2pytorch**: convert tensorflow .npz model trained in original repository to pytorch supported .tar format


