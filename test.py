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

def test_index(model_path):
    model = YOLO(model_path)
    
    # Validate the model
    metrics = model.val(split='test')  # no arguments needed, dataset and settings remembered
    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps   # a list contains map50-95 of each category

if __name__ == '__main__':
    test_index('runs/detect/train/weights/best.pt')