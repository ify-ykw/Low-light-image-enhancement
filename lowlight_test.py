import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time


def lowlight(image_path, output_folder):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} does not exist!")
        return

    if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        print(f"Skipping non-image file: {image_path}")
        return
    
    data_lowlight = Image.open(image_path)
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1).cuda().unsqueeze(0)

    DCE_net = model.enhance_net_nopool().cuda()
    DCE_net.load_state_dict(torch.load('/content/Low-light-image-enhancement/snapshots/Epoch_1_Iter_120.pth'))

    start = time.time()
    _, enhanced_image, _ = DCE_net(data_lowlight)
    end_time = time.time() - start

    print(f"Enhancement Time: {end_time:.3f} sec")
    
    image_filename = os.path.basename(image_path)
    result_path = os.path.join(output_folder, image_filename)

    os.makedirs(output_folder, exist_ok=True)
    torchvision.utils.save_image(enhanced_image, result_path)
    print(f"Saved: {result_path}")

if __name__ == '__main__':
    test_data_path = '/content/Low-light-image-enhancement/__data/test_data'
    result_base_path = '/content/Low-light-image-enhancement/__data/result'
    
    if not os.path.exists(test_data_path):
        print(f"Error: {test_data_path} does not exist!")
        exit(1)
    
    subfolders = [f for f in os.listdir(test_data_path) if os.path.isdir(os.path.join(test_data_path, f))]
    
    if not subfolders:
        print("No subfolders found in test_data!")
        exit(1)
    
    for subfolder in subfolders:
        input_folder = os.path.join(test_data_path, subfolder)
        output_folder = os.path.join(result_base_path, subfolder)
        
        image_list = glob.glob(os.path.join(input_folder, '*'))
        
        for image in image_list:
            print(f"Processing: {image}")
            lowlight(image, output_folder)
    
    print("All images processed!")

