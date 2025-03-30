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


def lowlight(image_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    if not os.path.exists(image_path):
        print(f" error: {image_path} does not exist！")
        return

    if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        print(f" Skip non-image files: {image_path}")
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

    print(f" Enhancement Time: {end_time:.3f} sec")
    print(f" Enhanced Image Shape: {enhanced_image.shape}")
    print(f" Max Pixel Value: {enhanced_image.max()}")

    image_filename = os.path.basename(image_path)
    result_path = os.path.join('/content/Low-light-image-enhancement/__data/result', image_filename)

    if not os.path.exists(os.path.dirname(result_path)):
        os.makedirs(os.path.dirname(result_path))

    if enhanced_image is not None:
        print(f" Saving image to {result_path}")
        torchvision.utils.save_image(enhanced_image, result_path)
    else:
        print(" Error: enhanced_image is None, skipping save!")

if __name__ == '__main__':
    with torch.no_grad():
        filePath = '/content/Low-light-image-enhancement/__data/test_data'
        if not os.path.exists(filePath):
            print(f" Error: {filePath} does not exist!")
            exit(1)

        file_list = os.listdir(filePath)
        if not file_list:
            print(" No files found in test_data!")
            exit(1)

        for file_name in file_list:
            test_list = glob.glob(os.path.join(filePath, file_name, "*"))
            for image in test_list:
                print(f" Processing: {image}")
                lowlight(image)

    print(" All images processed!")
