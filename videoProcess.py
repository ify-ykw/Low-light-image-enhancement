import cv2 
import os
import torch
import numpy as np
import torchvision
import model
from PIL import Image
import glob
import time
from runpy import run_path
from skimage import img_as_ubyte
import torch.nn.functional as F
import shutil

# 初始化 MIRNetv2 模型
def load_denoise_model():
    parameters = {
        'inp_channels': 3, 'out_channels': 3, 'n_feat': 80,
        'chan_factor': 1.5, 'n_RRG': 4, 'n_MRB': 2, 'height': 3,
        'width': 2, 'bias': False, 'scale': 1, 'task': 'real_denoising'
    }

    model_script_path = "/content/MIRNetv2/basicsr/models/archs/mirnet_v2_arch.py"
    weights_path = "/content/MIRNetv2/Real_Denoising/pretrained_models/real_denoising.pth"

    if not os.path.exists(model_script_path):
        raise FileNotFoundError(f"找不到模型脚本文件: {model_script_path}")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"找不到权重文件: {weights_path}")

    load_arch = run_path(model_script_path)
    model = load_arch['MIRNet_v2'](**parameters).cuda()
    checkpoint = torch.load(weights_path, map_location='cuda')
    model.load_state_dict(checkpoint['params'])
    model.eval()
    return model

# 加载 MIRNetv2 模型
denoise_model = load_denoise_model()

def lowlight(image_path, output_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # 读取原始图像
    data_lowlight = Image.open(image_path)
    original_size = data_lowlight.size  # (width, height)
    
    # **根据原始尺寸确定缩放目标**
    if original_size[0] > original_size[1]:  # 横屏
        target_size = (640, 480)
    else:  # 竖屏
        target_size = (480, 640)

    # **保持原始宽高比缩小**
    data_lowlight.thumbnail(target_size)


    # 转换成 Tensor
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1).cuda().unsqueeze(0)

    # **低光增强**
    DCE_net = model.enhance_net_nopool().cuda()
    DCE_net.load_state_dict(torch.load('/content/Low-light-image-enhancement/snapshots/Epoch_1_Iter_120.pth'))
    with torch.no_grad():
        _, enhanced_image, _ = DCE_net(data_lowlight)

    # 转换回 NumPy
    enhanced_image = enhanced_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    enhanced_image = (enhanced_image * 255).astype(np.uint8)

    # **使用 MIRNetv2 进行降噪**
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
    input_tensor = torch.from_numpy(enhanced_image).float().div(255.).permute(2, 0, 1).unsqueeze(0).cuda()

    with torch.no_grad():
        restored = denoise_model(input_tensor)
        restored = torch.clamp(restored, 0, 1)
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

    # **恢复到原始尺寸**
    restored_image_pil = Image.fromarray(restored)
    restored_image_pil = restored_image_pil.resize(original_size)
    
    # 保存结果
    restored_image_pil.save(output_path)

def process_video(input_video_path, output_video_path, temp_dir):
    original_frames_dir = os.path.join(temp_dir, "lowLight")
    enhanced_frames_dir = os.path.join(temp_dir, "result")

    os.makedirs(original_frames_dir, exist_ok=True)
    os.makedirs(enhanced_frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video: {width}x{height}, FPS: {fps}, Total frames: {frame_count}")

    frame_index = 0
    original_frame_paths = []
    enhanced_frame_paths = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 保存原始帧
        original_frame_path = os.path.join(original_frames_dir, f"frame_{frame_index:05d}.png")
        cv2.imwrite(original_frame_path, frame)
        original_frame_paths.append(original_frame_path)

        # 设定增强后的输出路径
        enhanced_frame_path = os.path.join(enhanced_frames_dir, f"frame_{frame_index:05d}.png")
        enhanced_frame_paths.append(enhanced_frame_path)

        frame_index += 1

    cap.release()
    print("所有帧已提取，开始低光增强并降噪。")

    # **使用 Zero-DCE + MIRNetv2 处理每一帧**
    for i in range(len(original_frame_paths)):
        print(f"Processing: {original_frame_paths[i]}")
        lowlight(original_frame_paths[i], enhanced_frame_paths[i])  # 移除 target_size 参数

    print("所有帧已处理完毕，开始合成新视频。")

    # **将处理后的帧合成视频**
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # if width > height:
      # out = cv2.VideoWriter(output_video_path, fourcc, fps, (640, 480))  # 橫屏輸出
    # else:
      # out = cv2.VideoWriter(output_video_path, fourcc, fps, (480, 640))  # 豎屏輸出

#1
    for enhanced_frame_path in enhanced_frame_paths:
      frame = cv2.imread(enhanced_frame_path)
      out.write(frame)


#1
    out.release()
    print(f"增强视频已保存至 {output_video_path}")

    # **可选: 删除临时文件**
    shutil.rmtree(temp_dir)
    print("临时文件已删除。")


if __name__ == "__main__":
    input_video = "/content/Low-light-image-enhancement/video/lowLightVideo.mp4"
    output_video = "/content/Low-light-image-enhancement/video/enhanceVideo.mp4"
    temp_directory = "/content/Low-light-image-enhancement/temp_frames"

    process_video(input_video, output_video, temp_directory)
