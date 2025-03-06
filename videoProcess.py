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

# 初始化 MIRNetv2 模型
def load_denoise_model():
    parameters = {
        'inp_channels':3, 'out_channels':3, 'n_feat':80,
        'chan_factor':1.5, 'n_RRG':4, 'n_MRB':2, 'height':3,
        'width':2, 'bias':False, 'scale':1, 'task': 'real_denoising'
    }

    # **使用绝对路径**
    model_script_path = "/content/MIRNetv2/basicsr/models/archs/mirnet_v2_arch.py"
    weights_path = "/content/MIRNetv2/Real_Denoising/pretrained_models/real_denoising.pth"

    if not os.path.exists(model_script_path):
        raise FileNotFoundError(f"找不到模型脚本文件: {model_script_path}")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"找不到权重文件: {weights_path}")

    # **改成使用绝对路径**
    load_arch = run_path(model_script_path)

    model = load_arch['MIRNet_v2'](**parameters).cuda()
    checkpoint = torch.load(weights_path, weights_only=True)
    model.load_state_dict(checkpoint['params'])
    model.eval()
    return model

# 加載 MIRNetv2 模型
denoise_model = load_denoise_model()

def lowlight(image_path, output_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_lowlight = Image.open(image_path)
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1).cuda().unsqueeze(0)

    # 加載 Zero-DCE 模型
    DCE_net = model.enhance_net_nopool().cuda()
    DCE_net.load_state_dict(torch.load('/content/Low-light-image-enhancement/snapshots/Epoch_1_Iter_120.pth'))

    # 低光增強
    with torch.no_grad():
        _, enhanced_image, _ = DCE_net(data_lowlight)

    # 轉換為 NumPy 格式
    enhanced_image = enhanced_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    enhanced_image = (enhanced_image * 255).astype(np.uint8)

    # 使用 MIRNetv2 進行降噪
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
    input_tensor = torch.from_numpy(enhanced_image).float().div(255.).permute(2,0,1).unsqueeze(0).cuda()

    with torch.no_grad():
        restored = denoise_model(input_tensor)
        restored = torch.clamp(restored, 0, 1)
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

    # 儲存降噪後的圖像
    restored_image_pil = Image.fromarray(restored)
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

        # 保存原始幀
        original_frame_path = os.path.join(original_frames_dir, f"frame_{frame_index:05d}.png")
        cv2.imwrite(original_frame_path, frame)
        original_frame_paths.append(original_frame_path)

        # 設定增強後的輸出路徑
        enhanced_frame_path = os.path.join(enhanced_frames_dir, f"frame_{frame_index:05d}.png")
        enhanced_frame_paths.append(enhanced_frame_path)

        frame_index += 1

    cap.release()
    print("所有幀已擷取，開始進行低光增強並降噪。")

    # 使用 Zero-DCE + MIRNetv2 處理每一幀
    for i in range(len(original_frame_paths)):
        print(f"Processing: {original_frame_paths[i]}")
        lowlight(original_frame_paths[i], enhanced_frame_paths[i])

    print("所有幀已處理完畢，開始合成新影片。")

    # 將處理後的幀合成影片
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for enhanced_frame_path in enhanced_frame_paths:
        frame = cv2.imread(enhanced_frame_path)
        out.write(frame)

    out.release()
    print(f"降噪後的增強影片已儲存至 {output_video_path}")

    # **可選: 刪除 temp_frames 來節省空間**
    shutil.rmtree(temp_dir)
    print("臨時文件已刪除。")


if __name__ == "__main__":
    input_video = "/content/Low-light-image-enhancement/video/lowLightVideo.mp4"  # input video
    output_video = "/content/Low-light-image-enhancement/video/enhanceVideo.mp4"  # result
    temp_directory = "/content/Low-light-image-enhancement/temp_frames"

    process_video(input_video, output_video, temp_directory)
