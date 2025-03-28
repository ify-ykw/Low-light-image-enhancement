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

# Initialize MIRNetv2 
def load_denoise_model():
    parameters = {
        'inp_channels': 3, 'out_channels': 3, 'n_feat': 80,
        'chan_factor': 1.5, 'n_RRG': 4, 'n_MRB': 2, 'height': 3,
        'width': 2, 'bias': False, 'scale': 1, 'task': 'real_denoising'
    }

    model_script_path = "/content/MIRNetv2/basicsr/models/archs/mirnet_v2_arch.py"
    weights_path = "/content/MIRNetv2/Real_Denoising/pretrained_models/real_denoising.pth"

    if not os.path.exists(model_script_path):
        raise FileNotFoundError(f"Model script file not found: {model_script_path}")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    load_arch = run_path(model_script_path)
    model = load_arch['MIRNet_v2'](**parameters).cuda()
    checkpoint = torch.load(weights_path, map_location='cuda')
    model.load_state_dict(checkpoint['params'])
    model.eval()
    return model

# Load MIRNetv2 
denoise_model = load_denoise_model()

def lowlight(image_path, output_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Read original image ( RGB )
    data_lowlight = Image.open(image_path).convert("RGB")
    
    # Convert to Tensor
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1).cuda().unsqueeze(0)

    # Low-light enhancement
    DCE_net = model.enhance_net_nopool().cuda()
    DCE_net.load_state_dict(torch.load('/content/Low-light-image-enhancement/snapshots/Epoch_1_Iter_120.pth'))
    with torch.no_grad():
        _, enhanced_image, _ = DCE_net(data_lowlight)

    # Convert back to NumPy
    enhanced_image = enhanced_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    enhanced_image = (enhanced_image * 255).astype(np.uint8)

    # Convert to BGR for OpenCV compatibility
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)

    # Apply MIRNetv2 for denoising
    input_tensor = torch.from_numpy(enhanced_image).float().div(255.).permute(2, 0, 1).unsqueeze(0).cuda()

    with torch.no_grad():
        restored = denoise_model(input_tensor)
        restored = torch.clamp(restored, 0, 1)
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

    # Convert back to RGB to avoid color abnormalities
    restored = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)

    # Convert to PIL.Image
    restored_image_pil = Image.fromarray(restored)

    # Save result
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
    print(f"Video: {width}x{height}, FPS: {fps}, Total frames: {frame_count}")

    frame_index = 0
    original_frame_paths = []
    enhanced_frame_paths = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save original frame
        original_frame_path = os.path.join(original_frames_dir, f"frame_{frame_index:05d}.png")
        cv2.imwrite(original_frame_path, frame)
        original_frame_paths.append(original_frame_path)

        # Set enhanced output path
        enhanced_frame_path = os.path.join(enhanced_frames_dir, f"frame_{frame_index:05d}.png")
        enhanced_frame_paths.append(enhanced_frame_path)

        frame_index += 1

    cap.release()
    print("All frames extracted, starting low-light enhancement and denoising.")

    # Process each frame using Zero-DCE + MIRNetv2
    for i in range(len(original_frame_paths)):
        print(f"Processing: {original_frame_paths[i]}")
        lowlight(original_frame_paths[i], enhanced_frame_paths[i])  

    print("All frames processed, starting video compilation.")

    # Create video with original dimensions
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for enhanced_frame_path in enhanced_frame_paths:
        frame = cv2.imread(enhanced_frame_path)
        out.write(frame)

    out.release()
    print(f"Enhanced video saved to {output_video_path}")

    # Optional: Delete temporary files
    shutil.rmtree(temp_dir)
    print("Temporary files deleted.")

if __name__ == "__main__":
    input_video = "/content/Low-light-image-enhancement/video/lowLightVideo.mp4"
    output_video = "/content/Low-light-image-enhancement/video/enhanceVideo.mp4"
    temp_directory = "/content/Low-light-image-enhancement/temp_frames"

    process_video(input_video, output_video, temp_directory)
