import cv2
import os
import torch
import numpy as np
import torchvision
import model
from PIL import Image
import glob
import time

def lowlight(image_path, output_path):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_lowlight = Image.open(image_path)
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1).cuda().unsqueeze(0)

    # load model
    DCE_net = model.enhance_net_nopool().cuda()
    DCE_net.load_state_dict(torch.load('/content/Zero-DCE/Zero-DCE_code/snapshots/Epoch_1_Iter_120.pth'))

    # enhance
    with torch.no_grad():
        _, enhanced_image, _ = DCE_net(data_lowlight)

    # Convert to numpy format
    enhanced_image = enhanced_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    enhanced_image = (enhanced_image * 255).astype(np.uint8)

    # Save the enhanced image
    enhanced_image_pil = Image.fromarray(enhanced_image)
    enhanced_image_pil.save(output_path)

def process_video(input_video_path, output_video_path, temp_dir):
    # enhance each frame and save as a new video

    original_frames_dir = os.path.join(temp_dir, "lowLight")
    enhanced_frames_dir = os.path.join(temp_dir, "result")

    os.makedirs(original_frames_dir, exist_ok=True)
    os.makedirs(enhanced_frames_dir, exist_ok=True)

    # read video
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

        # original frame
        original_frame_path = os.path.join(original_frames_dir, f"frame_{frame_index:05d}.png")
        cv2.imwrite(original_frame_path, frame)
        original_frame_paths.append(original_frame_path)

        # enhance frame
        enhanced_frame_path = os.path.join(enhanced_frames_dir, f"frame_{frame_index:05d}.png")
        enhanced_frame_paths.append(enhanced_frame_path)

        frame_index += 1

    cap.release()
    print("All frames have been extracted, low-light enhancement processing begins.")

    # Use the lowlight method for each frame
    for i in range(len(original_frame_paths)):
        print(f"processing: {original_frame_paths[i]}")
        lowlight(original_frame_paths[i], enhanced_frame_paths[i])  # Process and save the enhanced image

    print("All frames have been enhanced, video synthesis begins.")

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for enhanced_frame_path in enhanced_frame_paths:
        frame = cv2.imread(enhanced_frame_path)
        out.write(frame)

    out.release()
    print(f"Enhanced video saved to {output_video_path}")

if __name__ == "__main__":
    input_video = "/content/Zero-DCE/Zero-DCE_code/video/lowLightVideo.mp4"  # input video
    output_video = "/content/Zero-DCE/Zero-DCE_code/video/enhanceVideo.mp4"  # result
    temp_directory = "/content/Zero-DCE/Zero-DCE_code/temp_frames"

    process_video(input_video, output_video, temp_directory)
