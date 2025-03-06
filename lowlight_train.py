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
import Myloss
import numpy as np
from torchvision import transforms

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DCE_net = model.enhance_net_nopool().to(device)
    DCE_net.apply(weights_init)
    
    if config.load_pretrain:
        try:
            DCE_net.load_state_dict(torch.load(config.pretrain_dir))
            print(f"Loaded pre-trained model from {config.pretrain_dir}")
        except Exception as e:
            print(f"Error loading pre-trained model: {e}")
            return
    
    # Debugging: Check if the path exists and list files
    if not os.path.exists(config.lowlight_images_path):
        print(f"Error: The path {config.lowlight_images_path} does not exist.")
        return
    else:
        print(f"Path {config.lowlight_images_path} exists.")
        print("Listing files in the directory:")
        print(os.listdir(config.lowlight_images_path))
    
    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)
    print(f"Total training examples: {len(train_dataset)}")
    
    if len(train_dataset) == 0:
        print("Error: No training examples found. Please check the dataset path and ensure it contains images.")
        return
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    L_color = Myloss.L_color()
    L_spa = Myloss.L_spa()
    L_exp = Myloss.L_exp(16, 0.6)
    L_TV = Myloss.L_TV()

    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    DCE_net.train()

    for epoch in range(config.num_epochs):
        for iteration, img_lowlight in enumerate(train_loader):
            img_lowlight = img_lowlight.to(device)
            enhanced_image_1, enhanced_image, A = DCE_net(img_lowlight)
            Loss_TV = 200 * L_TV(A)
            loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))
            loss_col = 5 * torch.mean(L_color(enhanced_image))
            loss_exp = 10 * torch.mean(L_exp(enhanced_image))
            loss = Loss_TV + loss_spa + loss_col + loss_exp
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(DCE_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            if (iteration + 1) % config.display_iter == 0:
                print(f"Epoch [{epoch+1}/{config.num_epochs}], Iteration [{iteration+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            if (iteration + 1) % config.snapshot_iter == 0:
                snapshot_path = os.path.join(config.snapshots_folder, f"Epoch_{epoch+1}_Iter_{iteration+1}.pth")
                torch.save(DCE_net.state_dict(), snapshot_path)
                print(f"Saved snapshot: {snapshot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lowlight_images_path', type=str, default="/content/Zero-DCE/Zero-DCE_code/data/train_data")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="/content/Zero-DCE/Zero-DCE_code/snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="/content/Zero-DCE/Zero-DCE_code/snapshots/Epoch99.pth")
    parser.add_argument('--num_epochs', type=int, default=400, help='Number of epochs to train the model')

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.makedirs(config.snapshots_folder)

    train(config)
