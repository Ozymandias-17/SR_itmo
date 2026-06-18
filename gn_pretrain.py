import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from nn_arch.RRDBNet_arch import RRDBNet
from dataloader import DF2KDataset

def train_psnr(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PSNR Pre-training on device: {device}")

    if args.mode == 'orig':
        nf_base, gc_base = 64, 32
        print(f"Mode 'orig': standard channels (nf={nf_base}, gc={gc_base}).")
    elif args.mode == 'rgb2':
        nf_base, gc_base = 32, 16
        print(f"Mode 'rgb2': halved channels (nf={nf_base}, gc={gc_base}).")

    netG = RRDBNet(in_nc=3, out_nc=3, nf=nf_base, nb=23, gc=gc_base).to(device)

    optimizer_G = optim.AdamW(netG.parameters(), lr=args.lr)
    scheduler_G = ReduceLROnPlateau(optimizer_G, mode='max', factor=0.5, patience=10)

    # Pixel Loss
    criterion_pixel = nn.L1Loss().to(device)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    train_loader = DataLoader(DF2KDataset('train', scale=args.scale, lr_patch_size=args.lr_patch_size), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(DF2KDataset('val', scale=args.scale), batch_size=1, shuffle=False)

    writer = SummaryWriter(log_dir=f'./logs/gn_base_{args.mode}')
    os.makedirs('./checkpoints', exist_ok=True)

    best_psnr = 0.0
    total_training_start_time = time.perf_counter()

    for epoch in range(args.epochs):
        netG.train()
        train_loss, train_psnr, train_ssim = 0.0, 0.0, 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")
        for lr_imgs, hr_imgs in loop:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            optimizer_G.zero_grad()
            sr_imgs = netG(lr_imgs)

            # L1 loss
            loss_pixel = criterion_pixel(sr_imgs, hr_imgs)
            
            loss_pixel.backward()
            optimizer_G.step()

            train_loss += loss_pixel.item()

            with torch.no_grad():
                sr_clamped = torch.clamp(sr_imgs, 0, 1)
                train_psnr += psnr_metric(sr_clamped, hr_imgs).item()

            loop.set_postfix(Loss=loss_pixel.item())

        avg_train_loss = train_loss / len(train_loader)
        avg_train_psnr = train_psnr / len(train_loader)

        # Валидация
        netG.eval()
        val_psnr, val_ssim, val_loss = 0.0, 0.0, 0.0

        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                sr_imgs = netG(lr_imgs)

                val_loss += criterion_pixel(sr_imgs, hr_imgs).item()
                
                sr_clamped = torch.clamp(sr_imgs, 0, 1)
                val_psnr += psnr_metric(sr_clamped, hr_imgs).item()
                val_ssim += ssim_metric(sr_clamped, hr_imgs).item()

        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)

        print(f"Val - PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f}")

        scheduler_G.step(val_psnr)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(netG.state_dict(), f'./checkpoints/pretrained_gn_{args.mode}.pth')
            print("Best PSNR model saved.")

        # Tensorboard
        writer.add_scalars('Loss/Pixel', {'train': avg_train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Metric/PSNR', {'train': avg_train_psnr, 'val': val_psnr}, epoch)
        writer.add_scalar('LR/Generator', optimizer_G.param_groups[0]['lr'], epoch)

    writer.close()
    print("PSNR pre-training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PSNR Pre-Training")
    parser.add_argument('--mode', type=str, default='orig', help='Model: orig, rgb2, yuv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr_patch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--scale', type=int, default=2)
    
    parsed_args = parser.parse_args()
    train_psnr(parsed_args)