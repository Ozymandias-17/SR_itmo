import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import vgg19, VGG19_Weights
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
from piq import DISTS

from nn_arch.RRDBNet_arch import RRDBNet
from nn_arch.VGG_feat import VGGFeatureExtractor
from nn_arch.Discriminator import Discriminator
from dataloader import DF2KDataset


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Train on device: {device}")

    # Инициализация моделей
    netG = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23).to(device)
    netD = Discriminator().to(device)
    vgg_extractor = VGGFeatureExtractor().to(device)
    vgg_extractor.eval()
    for p in vgg_extractor.parameters():
        p.requires_grad = False
    
    # netG.load_state_dict(torch.load('pretrained_psnr_esrgan.pth'))

    # Оптимизаторы
    optimizer_G = optim.AdamW(netG.parameters(), lr=args.lr)
    optimizer_D = optim.AdamW(netD.parameters(), lr=args.lr)

    scheduler_G = ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=10)
    scheduler_D = ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=10)

    # Функции потерь
    criterion_pixel = nn.L1Loss().to(device)
    criterion_adv = nn.BCEWithLogitsLoss().to(device)
    criterion_perceptual = nn.L1Loss().to(device) # L1 между фичами VGG

    # Инициализация метрик
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_ = lpips.LPIPS(net='vgg').to(device) # LPIPS требует значения в диапазоне [-1, 1]
    lpips_.eval()
    for p in lpips_.parameters():
        p.requires_grad = False
        
    dists_metric = DISTS().to(device)
    dists_metric.eval()
    for p in dists_metric.parameters():
        p.requires_grad = False

    lambda_perceptual = 1.0
    lambda_adv = 5e-3
    lambda_pixel = 1e-2

    train_loader = DataLoader(DF2KDataset('train', scale=args.scale, lr_patch_size=args.lr_patch_size), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(DF2KDataset('val', scale=args.scale), batch_size=1, shuffle=False)

    # Логирование
    writer = SummaryWriter(log_dir=f'./logs/esrgan_{args.mode}')
    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    best_lpips = float('inf')

    total_training_start_time = time.perf_counter()

    for epoch in range(args.epochs):
        netG.train()
        netD.train()

        train_loss = 0.0
        train_psnr, train_ssim = 0.0, 0.0
        total_fps = 0.0
        epoch_start_time = time.perf_counter()
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")
        for lr_imgs, hr_imgs in loop:
            step_start_time = time.perf_counter()

            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            sr_imgs = netG(lr_imgs)

            with torch.no_grad():
                train_psnr += psnr_metric(torch.clamp(sr_imgs, 0, 1), hr_imgs).item()
                train_ssim += ssim_metric(torch.clamp(sr_imgs, 0, 1), hr_imgs).item()

            # дискриминатор
            optimizer_D.zero_grad()
            
            pred_real = netD(hr_imgs)
            pred_fake = netD(sr_imgs.detach())
            
            loss_d_real = criterion_adv(pred_real - torch.mean(pred_fake), torch.ones_like(pred_real))
            loss_d_fake = criterion_adv(pred_fake - torch.mean(pred_real), torch.zeros_like(pred_fake))
            loss_D = (loss_d_real + loss_d_fake) / 2
            
            loss_D.backward()
            optimizer_D.step()

            # генератор
            optimizer_G.zero_grad()
            pred_fake_g = netD(sr_imgs)
            pred_real_g = netD(hr_imgs).detach()

            # Pixel Loss (L1)
            loss_pixel = criterion_pixel(sr_imgs, hr_imgs)
            
            # Perceptual Loss (VGG)
            real_features = vgg_extractor(hr_imgs).detach()
            fake_features = vgg_extractor(sr_imgs)
            loss_percep = criterion_perceptual(fake_features, real_features)
            
            # Relativistic Adversarial Loss
            loss_g_real = criterion_adv(pred_real_g - torch.mean(pred_fake_g), torch.zeros_like(pred_real_g))
            loss_g_fake = criterion_adv(pred_fake_g - torch.mean(pred_real_g), torch.ones_like(pred_fake_g))
            loss_adv = (loss_g_real + loss_g_fake) / 2
            
            # лосс Генератора
            loss_G = (lambda_pixel * loss_pixel) + (lambda_perceptual * loss_percep) + (lambda_adv * loss_adv)
            
            loss_G.backward()
            optimizer_G.step()

            train_loss += loss_G.item()

            step_time = time.perf_counter() - step_start_time
            current_fps = args.batch_size / step_time
            total_fps += current_fps

            loop.set_postfix(L_G=loss_G.item(), L_D=loss_D.item(), FPS=f"{current_fps:.1f}")

        avg_fps = total_fps / len(train_loader)
        avg_train_loss = train_loss / len(train_loader)
        avg_train_psnr = train_psnr / len(train_loader)
        avg_train_ssim = train_ssim / len(train_loader)
        epoch_duration = time.perf_counter() - epoch_start_time

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Performance/Avg_FPS', avg_fps, epoch)
        writer.add_scalar('Performance/Epoch_Duration_sec', epoch_duration, epoch)

        print(f"\nAvg FPS = {avg_fps:.2f} | Epoch time = {epoch_duration:.2f} sec.")

        # Валидация
        netG.eval()
        netD.eval()

        val_psnr, val_ssim, val_lpips, val_dists = 0.0, 0.0, 0.0, 0.0
        val_loss = 0.0

        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                sr_imgs = netG(lr_imgs)

                pred_fake_g = netD(sr_imgs)
                pred_real_g = netD(hr_imgs)

                loss_pixel = criterion_pixel(sr_imgs, hr_imgs)

                real_features = vgg_extractor(hr_imgs)
                fake_features = vgg_extractor(sr_imgs)
                loss_percep = criterion_perceptual(fake_features, real_features)

                loss_g_real = criterion_adv(pred_real_g - torch.mean(pred_fake_g), torch.zeros_like(pred_real_g))
                loss_g_fake = criterion_adv(pred_fake_g - torch.mean(pred_real_g), torch.ones_like(pred_fake_g))
                loss_adv = (loss_g_real + loss_g_fake) / 2
                loss_G_val = (lambda_pixel * loss_pixel) + (lambda_perceptual * loss_percep) + (lambda_adv * loss_adv)

                val_loss += loss_G_val.item()

                sr_imgs_clamped = torch.clamp(sr_imgs, 0, 1) # [0, 1]

                val_psnr += psnr_metric(sr_imgs_clamped, hr_imgs).item()
                val_ssim += ssim_metric(sr_imgs_clamped, hr_imgs).item()
                val_dists += dists_metric(sr_imgs_clamped, hr_imgs).item()
                val_lpips += lpips_(sr_imgs_clamped * 2 - 1, hr_imgs * 2 - 1).item()  # [-1, 1]

        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)
        val_lpips /= len(val_loader)
        val_dists /= len(val_loader)

        print(f"Val - PSNR: {val_psnr:.2f} | Val - SSIM: {val_ssim:.2f} | LPIPS: {val_lpips:.4f} | DISTS: {val_dists:.4f}")

        scheduler_G.step(val_lpips)
        scheduler_D.step(val_lpips)

        current_lr_G = optimizer_G.param_groups[0]['lr']
        current_lr_D = optimizer_D.param_groups[0]['lr']

        # Сохранение по лучшей перцептивной метрике
        if val_lpips < best_lpips:
            best_lpips = val_lpips
            torch.save(netG.state_dict(), f'./checkpoints/esrgan_{args.mode}.pth')
            print("Best model is saved.")

        # Tensorboard
        writer.add_scalars('Loss/Total', {'train': avg_train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Metric/PSNR', {'train': avg_train_psnr, 'val': val_psnr}, epoch)
        writer.add_scalars('Metric/SSIM', {'train': avg_train_ssim, 'val': val_ssim}, epoch)
        writer.add_scalar('Metric/Val_LPIPS', val_lpips, epoch)
        writer.add_scalar('Metric/Val_DISTS', val_dists, epoch)
        writer.add_scalar('LR/Generator', current_lr_G, epoch)
        writer.add_scalar('LR/Discriminator', current_lr_D, epoch)

    total_training_time = time.perf_counter() - total_training_start_time
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)

    writer.add_scalar('Performance/Total_Training_Time_sec', total_training_time, 0)
    writer.close()

    print(f"Training is over, total time: {hours}h {minutes}m {seconds}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ESRGAN Training")

    parser.add_argument('--mode', type=str, default='orig', 
                        help='Model: orig, rgb2, yuv (default: orig)')
    parser.add_argument('--epochs', type=int, default=200, 
                        help='Number of epochs (default: 200)')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='default: 32')
    parser.add_argument('--lr_patch_size', type=int, default=64, 
                        help='Patch size of LR images (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Initial Learning Rate (default: 1e-4)')
    parser.add_argument('--scale', type=int, default=2, 
                        help='Scale parametr (default: 2)')
    
    parsed_args = parser.parse_args()
    
    train(parsed_args)