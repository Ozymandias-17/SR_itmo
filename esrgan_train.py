import os
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
    print(f"Обучение на устройстве: {device}")

    # Инициализация моделей
    netG = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23).to(device)
    netD = Discriminator().to(device)
    vgg_extractor = VGGFeatureExtractor().to(device)
    vgg_extractor.eval()
    for p in vgg_extractor.parameters():
        p.requires_grad = False
    
    # netG.load_state_dict(torch.load('pretrained_psnr_esrgan.pth'))

    # Оптимизаторы
    optimizer_G = optim.AdamW(netG.parameters(), lr=args.initial_lr)
    optimizer_D = optim.AdamW(netD.parameters(), lr=args.initial_lr)

    scheduler_G = ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=5)
    scheduler_D = ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=5)

    # Функции потерь
    criterion_pixel = nn.L1Loss().to(device)
    criterion_adv = nn.BCEWithLogitsLoss().to(device)
    criterion_perceptual = nn.L1Loss().to(device) # L1 между фичами VGG

    # Инициализация метрик
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_ = lpips.LPIPS(net='vgg').to(device) # LPIPS требует значения в диапазоне [-1, 1]
    dists_metric = DISTS().to(device)

    lambda_perceptual = 1.0
    lambda_adv = 5e-3
    lambda_pixel = 1e-2

    train_loader = DataLoader(DF2KDataset('train', scale=args.scale, lr_patch_size=args.lr_patch_size), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(DF2KDataset('val', scale=args.scale), batch_size=1, shuffle=False)

    # Логирование
    writer = SummaryWriter(log_dir='./logs/esrgan_experiment')
    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    best_lpips = float('inf')

    for epoch in range(args.epochs):
        netG.train()
        netD.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")
        for lr_imgs, hr_imgs in loop:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            # дискриминатор
            optimizer_D.zero_grad()
            
            sr_imgs = netG(lr_imgs).detach()
            
            pred_real = netD(hr_imgs)
            pred_fake = netD(sr_imgs)
            
            loss_d_real = criterion_adv(pred_real, torch.ones_like(pred_real))
            loss_d_fake = criterion_adv(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_d_real + loss_d_fake) / 2
            
            loss_D.backward()
            optimizer_D.step()

            # генератор
            optimizer_G.zero_grad()
            sr_imgs = netG(lr_imgs)
            pred_fake_g = netD(sr_imgs)

            # Pixel Loss (L1)
            loss_pixel = criterion_pixel(sr_imgs, hr_imgs)
            
            # Perceptual Loss (VGG)
            real_features = vgg_extractor(hr_imgs).detach()
            fake_features = vgg_extractor(sr_imgs)
            loss_percep = criterion_perceptual(fake_features, real_features)
            
            loss_adv = criterion_adv(pred_fake_g, torch.ones_like(pred_fake_g))
            
            # лосс Генератора
            loss_G = (lambda_pixel * loss_pixel) + (lambda_perceptual * loss_percep) + (lambda_adv * loss_adv)
            
            loss_G.backward()
            optimizer_G.step()

            train_loss += loss_G.item()

            loop.set_postfix(L_G=loss_G.item(), L_D=loss_D.item())

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/Train (L2)', avg_train_loss, epoch)

        # Валидация
        netG.eval()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        val_psnr, val_ssim, val_lpips, val_dists = 0.0, 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                sr_imgs = torch.clamp(netG(lr_imgs), 0, 1) # [0, 1]

                val_psnr += psnr_metric(sr_imgs, hr_imgs).item()
                val_ssim += ssim_metric(sr_imgs, hr_imgs).item()
                val_dists += dists_metric(sr_imgs, hr_imgs).item()
                val_lpips += lpips_(sr_imgs * 2 - 1, hr_imgs * 2 - 1).item()  # [-1, 1]

        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)
        val_lpips /= len(val_loader)
        val_dists /= len(val_loader)

        print(f"Val - PSNR: {val_psnr:.2f} | Val - SSSIM: {val_ssim:.2f} | LPIPS: {val_lpips:.4f} | DISTS: {val_dists:.4f}")

        scheduler_G.step(val_lpips)
        scheduler_D.step(val_lpips)

        current_lr_G = optimizer_G.param_groups[0]['lr']
        current_lr_D = optimizer_D.param_groups[0]['lr']

        # Сохранение по лучшей перцептивной метрике
        if val_lpips < best_lpips:
            best_lpips = val_lpips
            torch.save(netG.state_dict(), './checkpoints/best_esrgan.pth')
            print("Лучшая модель сохранена")

        # Tensorboard
        writer.add_scalar('Metric/Val_PSNR', val_psnr, epoch)
        writer.add_scalar('Metric/Val_SSIM', val_ssim, epoch)
        writer.add_scalar('Metric/Val_LPIPS', val_lpips, epoch)
        writer.add_scalar('Metric/Val_DISTS', val_dists, epoch)
        writer.add_scalar('LR/Generator', current_lr_G, epoch)
        writer.add_scalar('LR/Discriminator', current_lr_D, epoch)

    writer.close()
    print("Обучение завершено.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ESRGAN Training")
    
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Количество эпох обучения (по умолчанию: 10)')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Размер батча (по умолчанию: 32)')
    parser.add_argument('--lr_patch_size', type=int, default=64, 
                        help='Размер патча низкого разрешения LR (по умолчанию: 64)')
    parser.add_argument('--initial_lr', type=float, default=1e-4, 
                        help='Начальный Learning Rate для AdamW (по умолчанию: 1e-4)')
    parser.add_argument('--scale', type=int, default=2, 
                        help='Коэффициент масштабирования(по умолчанию: 2)')
    
    parsed_args = parser.parse_args()
    
    train(parsed_args)