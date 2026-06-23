import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lr_scheduler import CosineAnnealingRestartLR
from torch.utils.tensorboard import SummaryWriter
import logging
from utils import AverageMeter, seed_everything
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from nn_arch.RRDBNet_arch import RRDBNet
from nn_arch.YUV_net_arch import YUV_Generator, rgb_to_yuv
from dataloader import DF2KDataset

def train_psnr(args):
    seed_everything(42)
    
    logger = logging.getLogger('gn_net')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"PSNR Pre-training on device: {device}")

    if args.model == 'rgb':
        nf_base, gc_base = 64, 32
        netG = RRDBNet(in_nc=3, out_nc=3, nf=nf_base, nb=23, gc=gc_base).to(device)
        print(f"Model 'rgb': standard channels (nf={nf_base}, gc={gc_base}).")
    elif args.model == 'rgb_lpips':
        nf_base, gc_base = 64, 32
        netG = RRDBNet(in_nc=3, out_nc=3, nf=nf_base, nb=23, gc=gc_base).to(device)
        print(f"Model 'rgb_lpips': standard channels (nf={nf_base}, gc={gc_base}) with lpips loss.")
    elif args.model == 'yuv':
        nf_base, gc_base = 64, 32
        netG = YUV_Generator(in_nc=1, out_nc=1, nf=nf_base, nb=23, gc=gc_base).to(device)
        print(f"Model 'yuv': prior channel y (in_nc=1, out_nc=1, nf={nf_base}, gc={gc_base}).")
    else:
        raise ValueError(f"Unknown model {args.model}")
    
    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(DF2KDataset('train', scale=args.scale, lr_patch_size=args.lr_patch_size), batch_size=args.batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(DF2KDataset('val', scale=args.scale), batch_size=1, shuffle=False, generator=g)

    steps_per_epoch = len(train_loader)
    total_iters = args.epochs * steps_per_epoch
    period_length = total_iters // 4
    periods = [period_length, period_length, period_length, total_iters - (period_length * 3)]
    optimizer_G = optim.AdamW(netG.parameters(), lr=args.lr)
    scheduler_G = CosineAnnealingRestartLR(optimizer_G, periods=periods, restart_weights=[1.0, 1.0, 1.0, 1.0], eta_min=1e-7)

    # Pixel Loss
    criterion_pixel = nn.L1Loss().to(device)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    writer = SummaryWriter(log_dir=f'./logs/gn_base_{args.model}')
    os.makedirs('./checkpoints', exist_ok=True)

    timer_iter = AverageMeter()
    loss_meter = AverageMeter()

    best_psnr = 0.0
    print_freq = 100
    current_iter = 0
    total_fps = 0.0
    total_training_start_time = time.perf_counter()

    for epoch in range(args.epochs):
        netG.train()
        train_loss, train_psnr, train_ssim = 0.0, 0.0, 0.0

        iter_start_time = time.time()
        epoch_start_time = time.perf_counter()

        for lr_imgs, hr_imgs in train_loader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            step_start_time = time.perf_counter()
            step_iter_time = time.time()
            current_iter += 1

            optimizer_G.zero_grad()
            sr_imgs = netG(lr_imgs)

            # L1 loss
            if args.model == 'yuv':
                yuv_sr = rgb_to_yuv(sr_imgs)
                yuv_hr = rgb_to_yuv(hr_imgs)
                y_sr = yuv_sr[:, 0:1, :, :]
                y_hr = yuv_hr[:, 0:1, :, :]

                loss_pixel = criterion_pixel(y_sr, y_hr)

            else:
                loss_pixel = criterion_pixel(sr_imgs, hr_imgs)
            
            loss_pixel.backward()
            optimizer_G.step()
            scheduler_G.step()

            train_loss += loss_pixel.item()

            loss_meter.update(loss_pixel.item(), n=lr_imgs.size(0))
            iter_time = time.time() - step_iter_time
            timer_iter.update(iter_time)
            step_time = time.perf_counter() - step_start_time
            current_fps = lr_imgs.size(0) / step_time
            total_fps += current_fps

            if current_iter % print_freq == 0:
                current_lr = optimizer_G.param_groups[0]['lr']
                log_message = (f"epoch: {epoch + 1:3d}, iter: {current_iter:7d}/{total_iters}, "
                f"lr: ({current_lr:.4e},), "
                f"time: {timer_iter.avg:.4f}, "
                f"l_pix: {loss_meter.avg:.4f}")
                logger.info(log_message)
                timer_iter.reset()
                loss_meter.reset()

            with torch.no_grad():
                sr_clamped = torch.clamp(sr_imgs, 0, 1)
                train_psnr += psnr_metric(sr_clamped, hr_imgs).item()
                train_ssim += ssim_metric(sr_clamped, hr_imgs).item()

            iter_start_time = time.time()

        avg_fps = total_fps / len(train_loader)
        avg_train_loss = train_loss / len(train_loader)
        avg_train_psnr = train_psnr / len(train_loader)
        avg_train_ssim = train_ssim / len(train_loader)
        epoch_duration = time.perf_counter() - epoch_start_time

        writer.add_scalar('Performance/Avg_FPS', avg_fps, epoch)
        writer.add_scalar('Performance/Epoch_Duration_sec', epoch_duration, epoch)

        logger.info(f"Avg FPS = {avg_fps:.2f} | Epoch time = {epoch_duration:.2f} sec.")

        # Валидация
        netG.eval()
        psnr_metric.reset()
        ssim_metric.reset()
        val_psnr, val_ssim, val_loss = 0.0, 0.0, 0.0

        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                sr_imgs = netG(lr_imgs)

                if args.model == 'yuv':
                    yuv_sr = rgb_to_yuv(sr_imgs)
                    yuv_hr = rgb_to_yuv(hr_imgs)
                    y_sr = yuv_sr[:, 0:1, :, :]
                    y_hr = yuv_hr[:, 0:1, :, :]

                    val_loss += criterion_pixel(y_sr, y_hr)

                else:
                    val_loss += criterion_pixel(sr_imgs, hr_imgs)
                
                sr_clamped = torch.clamp(sr_imgs, 0, 1)
                val_psnr += psnr_metric(sr_clamped, hr_imgs).item()
                val_ssim += ssim_metric(sr_clamped, hr_imgs).item()

        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)

        logger.info(f"Validation: Epoch: {epoch + 1}, Val_PSNR: {val_psnr:.4f}, Val_SSIM: {val_ssim:.4f}, Loss: {val_loss:.4f}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(netG.state_dict(), f'./checkpoints/pretrained_gn_{args.model}.pth')
            logger.info(f"Best PSNR model saved to ./checkpoints/pretrained_gn_{args.model}.pth")

        # Tensorboard
        writer.add_scalars('Loss/Pixel', {'train': avg_train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Metric/PSNR', {'train': avg_train_psnr, 'val': val_psnr}, epoch)
        writer.add_scalars('Metric/SSIM', {'train': avg_train_ssim, 'val': val_ssim}, epoch)
        writer.add_scalar('LR/Generator', optimizer_G.param_groups[0]['lr'], epoch)

        psnr_metric.reset()
        ssim_metric.reset() 

    total_training_time = time.perf_counter() - total_training_start_time
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)

    writer.add_scalar('Performance/Total_Training_Time_sec', total_training_time, 0)

    writer.close()
    logger.info(f"Training is over, total time: {hours}h {minutes}m {seconds}s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PSNR Pre-Training")
    parser.add_argument('--model', type=str, default='rgb', help='Model: rgb, rgb_lpips, yuv')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr_patch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--scale', type=int, default=2)

    parsed_args = parser.parse_args()

    log_dir = "train_logs"
    os.makedirs(log_dir, exist_ok=True)

    log_format = '%(asctime)s INFO: %(message)s'
    logging.basicConfig(level=logging.INFO,
                        format=log_format,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(os.path.join(log_dir, f"gn_{parsed_args.model}.log"), encoding='utf-8')])

    train_psnr(parsed_args)