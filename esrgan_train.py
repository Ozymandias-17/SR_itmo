import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.models import vgg19, VGG19_Weights
from torch.utils.tensorboard import SummaryWriter
import logging
from utils import AverageMeter, seed_everything

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
from piq import DISTS

from nn_arch.RRDBNet_arch import RRDBNet
from nn_arch.YUV_net_arch import YUV_Generator, rgb_to_yuv
from nn_arch.VGG_feat import VGGFeatureExtractor
from nn_arch.Discriminator import VGGDiscriminator
from dataloader import DF2KDataset


def train(args):
    seed_everything(42)

    logger = logging.getLogger('ESRGAN')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Train on device: {device}")

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
    
    # Инициализация моделей
    netD = VGGDiscriminator(in_nc=3, nf=nf_base).to(device)
    vgg_extractor = VGGFeatureExtractor().to(device)
    vgg_extractor.eval()
    for p in vgg_extractor.parameters():
        p.requires_grad = False
    
    # netG.load_state_dict(torch.load(f'./checkpoints/pretrained_gn_{args.model}.pth'))

    '''
    pretrained_path = f'./checkpoints/ESRGAN_PSNR_SRx4_DF2K_official-150ff491.pth'
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained generator weights from {pretrained_path}...")
        state_dict = torch.load(pretrained_path, map_location=device)
        
        if args.model == 'yuv':
            has_prefix = any(k.startswith('y_net.') for k in state_dict.keys())
            
            if not has_prefix:
                # префикс 'y_net.'
                new_state_dict = {}
                for k, v in state_dict.items():
                    # обработка слоев ввода/вывода, если они имели 3 канала (RGB), а стали 1 (Y)
                    if k == 'conv_first.weight' and v.shape[1] == 3:
                        print("Adapting conv_first from 3 channels to 1 channel (YUV)...")
                        v_y = 0.299 * v[:, 0:1, :, :] + 0.587 * v[:, 1:2, :, :] + 0.114 * v[:, 2:3, :, :]
                        v = v_y
                    if k == 'conv_last.weight' and v.shape[0] == 3:
                        print("Adapting conv_last from 3 channels to 1 channel (YUV)...")
                        v = v.mean(dim=0, keepdim=True)
                        
                    new_state_dict[f'y_net.{k}'] = v
                state_dict = new_state_dict
            
            netG.load_state_dict(state_dict, strict=False)
        else:
            netG.load_state_dict(state_dict, strict=True)
        
        print("Generator weights loaded successfully")
    
    else:
        print(f"Warning: Pretrained weights {pretrained_path} not found. Training from scratch.")

    '''

    # Оптимизаторы
    steps_per_epoch = len(train_loader)
    total_iters = args.epochs * steps_per_epoch
    optimizer_G = optim.AdamW(netG.parameters(), lr=args.lr)
    optimizer_D = optim.AdamW(netD.parameters(), lr=args.lr)
    scheduler_G = MultiStepLR(optimizer_G, milestones=[int(total_iters * 0.125), int(total_iters * 0.25), int(total_iters * 0.5), int(total_iters * 0.75)], gamma=0.5) # сохранение пропорций
    scheduler_D = MultiStepLR(optimizer_D, milestones=[int(total_iters * 0.125), int(total_iters * 0.25), int(total_iters * 0.5), int(total_iters * 0.75)], gamma=0.5)

    # Функции потерь
    criterion_pixel = nn.L1Loss().to(device)
    criterion_adv = nn.BCEWithLogitsLoss().to(device)
    criterion_perceptual = nn.L1Loss().to(device) # L1 между фичами VGG

    # Инициализация метрик
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_ = lpips.LPIPS(net='vgg').to(device).eval().requires_grad_(False) # LPIPS требует значения в диапазоне [-1, 1]

    dists_metric = DISTS().to(device)
    dists_metric.eval()
    for p in dists_metric.parameters():
        p.requires_grad = False

    lambda_perceptual = 1.0
    lambda_adv = 5e-3
    lambda_pixel = 1e-2

    # Логирование
    writer = SummaryWriter(log_dir=f'./logs/esrgan_{args.model}')
    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    timer_iter = AverageMeter()
    loss_meter = AverageMeter()

    best_lpips = float('inf')
    print_freq = 100
    current_iter = 0
    total_training_start_time = time.perf_counter()

    for epoch in range(args.epochs):
        netG.train()
        netD.train()

        train_loss = 0.0
        train_psnr, train_ssim = 0.0, 0.0
        total_fps = 0.0
        epoch_start_time = time.perf_counter()
        iter_start_time = time.time()
        
        for lr_imgs, hr_imgs in train_loader:
            step_start_time = time.perf_counter()
            step_iter_time = time.time()
            current_iter += 1

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
            scheduler_D.step()

            # генератор
            for p in netD.parameters():
                p.requires_grad = False

            optimizer_G.zero_grad()
            pred_fake_g = netD(sr_imgs)
            with torch.no_grad():
                pred_real_g = netD(hr_imgs)

            # Pixel Loss (L1), Perceptual Loss (VGG)
            if args.model == 'rgb':
                loss_pixel = criterion_pixel(sr_imgs, hr_imgs)

                real_features = vgg_extractor(hr_imgs).detach()
                fake_features = vgg_extractor(sr_imgs)
                loss_percep = criterion_perceptual(fake_features, real_features)

            elif args.model == 'rgb_lpips':
                loss_pixel = criterion_pixel(sr_imgs, hr_imgs)
                loss_percep = lpips_.forward(sr_imgs, hr_imgs, normalize=True).mean()

            elif args.model == 'yuv':
                yuv_sr = rgb_to_yuv(sr_imgs)
                yuv_hr = rgb_to_yuv(hr_imgs)
                y_sr = yuv_sr[:, 0:1, :, :]
                y_hr = yuv_hr[:, 0:1, :, :]

                loss_pixel = criterion_pixel(y_sr, y_hr)

                real_features = vgg_extractor(hr_imgs).detach()
                fake_features = vgg_extractor(sr_imgs)
                loss_percep = criterion_perceptual(fake_features, real_features)

            else:
                raise ValueError(f"Unknown model {args.model}")

            # Relativistic Adversarial Loss
            loss_g_real = criterion_adv(pred_real_g - torch.mean(pred_fake_g), torch.zeros_like(pred_real_g))
            loss_g_fake = criterion_adv(pred_fake_g - torch.mean(pred_real_g), torch.ones_like(pred_fake_g))
            loss_adv = (loss_g_real + loss_g_fake) / 2

            # Generator Total Loss
            loss_G = (lambda_pixel * loss_pixel) + (lambda_perceptual * loss_percep) + (lambda_adv * loss_adv)
            
            loss_G.backward()
            optimizer_G.step()
            scheduler_G.step()

            for p in netD.parameters():
                p.requires_grad = True

            train_loss += loss_G.item()

            step_time = time.perf_counter() - step_start_time
            current_fps = lr_imgs.size(0) / step_time
            total_fps += current_fps
            loss_meter.update(loss_G.item(), n=lr_imgs.size(0))
            iter_time = time.time() - step_iter_time
            timer_iter.update(iter_time)

            if current_iter % print_freq == 0:
                current_lr = optimizer_G.param_groups[0]['lr']
                log_message = (f"epoch: {epoch + 1:3d}, iter: {current_iter:7d}/{total_iters}, "
                f"lr: ({current_lr:.4e},), "
                f"time: {timer_iter.avg:.4f}, "
                f"loss_G: {loss_meter.avg:.4f}")
                logger.info(log_message)
                timer_iter.reset()
                loss_meter.reset()

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
        netD.eval()

        psnr_metric.reset()
        ssim_metric.reset()

        val_psnr, val_ssim, val_lpips, val_dists = 0.0, 0.0, 0.0, 0.0
        val_loss = 0.0

        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                sr_imgs = netG(lr_imgs)

                pred_fake_g = netD(sr_imgs)
                pred_real_g = netD(hr_imgs)

                if args.model == 'rgb':
                    loss_pixel = criterion_pixel(sr_imgs, hr_imgs)

                    real_features = vgg_extractor(hr_imgs).detach()
                    fake_features = vgg_extractor(sr_imgs)
                    loss_percep = criterion_perceptual(fake_features, real_features)

                elif args.model == 'rgb_lpips':
                    loss_pixel = criterion_pixel(sr_imgs, hr_imgs)
                    loss_percep = lpips_.forward(sr_imgs, hr_imgs, normalize=True).mean()

                elif args.model == 'yuv':
                    yuv_sr = rgb_to_yuv(sr_imgs)
                    yuv_hr = rgb_to_yuv(hr_imgs)
                    y_sr = yuv_sr[:, 0:1, :, :]
                    y_hr = yuv_hr[:, 0:1, :, :]

                    loss_pixel = criterion_pixel(y_sr, y_hr)

                    real_features = vgg_extractor(hr_imgs).detach()
                    fake_features = vgg_extractor(sr_imgs)
                    loss_percep = criterion_perceptual(fake_features, real_features)

                else:
                    raise ValueError(f"Unknown model {args.model}")

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

        logger.info(f"Validation: Epoch: {epoch + 1}, Val_PSNR: {val_psnr:.2f}, Val_SSIM: {val_ssim:.2f}, LPIPS: {val_lpips:.4f}, DISTS: {val_dists:.4f}")

        current_lr_G = optimizer_G.param_groups[0]['lr']
        current_lr_D = optimizer_D.param_groups[0]['lr']

        # Сохранение по лучшей перцептивной метрике
        if val_lpips < best_lpips:
            best_lpips = val_lpips
            torch.save(netG.state_dict(), f'./checkpoints/esrgan_{args.model}.pth')
            logger.info(f"Best ESRGAN model saved to ./checkpoints/esrgan_{args.model}.pth")

        # Tensorboard
        writer.add_scalars('Loss/Total', {'train': avg_train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Metric/PSNR', {'train': avg_train_psnr, 'val': val_psnr}, epoch)
        writer.add_scalars('Metric/SSIM', {'train': avg_train_ssim, 'val': val_ssim}, epoch)
        writer.add_scalar('Metric/Val_LPIPS', val_lpips, epoch)
        writer.add_scalar('Metric/Val_DISTS', val_dists, epoch)
        writer.add_scalar('LR/Generator', current_lr_G, epoch)
        writer.add_scalar('LR/Discriminator', current_lr_D, epoch)

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
    parser = argparse.ArgumentParser(description="ESRGAN Training")

    parser.add_argument('--model', type=str, default='rgb', 
                        help='Model: rgb, rgb_lpips, yuv (default: rgb)')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='default: 16')
    parser.add_argument('--lr_patch_size', type=int, default=16, 
                        help='Patch size of LR images (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Initial Learning Rate (default: 1e-4)')
    parser.add_argument('--scale', type=int, default=2, 
                        help='Scale parametr (default: 2)')
    
    parsed_args = parser.parse_args()

    log_dir = "train_logs"
    os.makedirs(log_dir, exist_ok=True)

    log_format = '%(asctime)s INFO: %(message)s'
    logging.basicConfig(level=logging.INFO,
                        format=log_format,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(os.path.join(log_dir, f"esrgan_{parsed_args.model}.log"), encoding='utf-8')])
    
    train(parsed_args)